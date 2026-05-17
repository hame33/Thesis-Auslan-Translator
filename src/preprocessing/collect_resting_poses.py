#!/usr/bin/env python3
"""
collect_resting_poses.py
=========================
Webcam tool for collecting neutral/resting pose clips to augment the
non-detection class. Fixes the YES-bias problem by giving the model
examples of genuinely neutral pose (hands at rest, not signing).

Keyboard shortcuts:
    Space  — Record a resting pose clip (0.3–1.0s random duration)
    H      — Hold-to-record mode (hold Space, release to stop)
    D      — Discard last saved clip
    Q      — Quit

Usage:
    python src/tools/collect_resting_poses.py
    python src/tools/collect_resting_poses.py --count 50

The tool saves .mp4 clips to data/non_detection_clips/ and immediately
extracts MediaPipe features to data/non_detection_clips_features/.
It also appends rows to data/manifests/non_detection_manifest.xlsx so
the training pipeline picks them up automatically.

After collecting, re-run augmentation:
    python src/augmentation/augment_poses.py --features-dir data/non_detection_clips_features

Then retrain.
"""

import argparse
import random
import time
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# ── Repo root ─────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]

# ── DEFAULTS — edit once ──────────────────────────────────────────────────────
DEFAULTS = {
    "output_dir":    str(REPO_ROOT / "data" / "non_detection_clips"),
    "features_dir":  str(REPO_ROOT / "data" / "non_detection_clips_features"),
    "manifest":      str(REPO_ROOT / "data" / "manifests" / "non_detection_manifest.xlsx"),
    "min_dur":       0.3,
    "max_dur":       1.0,
    "fps":           30,
    "target_count":  50,   # suggested collection target
}
# ─────────────────────────────────────────────────────────────────────────────

# ── Feature layout (must match training) ─────────────────────────────────────
POSE_N, LHAND_N, RHAND_N = 33, 21, 21
POSE_DIMS  = POSE_N  * 4
LHAND_DIMS = LHAND_N * 3
RHAND_DIMS = RHAND_N * 3

mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def extract_features(frames_bgr: list, holistic) -> np.ndarray:
    """Extract MediaPipe Holistic features from a list of BGR frames."""
    feats = []
    for bgr in frames_bgr:
        rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        def pose_feat(lms):
            if lms is None: return np.zeros(POSE_DIMS, np.float32)
            return np.array([v for lm in lms.landmark
                             for v in (lm.x, lm.y, lm.z, lm.visibility)], np.float32)

        def hand_feat(lms):
            if lms is None: return np.zeros(LHAND_DIMS, np.float32)
            return np.array([v for lm in lms.landmark
                             for v in (lm.x, lm.y, lm.z)], np.float32)

        feats.append(np.concatenate([
            pose_feat(results.pose_landmarks),
            hand_feat(results.left_hand_landmarks),
            hand_feat(results.right_hand_landmarks),
        ]))
    return np.stack(feats)  # [T, 258]


def save_clip(frames_bgr: list, fps: int, out_path: Path):
    """Write frames to an mp4 file using ffmpeg via OpenCV."""
    h, w = frames_bgr[0].shape[:2]
    tmp = str(out_path).replace(".mp4", "_tmp.mp4")
    writer = cv2.VideoWriter(
        tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    for f in frames_bgr:
        writer.write(f)
    writer.release()
    # Re-encode with ffmpeg for browser compat
    subprocess.run(
        ["ffmpeg", "-y", "-i", tmp,
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
         "-movflags", "+faststart", "-an", "-loglevel", "error",
         str(out_path)],
        check=True
    )
    Path(tmp).unlink(missing_ok=True)


def append_manifest(manifest_path: Path, row: dict):
    if manifest_path.exists():
        try:
            df = pd.read_excel(manifest_path)
        except Exception:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_excel(manifest_path, index=False)


def get_existing_count(manifest_path: Path) -> int:
    """Count resting-pose clips already in manifest."""
    if not manifest_path.exists():
        return 0
    try:
        df = pd.read_excel(manifest_path)
        return len(df[df.get("source", df.get("glosses_blocked", pd.Series(dtype=str)))
                      .astype(str).str.contains("resting", case=False, na=False)])
    except Exception:
        return 0


def next_clip_name(output_dir: Path) -> str:
    """Generate a unique clip filename."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:20]
    return f"resting_{ts}_nondet_1.mp4"


def draw_ui(frame, state: dict):
    """Draw recording state overlay onto frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Status bar at top
    if state.get("waiting"):
        bar_color = (60, 40, 10)
    elif state["recording"]:
        bar_color = (0, 0, 200)
    else:
        bar_color = (0, 0, 100)
    cv2.rectangle(overlay, (0, 0), (w, 50), bar_color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    if state.get("waiting"):
        status = "⏸  PRESS SPACE TO START"
        col = (60, 200, 255)
    elif state["recording"]:
        status = "● REC"
        col = (0, 80, 255)
    else:
        status = "○ READY"
        col = (200, 200, 200)
    cv2.putText(frame, status, (12, 34),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, col, 2)

    count_str = f"Saved: {state['saved']}  |  Target: {state['target']}"
    cv2.putText(frame, count_str, (w - 340, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)

    # Instructions at bottom
    cv2.rectangle(frame, (0, h - 90), (w, h), (20, 20, 20), -1)
    lines = [
        "AUTO MODE  |  D: discard last  |  Q: quit",
        "Just hold natural resting poses — recording automatically",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (12, h - 60 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 150, 150), 1)

    # Recording / countdown progress bar
    if state["recording"] and state["record_target_frames"] > 0:
        pct = min(1.0, state["record_frames"] / state["record_target_frames"])
        cv2.rectangle(frame, (0, h - 94), (w, h - 90), (40, 40, 40), -1)
        cv2.rectangle(frame, (0, h - 94), (int(w * pct), h - 90), (0, 200, 100), -1)
    elif state.get("in_gap") and state.get("next_action_time") and state.get("now"):
        # Show countdown to next clip during gap
        remaining = max(0.0, state["next_action_time"] - state["now"])
        total_gap = 1.5   # approximate max gap for display
        pct = 1.0 - min(1.0, remaining / total_gap)
        cv2.rectangle(frame, (0, h - 94), (w, h - 90), (40, 40, 40), -1)
        cv2.rectangle(frame, (0, h - 94), (int(w * pct), h - 90), (80, 80, 200), -1)
        cv2.putText(frame, f"next in {remaining:.1f}s", (w - 130, h - 96),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (120, 120, 200), 1)

    # Last save confirmation
    if state.get("last_saved_msg") and (time.time() - state.get("last_saved_time", 0)) < 2.0:
        cv2.putText(frame, state["last_saved_msg"], (12, h - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 120), 2)

    return frame


def main():
    parser = argparse.ArgumentParser(description="Resting pose collector for non-detection class")
    parser.add_argument("--output-dir",   default=DEFAULTS["output_dir"])
    parser.add_argument("--features-dir", default=DEFAULTS["features_dir"])
    parser.add_argument("--manifest",     default=DEFAULTS["manifest"])
    parser.add_argument("--min-dur",      type=float, default=DEFAULTS["min_dur"])
    parser.add_argument("--max-dur",      type=float, default=DEFAULTS["max_dur"])
    parser.add_argument("--fps",          type=int,   default=DEFAULTS["fps"])
    parser.add_argument("--count",    type=int,   default=DEFAULTS["target_count"],
                        help="Target number of clips to collect (display only)")
    parser.add_argument("--gap-min",  type=float, default=0.5,
                        help="Minimum pause between clips in seconds (default 0.5)")
    parser.add_argument("--gap-max",  type=float, default=1.5,
                        help="Maximum pause between clips in seconds (default 1.5)")
    args = parser.parse_args()

    output_dir   = Path(args.output_dir);   output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = Path(args.features_dir); features_dir.mkdir(parents=True, exist_ok=True)
    manifest     = Path(args.manifest)

    print(f"\n{'='*55}")
    print(f"  Resting Pose Collector")
    print(f"{'='*55}")
    print(f"  Output:   {output_dir}")
    print(f"  Features: {features_dir}")
    print(f"  Manifest: {manifest}")
    print(f"  Clip duration: {args.min_dur}–{args.max_dur}s")
    print(f"  Target: {args.count} clips")
    print(f"{'='*55}")
    print(f"\n  Press SPACE to record.  Q to quit.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Could not open webcam.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS)) or args.fps

    state = {
        "recording":            False,
        "record_frames":        0,
        "record_target_frames": 0,
        "saved":                0,
        "target":               args.count,
        "buffer":               [],
        "last_saved_msg":       "",
        "last_saved_time":      0.0,
        "last_clip_path":       None,
        "last_feat_path":       None,
        "in_gap":               True,
        "waiting":              True,
        "next_action_time":     0.0,
        "now":                  0.0,
    }

    holistic = mp_holistic.Holistic(
        static_image_mode=False, model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # ── Auto-record mode ──────────────────────────────────────────────────────
    gap_min  = args.gap_min
    gap_max  = args.gap_max
    waiting  = True   # waiting for Space before any recording starts
    next_action_time = time.time()
    in_gap   = True

    print(f"  Press SPACE when ready to begin auto-recording.")
    print(f"  Vary your resting pose naturally. Press Q to stop.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = cv2.flip(frame, 1)
        now     = time.time()

        # Draw MediaPipe landmarks
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                display, results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                display, results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
            )
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                display, results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
            )

        # ── Auto state machine ─────────────────────────────────────────────
        if waiting:
            pass   # just show the waiting screen, do nothing
        elif in_gap:
            if now >= next_action_time:
                dur = random.uniform(args.min_dur, args.max_dur)
                state["record_target_frames"] = max(5, int(dur * actual_fps))
                state["record_frames"]        = 0
                state["buffer"]               = []
                state["recording"]            = True
                in_gap = False
        else:
            if state["recording"]:
                state["buffer"].append(frame.copy())
                state["record_frames"] += 1
                if state["record_frames"] >= state["record_target_frames"]:
                    state["recording"] = False
                    try:
                        _save_clip(state, output_dir, features_dir, manifest,
                                   actual_fps, args, holistic)
                    except KeyboardInterrupt:
                        print("\n  Interrupted during save — exiting cleanly.")
                        cap.release(); holistic.close(); cv2.destroyAllWindows()
                        print(f"  Saved {state['saved']} clips total.")
                        return
                    gap = random.uniform(gap_min, gap_max)
                    next_action_time = now + gap
                    in_gap = True

        state["waiting"]           = waiting
        state["in_gap"]            = in_gap
        state["next_action_time"]  = next_action_time
        state["now"]               = now

        display = draw_ui(display, state)
        cv2.imshow("Resting Pose Collector — Auto Mode", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord(' ') and waiting:
            waiting = False
            next_action_time = now + 1.0  # 1s grace after pressing space
            print("  Starting in 1 second…")
        elif key == ord('d'):
            # Manual discard of last clip
            if state["last_clip_path"] and Path(state["last_clip_path"]).exists():
                Path(state["last_clip_path"]).unlink()
                if state["last_feat_path"] and Path(state["last_feat_path"]).exists():
                    Path(state["last_feat_path"]).unlink()
                if manifest.exists():
                    df = pd.read_excel(manifest)
                    if len(df) > 0:
                        df = df.iloc[:-1]
                        df.to_excel(manifest, index=False)
                state["saved"]           = max(0, state["saved"] - 1)
                state["last_saved_msg"]  = "✗ Discarded"
                state["last_saved_time"] = now
                print("  Discarded last clip.")

    cap.release()
    holistic.close()
    cv2.destroyAllWindows()
    print(f"\n  Done. Collected {state['saved']} resting-pose clips.")
    print(f"  Next steps:")
    print(f"    1. Augment:")
    print(f"       python src/augmentation/augment_poses.py --features-dir {args.features_dir}")
    print(f"    2. Retrain:")
    print(f"       python src/v2g/train_video2gloss.py --config experiments/configs/exp_clean_with_nondet_v2.yaml")


def _save_clip(state, output_dir, features_dir, manifest, fps, args, holistic):
    """Save accumulated frames as a clip + extract features + update manifest."""
    frames = state["buffer"]
    if len(frames) < 3:
        print("  Too short, discarded.")
        return

    clip_name = next_clip_name(output_dir)
    clip_path = output_dir / clip_name
    feat_path = features_dir / (Path(clip_name).stem + ".npy")

    # Save video
    try:
        save_clip(frames, fps, clip_path)
    except Exception as e:
        print(f"  [error] Could not save clip: {e}")
        return

    # Extract features
    try:
        feat_array = extract_features(frames, holistic)
        np.save(feat_path, feat_array)
    except Exception as e:
        print(f"  [error] Feature extraction failed: {e}")
        clip_path.unlink(missing_ok=True)
        return

    dur = len(frames) / fps

    # Append to manifest
    row = {
        "source_clip":       "resting_pose_webcam",
        "start_sec":         0.0,
        "end_sec":           round(dur, 3),
        "duration_sec":      round(dur, 3),
        "glosses_blocked":   "resting",
        "n_blocked_windows": 0,
        "output_file":       clip_name,
        "status":            "accepted",
    }
    append_manifest(manifest, row)

    state["saved"]           += 1
    state["last_clip_path"]   = str(clip_path)
    state["last_feat_path"]   = str(feat_path)
    state["last_saved_msg"]   = f"✓ Saved {clip_name[:30]}…  ({state['saved']} total)"
    state["last_saved_time"]  = time.time()

    print(f"  ✓ [{state['saved']}] {clip_name}  shape={feat_array.shape}")


if __name__ == "__main__":
    main()