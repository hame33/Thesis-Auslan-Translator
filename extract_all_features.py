import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────
MANIFEST_CSV = "/Users/hamishdawson/Desktop/Thesis/Thesis-Auslan-Translator/gloss_clips_manifest.xlsx"
VIDEO_DIR = Path("/Users/hamishdawson/Desktop/Thesis/Thesis-Auslan-Translator/gloss_clips")
FEATURES_DIR = Path("/Users/hamishdawson/Desktop/Thesis/Thesis-Auslan-Translator/gloss_clips_features")
# ───────────────────────────────────────────────────────────────

FEATURES_DIR.mkdir(parents=True, exist_ok=True)
mp_holistic = mp.solutions.holistic


def flatten_pose_landmarks(pose_landmarks):
    if pose_landmarks is None:
        return np.zeros(33 * 4, dtype=np.float32)
    out = []
    for lm in pose_landmarks.landmark:
        out.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(out, dtype=np.float32)


def flatten_hand_landmarks(hand_landmarks):
    if hand_landmarks is None:
        return np.zeros(21 * 3, dtype=np.float32)
    out = []
    for lm in hand_landmarks.landmark:
        out.extend([lm.x, lm.y, lm.z])
    return np.array(out, dtype=np.float32)


def extract_sign_features(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    features = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            pose = flatten_pose_landmarks(results.pose_landmarks)
            lh   = flatten_hand_landmarks(results.left_hand_landmarks)
            rh   = flatten_hand_landmarks(results.right_hand_landmarks)
            features.append(np.concatenate([pose, lh, rh], axis=0))

    cap.release()

    if not features:
        raise ValueError(f"No frames processed for {video_path}")

    return np.stack(features)


def main():
    df = pd.read_excel(MANIFEST_CSV)
    df = df[df['status'] == 'saved'].reset_index(drop=True)

    total    = len(df)
    success  = 0
    failed   = 0
    skipped  = 0

    print(f"Total clips in manifest: {total}")
    print(f"Features dir: {FEATURES_DIR}")
    print(f"Video dir:    {VIDEO_DIR}\n")

    for i, (_, row) in enumerate(df.iterrows(), 1):
        clip_name  = row["output_file"]
        gloss_name = row["gloss"]
        video_path = VIDEO_DIR    / f"{clip_name}"
        save_path  = FEATURES_DIR / f"{Path(clip_name).stem}.npy"

        # Already done — skip
        if save_path.exists():
            skipped += 1
            continue

        if not video_path.exists():
            print(f"[{i}/{total}] Missing video: {video_path}")
            failed += 1
            continue

        try:
            features = extract_sign_features(video_path)
            np.save(save_path, features)
            print(f"[{i}/{total}] Saved {save_path.name}  shape={features.shape}")
            success += 1
        except Exception as e:
            print(f"[{i}/{total}] Failed on {clip_name}: {e}")
            failed += 1

        # Progress summary every 500 clips
        if i % 500 == 0:
            remaining = total - skipped - success - failed
            print(f"\n--- Progress: {success} done, {skipped} skipped, "
                  f"{failed} failed, ~{remaining} remaining ---\n")

    print("\n=== DONE ===")
    print(f"Successful:      {success}")
    print(f"Skipped existing:{skipped}")
    print(f"Failed:          {failed}")
    print(f"Total .npy files:{sum(1 for _ in FEATURES_DIR.glob('*.npy'))}")


if __name__ == "__main__":
    main()