#!/usr/bin/env python3
"""
sliding_window_demo.py
=======================
Real-time Auslan sign-language detector using a sliding window over
webcam frames. Runs a local Flask-SocketIO server; open the browser
at http://localhost:5050 to use.

Requirements (add to your env if missing):
    pip install flask flask-socketio opencv-python mediapipe

Edit the DEFAULTS block to point at your model checkpoint and label map.

Usage:
    python sliding_window_demo.py
    python sliding_window_demo.py --model results/.../best_model.pt \
                                   --label-map results/.../label_map.json \
                                   --config experiments/configs/exp_clean_with_nondet.yaml
"""

import argparse
import base64
import collections
import json
import sys
import threading
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import yaml
from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit

# ── Repo imports ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# ── DEFAULTS — edit these ─────────────────────────────────────────────────────
DEFAULTS = {
    "model":      str(sorted((REPO_ROOT / "results").glob("**/best_model.pt"))[-1])
                  if list((REPO_ROOT / "results").glob("**/best_model.pt")) else "",
    "label_map":  str(sorted((REPO_ROOT / "results").glob("**/label_map.json"))[-1])
                  if list((REPO_ROOT / "results").glob("**/label_map.json")) else "",
    "config":     str(REPO_ROOT / "experiments" / "configs" / "exp_clean_with_nondet.yaml"),
    "port":       5050,
    "threshold":  0.35,
    "window":     60,
    "stride":     10,
}
# ─────────────────────────────────────────────────────────────────────────────

# ── Feature layout (must match training) ─────────────────────────────────────
POSE_N, LHAND_N, RHAND_N = 33, 21, 21
POSE_DIMS  = POSE_N  * 4
LHAND_DIMS = LHAND_N * 3
RHAND_DIMS = RHAND_N * 3
TOTAL_DIMS = POSE_DIMS + LHAND_DIMS + RHAND_DIMS  # 258


# ── Model (copy of architecture from train_video2gloss.py) ───────────────────

class Video2GlossTransformer(nn.Module):
    def __init__(self, cfg, n_classes):
        super().__init__()
        m = cfg["model"]
        feat_dim, proj_dim = m["frame_feat_dim"], m["proj_dim"]
        n_heads, n_layers  = m["n_heads"], m["n_layers"]
        dropout, max_frames = m["dropout"], m["max_frames"]
        self.max_frames = max_frames
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, proj_dim), nn.LayerNorm(proj_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.pos_emb = nn.Embedding(max_frames + 1, proj_dim)
        enc = nn.TransformerEncoderLayer(
            d_model=proj_dim, nhead=n_heads,
            dim_feedforward=proj_dim * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.classifier  = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(proj_dim // 2, n_classes),
        )

    def forward(self, x, mask):
        B, T, _ = x.shape
        x = self.input_proj(x)
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1).clamp(max=self.max_frames)
        x = x + self.pos_emb(pos)
        x = self.transformer(x, src_key_padding_mask=mask)
        mf = (~mask).float().unsqueeze(-1)
        x  = (x * mf).sum(1) / mf.sum(1).clamp(min=1)
        return self.classifier(x)


# ── MediaPipe extractor ───────────────────────────────────────────────────────

class FrameExtractor:
    def __init__(self):
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False, model_complexity=1,
            smooth_landmarks=True, min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract(self, bgr_frame: np.ndarray):
        """Returns (feat_vector, landmarks_dict) where landmarks_dict has
        normalised x/y coords for pose, left_hand, right_hand."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        r   = self.holistic.process(rgb)

        def pose_feat(lms):
            if lms is None: return np.zeros(POSE_DIMS, np.float32)
            return np.array([v for lm in lms.landmark for v in (lm.x, lm.y, lm.z, lm.visibility)], np.float32)

        def hand_feat(lms):
            if lms is None: return np.zeros(LHAND_DIMS, np.float32)
            return np.array([v for lm in lms.landmark for v in (lm.x, lm.y, lm.z)], np.float32)

        feat = np.concatenate([pose_feat(r.pose_landmarks),
                                hand_feat(r.left_hand_landmarks),
                                hand_feat(r.right_hand_landmarks)])

        # Build landmark overlay data (only upper-body pose indices 0-24)
        # Pose: filter low-visibility landmarks (they have a real visibility score)
        def pose_lms(lms, indices):
            if lms is None: return []
            pts = lms.landmark
            return [[round(pts[i].x, 4), round(pts[i].y, 4)]
                    for i in indices if pts[i].visibility > 0.25]

        # Hands: always return all 21 points — visibility is not meaningful for hands
        def hand_lms(lms):
            if lms is None: return []
            return [[round(lm.x, 4), round(lm.y, 4)] for lm in lms.landmark]

        UPPER_POSE = list(range(25))  # head + upper body
        landmarks = {
            "pose":  pose_lms(r.pose_landmarks, UPPER_POSE),
            "lhand": hand_lms(r.left_hand_landmarks),
            "rhand": hand_lms(r.right_hand_landmarks),
        }
        return feat, landmarks

    def close(self):
        self.holistic.close()


# ── Classifier ────────────────────────────────────────────────────────────────

class SingleWindowClassifier:
    """One sliding window buffer — classify every `stride` frames."""
    def __init__(self, model, label_map, cfg, device, threshold, window, stride,
                 nd_threshold=None):
        self.model        = model
        self.label_map    = label_map
        self.idx2gloss    = {v: k for k, v in label_map.items()}
        self.device       = device
        self.threshold    = threshold
        self.nd_threshold = nd_threshold if nd_threshold is not None else threshold
        self.window       = window
        self.stride       = stride
        self.max_frames   = cfg["model"]["max_frames"]
        self.buffer       = collections.deque(maxlen=window)
        self.frame_count  = 0
        self.last_top     = None   # (gloss, confidence, nd_prob) or None
        self.full_probs   = {}     # {gloss: prob} updated each classify call

    def push(self, feat: np.ndarray):
        self.buffer.append(feat)
        self.frame_count += 1
        # Minimum fill: 5 frames for fast window, else window//8 — lets short signs like READY fire
        min_fill = max(5, self.window // 8)
        if self.frame_count % self.stride == 0 and len(self.buffer) >= min_fill:
            self._classify()
        return self.last_top

    def _classify(self):
        frames = np.stack(list(self.buffer)).astype(np.float32)
        T = len(frames)
        if T > self.max_frames:
            idx = np.linspace(0, T - 1, self.max_frames, dtype=int)
            frames = frames[idx]; T = self.max_frames
        t    = torch.from_numpy(frames).unsqueeze(0).to(self.device)
        mask = torch.zeros(1, T, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            probs = torch.sigmoid(self.model(t, mask)).squeeze(0).cpu().numpy()
        preds = sorted(
            [(self.idx2gloss[i], float(probs[i])) for i in range(len(probs))
             if probs[i] >= self.threshold and self.idx2gloss[i] != "NON_DETECTION"],
            key=lambda x: -x[1],
        )
        nd_prob = float(probs[self.label_map["NON_DETECTION"]]) if "NON_DETECTION" in self.label_map else 0.0
        # full_probs: dict {gloss: prob} for all classes including NON_DETECTION
        self.full_probs = {self.idx2gloss[i]: round(float(probs[i]), 3) for i in range(len(probs))}
        # If NON_DETECTION fires above its threshold, suppress all gloss predictions
        if nd_prob >= self.nd_threshold:
            self.last_top = (None, 0.0, round(nd_prob, 3))
        else:
            self.last_top = (preds[0][0], round(preds[0][1], 3), round(nd_prob, 3)) if preds else (None, 0.0, round(nd_prob, 3))

        # Console diagnostics every 20 strides so you can see raw probs in terminal
        if self.frame_count % (self.stride * 20) == 0:
            top3 = sorted(self.full_probs.items(), key=lambda x: -x[1])[:4]
            top3_str = "  ".join(f"{g}={v:.2f}" for g, v in top3)
            nd_str = f"NON_DET={nd_prob:.2f}(thr={self.nd_threshold:.2f})"
            result = self.last_top[0] if self.last_top and self.last_top[0] else "—"
            print(f"  [W{self.window}] top: {top3_str}  {nd_str}  → {result}")

    def reset(self):
        self.buffer.clear()
        self.frame_count = 0
        self.last_top = None


class DualWindowClassifier:
    """
    Two-stage sliding window detector.

    fast_window  (short)  — nominates a candidate gloss quickly.
    slow_window  (long)   — must agree before the gloss is confirmed.

    A gloss is committed to the transcript only when both windows
    independently predict the same non-NON_DETECTION gloss in the same
    stride cycle. This eliminates single-window jitter.
    """
    def __init__(self, model, label_map, cfg, device, threshold,
                 fast_window, slow_window, stride, nd_threshold=None):
        nd_t = nd_threshold if nd_threshold is not None else threshold
        self.fast = SingleWindowClassifier(model, label_map, cfg, device, threshold, fast_window, stride, nd_threshold=nd_t)
        self.slow = SingleWindowClassifier(model, label_map, cfg, device, threshold, slow_window, stride, nd_threshold=nd_t)
        self.threshold    = threshold
        self.nd_threshold = nd_t
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.stride      = stride
        # State
        self.candidate = None
        self.confirmed = None
        self.fast_conf = 0.0
        self.slow_conf = 0.0
        self.nondet    = 0.0

    def push(self, feat: np.ndarray) -> dict:
        fast_result = self.fast.push(feat)
        slow_result = self.slow.push(feat)

        fast_gloss = fast_result[0] if fast_result else None
        slow_gloss = slow_result[0] if slow_result else None
        nd_fast    = fast_result[2] if fast_result else 0.0
        nd_slow    = slow_result[2] if slow_result else 0.0
        self.nondet = max(nd_fast, nd_slow)

        self.candidate = fast_gloss

        # Confirmation rule:
        #   Fast window predicts a real non-YES gloss  → CANDIDATE
        #   Slow window may say YES or None (resting)  → IGNORE (background class)
        #   Slow window says a DIFFERENT real gloss with high confidence → BLOCK
        #   Otherwise → CONFIRM from fast window alone
        #
        # This fixes the core problem: the slow window averages over several
        # seconds including resting frames and almost always outputs YES or
        # NON_DETECTION, which should never block a confident fast prediction.
        BACKGROUND = {None, "YES"}

        if fast_gloss and fast_gloss != "YES":
            slow_contradicts = (
                slow_gloss is not None
                and slow_gloss not in BACKGROUND
                and slow_gloss != fast_gloss
                and (slow_result[1] if slow_result else 0.0) > 0.6
            )
            if not slow_contradicts:
                self.confirmed = fast_gloss
                self.fast_conf = fast_result[1]
                self.slow_conf = slow_result[1] if slow_gloss == fast_gloss else 0.0
            else:
                # Slow window strongly disagrees with a real different gloss
                self.confirmed = None
                self.fast_conf = fast_result[1]
                self.slow_conf = slow_result[1] if slow_result else 0.0
        else:
            # Fast says YES (resting) or None — nothing to confirm
            self.confirmed = None
            self.fast_conf = fast_result[1] if fast_result else 0.0
            self.slow_conf = slow_result[1] if slow_result else 0.0

        return {
            "candidate":   self.candidate,
            "confirmed":   self.confirmed,
            "fast_conf":   round(self.fast_conf, 3),
            "slow_conf":   round(self.slow_conf, 3),
            "nondet":      round(self.nondet, 3),
            "fast_probs":  self.fast.full_probs,
            "slow_probs":  self.slow.full_probs,
            "fast_window": self.fast_window,
            "slow_window": self.slow_window,
            "n_fast":      len(self.fast.buffer),
            "n_slow":      len(self.slow.buffer),
        }

    def update_params(self, fast_window=None, slow_window=None, stride=None,
                      threshold=None, nd_threshold=None):
        if stride is not None:
            self.stride = stride
            self.fast.stride = stride
            self.slow.stride = stride
        if threshold is not None:
            self.threshold = threshold
            self.fast.threshold = threshold
            self.slow.threshold = threshold
        if nd_threshold is not None:
            self.nd_threshold = nd_threshold
            self.fast.nd_threshold = nd_threshold
            self.slow.nd_threshold = nd_threshold
        if fast_window is not None:
            self.fast_window = fast_window
            self.fast.window = fast_window
            self.fast.buffer = collections.deque(maxlen=fast_window)
        if slow_window is not None:
            self.slow_window = slow_window
            self.slow.window = slow_window
            self.slow.buffer = collections.deque(maxlen=slow_window)



# ── Flask / SocketIO app ──────────────────────────────────────────────────────

app       = Flask(__name__)
socketio  = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Globals set at startup
G = {}   # model, classifier, extractor, transcript


@app.route("/")
def index():
    return render_template_string(HTML)


@socketio.on("frame")
def handle_frame(data):
    """Receive a base64 JPEG frame from the browser, extract features, classify."""
    try:
        img_bytes = base64.b64decode(data["image"].split(",")[1])
        nparr     = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        feat, landmarks = G["extractor"].extract(frame)
        result = G["classifier"].push(feat)

        # Only commit to transcript when BOTH windows agree (confirmed)
        confirmed = result["confirmed"]
        if confirmed:
            transcript = G["transcript"]
            if not transcript or transcript[-1]["gloss"] != confirmed:
                transcript.append({
                    "gloss": confirmed,
                    "conf":  max(result["fast_conf"], result["slow_conf"]),
                })
                if len(transcript) > 50:
                    transcript.pop(0)

        emit("prediction", {
            "candidate":   result["candidate"],
            "confirmed":   result["confirmed"],
            "fast_conf":   result["fast_conf"],
            "slow_conf":   result["slow_conf"],
            "nondet":      result["nondet"],
            "fast_probs":  result["fast_probs"],
            "slow_probs":  result["slow_probs"],
            "fast_window": result["fast_window"],
            "slow_window": result["slow_window"],
            "n_fast":      result["n_fast"],
            "n_slow":      result["n_slow"],
            "transcript":  G["transcript"][-10:],
            "landmarks":   landmarks,
        })

    except Exception as e:
        emit("error", {"msg": str(e)})


@socketio.on("update_params")
def handle_params(data):
    G["classifier"].update_params(
        fast_window=data.get("fast_window"),
        slow_window=data.get("slow_window"),
        stride=data.get("stride"),
        threshold=data.get("threshold"),
        nd_threshold=data.get("nd_threshold"),
    )
    emit("params_updated", {
        "fast_window":  G["classifier"].fast_window,
        "slow_window":  G["classifier"].slow_window,
        "stride":       G["classifier"].stride,
        "threshold":    G["classifier"].threshold,
        "nd_threshold": G["classifier"].nd_threshold,
    })


@socketio.on("clear_transcript")
def handle_clear():
    G["transcript"].clear()
    emit("transcript_cleared")


# ── HTML / CSS / JS UI ────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Auslan Live Detector</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

  :root {
    --bg:      #0a0a0f;
    --panel:   #111118;
    --border:  #1e1e2e;
    --accent:  #00ffc8;
    --accent2: #7b5ea7;
    --warn:    #ff6b6b;
    --yellow:  #f6c343;
    --text:    #e8e8f0;
    --muted:   #55556a;
    --radius:  12px;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Syne', sans-serif;
    min-height: 100vh;
    display: grid;
    grid-template-rows: auto 1fr auto;
    overflow: hidden;
  }

  /* ── Header ── */
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 28px;
    border-bottom: 1px solid var(--border);
    background: var(--panel);
  }
  .logo {
    font-size: 1.1rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent);
  }
  .logo span { color: var(--text); }
  .status-pill {
    display: flex; align-items: center; gap: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    padding: 5px 14px;
    border-radius: 20px;
    border: 1px solid var(--border);
    background: var(--bg);
    color: var(--muted);
    transition: all 0.3s;
  }
  .status-pill.live { border-color: var(--accent); color: var(--accent); }
  .status-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--muted); transition: background 0.3s;
  }
  .status-pill.live .status-dot {
    background: var(--accent);
    box-shadow: 0 0 6px var(--accent);
    animation: pulse 1.4s ease-in-out infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

  /* ── Main layout ── */
  main {
    display: grid;
    grid-template-columns: 1fr 380px;
    gap: 0;
    overflow: hidden;
    height: calc(100vh - 57px - 52px);
  }

  /* ── Video panel ── */
  .video-wrap {
    position: relative;
    background: #000;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
  }
  #webcam {
    width: 100%; height: 100%;
    object-fit: contain;
    transform: scaleX(-1);  /* mirror */
  }

  /* Overlay badge */
  .overlay-badge {
    position: absolute;
    bottom: 28px;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    pointer-events: none;
  }
  .gloss-display {
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--accent);
    text-shadow: 0 0 40px rgba(0,255,200,0.5);
    transition: all 0.2s;
    min-height: 4rem;
    text-align: center;
  }
  .gloss-display.empty { color: rgba(255,255,255,0.12); font-size: 1rem; letter-spacing:0.2em; }
  .conf-bar-wrap {
    width: 220px;
    height: 4px;
    background: rgba(255,255,255,0.1);
    border-radius: 2px;
    overflow: hidden;
  }
  .conf-bar {
    height: 100%;
    background: var(--accent);
    border-radius: 2px;
    transition: width 0.2s ease;
    box-shadow: 0 0 8px var(--accent);
  }
  .conf-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: rgba(255,255,255,0.5);
  }

  /* NON_DETECTION meter */
  .nondet-meter {
    position: absolute;
    top: 16px;
    right: 16px;
    background: rgba(10,10,15,0.75);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 14px;
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    backdrop-filter: blur(8px);
  }
  .nondet-meter span { color: var(--warn); }

  /* ── Right panel ── */
  .right-panel {
    display: grid;
    grid-template-rows: auto 1fr auto;
    border-left: 1px solid var(--border);
    background: var(--panel);
    overflow: hidden;
  }

  /* Controls */
  .controls {
    padding: 18px 20px;
    border-bottom: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    gap: 14px;
  }
  .ctrl-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
  }
  .ctrl-label .val {
    color: var(--accent);
    font-family: 'Space Mono', monospace;
  }
  input[type=range] {
    -webkit-appearance: none;
    width: 100%;
    height: 3px;
    background: var(--border);
    border-radius: 2px;
    outline: none;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 14px; height: 14px;
    border-radius: 50%;
    background: var(--accent);
    cursor: pointer;
    box-shadow: 0 0 6px var(--accent);
  }

  /* Transcript */
  .transcript-header {
    padding: 14px 20px 10px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border);
  }
  .btn-clear {
    background: none;
    border: 1px solid var(--border);
    color: var(--muted);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.7rem;
    font-family: 'Space Mono', monospace;
    cursor: pointer;
    transition: all 0.15s;
  }
  .btn-clear:hover { border-color: var(--warn); color: var(--warn); }

  .transcript-body {
    overflow-y: auto;
    padding: 14px 20px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .t-entry {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 12px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    animation: slide-in 0.2s ease;
  }
  @keyframes slide-in { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:none; } }
  .t-gloss {
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    color: var(--accent);
  }
  .t-conf {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
  }
  .transcript-empty {
    color: var(--muted);
    font-size: 0.8rem;
    text-align: center;
    margin-top: 32px;
    line-height: 1.6;
  }

  /* Start button */
  .bottom-bar {
    padding: 12px 20px;
    border-top: 1px solid var(--border);
  }
  .btn-start {
    width: 100%;
    padding: 13px;
    background: var(--accent);
    color: #000;
    border: none;
    border-radius: var(--radius);
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    cursor: pointer;
    transition: opacity 0.15s, transform 0.1s;
  }
  .btn-start:hover { opacity: 0.88; }
  .btn-start:active { transform: scale(0.98); }
  .btn-start.running {
    background: var(--border);
    color: var(--warn);
    border: 1px solid var(--warn);
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  /* ── Tab bar ── */
  .tab-bar {
    display: flex;
    border-bottom: 1px solid var(--border);
    background: var(--surface2, #16161f);
    flex-shrink: 0;
  }
  .tab-btn {
    flex: 1;
    padding: 10px 0;
    background: none;
    border: none;
    border-bottom: 2px solid transparent;
    color: var(--muted);
    font-family: 'Syne', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    cursor: pointer;
    transition: all 0.15s;
  }
  .tab-btn:hover { color: var(--text); }
  .tab-btn.active { color: var(--accent); border-bottom-color: var(--accent); }

  .tab-pane {
    display: flex;
    flex-direction: column;
    flex: 1;
    overflow: hidden;
  }

  /* ── Debug panel ── */


  .debug-section-title {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .debug-section-title span {
    font-family: 'Space Mono', monospace;
    color: var(--accent);
    font-size: 0.68rem;
  }

  /* Probability bars */
  .prob-rows { display: flex; flex-direction: column; gap: 5px; }
  .prob-row {
    display: grid;
    grid-template-columns: 80px 1fr 1fr 52px 52px;
    align-items: center;
    gap: 6px;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
  }
  .prob-label { color: var(--text); font-weight: 600; letter-spacing: 0.04em; }
  .prob-label.nondet { color: var(--warn); }
  .bar-wrap {
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
  }
  .bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.15s ease;
  }
  .bar-fill.fast { background: #5b8cff; }
  .bar-fill.slow { background: var(--accent2); }
  .bar-fill.nondet-fast { background: var(--warn); }
  .bar-fill.nondet-slow { background: #ff9999; }
  .prob-val { color: var(--muted); text-align: right; }
  .prob-val.hot { color: var(--accent); font-weight: 700; }

  /* Window fill meters */
  .window-meters { display: flex; flex-direction: column; gap: 16px; }
  .meter-group { display: flex; flex-direction: column; gap: 6px; }
  .meter-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--muted);
    display: flex;
    justify-content: space-between;
  }
  .meter-label .count { font-family: 'Space Mono', monospace; color: var(--text); }
  .meter-bar-wrap {
    height: 10px;
    background: var(--border);
    border-radius: 5px;
    overflow: hidden;
  }
  .meter-bar-fill {
    height: 100%;
    border-radius: 5px;
    transition: width 0.1s ease;
  }
  .meter-bar-fill.fast { background: #5b8cff; }
  .meter-bar-fill.slow { background: var(--accent2); }

  .state-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    margin-top: 10px;
  }
  .state-badge.confirmed { background: rgba(0,255,200,0.15); color: var(--accent); border: 1px solid var(--accent); }
  .state-badge.nominated { background: rgba(246,195,67,0.15); color: var(--yellow); border: 1px solid var(--yellow); }
  .state-badge.silent { background: var(--border); color: var(--muted); border: 1px solid var(--border); }

  /* History log */
  .history-log {
    display: flex;
    flex-direction: column;
    gap: 3px;
    max-height: 260px;
    overflow-y: auto;
  }
  .log-row {
    display: grid;
    grid-template-columns: 36px 1fr 1fr 60px;
    gap: 8px;
    padding: 4px 8px;
    border-radius: 4px;
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    color: var(--muted);
    border: 1px solid transparent;
    animation: slide-in 0.15s ease;
  }
  .log-row.confirmed-row { background: rgba(0,255,200,0.05); border-color: rgba(0,255,200,0.2); color: var(--text); }
  .log-row .log-fast { color: #5b8cff; }
  .log-row .log-slow { color: var(--accent2); }
  .log-row .log-tick { color: var(--muted); }
  .log-row .log-badge { font-weight: 700; }
  .log-row.confirmed-row .log-badge { color: var(--accent); }
</style>
</head>
<body>

<header>
  <div class="logo">Auslan <span>Detector</span></div>
  <div class="status-pill" id="status-pill">
    <div class="status-dot"></div>
    <span id="status-text">OFFLINE</span>
  </div>
</header>

<main>
  <!-- Video -->
  <div class="video-wrap">
    <video id="webcam" autoplay playsinline muted></video>
    <!-- Landmark overlay canvas (debug mode only) -->
    <canvas id="lm-canvas" style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;display:none"></canvas>
    <div class="overlay-badge">
      <div class="gloss-display empty" id="gloss-display">WAITING</div>
      <div class="conf-bar-wrap">
        <div class="conf-bar" id="conf-bar" style="width:0%"></div>
      </div>
      <div class="conf-label" id="conf-label">—</div>
    </div>
    <div class="nondet-meter">
      NON-DET &nbsp;<span id="nondet-val">—</span>
    </div>
  </div>

  <!-- Right panel -->
  <div class="right-panel">

    <!-- Tab bar -->
    <div class="tab-bar">
      <button class="tab-btn active" id="tab-controls" onclick="switchTab('controls')">Controls</button>
      <button class="tab-btn" id="tab-debug" onclick="switchTab('debug')">Debug</button>
    </div>

    <!-- Controls tab -->
    <div class="tab-pane" id="pane-controls">
      <div class="controls">
        <div>
          <div class="ctrl-label">Fast window <span class="val" id="fast-window-val">30 frames</span></div>
          <input type="range" id="sl-fast-window" min="10" max="90" step="5" value="30">
        </div>
        <div>
          <div class="ctrl-label">Slow window <span class="val" id="slow-window-val">90 frames</span></div>
          <input type="range" id="sl-slow-window" min="30" max="180" step="5" value="90">
        </div>
        <div>
          <div class="ctrl-label">Stride <span class="val" id="stride-val">10 frames</span></div>
          <input type="range" id="sl-stride" min="1" max="30" step="1" value="10">
        </div>
        <div>
          <div class="ctrl-label">Gloss threshold <span class="val" id="thresh-val">0.25</span></div>
          <input type="range" id="sl-thresh" min="0.10" max="0.90" step="0.05" value="0.25">
        </div>
        <div>
          <div class="ctrl-label">NON-DET threshold <span class="val" id="ndthresh-val">0.55</span></div>
          <input type="range" id="sl-ndthresh" min="0.10" max="0.95" step="0.05" value="0.55">
        </div>

      </div>
      <div>
        <div class="transcript-header">
          Running Transcript
          <button class="btn-clear" onclick="clearTranscript()">Clear</button>
        </div>
        <div class="transcript-body" id="transcript-body">
          <div class="transcript-empty" id="transcript-empty">
            Start the camera to begin<br>detecting Auslan signs.
          </div>
        </div>
      </div>
    </div>

    <!-- Debug tab -->
    <div class="tab-pane" id="pane-debug" style="display:none;gap:16px;overflow-y:auto;padding:14px 16px;">

      <!-- Prob bars -->
      <div>
        <div class="debug-section-title">
          Class probabilities
          <span><span style="color:#5b8cff">■</span> fast &nbsp;<span style="color:var(--accent2)">■</span> slow</span>
        </div>
        <div class="prob-rows" id="prob-rows"></div>
      </div>

      <!-- Window fill + state -->
      <div>
        <div class="debug-section-title">Window fill</div>
        <div class="window-meters">
          <div class="meter-group">
            <div class="meter-label">Fast <span class="count"><span id="n-fast">0</span>/<span id="cap-fast">30</span></span></div>
            <div class="meter-bar-wrap"><div class="meter-bar-fill fast" id="meter-fast" style="width:0%"></div></div>
          </div>
          <div class="meter-group">
            <div class="meter-label">Slow <span class="count"><span id="n-slow">0</span>/<span id="cap-slow">90</span></span></div>
            <div class="meter-bar-wrap"><div class="meter-bar-fill slow" id="meter-slow" style="width:0%"></div></div>
          </div>
          <div id="state-badge-wrap" style="margin-top:8px"></div>
          <div style="margin-top:12px;font-family:'Space Mono',monospace;font-size:0.66rem;line-height:1.9;color:var(--muted)">
            <div>Fast: <span id="dbg-fast-top" style="color:#5b8cff">—</span></div>
            <div>Slow: <span id="dbg-slow-top" style="color:var(--accent2)">—</span></div>
            <div>NON-DET: <span id="dbg-nondet" style="color:var(--warn)">—</span></div>
          </div>
        </div>
      </div>

      <!-- Landmark toggle -->
      <div>
        <div class="debug-section-title">Pose overlay</div>
        <label style="display:flex;align-items:center;gap:10px;cursor:pointer;font-size:0.78rem;color:var(--muted)">
          <input type="checkbox" id="lm-toggle" onchange="toggleLandmarks(this.checked)"
            style="accent-color:var(--accent);width:16px;height:16px">
          Show MediaPipe landmarks on video
        </label>
      </div>

      <!-- Stride history -->
      <div style="flex:1;min-height:0">
        <div class="debug-section-title">
          Stride history
          <span style="cursor:pointer;color:var(--warn);font-size:0.65rem" onclick="clearHistory()">clear</span>
        </div>
        <div class="history-log" id="history-log" style="max-height:200px">
          <div style="color:var(--muted);font-size:0.7rem;text-align:center;margin-top:16px">Waiting for first stride…</div>
        </div>
      </div>

    </div>

    <div class="bottom-bar">
      <button class="btn-start" id="btn-start" onclick="toggleCamera()">
        Start Camera
      </button>
    </div>
  </div>
</main>

<script>
const socket = io();
let running  = false;
let stream   = null;
let sendLoop = null;

const video   = document.getElementById('webcam');
const canvas  = document.createElement('canvas');
const ctx     = canvas.getContext('2d');

// ── Controls ──────────────────────────────────────────────────────────────────
function setupSlider(id, valId, suffix, socketKey) {
  const sl = document.getElementById(id);
  const vl = document.getElementById(valId);
  vl.textContent = sl.value + suffix;
  sl.addEventListener('input', () => {
    vl.textContent = sl.value + suffix;
  });
  sl.addEventListener('change', () => {
    const params = {};
    params[socketKey] = parseFloat(sl.value);
    socket.emit('update_params', params);
  });
}
setupSlider('sl-fast-window', 'fast-window-val', ' frames', 'fast_window');
setupSlider('sl-slow-window', 'slow-window-val', ' frames', 'slow_window');
setupSlider('sl-stride',      'stride-val',      ' frames', 'stride');
setupSlider('sl-thresh',      'thresh-val',      '',        'threshold');
setupSlider('sl-ndthresh',    'ndthresh-val',    '',        'nd_threshold');


// ── Camera ────────────────────────────────────────────────────────────────────
async function toggleCamera() {
  if (running) { stopCamera(); return; }

  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width:640, height:480 }, audio: false });
    video.srcObject = stream;
    running = true;
    setStatus(true);

    const btn = document.getElementById('btn-start');
    btn.textContent = 'Stop Camera';
    btn.classList.add('running');

    // Send a frame every ~100ms (≈10fps to server; MediaPipe runs at native fps)
    sendLoop = setInterval(sendFrame, 100);

  } catch (err) {
    alert('Camera access denied: ' + err.message);
  }
}

function stopCamera() {
  if (stream) stream.getTracks().forEach(t => t.stop());
  clearInterval(sendLoop);
  running = false;
  setStatus(false);
  const btn = document.getElementById('btn-start');
  btn.textContent = 'Start Camera';
  btn.classList.remove('running');
  resetDisplay();
}

function sendFrame() {
  if (!video.videoWidth) return;
  canvas.width  = 320;
  canvas.height = 240;
  // Do NOT flip — training was done on natural unflipped videos.
  // The video display is CSS-mirrored for the user, but we send natural frames
  // to MediaPipe so coordinates match the training distribution.
  ctx.drawImage(video, 0, 0, 320, 240);
  const b64 = canvas.toDataURL('image/jpeg', 0.7);
  socket.emit('frame', { image: b64 });
}

// ── Socket events ─────────────────────────────────────────────────────────────
let strideCount = 0;
const historyRows = [];
const MAX_HISTORY = 80;
let showLandmarks = false;
let activeTab = 'controls';

socket.on('prediction', data => {
  updateOverlay(data.candidate, data.confirmed, data.fast_conf, data.slow_conf);
  document.getElementById('nondet-val').textContent =
    data.nondet > 0 ? (data.nondet * 100).toFixed(0) + '%' : '—';
  updateTranscript(data.transcript);
  updateDebug(data);  // always update so data is ready when tab is opened
  if (showLandmarks && data.landmarks) drawLandmarks(data.landmarks);
});

// ── Tab switching ──────────────────────────────────────────────────────────────
function switchTab(tab) {
  activeTab = tab;
  // Controls pane
  const cp = document.getElementById('pane-controls');
  cp.style.display = tab === 'controls' ? '' : 'none';
  // Debug pane — use flex so the column layout works
  const dp = document.getElementById('pane-debug');
  dp.style.display = tab === 'debug' ? 'flex' : 'none';
  dp.style.flexDirection = 'column';
  document.getElementById('tab-controls').classList.toggle('active', tab === 'controls');
  document.getElementById('tab-debug').classList.toggle('active', tab === 'debug');
  // Show/hide landmark canvas based on checkbox state when switching tabs
  if (tab === 'debug') {
    const checked = document.getElementById('lm-toggle');
    if (checked && checked.checked) document.getElementById('lm-canvas').style.display = 'block';
  }
}

// ── Landmark overlay ───────────────────────────────────────────────────────────
function toggleLandmarks(on) {
  showLandmarks = on;
  const c = document.getElementById('lm-canvas');
  c.style.display = on ? 'block' : 'none';
  if (!on) {
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, c.width, c.height);
  }
}

function drawLandmarks(lm) {
  const c   = document.getElementById('lm-canvas');
  const vid = document.getElementById('webcam');
  if (!vid.videoWidth) return;

  // Size canvas to the container (not native video res)
  const vw = vid.clientWidth  || vid.offsetWidth;
  const vh = vid.clientHeight || vid.offsetHeight;
  if (!vw || !vh) return;
  c.width  = vw;
  c.height = vh;

  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, vw, vh);

  // Compute letterbox offsets — video uses object-fit:contain so it may have
  // black bars. MediaPipe coords are relative to the actual video frame, not
  // the container, so we need to offset into the rendered area.
  const nw = vid.videoWidth, nh = vid.videoHeight;
  // object-fit:contain — scale to fit, letterboxed (may have black bars)
  const scale = Math.min(vw / nw, vh / nh);
  const dw = nw * scale, dh = nh * scale;
  const ox = (vw - dw) / 2, oy = (vh - dh) / 2;

  // Video is CSS scaleX(-1), MediaPipe coords are natural — flip x for display
  function px(pt) { return [ox + (1 - pt[0]) * dw, oy + pt[1] * dh]; }

  function dot(pt, color, r) {
    const [x, y] = px(pt);
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }
  function line(pts, color) {
    if (pts.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const [x0, y0] = px(pts[0]);
    ctx.moveTo(x0, y0);
    for (let i = 1; i < pts.length; i++) {
      const [xi, yi] = px(pts[i]);
      ctx.lineTo(xi, yi);
    }
    ctx.stroke();
  }

  // Pose dots (upper body, white)
  (lm.pose || []).forEach(pt => dot(pt, 'rgba(255,255,255,0.7)', 4));

  // Hand connections + dots
  const HAND_CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],[11,12],
    [0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20],
    [5,9],[9,13],[13,17]
  ];
  function drawHand(pts, color) {
    if (!pts || pts.length === 0) return;
    HAND_CONNECTIONS.forEach(([a, b]) => {
      if (pts[a] && pts[b]) line([pts[a], pts[b]], color + '99');
    });
    pts.forEach(pt => dot(pt, color, 3));
  }
  drawHand(lm.lhand, '#5b8cff');  // left hand blue
  drawHand(lm.rhand, '#00ffc8');  // right hand green
}

function updateDebug(data) {
  // Always accumulate history; only paint DOM when debug tab is active
  strideCount++;
  historyRows.unshift({
    tick:      strideCount,
    fast:      data.candidate,
    fastConf:  data.fast_conf,
    slow:      data.confirmed ? data.confirmed : null,
    slowConf:  data.slow_conf,
    confirmed: !!data.confirmed,
  });
  if (historyRows.length > MAX_HISTORY) historyRows.pop();

  if (activeTab !== 'debug') return;  // skip DOM paint when tab hidden

  // ── Probability bars ──
  const allGlosses = data.fast_probs ? Object.keys(data.fast_probs).sort() : [];
  const probRows = document.getElementById('prob-rows');
  probRows.innerHTML = allGlosses.map(g => {
    const fp = data.fast_probs ? (data.fast_probs[g] || 0) : 0;
    const sp = data.slow_probs ? (data.slow_probs[g] || 0) : 0;
    const isNd = g === 'NON_DETECTION';
    const isHot = g === data.confirmed;
    const fpPct = (fp * 100).toFixed(0);
    const spPct = (sp * 100).toFixed(0);
    return `<div class="prob-row">
      <span class="prob-label${isNd ? ' nondet' : ''}">${g.replace('_',' ')}</span>
      <div class="bar-wrap"><div class="bar-fill ${isNd ? 'nondet-fast' : 'fast'}" style="width:${fpPct}%"></div></div>
      <div class="bar-wrap"><div class="bar-fill ${isNd ? 'nondet-slow' : 'slow'}" style="width:${spPct}%"></div></div>
      <span class="prob-val${isHot ? ' hot' : ''}">${fpPct}%</span>
      <span class="prob-val${isHot ? ' hot' : ''}">${spPct}%</span>
    </div>`;
  }).join('');

  // ── Window fill meters ──
  const nFast = data.n_fast || 0;
  const nSlow = data.n_slow || 0;
  const capFast = data.fast_window || 30;
  const capSlow = data.slow_window || 90;
  document.getElementById('n-fast').textContent = nFast;
  document.getElementById('n-slow').textContent = nSlow;
  document.getElementById('cap-fast').textContent = capFast;
  document.getElementById('cap-slow').textContent = capSlow;
  document.getElementById('meter-fast').style.width = (nFast/capFast*100) + '%';
  document.getElementById('meter-slow').style.width = (nSlow/capSlow*100) + '%';

  // ── State badge + streak progress ──
  const badgeWrap = document.getElementById('state-badge-wrap');
  if (data.confirmed) {
    badgeWrap.innerHTML = `<span class="state-badge confirmed">✓ CONFIRMED: ${data.confirmed}</span>`;
  } else if (data.candidate && data.candidate !== 'YES') {
    badgeWrap.innerHTML = `<span class="state-badge nominated">→ ${data.candidate} ${(data.fast_conf*100).toFixed(0)}%</span>`;
  } else {
    badgeWrap.innerHTML = '<span class="state-badge silent">— silent</span>';
  }

  // ── Last stride summary ──
  const fastTop = data.candidate ? `${data.candidate} ${(data.fast_conf*100).toFixed(0)}%` : '—';
  const slowTop = data.confirmed ? `${data.confirmed} ${(data.slow_conf*100).toFixed(0)}%` : (data.candidate ? 'no match' : '—');
  document.getElementById('dbg-fast-top').textContent = fastTop;
  document.getElementById('dbg-slow-top').textContent = slowTop;
  document.getElementById('dbg-nondet').textContent = data.nondet > 0
    ? (data.nondet * 100).toFixed(1) + '%' : '—';

  // ── Rolling history render ──
  const log = document.getElementById('history-log');
  log.innerHTML = historyRows.map(r => `
    <div class="log-row${r.confirmed ? ' confirmed-row' : ''}">
      <span class="log-tick">#${r.tick}</span>
      <span class="log-fast">${r.fast ? r.fast + ' ' + (r.fastConf*100).toFixed(0)+'%' : '—'}</span>
      <span class="log-slow">${r.slow ? r.slow + ' ' + (r.slowConf*100).toFixed(0)+'%' : '—'}</span>
      <span class="log-badge">${r.confirmed ? '✓' : ''}</span>
    </div>`).join('');
}

// toggleDebug replaced by switchTab()

function clearHistory() {
  historyRows.length = 0;
  document.getElementById('history-log').innerHTML =
    '<div style="color:var(--muted);font-size:0.7rem;text-align:center;margin-top:16px">Cleared.</div>';
}

socket.on('transcript_cleared', () => {
  document.getElementById('transcript-body').innerHTML =
    '<div class="transcript-empty" id="transcript-empty">Transcript cleared.</div>';
});

// ── Display ───────────────────────────────────────────────────────────────────
function updateOverlay(candidate, confirmed, fastConf, slowConf) {
  const gd = document.getElementById('gloss-display');
  const cb = document.getElementById('conf-bar');
  const cl = document.getElementById('conf-label');

  if (confirmed) {
    // Both windows agree — show confirmed state in accent colour
    gd.textContent = confirmed;
    gd.classList.remove('empty');
    gd.style.color = 'var(--accent)';
    cb.style.width = (Math.max(fastConf, slowConf) * 100) + '%';
    cb.style.background = 'var(--accent)';
    cl.textContent = slowConf > 0
      ? `✓ fast ${(fastConf*100).toFixed(0)}% · slow ${(slowConf*100).toFixed(0)}%`
      : `✓ fast ${(fastConf*100).toFixed(0)}% · slow neutral`;
  } else if (candidate) {
    // Fast window nominated a non-YES gloss — show streak progress
    gd.textContent = candidate + '?';
    gd.classList.remove('empty');
    gd.style.color = 'var(--yellow)';
    cb.style.width = (fastConf * 100) + '%';
    cb.style.background = 'var(--yellow)';
    cl.textContent = `nominated · building streak…`;
  } else {
    gd.textContent = '—';
    gd.classList.add('empty');
    gd.style.color = '';
    cb.style.width = '0%';
    cb.style.background = 'var(--accent)';
    cl.textContent = 'No sign detected';
  }
}

function updateTranscript(entries) {
  if (!entries || entries.length === 0) return;
  const body = document.getElementById('transcript-body');
  const empty = document.getElementById('transcript-empty');
  if (empty) empty.remove();

  body.innerHTML = entries.map(e => `
    <div class="t-entry">
      <span class="t-gloss">${e.gloss}</span>
      <span class="t-conf">${(e.conf * 100).toFixed(1)}%</span>
    </div>
  `).join('');
  body.scrollTop = body.scrollHeight;
}

function clearTranscript() {
  socket.emit('clear_transcript');
}

function resetDisplay() {
  updateOverlay(null, null, 0, 0);
  document.getElementById('nondet-val').textContent = '—';
}

function setStatus(live) {
  const pill = document.getElementById('status-pill');
  const text = document.getElementById('status-text');
  pill.classList.toggle('live', live);
  text.textContent = live ? 'LIVE' : 'OFFLINE';
}
</script>
</body>
</html>"""


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Auslan sliding window live demo")
    parser.add_argument("--model",      default=DEFAULTS["model"],     help="Path to best_model.pt")
    parser.add_argument("--label-map",  default=DEFAULTS["label_map"], help="Path to label_map.json")
    parser.add_argument("--config",     default=DEFAULTS["config"],    help="Path to experiment YAML config")
    parser.add_argument("--port",       type=int,   default=DEFAULTS["port"])
    parser.add_argument("--threshold",    type=float, default=DEFAULTS["threshold"])
    parser.add_argument("--nd-threshold", type=float, default=None,
                        help="Confidence threshold for NON_DETECTION (default: read from config or same as --threshold)")
    parser.add_argument("--fast-window",  type=int,   default=30,  help="Fast (nominating) window size in frames")
    parser.add_argument("--slow-window",  type=int,   default=90,  help="Slow (confirming) window size in frames")
    parser.add_argument("--stride",       type=int,   default=DEFAULTS["stride"])
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Load label map ────────────────────────────────────────────────────────
    if not args.label_map:
        raise FileNotFoundError("No label_map.json found. Pass --label-map.")
    with open(args.label_map) as f:
        label_map = json.load(f)
    print(f"  Vocabulary ({len(label_map)}): {sorted(label_map.keys())}")

    # ── Load model ────────────────────────────────────────────────────────────
    if not args.model:
        raise FileNotFoundError("No best_model.pt found. Pass --model.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = Video2GlossTransformer(cfg, len(label_map)).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"  Model loaded from {args.model}")
    print(f"  Device: {device}")

    # ── Init globals ──────────────────────────────────────────────────────────
    G["extractor"]  = FrameExtractor()
    # Read NON_DETECTION threshold — CLI > config > same as gloss threshold
    nd_thresh = args.nd_threshold
    if nd_thresh is None:
        nd_thresh = float(cfg.get("training", {}).get("non_detection_threshold", args.threshold))
    print(f"  Gloss threshold:  {args.threshold}")
    print(f"  NON_DET threshold: {nd_thresh}")

    G["classifier"] = DualWindowClassifier(
        model=model, label_map=label_map, cfg=cfg, device=device,
        threshold=args.threshold,
        fast_window=args.fast_window,
        slow_window=args.slow_window,
        stride=args.stride,
        nd_threshold=nd_thresh,
    )
    G["transcript"] = []

    print(f"\n{'='*50}")
    print(f"  Auslan Live Detector (Dual Window)")
    print(f"{'='*50}")
    print(f"  Fast window: {args.fast_window} frames")
    print(f"  Slow window: {args.slow_window} frames")
    print(f"  Stride:      {args.stride} frames")
    print(f"  Threshold:   {args.threshold}")
    print(f"  → Open http://localhost:{args.port}")
    print(f"{'='*50}\n")

    socketio.run(app, host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()