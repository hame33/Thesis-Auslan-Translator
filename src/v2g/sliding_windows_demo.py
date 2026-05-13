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

    def extract(self, bgr_frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        r   = self.holistic.process(rgb)

        def pose(lms):
            if lms is None: return np.zeros(POSE_DIMS, np.float32)
            return np.array([v for lm in lms.landmark for v in (lm.x, lm.y, lm.z, lm.visibility)], np.float32)

        def hand(lms):
            if lms is None: return np.zeros(LHAND_DIMS, np.float32)
            return np.array([v for lm in lms.landmark for v in (lm.x, lm.y, lm.z)], np.float32)

        return np.concatenate([pose(r.pose_landmarks),
                                hand(r.left_hand_landmarks),
                                hand(r.right_hand_landmarks)])

    def close(self):
        self.holistic.close()


# ── Classifier ────────────────────────────────────────────────────────────────

class SlidingWindowClassifier:
    def __init__(self, model, label_map, cfg, device, threshold, window, stride):
        self.model      = model
        self.label_map  = label_map
        self.idx2gloss  = {v: k for k, v in label_map.items()}
        self.cfg        = cfg
        self.device     = device
        self.threshold  = threshold
        self.window     = window
        self.stride     = stride
        self.max_frames = cfg["model"]["max_frames"]

        self.buffer     = collections.deque(maxlen=window)
        self.frame_count = 0
        self.last_result = {"gloss": None, "confidence": 0.0, "all": []}

    def push(self, feat: np.ndarray):
        """Add one frame of features; run classifier every `stride` frames."""
        self.buffer.append(feat)
        self.frame_count += 1
        if self.frame_count % self.stride == 0 and len(self.buffer) >= max(2, self.window // 4):
            self._classify()
        return self.last_result

    def _classify(self):
        frames = np.stack(list(self.buffer)).astype(np.float32)  # [T, 258]
        T = len(frames)
        if T > self.max_frames:
            idx = np.linspace(0, T - 1, self.max_frames, dtype=int)
            frames = frames[idx]
            T = self.max_frames

        t = torch.from_numpy(frames).unsqueeze(0).to(self.device)   # [1, T, 258]
        mask = torch.zeros(1, T, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            logits = self.model(t, mask)
            probs  = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        all_preds = sorted(
            [(self.idx2gloss[i], float(probs[i])) for i in range(len(probs))
             if probs[i] >= self.threshold],
            key=lambda x: -x[1],
        )
        # Filter out NON_DETECTION from display (it's an internal class)
        display = [(g, c) for g, c in all_preds if g != "NON_DETECTION"]

        top = display[0] if display else (None, 0.0)
        self.last_result = {
            "gloss":      top[0],
            "confidence": round(top[1], 3),
            "all":        [(g, round(c, 3)) for g, c in display],
            "nondet":     round(float(probs[self.label_map.get("NON_DETECTION", 0)]), 3)
                          if "NON_DETECTION" in self.label_map else 0.0,
        }

    def update_params(self, window=None, stride=None, threshold=None):
        if window    is not None: self.window    = window;    self.buffer = collections.deque(maxlen=window)
        if stride    is not None: self.stride    = stride
        if threshold is not None: self.threshold = threshold


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

        feat   = G["extractor"].extract(frame)
        result = G["classifier"].push(feat)

        # Append to rolling transcript if confident gloss detected
        gloss = result["gloss"]
        if gloss and result["confidence"] >= G["classifier"].threshold:
            transcript = G["transcript"]
            if not transcript or transcript[-1]["gloss"] != gloss:
                transcript.append({"gloss": gloss, "conf": result["confidence"]})
                if len(transcript) > 50:
                    transcript.pop(0)

        emit("prediction", {
            "gloss":      result["gloss"],
            "confidence": result["confidence"],
            "all":        result["all"],
            "nondet":     result["nondet"],
            "transcript": G["transcript"][-10:],  # last 10 entries
        })

    except Exception as e:
        emit("error", {"msg": str(e)})


@socketio.on("update_params")
def handle_params(data):
    G["classifier"].update_params(
        window=data.get("window"),
        stride=data.get("stride"),
        threshold=data.get("threshold"),
    )
    emit("params_updated", {
        "window":    G["classifier"].window,
        "stride":    G["classifier"].stride,
        "threshold": G["classifier"].threshold,
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
    object-fit: cover;
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
    <div class="controls">
      <div>
        <div class="ctrl-label">Window size <span class="val" id="window-val">60 frames</span></div>
        <input type="range" id="sl-window" min="15" max="150" step="5" value="60">
      </div>
      <div>
        <div class="ctrl-label">Stride <span class="val" id="stride-val">10 frames</span></div>
        <input type="range" id="sl-stride" min="1" max="30" step="1" value="10">
      </div>
      <div>
        <div class="ctrl-label">Threshold <span class="val" id="thresh-val">0.35</span></div>
        <input type="range" id="sl-thresh" min="0.10" max="0.90" step="0.05" value="0.35">
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
setupSlider('sl-window', 'window-val', ' frames', 'window');
setupSlider('sl-stride', 'stride-val', ' frames', 'stride');
setupSlider('sl-thresh', 'thresh-val', '',        'threshold');

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
  // Mirror-flip to match webcam display
  ctx.save();
  ctx.scale(-1, 1);
  ctx.drawImage(video, -320, 0, 320, 240);
  ctx.restore();
  const b64 = canvas.toDataURL('image/jpeg', 0.7);
  socket.emit('frame', { image: b64 });
}

// ── Socket events ─────────────────────────────────────────────────────────────
socket.on('prediction', data => {
  updateOverlay(data.gloss, data.confidence);
  document.getElementById('nondet-val').textContent =
    data.nondet > 0 ? (data.nondet * 100).toFixed(0) + '%' : '—';
  updateTranscript(data.transcript);
});

socket.on('transcript_cleared', () => {
  document.getElementById('transcript-body').innerHTML =
    '<div class="transcript-empty" id="transcript-empty">Transcript cleared.</div>';
});

// ── Display ───────────────────────────────────────────────────────────────────
function updateOverlay(gloss, conf) {
  const gd = document.getElementById('gloss-display');
  const cb = document.getElementById('conf-bar');
  const cl = document.getElementById('conf-label');

  if (gloss) {
    gd.textContent = gloss;
    gd.classList.remove('empty');
    cb.style.width = (conf * 100) + '%';
    cl.textContent = (conf * 100).toFixed(1) + '% confidence';
  } else {
    gd.textContent = '—';
    gd.classList.add('empty');
    cb.style.width = '0%';
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
  updateOverlay(null, 0);
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
    parser.add_argument("--threshold",  type=float, default=DEFAULTS["threshold"])
    parser.add_argument("--window",     type=int,   default=DEFAULTS["window"])
    parser.add_argument("--stride",     type=int,   default=DEFAULTS["stride"])
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
    G["classifier"] = SlidingWindowClassifier(
        model=model, label_map=label_map, cfg=cfg, device=device,
        threshold=args.threshold, window=args.window, stride=args.stride,
    )
    G["transcript"] = []

    print(f"\n{'='*50}")
    print(f"  Auslan Live Detector")
    print(f"{'='*50}")
    print(f"  Window:    {args.window} frames")
    print(f"  Stride:    {args.stride} frames")
    print(f"  Threshold: {args.threshold}")
    print(f"  → Open http://localhost:{args.port}")
    print(f"{'='*50}\n")

    socketio.run(app, host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()