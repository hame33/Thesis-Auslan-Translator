#!/usr/bin/env python3
"""
Auslan Gloss Clip Annotator
============================
Usage:
    python annotator.py --gloss HELLO --video-dir /path/to/videos --output-dir ./clips --excel Auslan-Daily.xlsx

Keyboard shortcuts in the browser:
    Space       — Play / Pause
    S           — Set start time to current position
    E           — Set end time to current position
    Enter       — Save clip
    X / Delete  — Skip clip
    [ / ]       — Step back / forward 0.1s
    , / .       — Step back / forward 1s
"""

import argparse
import json
import os
import subprocess
import threading
import webbrowser
from pathlib import Path

import pandas as pd
import re
from flask import Flask, Response, jsonify, request, stream_with_context

app = Flask(__name__)

# ─── Global State ──────────────────────────────────────────────────────────────
STATE = {
    "gloss": None,
    "clips": [],          # list of {clip_name, signer_id, subtitle, video_path}
    "current_idx": 0,
    "video_dir": None,
    "output_dir": None,
    "manifest_path": None,
    "done_keys": set(),   # set of clip_names already in manifest
    "transcode_cache": None,  # dir for H.264 transcoded copies
}


# ─── Helpers ───────────────────────────────────────────────────────────────────

def load_manifest():
    path = STATE["manifest_path"]
    done = set()
    if Path(path).exists():
        try:
            df = pd.read_excel(path)
            for _, row in df.iterrows():
                if str(row.get("gloss", "")).upper() == STATE["gloss"].upper():
                    done.add(str(row["source_clip"]))
        except Exception:
            pass
    STATE["done_keys"] = done


def append_manifest(row_dict):
    path = STATE["manifest_path"]
    if Path(path).exists():
        try:
            df = pd.read_excel(path)
        except Exception:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    new_row = pd.DataFrame([row_dict])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_excel(path, index=False)


def find_clips(excel_path, gloss):
    df = pd.read_excel(excel_path)
    # Match whole-word gloss (space-delimited)
    gloss_upper = gloss.upper()
    mask = df["gloss"].apply(
        lambda g: gloss_upper in [tok.strip().upper() for tok in str(g).split()]
        if pd.notna(g) else False
    )
    matched = df[mask].copy()
    clips = []
    for _, row in matched.iterrows():
        clip_name = str(row["Video_Clip_Name"])
        signer_id = row.get("Signer_ID", "")
        subtitle = str(row.get("Subtitle", ""))
        all_glosses = str(row.get("gloss", ""))
        video_path = Path(STATE["video_dir"]) / f"{clip_name}_signer.mp4"
        # Only include if video exists
        if video_path.exists():
            clips.append({
                "clip_name": clip_name,
                "signer_id": signer_id,
                "subtitle": subtitle,
                "all_glosses": all_glosses,
                "video_path": str(video_path),
            })
    return clips


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return HTML_PAGE


@app.route("/api/status")
def api_status():
    clips = STATE["clips"]
    idx = STATE["current_idx"]
    done_count = len([c for c in clips if c["clip_name"] in STATE["done_keys"]])

    # Find the next un-done clip starting from current_idx
    start = idx
    for offset in range(len(clips)):
        i = (start + offset) % len(clips)
        if clips[i]["clip_name"] not in STATE["done_keys"]:
            STATE["current_idx"] = i
            clip = clips[i]
            pending = sum(1 for c in clips if c["clip_name"] not in STATE["done_keys"])
            return jsonify({
                "gloss": STATE["gloss"],
                "clip_name": clip["clip_name"],
                "signer_id": clip["signer_id"],
                "subtitle": clip["subtitle"],
                "all_glosses": clip["all_glosses"],
                "current_idx": i,
                "total": len(clips),
                "done": done_count,
                "pending": pending,
                "exhausted": False,
            })

    return jsonify({
        "gloss": STATE["gloss"],
        "total": len(clips),
        "done": done_count,
        "pending": 0,
        "exhausted": True,
    })


def get_transcoded_path(original_path):
    """Return path to H.264 version, transcoding if needed (~0.7s per clip)."""
    cache_dir = Path(STATE["transcode_cache"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / (Path(original_path).stem + "_h264.mp4")
    if not cached.exists():
        subprocess.run([
            "ffmpeg", "-y", "-i", original_path,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-movflags", "+faststart", "-an",
            str(cached),
        ], capture_output=True, check=True)
    return str(cached)


def serve_video_file(path):
    """Serve a video file with proper HTTP range request support."""
    file_size = os.path.getsize(path)
    range_header = request.headers.get("Range")

    if range_header:
        match = re.search(r"bytes=(\d+)-(\d*)", range_header)
        byte_start = int(match.group(1)) if match else 0
        byte_end = int(match.group(2)) if match and match.group(2) else file_size - 1
        byte_end = min(byte_end, file_size - 1)
        length = byte_end - byte_start + 1

        def generate_range():
            with open(path, "rb") as f:
                f.seek(byte_start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        return Response(
            stream_with_context(generate_range()), status=206,
            headers={
                "Content-Range": f"bytes {byte_start}-{byte_end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(length),
                "Content-Type": "video/mp4",
            },
        )

    def generate_full():
        with open(path, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                yield chunk

    return Response(
        stream_with_context(generate_full()), status=200,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Content-Type": "video/mp4",
        },
    )


@app.route("/api/video")
def api_video():
    clips = STATE["clips"]
    idx = STATE["current_idx"]
    if idx >= len(clips):
        return "No video", 404
    original_path = clips[idx]["video_path"]
    try:
        path = get_transcoded_path(original_path)
    except subprocess.CalledProcessError as e:
        return f"Transcode failed: {e}", 500
    return serve_video_file(path)


@app.route("/api/save", methods=["POST"])
def api_save():
    data = request.json
    start = float(data.get("start", 0))
    end = float(data.get("end", 0))

    if end <= start:
        return jsonify({"ok": False, "error": "End must be after start"}), 400

    clips = STATE["clips"]
    idx = STATE["current_idx"]
    clip = clips[idx]
    gloss = STATE["gloss"]
    out_dir = Path(STATE["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build unique output filename (handle multiple instances of same gloss in same clip)
    base = f"{clip['clip_name']}_signer_{gloss.upper()}"
    counter = 1
    while (out_dir / f"{base}_{counter}.mp4").exists():
        counter += 1
    out_filename = f"{base}_{counter}.mp4"
    out_path = out_dir / out_filename

    # Trim with ffmpeg (re-encode for accurate cutting)
    cmd = [
        "ffmpeg", "-y",
        "-i", clip["video_path"],
        "-ss", str(start),
        "-to", str(end),
        "-c:v", "libx264", "-c:a", "aac",
        "-loglevel", "error",
        str(out_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return jsonify({"ok": False, "error": result.stderr}), 500
    except FileNotFoundError:
        return jsonify({"ok": False, "error": "ffmpeg not found — please install ffmpeg"}), 500

    # Append to manifest
    row = {
        "source_clip": clip["clip_name"],
        "gloss": gloss,
        "all_glosses_in_clip": clip["all_glosses"],
        "subtitle": clip["subtitle"],
        "signer_id": clip["signer_id"],
        "start_sec": round(start, 3),
        "end_sec": round(end, 3),
        "duration_sec": round(end - start, 3),
        "output_file": out_filename,
        "status": "saved",
    }
    append_manifest(row)
    STATE["done_keys"].add(clip["clip_name"])
    STATE["current_idx"] = idx + 1

    return jsonify({"ok": True, "saved_as": out_filename})


@app.route("/api/skip", methods=["POST"])
def api_skip():
    clips = STATE["clips"]
    idx = STATE["current_idx"]
    clip = clips[idx]

    row = {
        "source_clip": clip["clip_name"],
        "gloss": STATE["gloss"],
        "all_glosses_in_clip": clip["all_glosses"],
        "subtitle": clip["subtitle"],
        "signer_id": clip["signer_id"],
        "start_sec": None,
        "end_sec": None,
        "duration_sec": None,
        "output_file": None,
        "status": "skipped",
    }
    append_manifest(row)
    STATE["done_keys"].add(clip["clip_name"])
    STATE["current_idx"] = idx + 1

    return jsonify({"ok": True})


# ─── HTML UI ───────────────────────────────────────────────────────────────────

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Auslan Gloss Annotator</title>
<style>
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #22263a;
    --border: #2e3347;
    --accent: #5b8cff;
    --accent2: #7c5cff;
    --green: #3ecf8e;
    --red: #f56565;
    --yellow: #f6c343;
    --text: #e2e8f0;
    --muted: #7b8aad;
    --radius: 10px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', system-ui, sans-serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 24px 16px;
  }

  .header {
    width: 100%;
    max-width: 900px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 20px;
  }
  .gloss-badge {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: white;
    font-size: 1.3rem;
    font-weight: 700;
    padding: 6px 18px;
    border-radius: 30px;
    letter-spacing: 0.05em;
  }
  .progress-info {
    font-size: 0.85rem;
    color: var(--muted);
    text-align: right;
    line-height: 1.6;
  }
  .progress-bar-wrap {
    width: 100%;
    max-width: 900px;
    background: var(--surface2);
    border-radius: 4px;
    height: 6px;
    margin-bottom: 20px;
  }
  .progress-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    border-radius: 4px;
    transition: width 0.4s ease;
  }

  .card {
    width: 100%;
    max-width: 900px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    margin-bottom: 16px;
  }
  .card-header {
    background: var(--surface2);
    padding: 12px 20px;
    display: flex;
    gap: 24px;
    align-items: center;
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
  }
  .meta-item { font-size: 0.82rem; }
  .meta-label { color: var(--muted); margin-right: 4px; }
  .meta-value { color: var(--text); font-weight: 600; }

  .gloss-tokens {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    padding: 12px 20px;
    border-bottom: 1px solid var(--border);
    background: var(--bg);
  }
  .token {
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    background: var(--surface2);
    color: var(--muted);
    border: 1px solid var(--border);
    letter-spacing: 0.04em;
  }
  .token.target {
    background: rgba(91, 140, 255, 0.2);
    color: var(--accent);
    border-color: var(--accent);
  }

  video {
    width: 100%;
    display: block;
    background: #000;
    max-height: 420px;
    object-fit: contain;
  }

  .timeline {
    position: relative;
    margin: 12px 20px;
    height: 36px;
    background: var(--surface2);
    border-radius: 6px;
    border: 1px solid var(--border);
    cursor: pointer;
    overflow: hidden;
  }
  .timeline-fill {
    position: absolute;
    top: 0; bottom: 0; left: 0;
    background: rgba(91, 140, 255, 0.25);
    pointer-events: none;
    transition: width 0.05s linear;
  }
  .timeline-selection {
    position: absolute;
    top: 0; bottom: 0;
    background: rgba(62, 207, 142, 0.35);
    border-left: 2px solid var(--green);
    border-right: 2px solid var(--green);
    pointer-events: none;
  }
  .timeline-cursor {
    position: absolute;
    top: 0; bottom: 0;
    width: 2px;
    background: white;
    pointer-events: none;
    transform: translateX(-50%);
  }
  .timeline-label {
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
    font-size: 0.72rem;
    color: var(--muted);
    pointer-events: none;
    padding: 0 10px;
  }

  .controls {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    padding: 16px 20px;
  }
  .time-group { display: flex; flex-direction: column; gap: 6px; }
  .time-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .time-label .kbd { 
    background: var(--surface2); 
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 0.7rem;
    color: var(--text);
    font-family: monospace;
  }
  .time-input-row { display: flex; gap: 8px; align-items: center; }
  .time-input {
    flex: 1;
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 1rem;
    font-family: monospace;
    font-weight: 600;
    transition: border-color 0.2s;
  }
  .time-input:focus { outline: none; border-color: var(--accent); }
  .set-btn {
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 8px 14px;
    border-radius: 6px;
    font-size: 0.8rem;
    cursor: pointer;
    white-space: nowrap;
    transition: background 0.15s, border-color 0.15s;
  }
  .set-btn:hover { background: var(--border); border-color: var(--accent); }

  .subtitle-row {
    padding: 0 20px 14px;
    font-size: 0.88rem;
    color: var(--muted);
    font-style: italic;
  }
  .subtitle-row span { color: var(--text); }

  .action-bar {
    width: 100%;
    max-width: 900px;
    display: flex;
    gap: 12px;
  }
  .btn {
    flex: 1;
    padding: 14px;
    border: none;
    border-radius: var(--radius);
    font-size: 1rem;
    font-weight: 700;
    cursor: pointer;
    transition: opacity 0.15s, transform 0.1s;
    letter-spacing: 0.03em;
  }
  .btn:active { transform: scale(0.98); }
  .btn:disabled { opacity: 0.45; cursor: not-allowed; }
  .btn-save {
    background: linear-gradient(135deg, var(--green), #2bb57c);
    color: #071a11;
  }
  .btn-skip {
    background: var(--surface);
    color: var(--muted);
    border: 1px solid var(--border);
  }
  .btn-skip:hover:not(:disabled) { border-color: var(--red); color: var(--red); }

  .toast {
    position: fixed;
    bottom: 30px;
    left: 50%;
    transform: translateX(-50%) translateY(80px);
    background: var(--surface2);
    border: 1px solid var(--border);
    padding: 12px 24px;
    border-radius: 30px;
    font-size: 0.9rem;
    font-weight: 600;
    transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    z-index: 999;
    white-space: nowrap;
  }
  .toast.show { transform: translateX(-50%) translateY(0); }
  .toast.success { border-color: var(--green); color: var(--green); }
  .toast.error { border-color: var(--red); color: var(--red); }
  .toast.info { border-color: var(--accent); color: var(--accent); }

  .done-screen {
    text-align: center;
    padding: 60px 20px;
  }
  .done-screen h2 { font-size: 2rem; margin-bottom: 12px; color: var(--green); }
  .done-screen p { color: var(--muted); }

  .shortcuts {
    width: 100%;
    max-width: 900px;
    margin-top: 20px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px 20px;
    display: flex;
    flex-wrap: wrap;
    gap: 12px 24px;
  }
  .shortcut { font-size: 0.78rem; color: var(--muted); }
  .shortcut .kbd {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1px 7px;
    font-family: monospace;
    color: var(--text);
    font-size: 0.76rem;
  }
</style>
</head>
<body>

<div class="header">
  <div>
    <div style="font-size:0.75rem; color:var(--muted); margin-bottom:4px;">Annotating gloss</div>
    <div class="gloss-badge" id="gloss-badge">—</div>
  </div>
  <div class="progress-info" id="progress-info">Loading…</div>
</div>

<div class="progress-bar-wrap">
  <div class="progress-bar-fill" id="progress-bar" style="width:0%"></div>
</div>

<!-- Static persistent layout — only content inside cards is updated -->
<div id="clip-card" class="card" style="display:none">
  <div class="card-header">
    <div class="meta-item"><span class="meta-label">Clip</span><span class="meta-value" id="meta-clip">—</span></div>
    <div class="meta-item"><span class="meta-label">Signer</span><span class="meta-value" id="meta-signer">—</span></div>
    <div class="meta-item"><span class="meta-label">Index</span><span class="meta-value" id="meta-idx">—</span></div>
  </div>
  <div class="gloss-tokens" id="gloss-tokens"></div>
  <video id="vid" preload="auto" controls playsinline muted style="width:100%;display:block;background:#000;max-height:420px;object-fit:contain"></video>
  <div class="timeline" id="timeline">
    <div class="timeline-fill" id="tl-fill" style="width:0%"></div>
    <div class="timeline-selection" id="tl-sel" style="display:none"></div>
    <div class="timeline-cursor" id="tl-cursor" style="left:0%"></div>
    <div class="timeline-label">Click to seek · drag to scrub</div>
  </div>
  <div class="controls">
    <div class="time-group">
      <div class="time-label">Start <span class="kbd">S</span></div>
      <div class="time-input-row">
        <input class="time-input" id="inp-start" type="number" step="0.001" min="0" value="0.000">
        <button class="set-btn" onclick="setFromVideo('start')">Set to now</button>
      </div>
    </div>
    <div class="time-group">
      <div class="time-label">End <span class="kbd">E</span></div>
      <div class="time-input-row">
        <input class="time-input" id="inp-end" type="number" step="0.001" min="0" value="0.000">
        <button class="set-btn" onclick="setFromVideo('end')">Set to now</button>
      </div>
    </div>
  </div>
  <div class="subtitle-row">Subtitle: <span id="meta-subtitle"></span></div>
</div>

<div id="done-card" class="card done-screen" style="display:none">
  <h2>✓ All done!</h2>
  <p id="done-msg"></p>
  <p style="margin-top:8px;">You can close this window or restart with a new gloss.</p>
</div>

<div class="action-bar" id="action-bar" style="display:none">
  <button class="btn btn-save" id="btn-save" onclick="saveClip()">
    ✓ Save Clip &nbsp;<small style="opacity:.7;font-weight:400">[Enter]</small>
  </button>
  <button class="btn btn-skip" id="btn-skip" onclick="skipClip()">
    Skip &nbsp;<small style="opacity:.7;font-weight:400">[X]</small>
  </button>
</div>

<div class="toast" id="toast"></div>

<div class="shortcuts">
  <span class="shortcut"><span class="kbd">Space</span> Play/Pause</span>
  <span class="shortcut"><span class="kbd">S</span> Set start</span>
  <span class="shortcut"><span class="kbd">E</span> Set end</span>
  <span class="shortcut"><span class="kbd">Enter</span> Save</span>
  <span class="shortcut"><span class="kbd">X</span> Skip</span>
  <span class="shortcut"><span class="kbd">←</span><span class="kbd">→</span> ±0.5s</span>
  <span class="shortcut"><span class="kbd">[</span><span class="kbd">]</span> ±0.1s</span>
  <span class="shortcut"><span class="kbd">,</span><span class="kbd">.</span> ±1s</span>
</div>

<script>
let currentStatus = {};
let startTime = 0;
let endTime = 0;
let saving = false;

const $ = id => document.getElementById(id);
const vid = document.getElementById('vid');

// Wire up persistent video events once
vid.addEventListener('timeupdate', () => {
  if (!vid.duration) return;
  const pct = (vid.currentTime / vid.duration) * 100;
  $('tl-fill').style.width = pct + '%';
  $('tl-cursor').style.left = pct + '%';
});

vid.addEventListener('loadedmetadata', () => {
  endTime = vid.duration;
  $('inp-end').value = fmtTime(endTime);
  updateSelectionOverlay();
  vid.play().catch(() => {});
});

vid.addEventListener('error', () => {
  console.error('Video error:', vid.error);
});

// Timeline — wired once
const tl = $('timeline');
let dragging = false;
function seekFromEvent(e) {
  const rect = tl.getBoundingClientRect();
  const x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
  const ratio = Math.max(0, Math.min(1, x / rect.width));
  if (vid.duration) vid.currentTime = ratio * vid.duration;
}
tl.addEventListener('mousedown', e => { dragging = true; seekFromEvent(e); });
document.addEventListener('mousemove', e => { if (dragging) seekFromEvent(e); });
document.addEventListener('mouseup', () => dragging = false);

$('inp-start').addEventListener('input', e => { startTime = parseTime(e.target.value); updateSelectionOverlay(); });
$('inp-end').addEventListener('input', e => { endTime = parseTime(e.target.value); updateSelectionOverlay(); });

function showToast(msg, type = 'info', duration = 2500) {
  const t = $('toast');
  t.textContent = msg;
  t.className = `toast ${type} show`;
  setTimeout(() => t.classList.remove('show'), duration);
}

function fmtTime(s) { return isNaN(s) ? '0.000' : s.toFixed(3); }
function parseTime(s) { const v = parseFloat(s); return isNaN(v) ? 0 : Math.max(0, v); }

function updateSelectionOverlay() {
  if (!vid.duration) return;
  const sel = $('tl-sel');
  const l = (startTime / vid.duration) * 100;
  const r = (endTime / vid.duration) * 100;
  if (endTime > startTime) {
    sel.style.left = l + '%';
    sel.style.width = (r - l) + '%';
    sel.style.display = 'block';
  } else {
    sel.style.display = 'none';
  }
}

function buildUI(st) {
  currentStatus = st;
  $('gloss-badge').textContent = st.gloss || '—';

  if (st.exhausted) {
    $('clip-card').style.display = 'none';
    $('action-bar').style.display = 'none';
    $('done-card').style.display = 'block';
    $('done-msg').textContent = `${st.done} clips processed for ${st.gloss}.`;
    $('progress-info').textContent = `${st.done} / ${st.total} done`;
    $('progress-bar').style.width = '100%';
    return;
  }

  // Update metadata fields
  $('meta-clip').textContent = st.clip_name;
  $('meta-signer').textContent = st.signer_id;
  $('meta-idx').textContent = `${st.current_idx + 1} / ${st.total}`;
  $('meta-subtitle').textContent = st.subtitle;

  const tokens = (st.all_glosses || '').split(' ').filter(Boolean);
  $('gloss-tokens').innerHTML = tokens.map(t =>
    `<span class="token${t.toUpperCase() === st.gloss.toUpperCase() ? ' target' : ''}">${t}</span>`
  ).join('');

  const pct = st.total > 0 ? (st.done / st.total) * 100 : 0;
  $('progress-bar').style.width = pct + '%';
  $('progress-info').innerHTML = `
    <strong>${st.pending}</strong> remaining &nbsp;|&nbsp;
    ${st.done} done &nbsp;|&nbsp; ${st.total} total<br>
    <span style="color:var(--muted)">${st.clip_name}</span>`;

  // Reset times
  startTime = 0;
  endTime = 0;
  $('inp-start').value = '0.000';
  $('inp-end').value = '0.000';
  $('tl-sel').style.display = 'none';
  $('tl-fill').style.width = '0%';
  $('tl-cursor').style.left = '0%';

  // Load new video — this is the reliable way to update a persistent video element
  vid.src = `/api/video?idx=${st.current_idx}&t=${Date.now()}`;
  vid.load();

  $('clip-card').style.display = 'block';
  $('action-bar').style.display = 'flex';
  $('done-card').style.display = 'none';
  $('btn-save').disabled = false;
  $('btn-skip').disabled = false;
  saving = false;
}

function setFromVideo(which) {
  const t = vid.currentTime;
  if (which === 'start') {
    startTime = t;
    $('inp-start').value = fmtTime(t);
  } else {
    endTime = t;
    $('inp-end').value = fmtTime(t);
  }
  updateSelectionOverlay();
}

async function saveClip() {
  if (saving) return;
  const s = parseTime($('inp-start').value);
  const e = parseTime($('inp-end').value);
  if (e <= s) { showToast('End must be after start', 'error'); return; }

  saving = true;
  $('btn-save').disabled = true;
  $('btn-save').textContent = 'Saving…';

  try {
    const res = await fetch('/api/save', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({start: s, end: e})
    });
    const data = await res.json();
    if (data.ok) {
      showToast(`✓ Saved: ${data.saved_as}`, 'success');
      setTimeout(() => loadStatus(), 400);
    } else {
      showToast('Error: ' + data.error, 'error', 4000);
    }
  } catch (err) {
    showToast('Network error', 'error');
  } finally {
    saving = false;
  }
}

async function skipClip() {
  if (saving) return;
  saving = true;
  $('btn-skip').disabled = true;
  try {
    await fetch('/api/skip', {method: 'POST'});
    showToast('Skipped', 'info', 1200);
    setTimeout(() => loadStatus(), 300);
  } finally {
    saving = false;
  }
}

async function loadStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    buildUI(data);
  } catch (e) {
    console.error(e);
  }
}

// Use capture:true so we intercept keys before the video element's native handler
window.addEventListener('keydown', e => {
  const tag = document.activeElement.tagName;
  const inInput = tag === 'INPUT' || tag === 'TEXTAREA';
  if (inInput) return;

  const dur = vid.duration || 0;

  switch (e.key) {
    case ' ':
      e.preventDefault(); e.stopPropagation();
      vid.paused ? vid.play() : vid.pause();
      break;
    case 's': case 'S':
      e.preventDefault(); e.stopPropagation();
      setFromVideo('start');
      break;
    case 'e': case 'E':
      e.preventDefault(); e.stopPropagation();
      setFromVideo('end');
      break;
    case 'x': case 'X': case 'Delete':
      e.preventDefault(); e.stopPropagation();
      skipClip();
      break;
    case 'Enter':
      e.preventDefault(); e.stopPropagation();
      saveClip();
      break;
    case '[':
      e.preventDefault(); e.stopPropagation();
      vid.currentTime = Math.max(0, vid.currentTime - 0.1);
      break;
    case ']':
      e.preventDefault(); e.stopPropagation();
      vid.currentTime = Math.min(dur, vid.currentTime + 0.1);
      break;
    case ',':
      e.preventDefault(); e.stopPropagation();
      vid.currentTime = Math.max(0, vid.currentTime - 1);
      break;
    case '.':
      e.preventDefault(); e.stopPropagation();
      vid.currentTime = Math.min(dur, vid.currentTime + 1);
      break;
    case 'ArrowLeft':
      e.preventDefault(); e.stopPropagation();
      vid.currentTime = Math.max(0, vid.currentTime - 0.5);
      break;
    case 'ArrowRight':
      e.preventDefault(); e.stopPropagation();
      vid.currentTime = Math.min(dur, vid.currentTime + 0.5);
      break;
  }
}, { capture: true });

loadStatus();
</script>
</body>
</html>"""


# ─── Entry Point ───────────────────────────────────────────────────────────────

def main():
    # ── Configure these paths once, then only --gloss is needed ─────────────
    DEFAULTS = {
        "video_dir":  "/Users/hamishdawson/Desktop/Thesis/Thesis-Auslan-Translator/Signer",
        "output_dir": "/Users/hamishdawson/Desktop/Thesis/Thesis-Auslan-Translator/gloss_clips",
        "manifest":   "/Users/hamishdawson/Desktop/Thesis/Thesis-Auslan-Translator/gloss_clips_manifest.xlsx",
        "excel":      "/Users/hamishdawson/Desktop/Thesis/Auslan-Daily_Communication_with_gloss_fixed.xlsx",
        "port":       5000,
    }
    # ─────────────────────────────────────────────────────────────────────────

    parser = argparse.ArgumentParser(
        description="Auslan Gloss Clip Annotator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--gloss", required=True, help="Gloss to annotate (e.g. HELLO)")
    parser.add_argument("--video-dir", default=DEFAULTS["video_dir"], help="Directory containing *_signer.mp4 files")
    parser.add_argument("--output-dir", default=DEFAULTS["output_dir"], help="Directory to save trimmed clips")
    parser.add_argument("--manifest", default=DEFAULTS["manifest"], help="Path to manifest Excel file")
    parser.add_argument("--excel", default=DEFAULTS["excel"], help="Path to gloss annotation Excel")
    parser.add_argument("--port", type=int, default=DEFAULTS["port"], help="Port for web server")
    args = parser.parse_args()

    gloss = args.gloss.upper()
    STATE["gloss"] = gloss
    STATE["video_dir"] = args.video_dir
    STATE["transcode_cache"] = str(Path(args.output_dir) / ".transcode_cache")
    STATE["output_dir"] = args.output_dir
    STATE["manifest_path"] = args.manifest

    print(f"\n{'='*55}")
    print(f"  Auslan Gloss Annotator")
    print(f"{'='*55}")
    print(f"  Gloss:      {gloss}")
    print(f"  Video dir:  {args.video_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Manifest:   {args.manifest}")
    print(f"{'='*55}")

    # Load clips
    print(f"\n⟳ Scanning Excel for '{gloss}'…")
    clips = find_clips(args.excel, gloss)
    if not clips:
        print(f"\n✗ No video files found for gloss '{gloss}'.")
        print(f"  (Either no matching rows, or video files missing in {args.video_dir})")
        return

    STATE["clips"] = clips
    load_manifest()

    already_done = len([c for c in clips if c["clip_name"] in STATE["done_keys"]])
    print(f"✓ Found {len(clips)} clips  ({already_done} already in manifest, {len(clips)-already_done} to go)")

    # Open browser after short delay
    url = f"http://localhost:{args.port}"
    print(f"\n→ Opening {url}\n")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()