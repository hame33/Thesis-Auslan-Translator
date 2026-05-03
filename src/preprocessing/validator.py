#!/usr/bin/env python3
"""
Auslan Gloss Validation Tool
==============================
Review test set clips, verify gloss annotations, and correct them in-place.

Usage:
    python validator.py

Opens browser at http://localhost:5001
Changes are saved back to the Excel file immediately on confirmation.
A validation manifest tracks which clips have been reviewed so you can resume.

Keyboard shortcuts:
    Space       — Play / Pause
    Enter       — Confirm (save current gloss as-is)
    E           — Focus the gloss edit box
    [ / ]       — ±0.1s
    , / .       — ±1s
    ← →         — ±0.5s
"""

import argparse
import json
import os
import re
import subprocess
import threading
import webbrowser
from pathlib import Path

import pandas as pd
from flask import Flask, Response, jsonify, request, stream_with_context

app = Flask(__name__)

# ── Configure these paths ──────────────────────────────────────────────────
EXCEL_PATH       = "/Users/hamishdawson/Desktop/Thesis/Auslan-Daily_Communication_with_gloss_fixed.xlsx"
VIDEO_DIR        = "/Users/hamishdawson/Desktop/Thesis/Thesis-Auslan-Translator/Signer"
VALIDATION_LOG   = "/Users/hamishdawson/Desktop/Thesis/Thesis-Auslan-Translator/validation_log.xlsx"
SPLIT_TO_REVIEW  = "test"    # change to "dev" or "train" if needed
PORT             = 5001

# Only show clips containing at least one of these glosses.
# Set to None to review all clips in the split.
FOCUS_GLOSSES = {'THINK', 'YES', 'KNOW', 'TIME', 'HELLO', 'GO', 'WHAT', 'GOOD'}
# ──────────────────────────────────────────────────────────────────────────

STATE = {
    "clips":         [],
    "current_idx":   0,
    "done_keys":     set(),   # clip names already validated
    "df":            None,    # full dataframe (mutable)
    "excel_path":    EXCEL_PATH,
    "video_dir":     VIDEO_DIR,
    "transcode_cache": None,
}


# ── Helpers ────────────────────────────────────────────────────────────────

def load_validation_log():
    done = set()
    if Path(VALIDATION_LOG).exists():
        try:
            log = pd.read_excel(VALIDATION_LOG)
            done = set(log["clip_name"].astype(str).tolist())
        except Exception:
            pass
    STATE["done_keys"] = done


def append_validation_log(clip_name, old_gloss, new_gloss, action):
    row = {
        "clip_name": clip_name,
        "old_gloss": old_gloss,
        "new_gloss": new_gloss,
        "action":    action,   # "confirmed", "edited", "skipped"
    }
    if Path(VALIDATION_LOG).exists():
        try:
            log = pd.read_excel(VALIDATION_LOG)
        except Exception:
            log = pd.DataFrame()
    else:
        log = pd.DataFrame()
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    log.to_excel(VALIDATION_LOG, index=False)


def find_next_clip():
    clips = STATE["clips"]
    idx = STATE["current_idx"]
    for offset in range(len(clips)):
        i = (idx + offset) % len(clips)
        if clips[i]["clip_name"] not in STATE["done_keys"]:
            STATE["current_idx"] = i
            return clips[i], i
    return None, -1


def get_video_path(clip_name):
    return Path(STATE["video_dir"]) / f"{clip_name}_signer.mp4"


def get_transcoded_path(original_path):
    cache_dir = Path(STATE["transcode_cache"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / (Path(original_path).stem + "_h264.mp4")
    if not cached.exists():
        subprocess.run([
            "ffmpeg", "-y", "-i", str(original_path),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-movflags", "+faststart", "-an", str(cached),
        ], capture_output=True, check=True)
    return str(cached)


def serve_video_file(path):
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

        return Response(stream_with_context(generate_range()), status=206, headers={
            "Content-Range":  f"bytes {byte_start}-{byte_end}/{file_size}",
            "Accept-Ranges":  "bytes",
            "Content-Length": str(length),
            "Content-Type":   "video/mp4",
        })

    def generate_full():
        with open(path, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                yield chunk

    return Response(stream_with_context(generate_full()), status=200, headers={
        "Accept-Ranges":  "bytes",
        "Content-Length": str(file_size),
        "Content-Type":   "video/mp4",
    })


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return HTML_PAGE


@app.route("/api/status")
def api_status():
    clips = STATE["clips"]
    done_count = len(STATE["done_keys"])
    pending = len([c for c in clips if c["clip_name"] not in STATE["done_keys"]])

    clip, idx = find_next_clip()
    if clip is None:
        return jsonify({"exhausted": True, "done": done_count, "total": len(clips)})

    return jsonify({
        "exhausted":   False,
        "clip_name":   clip["clip_name"],
        "subtitle":    clip["subtitle"],
        "gloss":       clip["gloss"],
        "signer_id":   clip["signer_id"],
        "current_idx": idx,
        "total":       len(clips),
        "done":        done_count,
        "pending":     pending,
    })


@app.route("/api/video")
def api_video():
    clip, _ = find_next_clip()
    if clip is None:
        return "No video", 404
    original = get_video_path(clip["clip_name"])
    if not original.exists():
        return f"Video not found: {original}", 404
    try:
        path = get_transcoded_path(str(original))
    except subprocess.CalledProcessError as e:
        return f"Transcode failed: {e}", 500
    return serve_video_file(path)


@app.route("/api/confirm", methods=["POST"])
def api_confirm():
    """Save gloss (edited or as-is) and advance."""
    data = request.json
    new_gloss = str(data.get("gloss", "")).strip()

    clip, idx = find_next_clip()
    if clip is None:
        return jsonify({"ok": False, "error": "No clip"}), 400

    old_gloss = clip["gloss"]
    action = "edited" if new_gloss != old_gloss else "confirmed"

    # Update the dataframe in memory and save to Excel
    df = STATE["df"]
    mask = df["Video_Clip_Name"] == clip["clip_name"]
    df.loc[mask, "gloss"] = new_gloss

    # Also update our clips list
    STATE["clips"][idx]["gloss"] = new_gloss

    # Save Excel
    try:
        df.to_excel(STATE["excel_path"], index=False)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Failed to save Excel: {e}"}), 500

    append_validation_log(clip["clip_name"], old_gloss, new_gloss, action)
    STATE["done_keys"].add(clip["clip_name"])
    STATE["current_idx"] = idx + 1

    return jsonify({"ok": True, "action": action, "old_gloss": old_gloss, "new_gloss": new_gloss})


@app.route("/api/skip", methods=["POST"])
def api_skip():
    """Skip this clip for now — comes back next session."""
    clip, idx = find_next_clip()
    if clip is None:
        return jsonify({"ok": False}), 400
    STATE["done_keys"].add(clip["clip_name"])
    STATE["current_idx"] = idx + 1
    return jsonify({"ok": True})


# ── HTML ───────────────────────────────────────────────────────────────────

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Auslan Gloss Validator</title>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d27; --surface2: #22263a;
    --border: #2e3347; --accent: #5b8cff; --accent2: #7c5cff;
    --green: #3ecf8e; --red: #f56565; --yellow: #f6c343;
    --text: #e2e8f0; --muted: #7b8aad; --radius: 10px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg); color: var(--text);
    font-family: 'Inter', system-ui, sans-serif;
    min-height: 100vh; display: flex; flex-direction: column;
    align-items: center; padding: 24px 16px;
  }
  .header {
    width: 100%; max-width: 900px; display: flex;
    align-items: center; justify-content: space-between; margin-bottom: 16px;
  }
  .title { font-size: 1.2rem; font-weight: 700; color: var(--accent); }
  .progress-info { font-size: 0.85rem; color: var(--muted); text-align: right; line-height: 1.6; }
  .progress-bar-wrap {
    width: 100%; max-width: 900px; background: var(--surface2);
    border-radius: 4px; height: 6px; margin-bottom: 16px;
  }
  .progress-bar-fill {
    height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent2));
    border-radius: 4px; transition: width 0.4s ease;
  }
  .card {
    width: 100%; max-width: 900px; background: var(--surface);
    border: 1px solid var(--border); border-radius: var(--radius);
    overflow: hidden; margin-bottom: 16px;
  }
  .card-header {
    background: var(--surface2); padding: 12px 20px;
    display: flex; gap: 24px; align-items: center;
    border-bottom: 1px solid var(--border); flex-wrap: wrap;
  }
  .meta-item { font-size: 0.82rem; }
  .meta-label { color: var(--muted); margin-right: 4px; }
  .meta-value { color: var(--text); font-weight: 600; }
  video {
    width: 100%; display: block; background: #000;
    max-height: 420px; object-fit: contain;
  }

  .subtitle-section {
    padding: 14px 20px; border-bottom: 1px solid var(--border);
    background: var(--bg);
  }
  .subtitle-label { font-size: 0.72rem; font-weight: 600; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; }
  .subtitle-text { font-size: 1.05rem; color: var(--text); font-style: italic; }

  .gloss-section { padding: 16px 20px; }
  .gloss-label {
    font-size: 0.72rem; font-weight: 600; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px;
    display: flex; align-items: center; gap: 8px;
  }
  .kbd {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 4px; padding: 1px 6px; font-size: 0.7rem;
    color: var(--text); font-family: monospace;
  }
  .gloss-input {
    width: 100%; background: var(--surface2); border: 2px solid var(--border);
    color: var(--text); padding: 12px 16px; border-radius: 8px;
    font-size: 1.1rem; font-family: monospace; font-weight: 700;
    letter-spacing: 0.04em; transition: border-color 0.2s;
  }
  .gloss-input:focus { outline: none; border-color: var(--accent); }
  .gloss-input.modified { border-color: var(--yellow); }

  .change-indicator {
    margin-top: 8px; font-size: 0.82rem; color: var(--yellow);
    min-height: 20px;
  }

  .action-bar {
    width: 100%; max-width: 900px; display: flex; gap: 12px;
  }
  .btn {
    flex: 1; padding: 14px; border: none; border-radius: var(--radius);
    font-size: 1rem; font-weight: 700; cursor: pointer;
    transition: opacity 0.15s, transform 0.1s; letter-spacing: 0.03em;
  }
  .btn:active { transform: scale(0.98); }
  .btn:disabled { opacity: 0.45; cursor: not-allowed; }
  .btn-confirm {
    background: linear-gradient(135deg, var(--green), #2bb57c);
    color: #071a11;
  }
  .btn-skip {
    flex: 0.5; background: var(--surface); color: var(--muted);
    border: 1px solid var(--border);
  }
  .btn-skip:hover:not(:disabled) { border-color: var(--yellow); color: var(--yellow); }

  .toast {
    position: fixed; bottom: 30px; left: 50%;
    transform: translateX(-50%) translateY(80px);
    background: var(--surface2); border: 1px solid var(--border);
    padding: 12px 24px; border-radius: 30px; font-size: 0.9rem;
    font-weight: 600; transition: transform 0.3s cubic-bezier(0.175,0.885,0.32,1.275);
    z-index: 999; white-space: nowrap;
  }
  .toast.show { transform: translateX(-50%) translateY(0); }
  .toast.success { border-color: var(--green); color: var(--green); }
  .toast.error { border-color: var(--red); color: var(--red); }
  .toast.info { border-color: var(--accent); color: var(--accent); }
  .toast.edited { border-color: var(--yellow); color: var(--yellow); }

  .done-screen { text-align: center; padding: 60px 20px; }
  .done-screen h2 { font-size: 2rem; margin-bottom: 12px; color: var(--green); }
  .done-screen p { color: var(--muted); }

  .shortcuts {
    width: 100%; max-width: 900px; margin-top: 16px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 14px 20px;
    display: flex; flex-wrap: wrap; gap: 12px 24px;
  }
  .shortcut { font-size: 0.78rem; color: var(--muted); }

  .timeline {
    position: relative; margin: 12px 20px; height: 32px;
    background: var(--surface2); border-radius: 6px;
    border: 1px solid var(--border); cursor: pointer; overflow: hidden;
  }
  .timeline-fill {
    position: absolute; top: 0; bottom: 0; left: 0;
    background: rgba(91,140,255,0.3); pointer-events: none;
  }
  .timeline-cursor {
    position: absolute; top: 0; bottom: 0; width: 2px;
    background: white; pointer-events: none; transform: translateX(-50%);
  }
</style>
</head>
<body>

<div class="header">
  <div class="title">Auslan Gloss Validator</div>
  <div class="progress-info" id="progress-info">Loading…</div>
</div>
<div class="progress-bar-wrap">
  <div class="progress-bar-fill" id="progress-bar" style="width:0%"></div>
</div>

<div id="clip-card" class="card" style="display:none">
  <div class="card-header">
    <div class="meta-item"><span class="meta-label">Clip</span><span class="meta-value" id="meta-clip">—</span></div>
    <div class="meta-item"><span class="meta-label">Signer</span><span class="meta-value" id="meta-signer">—</span></div>
    <div class="meta-item"><span class="meta-label">Index</span><span class="meta-value" id="meta-idx">—</span></div>
  </div>

  <video id="vid" preload="auto" controls playsinline muted
         style="width:100%;display:block;background:#000;max-height:420px;object-fit:contain"></video>

  <div class="timeline" id="timeline">
    <div class="timeline-fill" id="tl-fill" style="width:0%"></div>
    <div class="timeline-cursor" id="tl-cursor" style="left:0%"></div>
  </div>

  <div class="subtitle-section">
    <div class="subtitle-label">English Subtitle</div>
    <div class="subtitle-text" id="meta-subtitle">—</div>
  </div>

  <div class="gloss-section">
    <div class="gloss-label">
      Gloss Annotation
      <span class="kbd">E</span> to edit
    </div>
    <input class="gloss-input" id="gloss-input" type="text"
           placeholder="Enter corrected gloss sequence…"
           autocomplete="off" autocorrect="off" spellcheck="false">
    <div class="change-indicator" id="change-indicator"></div>
  </div>
</div>

<div id="done-card" class="card done-screen" style="display:none">
  <h2>✓ All done!</h2>
  <p id="done-msg"></p>
  <p style="margin-top:8px;">All test clips validated. Close this window.</p>
</div>

<div class="action-bar" id="action-bar" style="display:none">
  <button class="btn btn-confirm" id="btn-confirm" onclick="confirmClip()">
    ✓ Confirm &nbsp;<small style="opacity:.7;font-weight:400">[Enter]</small>
  </button>
  <button class="btn btn-skip" id="btn-skip" onclick="skipClip()">
    Skip &nbsp;<small style="opacity:.7;font-weight:400">[X]</small>
  </button>
</div>

<div class="toast" id="toast"></div>

<div class="shortcuts">
  <span class="shortcut"><span class="kbd">Space</span> Play/Pause</span>
  <span class="shortcut"><span class="kbd">E</span> Edit gloss</span>
  <span class="shortcut"><span class="kbd">Enter</span> Confirm</span>
  <span class="shortcut"><span class="kbd">X</span> Skip (comes back)</span>
  <span class="shortcut"><span class="kbd">← →</span> ±0.5s</span>
  <span class="shortcut"><span class="kbd">[ ]</span> ±0.1s</span>
  <span class="shortcut"><span class="kbd">, .</span> ±1s</span>
</div>

<script>
let currentStatus = {};
let originalGloss = '';
let saving = false;

const $ = id => document.getElementById(id);
const vid = $('vid');

// Video events — wired once
vid.addEventListener('timeupdate', () => {
  if (!vid.duration) return;
  const pct = (vid.currentTime / vid.duration) * 100;
  $('tl-fill').style.width = pct + '%';
  $('tl-cursor').style.left = pct + '%';
});
vid.addEventListener('loadedmetadata', () => {
  vid.play().catch(() => {});
});
vid.addEventListener('error', () => console.error('Video error:', vid.error));

// Timeline seek
const tl = $('timeline');
let dragging = false;
function seekFromEvent(e) {
  const rect = tl.getBoundingClientRect();
  const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  if (vid.duration) vid.currentTime = ratio * vid.duration;
}
tl.addEventListener('mousedown', e => { dragging = true; seekFromEvent(e); });
document.addEventListener('mousemove', e => { if (dragging) seekFromEvent(e); });
document.addEventListener('mouseup', () => dragging = false);

// Gloss input change indicator
$('gloss-input').addEventListener('input', () => {
  const current = $('gloss-input').value.trim();
  const changed = current !== originalGloss;
  $('gloss-input').className = 'gloss-input' + (changed ? ' modified' : '');
  $('change-indicator').textContent = changed
    ? `⚠ Changed from: "${originalGloss}"`
    : '';
});

function showToast(msg, type='info', duration=2200) {
  const t = $('toast');
  t.textContent = msg;
  t.className = `toast ${type} show`;
  setTimeout(() => t.classList.remove('show'), duration);
}

function buildUI(st) {
  currentStatus = st;

  if (st.exhausted) {
    $('clip-card').style.display = 'none';
    $('action-bar').style.display = 'none';
    $('done-card').style.display = 'block';
    $('done-msg').textContent = `${st.done} clips validated.`;
    $('progress-info').textContent = `${st.done} / ${st.total} done`;
    $('progress-bar').style.width = '100%';
    return;
  }

  // Update metadata
  $('meta-clip').textContent = st.clip_name;
  $('meta-signer').textContent = st.signer_id;
  $('meta-idx').textContent = `${st.current_idx + 1} / ${st.total}`;
  $('meta-subtitle').textContent = st.subtitle;

  // Set gloss input
  originalGloss = st.gloss;
  $('gloss-input').value = st.gloss;
  $('gloss-input').className = 'gloss-input';
  $('change-indicator').textContent = '';

  // Progress
  const pct = st.total > 0 ? (st.done / st.total) * 100 : 0;
  $('progress-bar').style.width = pct + '%';
  $('progress-info').innerHTML = `
    <strong>${st.pending}</strong> remaining &nbsp;|&nbsp;
    ${st.done} done &nbsp;|&nbsp; ${st.total} total<br>
    <span style="color:var(--muted)">${st.clip_name}</span>`;

  // Load new video
  vid.src = `/api/video?clip=${st.clip_name}&t=${Date.now()}`;
  vid.load();

  $('clip-card').style.display = 'block';
  $('action-bar').style.display = 'flex';
  $('done-card').style.display = 'none';
  $('btn-confirm').disabled = false;
  $('btn-skip').disabled = false;
  saving = false;
}

async function confirmClip() {
  if (saving) return;
  saving = true;
  $('btn-confirm').disabled = true;

  const gloss = $('gloss-input').value.trim().toUpperCase();
  const wasEdited = gloss !== originalGloss;

  try {
    const res = await fetch('/api/confirm', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({gloss})
    });
    const data = await res.json();
    if (data.ok) {
      if (wasEdited) {
        showToast(`✏ Saved: "${data.old_gloss}" → "${data.new_gloss}"`, 'edited', 3000);
      } else {
        showToast('✓ Confirmed', 'success', 1500);
      }
      setTimeout(() => loadStatus(), 400);
    } else {
      showToast('Error: ' + data.error, 'error', 4000);
      saving = false;
      $('btn-confirm').disabled = false;
    }
  } catch {
    showToast('Network error', 'error');
    saving = false;
  }
}

async function skipClip() {
  if (saving) return;
  saving = true;
  $('btn-skip').disabled = true;
  await fetch('/api/skip', {method: 'POST'});
  showToast('Skipped — comes back next session', 'info', 1800);
  setTimeout(() => loadStatus(), 300);
}

async function loadStatus() {
  const res = await fetch('/api/status');
  const data = await res.json();
  buildUI(data);
}

// Keyboard shortcuts
window.addEventListener('keydown', e => {
  const inInput = document.activeElement === $('gloss-input');
  const dur = vid.duration || 0;

  if (e.key === 'Enter' && !inInput) { e.preventDefault(); confirmClip(); return; }
  if (e.key === 'Enter' && inInput)  { e.preventDefault(); $('gloss-input').blur(); confirmClip(); return; }
  if (e.key === 'Escape' && inInput) { $('gloss-input').blur(); return; }

  if (inInput) return;  // don't steal other keys from input

  switch (e.key) {
    case ' ':
      e.preventDefault(); e.stopPropagation();
      vid.paused ? vid.play() : vid.pause(); break;
    case 'e': case 'E':
      e.preventDefault();
      $('gloss-input').focus();
      $('gloss-input').select(); break;
    case 'x': case 'X':
      e.preventDefault(); skipClip(); break;
    case 'ArrowLeft':
      e.preventDefault(); vid.currentTime = Math.max(0, vid.currentTime - 0.5); break;
    case 'ArrowRight':
      e.preventDefault(); vid.currentTime = Math.min(dur, vid.currentTime + 0.5); break;
    case '[':
      e.preventDefault(); vid.currentTime = Math.max(0, vid.currentTime - 0.1); break;
    case ']':
      e.preventDefault(); vid.currentTime = Math.min(dur, vid.currentTime + 0.1); break;
    case ',':
      e.preventDefault(); vid.currentTime = Math.max(0, vid.currentTime - 1); break;
    case '.':
      e.preventDefault(); vid.currentTime = Math.min(dur, vid.currentTime + 1); break;
  }
}, { capture: true });

loadStatus();
</script>
</body>
</html>"""


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    # Load Excel
    print(f"\nLoading Excel: {EXCEL_PATH}")
    if EXCEL_PATH.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(EXCEL_PATH)
    else:
        df = pd.read_csv(EXCEL_PATH)

    STATE["df"]         = df
    STATE["excel_path"] = EXCEL_PATH

    # Filter to the target split
    split_df = df[df["Split"] == SPLIT_TO_REVIEW].copy()
    clips = []
    skipped_no_video = 0
    skipped_no_focus = 0

    for _, row in split_df.iterrows():
        clip_name = str(row["Video_Clip_Name"])
        gloss_str = str(row.get("gloss", ""))

        # Focus filter — only include clips containing a focus gloss
        if FOCUS_GLOSSES is not None:
            tokens = {t.upper() for t in gloss_str.split()}
            if not tokens & FOCUS_GLOSSES:
                skipped_no_focus += 1
                continue

        video_path = Path(VIDEO_DIR) / f"{clip_name}_signer.mp4"
        if not video_path.exists():
            skipped_no_video += 1
            continue

        clips.append({
            "clip_name": clip_name,
            "subtitle":  str(row.get("Subtitle", "")),
            "gloss":     gloss_str,
            "signer_id": str(row.get("Signer_ID", "")),
        })

    STATE["clips"]             = clips
    STATE["video_dir"]         = VIDEO_DIR
    STATE["transcode_cache"]   = str(Path(VIDEO_DIR) / ".transcode_cache")

    load_validation_log()

    already_done = len([c for c in clips if c["clip_name"] in STATE["done_keys"]])

    print(f"Split:          {SPLIT_TO_REVIEW}")
    if FOCUS_GLOSSES:
        print(f"Focus glosses:  {', '.join(sorted(FOCUS_GLOSSES))}")
        print(f"Filtered out:   {skipped_no_focus} clips (no focus gloss)")
    print(f"Total clips:    {len(clips)}  ({already_done} already validated, {len(clips)-already_done} to go)")
    if skipped_no_video:
        print(f"Missing videos: {skipped_no_video} (skipped)")
    print(f"Validation log: {VALIDATION_LOG}")

    url = f"http://localhost:{PORT}"
    print(f"\n→ Opening {url}\n")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host="0.0.0.0", port=PORT, debug=False)


if __name__ == "__main__":
    main()