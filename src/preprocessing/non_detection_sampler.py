#!/usr/bin/env python3
"""
Auslan Non-Detection Clip Sampler
===================================
Reads the gloss-clips manifest, finds gaps in each source video that fall
OUTSIDE every clipped gloss window (+margin), randomly samples one clip per
video (duration 0.3–1.0 s), and presents them for review.

    Enter  — Accept clip (save to output dir + manifest)
    X      — Skip / reject

Usage:
    python non_detection_sampler.py

Edit the DEFAULTS block at the bottom of this file to configure paths.
"""

import argparse
import json
import os
import random
import re
import subprocess
import threading
import webbrowser
from pathlib import Path

import pandas as pd
from flask import Flask, Response, jsonify, request, stream_with_context

app = Flask(__name__)

# ─── Global State ──────────────────────────────────────────────────────────────
STATE = {
    "candidates":     [],   # list of candidate dicts (see build_candidates)
    "current_idx":    0,
    "done_keys":      set(),  # indices already handled in this session
    "output_dir":     None,
    "manifest_path":  None,   # non-detection manifest (output)
    "transcode_cache": None,
}

# ─── Sampling Logic ────────────────────────────────────────────────────────────

def get_video_duration(video_path: str) -> float:
    """Use ffprobe to get video duration in seconds."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def merge_windows(windows):
    """Merge overlapping/adjacent [start, end] windows."""
    if not windows:
        return []
    windows = sorted(windows)
    merged = [windows[0]]
    for s, e in windows[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def find_gaps(duration: float, blocked: list, margin: float, min_gap: float):
    """
    Return list of (gap_start, gap_end) tuples where a clip could be placed,
    given the video duration, blocked windows (with margin applied), and the
    minimum gap size required.
    """
    # Expand each window by margin
    expanded = [[max(0.0, s - margin), min(duration, e + margin)] for s, e in blocked]
    merged = merge_windows(expanded)

    # Build the free intervals
    gaps = []
    cursor = 0.0
    for s, e in merged:
        if s - cursor >= min_gap:
            gaps.append((cursor, s))
        cursor = e
    if duration - cursor >= min_gap:
        gaps.append((cursor, duration))
    return gaps


def sample_clip_from_gaps(gaps, min_dur=0.3, max_dur=1.0):
    """
    Pick a random clip of random duration [min_dur, max_dur] from within one
    of the provided gaps.  Returns (start, end, duration) or None if no gap
    is large enough.
    """
    # Weight gaps by their length so larger gaps are sampled more fairly
    eligible = [(s, e) for s, e in gaps if (e - s) >= min_dur]
    if not eligible:
        return None

    weights = [e - s for s, e in eligible]
    (gs, ge) = random.choices(eligible, weights=weights, k=1)[0]

    clip_dur = round(random.uniform(min_dur, min(max_dur, ge - gs)), 3)
    max_start = ge - clip_dur
    start = round(random.uniform(gs, max_start), 3)
    end = round(start + clip_dur, 3)
    return start, end, clip_dur


def build_candidates(
    gloss_manifest_path: str,
    video_dir: str,
    margin: float = 0.1,
    min_gap: float = 0.5,   # minimum free gap to bother sampling from
    min_dur: float = 0.3,
    max_dur: float = 1.0,
    seed: int = 42,
) -> list:
    """
    For every source video that has at least one *saved* gloss clip, attempt
    to sample one non-detection candidate.
    """
    random.seed(seed)

    df = pd.read_excel(gloss_manifest_path)
    # Only consider rows with a successfully saved clip
    saved = df[df["status"] == "saved"].copy()
    if saved.empty:
        return []

    # Group by source_clip → collect all blocked windows
    groups = saved.groupby("source_clip")
    candidates = []

    for source_clip, rows in groups:
        video_path = Path(video_dir) / f"{source_clip}_signer.mp4"
        if not video_path.exists():
            print(f"  [skip] video not found: {video_path}")
            continue

        # Collect every blocked window for this video (all glosses)
        blocked = []
        for _, row in rows.iterrows():
            s = row.get("start_sec")
            e = row.get("end_sec")
            if pd.notna(s) and pd.notna(e):
                blocked.append([float(s), float(e)])

        duration = get_video_duration(str(video_path))
        if duration <= 0:
            print(f"  [skip] could not read duration: {video_path}")
            continue

        gaps = find_gaps(duration, blocked, margin=margin, min_gap=min_gap)
        result = sample_clip_from_gaps(gaps, min_dur=min_dur, max_dur=max_dur)
        if result is None:
            print(f"  [skip] no usable gap in {source_clip} (duration={duration:.1f}s, {len(blocked)} windows)")
            continue

        start, end, clip_dur = result
        # Collect the gloss labels that were blocked (for display)
        glosses_in_video = sorted(rows["gloss"].dropna().unique().tolist())

        candidates.append({
            "source_clip":      source_clip,
            "video_path":       str(video_path),
            "start":            start,
            "end":              end,
            "duration":         clip_dur,
            "glosses_blocked":  glosses_in_video,
            "video_duration":   duration,
            "n_blocked_windows": len(blocked),
        })

    return candidates


# ─── Manifest helpers ──────────────────────────────────────────────────────────

def load_done_keys(manifest_path: str) -> set:
    done = set()
    if Path(manifest_path).exists():
        try:
            df = pd.read_excel(manifest_path)
            for _, row in df.iterrows():
                key = f"{row.get('source_clip')}_{row.get('start_sec')}"
                done.add(key)
        except Exception:
            pass
    return done


def append_manifest(manifest_path: str, row_dict: dict):
    if Path(manifest_path).exists():
        try:
            df = pd.read_excel(manifest_path)
        except Exception:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
    df.to_excel(manifest_path, index=False)


def candidate_key(c: dict) -> str:
    return f"{c['source_clip']}_{c['start']}"


# ─── Video serving ─────────────────────────────────────────────────────────────

def get_transcoded_path(original_path: str) -> str:
    cache_dir = Path(STATE["transcode_cache"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(original_path).stem
    cached = cache_dir / f"{stem}_h264.mp4"
    if not cached.exists():
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", original_path,
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                "-movflags", "+faststart", "-an",
                str(cached),
            ],
            capture_output=True, check=True,
        )
    return str(cached)


def serve_video_range(path: str):
    file_size = os.path.getsize(path)
    range_header = request.headers.get("Range")
    if range_header:
        m = re.search(r"bytes=(\d+)-(\d*)", range_header)
        byte_start = int(m.group(1)) if m else 0
        byte_end = int(m.group(2)) if m and m.group(2) else file_size - 1
        byte_end = min(byte_end, file_size - 1)
        length = byte_end - byte_start + 1

        def gen_range():
            with open(path, "rb") as f:
                f.seek(byte_start)
                rem = length
                while rem > 0:
                    chunk = f.read(min(65536, rem))
                    if not chunk:
                        break
                    rem -= len(chunk)
                    yield chunk

        return Response(
            stream_with_context(gen_range()), status=206,
            headers={
                "Content-Range":  f"bytes {byte_start}-{byte_end}/{file_size}",
                "Accept-Ranges":  "bytes",
                "Content-Length": str(length),
                "Content-Type":   "video/mp4",
            },
        )

    def gen_full():
        with open(path, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                yield chunk

    return Response(
        stream_with_context(gen_full()), status=200,
        headers={
            "Accept-Ranges":  "bytes",
            "Content-Length": str(file_size),
            "Content-Type":   "video/mp4",
        },
    )


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return HTML_PAGE


@app.route("/api/status")
def api_status():
    candidates = STATE["candidates"]
    done_keys  = STATE["done_keys"]
    idx        = STATE["current_idx"]
    done_count = len(done_keys)

    # Advance to the next un-done candidate
    for offset in range(len(candidates)):
        i = (idx + offset) % len(candidates)
        c = candidates[i]
        if candidate_key(c) not in done_keys:
            STATE["current_idx"] = i
            pending = sum(1 for c2 in candidates if candidate_key(c2) not in done_keys)
            return jsonify({
                "exhausted":         False,
                "current_idx":       i,
                "total":             len(candidates),
                "done":              done_count,
                "pending":           pending,
                "source_clip":       c["source_clip"],
                "start":             c["start"],
                "end":               c["end"],
                "duration":          c["duration"],
                "glosses_blocked":   c["glosses_blocked"],
                "video_duration":    c["video_duration"],
                "n_blocked_windows": c["n_blocked_windows"],
            })

    return jsonify({
        "exhausted": True,
        "total":     len(candidates),
        "done":      done_count,
        "pending":   0,
    })


@app.route("/api/preview_video")
def api_preview_video():
    """Serve a short trimmed preview (the candidate clip window) for instant playback."""
    candidates = STATE["candidates"]
    idx = STATE["current_idx"]
    if idx >= len(candidates):
        return "No video", 404

    c = candidates[idx]
    cache_dir = Path(STATE["transcode_cache"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    preview_name = f"{c['source_clip']}_{c['start']:.3f}_{c['end']:.3f}_preview.mp4"
    preview_path = cache_dir / preview_name

    if not preview_path.exists():
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", c["video_path"],
                    "-ss", str(c["start"]),
                    "-to", str(c["end"]),
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                    "-movflags", "+faststart", "-an",
                    str(preview_path),
                ],
                capture_output=True, check=True, timeout=30,
            )
        except subprocess.CalledProcessError as e:
            return f"Preview encode failed: {e}", 500

    return serve_video_range(str(preview_path))


@app.route("/api/context_video")
def api_context_video():
    """Serve the full source video (transcoded) for context."""
    candidates = STATE["candidates"]
    idx = STATE["current_idx"]
    if idx >= len(candidates):
        return "No video", 404
    c = candidates[idx]
    try:
        path = get_transcoded_path(c["video_path"])
    except subprocess.CalledProcessError as e:
        return f"Transcode failed: {e}", 500
    return serve_video_range(path)


@app.route("/api/accept", methods=["POST"])
def api_accept():
    candidates = STATE["candidates"]
    idx = STATE["current_idx"]
    c = candidates[idx]
    out_dir = Path(STATE["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Unique output filename
    base = f"{c['source_clip']}_nondet"
    counter = 1
    while (out_dir / f"{base}_{counter}.mp4").exists():
        counter += 1
    out_filename = f"{base}_{counter}.mp4"
    out_path = out_dir / out_filename

    cmd = [
        "ffmpeg", "-y",
        "-i", c["video_path"],
        "-ss", str(c["start"]),
        "-to", str(c["end"]),
        "-c:v", "libx264", "-c:a", "aac",
        "-loglevel", "error",
        str(out_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return jsonify({"ok": False, "error": result.stderr}), 500
    except FileNotFoundError:
        return jsonify({"ok": False, "error": "ffmpeg not found"}), 500

    row = {
        "source_clip":       c["source_clip"],
        "start_sec":         c["start"],
        "end_sec":           c["end"],
        "duration_sec":      c["duration"],
        "glosses_blocked":   ", ".join(c["glosses_blocked"]),
        "n_blocked_windows": c["n_blocked_windows"],
        "output_file":       out_filename,
        "status":            "accepted",
    }
    append_manifest(STATE["manifest_path"], row)
    STATE["done_keys"].add(candidate_key(c))
    STATE["current_idx"] = idx + 1
    return jsonify({"ok": True, "saved_as": out_filename})


@app.route("/api/skip", methods=["POST"])
def api_skip():
    candidates = STATE["candidates"]
    idx = STATE["current_idx"]
    c = candidates[idx]

    row = {
        "source_clip":       c["source_clip"],
        "start_sec":         c["start"],
        "end_sec":           c["end"],
        "duration_sec":      c["duration"],
        "glosses_blocked":   ", ".join(c["glosses_blocked"]),
        "n_blocked_windows": c["n_blocked_windows"],
        "output_file":       None,
        "status":            "skipped",
    }
    append_manifest(STATE["manifest_path"], row)
    STATE["done_keys"].add(candidate_key(c))
    STATE["current_idx"] = idx + 1
    return jsonify({"ok": True})


# ─── HTML UI ───────────────────────────────────────────────────────────────────

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Non-Detection Sampler</title>
<style>
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #22263a;
    --border: #2e3347;
    --accent: #f6a623;
    --accent2: #f56565;
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
  .badge {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: #1a0a00;
    font-size: 1.1rem;
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

  .info-strip {
    padding: 10px 20px;
    background: rgba(246,166,35,0.07);
    border-bottom: 1px solid var(--border);
    font-size: 0.8rem;
    color: var(--muted);
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    align-items: center;
  }
  .info-strip .blocked-label {
    color: var(--accent);
    font-weight: 600;
  }
  .token {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.74rem;
    font-weight: 600;
    background: rgba(246,166,35,0.15);
    color: var(--accent);
    border: 1px solid rgba(246,166,35,0.35);
    letter-spacing: 0.04em;
    margin: 2px 3px 2px 0;
  }

  /* Two-panel layout: preview + context */
  .video-panels {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0;
    border-bottom: 1px solid var(--border);
  }
  .panel {
    display: flex;
    flex-direction: column;
  }
  .panel + .panel {
    border-left: 1px solid var(--border);
  }
  .panel-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    padding: 8px 14px 6px;
    background: var(--surface2);
    border-bottom: 1px solid var(--border);
  }
  .panel-label.highlight { color: var(--green); }
  video {
    width: 100%;
    display: block;
    background: #000;
    max-height: 340px;
    object-fit: contain;
  }

  /* Timeline for context video */
  .timeline {
    position: relative;
    margin: 10px 14px;
    height: 32px;
    background: var(--surface2);
    border-radius: 6px;
    border: 1px solid var(--border);
    overflow: hidden;
    cursor: pointer;
  }
  .tl-play { position:absolute; top:0; bottom:0; left:0; background:rgba(91,140,255,0.2); pointer-events:none; }
  .tl-blocked { position:absolute; top:0; bottom:0; background:rgba(246,166,35,0.25); border-left:1px solid var(--accent); border-right:1px solid var(--accent); pointer-events:none; }
  .tl-candidate { position:absolute; top:0; bottom:0; background:rgba(62,207,142,0.35); border-left:2px solid var(--green); border-right:2px solid var(--green); pointer-events:none; }
  .tl-cursor { position:absolute; top:0; bottom:0; width:2px; background:white; pointer-events:none; transform:translateX(-50%); }
  .tl-hint { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); font-size:0.7rem; color:var(--muted); pointer-events:none; white-space:nowrap; }

  .action-bar {
    width: 100%;
    max-width: 900px;
    display: flex;
    gap: 12px;
  }
  .btn {
    flex: 1;
    padding: 16px;
    border: none;
    border-radius: var(--radius);
    font-size: 1.05rem;
    font-weight: 700;
    cursor: pointer;
    transition: opacity 0.15s, transform 0.1s;
    letter-spacing: 0.03em;
  }
  .btn:active { transform: scale(0.98); }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-accept {
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
  .toast.error   { border-color: var(--red);   color: var(--red);   }
  .toast.info    { border-color: var(--accent); color: var(--accent); }

  .done-screen { text-align:center; padding:60px 20px; }
  .done-screen h2 { font-size:2rem; margin-bottom:12px; color:var(--green); }
  .done-screen p  { color:var(--muted); }

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

  .legend {
    display:flex; gap:16px; flex-wrap:wrap; padding: 8px 14px 10px;
    font-size: 0.75rem; color: var(--muted);
  }
  .legend-item { display:flex; align-items:center; gap:5px; }
  .legend-swatch { width:14px; height:14px; border-radius:3px; flex-shrink:0; }
</style>
</head>
<body>

<div class="header">
  <div>
    <div style="font-size:0.75rem;color:var(--muted);margin-bottom:4px;">Non-Detection Sampler</div>
    <div class="badge">NON-DETECTION CLASS</div>
  </div>
  <div class="progress-info" id="progress-info">Loading…</div>
</div>

<div class="progress-bar-wrap">
  <div class="progress-bar-fill" id="progress-bar" style="width:0%"></div>
</div>

<div id="clip-card" class="card" style="display:none">
  <div class="card-header">
    <div class="meta-item"><span class="meta-label">Source clip</span><span class="meta-value" id="meta-clip">—</span></div>
    <div class="meta-item"><span class="meta-label">Candidate window</span><span class="meta-value" id="meta-window">—</span></div>
    <div class="meta-item"><span class="meta-label">Duration</span><span class="meta-value" id="meta-dur">—</span></div>
    <div class="meta-item"><span class="meta-label">Index</span><span class="meta-value" id="meta-idx">—</span></div>
  </div>

  <div class="info-strip">
    <span>Glosses blocked in this video:</span>
    <span id="blocked-tokens"></span>
    <span id="meta-nblocked" style="margin-left:auto"></span>
  </div>

  <div class="video-panels">
    <!-- Left: looping preview of just the candidate clip -->
    <div class="panel">
      <div class="panel-label highlight">▶ Candidate clip (loops)</div>
      <video id="vid-preview" preload="auto" playsinline muted loop
             style="width:100%;display:block;background:#000;max-height:340px;object-fit:contain"></video>
    </div>
    <!-- Right: full source video for context -->
    <div class="panel">
      <div class="panel-label">Full source video (context)</div>
      <video id="vid-context" preload="metadata" controls playsinline muted
             style="width:100%;display:block;background:#000;max-height:300px;object-fit:contain"></video>
      <div class="timeline" id="timeline">
        <div class="tl-play"      id="tl-play"      style="width:0%"></div>
        <div id="tl-blocked-container"></div>
        <div class="tl-candidate" id="tl-candidate" style="display:none"></div>
        <div class="tl-cursor"    id="tl-cursor"    style="left:0%"></div>
        <div class="tl-hint">Blocked windows (orange) · Candidate (green)</div>
      </div>
      <div class="legend">
        <div class="legend-item">
          <div class="legend-swatch" style="background:rgba(246,166,35,0.35);border:1px solid var(--accent)"></div>
          Clipped gloss windows
        </div>
        <div class="legend-item">
          <div class="legend-swatch" style="background:rgba(62,207,142,0.35);border:1px solid var(--green)"></div>
          Candidate (non-detection)
        </div>
      </div>
    </div>
  </div>
</div>

<div id="done-card" class="card done-screen" style="display:none">
  <h2>✓ All done!</h2>
  <p id="done-msg"></p>
  <p style="margin-top:8px;">You can close this window or re-run with a different manifest.</p>
</div>

<div class="action-bar" id="action-bar" style="display:none">
  <button class="btn btn-accept" id="btn-accept" onclick="acceptClip()">
    ✓ Accept &nbsp;<small style="opacity:.7;font-weight:400">[Enter]</small>
  </button>
  <button class="btn btn-skip" id="btn-skip" onclick="skipClip()">
    Skip / Reject &nbsp;<small style="opacity:.7;font-weight:400">[X]</small>
  </button>
</div>

<div class="toast" id="toast"></div>

<div class="shortcuts">
  <span class="shortcut"><span class="kbd">Enter</span> Accept clip</span>
  <span class="shortcut"><span class="kbd">X</span> Skip / Reject</span>
  <span class="shortcut"><span class="kbd">Space</span> Play/Pause context video</span>
  <span class="shortcut"><span class="kbd">←</span><span class="kbd">→</span> ±0.5s context</span>
</div>

<script>
let currentStatus = {};
let saving = false;
let blockedWindows = [];   // [{start, end}] for the current video
let videoDuration  = 0;
let candidateStart = 0;
let candidateEnd   = 0;

const $ = id => document.getElementById(id);
const vidPreview = $('vid-preview');
const vidContext  = $('vid-context');

// Context video timeline
vidContext.addEventListener('timeupdate', () => {
  if (!vidContext.duration) return;
  const pct = (vidContext.currentTime / vidContext.duration) * 100;
  $('tl-play').style.width = pct + '%';
  $('tl-cursor').style.left = pct + '%';
});

const tl = $('timeline');
let dragging = false;
function seekCtx(e) {
  const rect = tl.getBoundingClientRect();
  const x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
  const ratio = Math.max(0, Math.min(1, x / rect.width));
  if (vidContext.duration) vidContext.currentTime = ratio * vidContext.duration;
}
tl.addEventListener('mousedown', e => { dragging = true; seekCtx(e); });
document.addEventListener('mousemove', e => { if (dragging) seekCtx(e); });
document.addEventListener('mouseup', () => dragging = false);

function showToast(msg, type = 'info', duration = 2500) {
  const t = $('toast');
  t.textContent = msg;
  t.className = `toast ${type} show`;
  setTimeout(() => t.classList.remove('show'), duration);
}

function drawTimeline(dur, blocked, candStart, candEnd) {
  if (!dur) return;
  // Blocked overlays
  const container = $('tl-blocked-container');
  container.innerHTML = '';
  for (const w of blocked) {
    const div = document.createElement('div');
    div.className = 'tl-blocked';
    div.style.left  = (w.start / dur * 100) + '%';
    div.style.width = ((w.end - w.start) / dur * 100) + '%';
    container.appendChild(div);
  }
  // Candidate
  const cand = $('tl-candidate');
  cand.style.left    = (candStart / dur * 100) + '%';
  cand.style.width   = ((candEnd - candStart) / dur * 100) + '%';
  cand.style.display = 'block';
}

function buildUI(st) {
  currentStatus = st;

  if (st.exhausted) {
    $('clip-card').style.display  = 'none';
    $('action-bar').style.display = 'none';
    $('done-card').style.display  = 'block';
    $('done-msg').textContent = `${st.done} candidates processed.`;
    $('progress-info').textContent = `${st.done} / ${st.total} done`;
    $('progress-bar').style.width = '100%';
    return;
  }

  // Metadata
  $('meta-clip').textContent   = st.source_clip;
  $('meta-window').textContent = `${st.start.toFixed(3)}s → ${st.end.toFixed(3)}s`;
  $('meta-dur').textContent    = `${st.duration.toFixed(3)}s`;
  $('meta-idx').textContent    = `${st.current_idx + 1} / ${st.total}`;
  $('meta-nblocked').textContent = `${st.n_blocked_windows} blocked window${st.n_blocked_windows !== 1 ? 's' : ''}`;

  // Blocked tokens
  $('blocked-tokens').innerHTML = (st.glosses_blocked || [])
    .map(g => `<span class="token">${g}</span>`).join('');

  const pct = st.total > 0 ? (st.done / st.total * 100) : 0;
  $('progress-bar').style.width = pct + '%';
  $('progress-info').innerHTML  = `
    <strong>${st.pending}</strong> remaining &nbsp;|&nbsp;
    ${st.done} done &nbsp;|&nbsp; ${st.total} total<br>
    <span style="color:var(--muted)">${st.source_clip}</span>`;

  // Update state for timeline
  videoDuration  = st.video_duration;
  candidateStart = st.start;
  candidateEnd   = st.end;
  // We don't have blocked windows array from status; store candidate for timeline
  blockedWindows = [];   // re-drawn once context video loads

  // Load preview clip (loops automatically)
  vidPreview.src = `/api/preview_video?idx=${st.current_idx}&t=${Date.now()}`;
  vidPreview.load();
  vidPreview.play().catch(() => {});

  // Load full context video
  vidContext.src = `/api/context_video?idx=${st.current_idx}&t=${Date.now()}`;
  vidContext.load();

  // Seek context to candidate start once metadata loads
  vidContext.onloadedmetadata = () => {
    vidContext.currentTime = Math.max(0, candidateStart - 1.0);
    // Draw timeline (we only have candidate bounds, not all blocked windows from this route)
    drawTimeline(vidContext.duration, [], candidateStart, candidateEnd);
  };

  $('clip-card').style.display  = 'block';
  $('action-bar').style.display = 'flex';
  $('done-card').style.display  = 'none';
  $('btn-accept').disabled = false;
  $('btn-skip').disabled   = false;
  saving = false;
}

async function acceptClip() {
  if (saving) return;
  saving = true;
  $('btn-accept').disabled = true;
  $('btn-accept').textContent = 'Saving…';
  try {
    const res  = await fetch('/api/accept', { method: 'POST' });
    const data = await res.json();
    if (data.ok) {
      showToast(`✓ Saved: ${data.saved_as}`, 'success');
      setTimeout(loadStatus, 400);
    } else {
      showToast('Error: ' + data.error, 'error', 4000);
      saving = false;
      $('btn-accept').disabled = false;
      $('btn-accept').textContent = '✓ Accept   [Enter]';
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
  try {
    await fetch('/api/skip', { method: 'POST' });
    showToast('Skipped', 'info', 1200);
    setTimeout(loadStatus, 300);
  } finally {
    saving = false;
  }
}

async function loadStatus() {
  try {
    const res  = await fetch('/api/status');
    const data = await res.json();
    buildUI(data);
  } catch (e) { console.error(e); }
}

window.addEventListener('keydown', e => {
  const tag = document.activeElement.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA') return;

  switch (e.key) {
    case 'Enter':
      e.preventDefault(); acceptClip(); break;
    case 'x': case 'X': case 'Delete':
      e.preventDefault(); skipClip(); break;
    case ' ':
      e.preventDefault();
      vidContext.paused ? vidContext.play() : vidContext.pause();
      break;
    case 'ArrowLeft':
      e.preventDefault();
      vidContext.currentTime = Math.max(0, vidContext.currentTime - 0.5); break;
    case 'ArrowRight':
      e.preventDefault();
      vidContext.currentTime = Math.min(vidContext.duration || 0, vidContext.currentTime + 0.5); break;
  }
}, { capture: true });

loadStatus();
</script>
</body>
</html>"""


# ─── Entry Point ───────────────────────────────────────────────────────────────

def main():
    # ── Configure these paths — only edit this block ─────────────────────────
    REPO_ROOT = Path(__file__).resolve().parents[2]
    DEFAULTS = {
        "gloss_manifest":    str(REPO_ROOT / "data" / "manifests" / "gloss_clips_manifest.xlsx"),
        "video_dir":         str(REPO_ROOT / "Signer"),
        "output_dir":        str(REPO_ROOT / "data" / "non_detection_clips"),
        "nondet_manifest":   str(REPO_ROOT / "data" / "manifests" / "non_detection_manifest.xlsx"),
        "margin":            0.15,
        "min_gap":           0.5,
        "min_dur":           0.3,
        "max_dur":           1.0,
        "seed":              42,
        "port":              5001,
    }
    # ─────────────────────────────────────────────────────────────────────────

    parser = argparse.ArgumentParser(
        description="Auslan Non-Detection Clip Sampler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--gloss-manifest",  default=DEFAULTS["gloss_manifest"],
                        help="Path to gloss_clips_manifest.xlsx (output of annotator)")
    parser.add_argument("--video-dir",       default=DEFAULTS["video_dir"],
                        help="Directory containing *_signer.mp4 source videos")
    parser.add_argument("--output-dir",      default=DEFAULTS["output_dir"],
                        help="Directory to save accepted non-detection clips")
    parser.add_argument("--nondet-manifest", default=DEFAULTS["nondet_manifest"],
                        help="Path to write the non-detection manifest Excel")
    parser.add_argument("--margin",   type=float, default=DEFAULTS["margin"],
                        help="Seconds of margin to leave around each clipped window (default 0.1)")
    parser.add_argument("--min-gap",  type=float, default=DEFAULTS["min_gap"],
                        help="Minimum gap size to sample from (default 0.5s)")
    parser.add_argument("--min-dur",  type=float, default=DEFAULTS["min_dur"],
                        help="Minimum clip duration (default 0.3s)")
    parser.add_argument("--max-dur",  type=float, default=DEFAULTS["max_dur"],
                        help="Maximum clip duration (default 1.0s)")
    parser.add_argument("--seed",     type=int,   default=DEFAULTS["seed"],
                        help="Random seed for reproducible sampling (default 42)")
    parser.add_argument("--port",     type=int,   default=DEFAULTS["port"],
                        help="Web server port (default 5001)")
    args = parser.parse_args()

    STATE["output_dir"]    = args.output_dir
    STATE["manifest_path"] = args.nondet_manifest
    STATE["transcode_cache"] = str(Path(args.output_dir) / ".transcode_cache")

    print(f"\n{'='*58}")
    print(f"  Auslan Non-Detection Clip Sampler")
    print(f"{'='*58}")
    print(f"  Gloss manifest : {args.gloss_manifest}")
    print(f"  Video dir      : {args.video_dir}")
    print(f"  Output dir     : {args.output_dir}")
    print(f"  Non-det mfst   : {args.nondet_manifest}")
    print(f"  Margin         : {args.margin}s")
    print(f"  Clip duration  : {args.min_dur}–{args.max_dur}s (random)")
    print(f"  Seed           : {args.seed}")
    print(f"{'='*58}\n")

    print("⟳ Building candidates…")
    candidates = build_candidates(
        gloss_manifest_path=args.gloss_manifest,
        video_dir=args.video_dir,
        margin=args.margin,
        min_gap=args.min_gap,
        min_dur=args.min_dur,
        max_dur=args.max_dur,
        seed=args.seed,
    )

    if not candidates:
        print("\n✗ No candidates could be generated.")
        print("  Check that the gloss manifest has 'saved' rows and videos exist.")
        return

    STATE["candidates"] = candidates
    STATE["done_keys"]  = load_done_keys(args.nondet_manifest)

    already = len([c for c in candidates if candidate_key(c) in STATE["done_keys"]])
    print(f"✓ {len(candidates)} candidates  ({already} already in manifest, {len(candidates)-already} to review)\n")

    url = f"http://localhost:{args.port}"
    print(f"→ Opening {url}\n")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()