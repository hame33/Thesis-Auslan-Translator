#!/usr/bin/env python3
"""
Pose Sequence Augmentation
===========================
Augments MediaPipe .npy pose files in-place, generating new .npy files
alongside originals. Each augmentation is saved as a separate file so
your training pipeline can load them like any other sample.

Feature vector layout (per frame, 229 dims from your extraction script):
    [0:132]   — 33 pose landmarks × 4  (x, y, z, visibility)
    [132:195] — 21 left hand landmarks × 3  (x, y, z)
    [195:258] — 21 right hand landmarks × 3 (x, y, z)  ← wait, let me recheck
    
Actually from your script:
    pose:  33 × 4 = 132
    lhand: 21 × 3 = 63
    rhand: 21 × 3 = 63
    Total: 258

Usage:
    python augment_poses.py
"""

import argparse
import numpy as np
from pathlib import Path

# ── Configure this path ────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parents[2]
FEATURES_DIR = str(REPO_ROOT / "data" / "gloss_clips_features")
# ──────────────────────────────────────────────────────────────────────────

# ── Feature layout (must match your extraction script exactly) ─────────────
POSE_N    = 33
POSE_DIMS = POSE_N * 4        # x, y, z, visibility = 132
LHAND_N   = 21
LHAND_DIMS = LHAND_N * 3      # x, y, z = 63
RHAND_N   = 21
RHAND_DIMS = RHAND_N * 3      # x, y, z = 63
TOTAL_DIMS = POSE_DIMS + LHAND_DIMS + RHAND_DIMS  # 258

POSE_START  = 0
POSE_END    = POSE_DIMS                            # 132
LHAND_START = POSE_END                             # 132
LHAND_END   = LHAND_START + LHAND_DIMS             # 195
RHAND_START = LHAND_END                            # 195
RHAND_END   = RHAND_START + RHAND_DIMS             # 258


# ── Augmentation functions ──────────────────────────────────────────────────

def mirror(seq):
    """
    Horizontal flip — mirrors the signer left/right.
    - Flips x coordinates: x → 1 - x
    - Swaps left and right hands entirely (critical for sign language)
    - Flips pose x coords
    Returns: augmented sequence, same shape as input
    """
    aug = seq.copy()

    # Flip pose x (every 4th element starting at 0: x, y, z, vis → flip x only)
    for i in range(POSE_N):
        base = POSE_START + i * 4
        aug[:, base] = 1.0 - seq[:, base]  # flip x

    # Flip left hand x coords
    for i in range(LHAND_N):
        base = LHAND_START + i * 3
        aug[:, base] = 1.0 - seq[:, base]

    # Flip right hand x coords
    for i in range(RHAND_N):
        base = RHAND_START + i * 3
        aug[:, base] = 1.0 - seq[:, base]

    # Swap left and right hands (mirroring makes left hand become right hand)
    aug[:, LHAND_START:LHAND_END] = seq[:, RHAND_START:RHAND_END].copy()
    aug[:, RHAND_START:RHAND_END] = seq[:, LHAND_START:LHAND_END].copy()
    # But x coords need to stay flipped after the swap
    for i in range(LHAND_N):
        base = LHAND_START + i * 3
        aug[:, base] = 1.0 - aug[:, base]
    for i in range(RHAND_N):
        base = RHAND_START + i * 3
        aug[:, base] = 1.0 - aug[:, base]

    return aug


def speed_change(seq, factor):
    """
    Resample the sequence to change speed.
    factor < 1.0 = slower (more frames), factor > 1.0 = faster (fewer frames)
    Uses linear interpolation between frames.
    """
    T = seq.shape[0]
    new_T = max(2, int(round(T / factor)))
    old_indices = np.linspace(0, T - 1, new_T)
    aug = np.zeros((new_T, seq.shape[1]), dtype=seq.dtype)
    for j in range(seq.shape[1]):
        aug[:, j] = np.interp(old_indices, np.arange(T), seq[:, j])
    return aug


def temporal_crop(seq, crop_frac=0.1):
    """
    Randomly trim up to crop_frac of frames from start and end.
    Simulates slightly mis-timed annotation boundaries.
    """
    T = seq.shape[0]
    max_crop = max(1, int(T * crop_frac))
    start = np.random.randint(0, max_crop + 1)
    end   = T - np.random.randint(0, max_crop + 1)
    end   = max(end, start + 2)
    return seq[start:end]


def add_noise(seq, scale=0.005):
    """
    Add small Gaussian noise to landmark positions.
    Simulates natural jitter and imperfect landmark detection.
    scale=0.005 is subtle — coords are normalised 0-1 so this is 0.5% jitter.
    """
    aug = seq.copy()
    noise = np.random.normal(0, scale, seq.shape).astype(seq.dtype)
    # Only add noise to x,y,z — not visibility channel in pose
    for i in range(POSE_N):
        base = POSE_START + i * 4
        aug[:, base:base+3] += noise[:, base:base+3]
    aug[:, LHAND_START:LHAND_END] += noise[:, LHAND_START:LHAND_END]
    aug[:, RHAND_START:RHAND_END] += noise[:, RHAND_START:RHAND_END]
    return aug


# ── Main ────────────────────────────────────────────────────────────────────

def augment_file(path, args):
    """Generate all augmentations for a single .npy file."""
    seq = np.load(str(path))

    if seq.ndim != 2 or seq.shape[1] != TOTAL_DIMS:
        print(f"  ✗ Unexpected shape {seq.shape}, skipping {path.name}")
        return 0

    stem = path.stem
    parent = path.parent
    generated = 0

    def save_aug(aug, suffix):
        nonlocal generated
        out = parent / f"{stem}_{suffix}.npy"
        if not out.exists() or args.overwrite:
            np.save(str(out), aug.astype(np.float32))
            generated += 1

    # 1. Mirror
    if args.mirror:
        save_aug(mirror(seq), "mirror")

    # 2. Speed variants
    if args.speed:
        save_aug(speed_change(seq, 0.8), "slow")   # 20% slower
        save_aug(speed_change(seq, 1.2), "fast")   # 20% faster

    # 3. Temporal crop (run twice for two variants)
    if args.crop:
        np.random.seed(42)
        save_aug(temporal_crop(seq, 0.1), "crop1")
        np.random.seed(99)
        save_aug(temporal_crop(seq, 0.1), "crop2")

    # 4. Noise (run twice for variety)
    if args.noise:
        np.random.seed(7)
        save_aug(add_noise(seq, scale=0.004), "noise1")
        np.random.seed(13)
        save_aug(add_noise(seq, scale=0.007), "noise2")

    # 5. Mirror + speed (combined)
    if args.mirror and args.speed:
        save_aug(speed_change(mirror(seq), 0.8), "mirror_slow")
        save_aug(speed_change(mirror(seq), 1.2), "mirror_fast")

    return generated


def main():
    parser = argparse.ArgumentParser(description="Augment MediaPipe pose .npy files")
    parser.add_argument("--features-dir", default=FEATURES_DIR,
                        help="Directory containing .npy pose files")
    parser.add_argument("--mirror", action="store_true", default=True,
                        help="Horizontal flip augmentation (default: on)")
    parser.add_argument("--speed", action="store_true", default=True,
                        help="Speed ±20%% augmentation (default: on)")
    parser.add_argument("--crop", action="store_true", default=True,
                        help="Temporal crop augmentation (default: on)")
    parser.add_argument("--noise", action="store_true", default=True,
                        help="Gaussian noise augmentation (default: on)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing augmented files")
    parser.add_argument("--no-mirror", dest="mirror", action="store_false")
    parser.add_argument("--no-speed",  dest="speed",  action="store_false")
    parser.add_argument("--no-crop",   dest="crop",   action="store_false")
    parser.add_argument("--no-noise",  dest="noise",  action="store_false")
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    # Find all original files — exclude already-augmented ones
    suffixes = {"mirror", "slow", "fast", "crop1", "crop2",
                "noise1", "noise2", "mirror_slow", "mirror_fast"}
    originals = [
        p for p in sorted(features_dir.rglob("*.npy"))
        if p.stem.split("_")[-1] not in suffixes
    ]

    if not originals:
        print(f"No original .npy files found in {features_dir}")
        return

    # Count augmentations per file
    n_augs = (2 if args.mirror else 0) + \
             (2 if args.speed else 0) + \
             (2 if args.crop else 0) + \
             (2 if args.noise else 0) + \
             (2 if args.mirror and args.speed else 0)

    print(f"\nFound {len(originals)} original files")
    print(f"Augmentations per file: {n_augs}")
    print(f"Expected total new files: ~{len(originals) * n_augs}")
    print(f"Expected total dataset size: ~{len(originals) * (n_augs + 1)}\n")

    total_generated = 0
    for i, path in enumerate(originals):
        n = augment_file(path, args)
        total_generated += n
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(originals)}] {path.parent.name}/{path.name} → {n} new files")

    print(f"\n{'='*50}")
    print(f"  Done! Generated {total_generated} augmented files")
    print(f"  Total .npy files now: {sum(1 for _ in features_dir.rglob('*.npy'))}")
    print(f"{'='*50}")

    # Per-gloss summary
    print("\nPer-gloss counts (originals → total with augmentation):")
    from collections import Counter
    all_files = list(features_dir.glob("*.npy"))
    gloss_orig  = Counter()
    gloss_total = Counter()
    for p in all_files:
        # filename format: video_X_Y_signer_GLOSS_N[_suffix].npy
        parts = p.stem.split("_")
        is_aug = parts[-1] in suffixes
        # Extract gloss — it's the token after "signer"
        try:
            signer_idx = parts.index("signer")
            gloss = parts[signer_idx + 1]
        except (ValueError, IndexError):
            gloss = "unknown"
        gloss_total[gloss] += 1
        if not is_aug:
            gloss_orig[gloss] += 1
    for gloss in sorted(gloss_orig):
        print(f"  {gloss:<16} {gloss_orig[gloss]} → {gloss_total[gloss]}")


if __name__ == "__main__":
    main()