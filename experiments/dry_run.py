"""
experiments/dry_run.py

Validates all experiment configs without training anything.
Runs load_combined_manifest() for each config and prints a full breakdown
of what data would be used — splits, gloss distribution, feature file
availability, and any warnings.

Usage:
    python experiments/dry_run.py
    python experiments/dry_run.py --config experiments/configs/exp01_clean_only.yaml
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.v2g.train_video2gloss import load_config, load_combined_manifest

CONFIGS_DIR = REPO_ROOT / "experiments" / "configs"
ALL_CONFIGS = sorted(CONFIGS_DIR.glob("*.yaml"))


def check_feature_files(df: pd.DataFrame, cfg: dict) -> tuple[int, int]:
    """Return (found, missing) feature file counts for a dataframe."""
    data_cfg = cfg["data"]
    feat_dirs = [
        Path(data_cfg["auslan_daily_features_dir"]),
        Path(data_cfg["annotated_features_dir"]),
    ]
    found = missing = 0
    for clip_id in df["clip_id"]:
        exists = any((d / f"{clip_id}.npy").exists() for d in feat_dirs)
        if exists:
            found += 1
        else:
            missing += 1
    return found, missing


def validate_config(config_path: Path):
    cfg = load_config(str(config_path))
    exp_name = cfg["experiment_name"]

    print(f"\n{'='*65}")
    print(f"  {exp_name}")
    print(f"  {cfg.get('description', '').strip()[:80]}")
    print(f"{'='*65}")

    # ── Load splits ───────────────────────────────────────────────────────────
    try:
        train_df, dev_df, test_df = load_combined_manifest(cfg)
    except Exception as e:
        print(f"  ❌  FAILED to load manifest: {e}")
        return

    gloss_col = cfg["data"]["gloss_col"]

    for split_name, split_df in [("TRAIN", train_df), ("DEV", dev_df), ("TEST", test_df)]:
        if len(split_df) == 0:
            colour = "⚠️ " if split_name == "DEV" else "❌"
            print(f"\n  {colour} {split_name}: 0 rows")
            continue

        # Gloss counts
        gloss_counts = Counter()
        for g in split_df[gloss_col].dropna():
            gloss_counts.update(str(g).strip().split())

        # Feature file availability
        found, missing = check_feature_files(split_df, cfg)

        # Source breakdown
        source_counts = split_df["source"].value_counts().to_dict() if "source" in split_df else {}

        print(f"\n  {split_name}: {len(split_df)} rows  "
              f"| features: {found} ✅  {missing} ❌"
              + (f"  | sources: {source_counts}" if source_counts else ""))

        print(f"  {'GLOSS':<22} {'COUNT':>6}")
        print(f"  {'-'*30}")
        for gloss, count in sorted(gloss_counts.items(), key=lambda x: -x[1]):
            flag = ""
            if split_name == "TRAIN" and count < 5:
                flag = "  ⚠️  very low"
            elif split_name == "TEST" and count < 3:
                flag = "  ⚠️  barely enough to evaluate"
            print(f"  {gloss:<22} {count:>6}{flag}")

    # ── Cross-split gloss coverage ────────────────────────────────────────────
    train_glosses = set()
    for g in train_df[gloss_col].dropna():
        train_glosses.update(str(g).strip().split())

    test_glosses = set()
    for g in test_df[gloss_col].dropna():
        test_glosses.update(str(g).strip().split())

    unseen = test_glosses - train_glosses
    if unseen:
        print(f"\n  ❌  GLOSSES IN TEST NOT IN TRAIN: {sorted(unseen)}")
        print(f"      These will always score 0 — remove from test or add to train.")
    else:
        print(f"\n  ✅  All test glosses are present in train.")

    # ── Config summary ────────────────────────────────────────────────────────
    t = cfg["training"]
    m = cfg["model"]
    print(f"\n  Hyperparams: epochs={t['epochs']}  lr={t['lr']}  "
          f"batch={t['batch_size']}  min_gloss_freq={t['min_gloss_freq']}")
    print(f"  Model:       proj_dim={m['proj_dim']}  "
          f"n_layers={m['n_layers']}  n_heads={m['n_heads']}")
    print(f"  Split strategy: {cfg['data'].get('test_split_strategy', 'all_originals')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=None,
        help="Path to a single config YAML (default: run all configs)"
    )
    args = parser.parse_args()

    if args.config:
        validate_config(Path(args.config))
    else:
        print(f"Dry-running all {len(ALL_CONFIGS)} experiment configs...\n")
        for config_path in ALL_CONFIGS:
            validate_config(config_path)

    print(f"\n{'='*65}")
    print("  Dry run complete.")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()