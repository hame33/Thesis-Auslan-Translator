"""
experiments/run_all.py

Runs all (or selected) experiments in sequence.
Each experiment is a separate call to train_video2gloss.py with its own YAML config.
Results are saved independently under results/<timestamp>__<exp_name>/

Usage:
    # Run all 5 experiments
    python experiments/run_all.py

    # Run specific experiments by name
    python experiments/run_all.py --experiments exp01_clean_only exp03_full_auslan_daily

    # Dry run — print what would be run without executing
    python experiments/run_all.py --dry-run

    # Compare results after all runs
    python experiments/run_all.py --compare-only
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT   = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "experiments" / "configs"
TRAIN_SCRIPT = REPO_ROOT / "src" / "v2g" / "train_video2gloss.py"
RESULTS_DIR  = REPO_ROOT / "results"

# Ordered list — experiments run in this sequence
ALL_EXPERIMENTS = [
    "exp01_clean_only",
    "exp02_auslan_daily_manual_glosses",
    "exp03_full_auslan_daily",
    "exp04_manual_glosses_plus_clean",
    "exp05_full_auslan_daily_plus_clean",
]


def find_config(exp_name: str) -> Path:
    path = CONFIGS_DIR / f"{exp_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return path


def run_experiment(exp_name: str, dry_run: bool = False) -> bool:
    config_path = find_config(exp_name)
    cmd = [sys.executable, str(TRAIN_SCRIPT), "--config", str(config_path)]

    print(f"\n{'='*60}")
    print(f"  Running: {exp_name}")
    print(f"  Config:  {config_path}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    if dry_run:
        print("  [DRY RUN — skipping execution]")
        return True

    start = time.time()
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n  ✗ FAILED: {exp_name} (exit code {result.returncode})")
        return False

    mins, secs = divmod(int(elapsed), 60)
    print(f"\n  ✓ Completed: {exp_name} in {mins}m {secs}s")
    return True


def compare_results():
    """Print a summary table of all completed experiment results."""
    import json

    summaries = []
    for summary_file in sorted(RESULTS_DIR.glob("*/summary.json"), reverse=True):
        with open(summary_file) as f:
            data = json.load(f)
        summaries.append(data)

    if not summaries:
        print("No completed results found in results/")
        return

    # Keep only most recent run per experiment name
    seen = {}
    for s in summaries:
        name = s.get("experiment_name", "unknown")
        if name not in seen:
            seen[name] = s

    rows = []
    for name in ALL_EXPERIMENTS:
        if name in seen:
            s = seen[name]
            rows.append({
                "experiment":       name,
                "test_f1_macro":    s.get("test_f1_macro",    "—"),
                "test_f1_micro":    s.get("test_f1_micro",    "—"),
                "best_dev_f1_macro":s.get("best_dev_f1_macro","—"),
                "best_epoch":       s.get("best_epoch",       "—"),
                "n_classes":        s.get("n_classes",        "—"),
                "n_train":          s.get("n_train",          "—"),
                "duration":         s.get("duration_human",   "—"),
            })
        else:
            rows.append({"experiment": name, "test_f1_macro": "not run"})

    df = pd.DataFrame(rows)
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)

    # Save summary CSV
    out = RESULTS_DIR / "experiment_comparison.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved comparison to: {out}")


def main():
    parser = argparse.ArgumentParser(description="Run Auslan V2G experiments")
    parser.add_argument(
        "--experiments", nargs="+", default=None,
        help="Names of specific experiments to run (default: all)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing"
    )
    parser.add_argument(
        "--compare-only", action="store_true",
        help="Skip training, just print results comparison table"
    )
    parser.add_argument(
        "--skip-on-failure", action="store_true", default=True,
        help="Continue to next experiment if one fails (default: True)"
    )
    args = parser.parse_args()

    if args.compare_only:
        compare_results()
        return

    to_run = args.experiments if args.experiments else ALL_EXPERIMENTS

    # Validate experiment names upfront
    for name in to_run:
        find_config(name)  # raises if missing

    print(f"\nRunning {len(to_run)} experiment(s):")
    for name in to_run:
        print(f"  • {name}")

    failed = []
    total_start = time.time()

    for exp_name in to_run:
        success = run_experiment(exp_name, dry_run=args.dry_run)
        if not success:
            failed.append(exp_name)
            if not args.skip_on_failure:
                print("\nAborting due to failure (use --skip-on-failure to continue).")
                break

    total_elapsed = time.time() - total_start
    total_mins, total_secs = divmod(int(total_elapsed), 60)

    print(f"\n{'='*60}")
    print(f"All done in {total_mins}m {total_secs}s")
    if failed:
        print(f"  Failed experiments: {', '.join(failed)}")
    else:
        print(f"  All experiments completed successfully.")
    print(f"{'='*60}\n")

    if not args.dry_run:
        compare_results()


if __name__ == "__main__":
    main()