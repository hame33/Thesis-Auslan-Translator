"""
experiment_logger.py
--------------------
Drop-in experiment tracker for V2G (and G2T) training runs.

Usage
-----
    from experiment_logger import ExperimentLogger

    logger = ExperimentLogger(
        experiment_name="v2g_clean_only",
        config={
            "model": "MediaPipe + LSTM",
            "dataset": "clean_clips_only",
            "num_classes": 87,
            "epochs": 30,
            "lr": 1e-3,
            "batch_size": 32,
            "augmentation": ["mirror", "speed", "temporal_crop"],
            "notes": "Baseline — no PGen data",
        }
    )

    # Inside your training loop:
    logger.log_epoch(epoch=1, train_loss=1.23, val_loss=1.10, val_acc=0.45, val_top5_acc=0.78)

    # At the end:
    logger.finish(test_acc=0.51, test_top5_acc=0.82, confusion_matrix=cm)
"""

import json
import csv
import os
import time
import datetime
import numpy as np
from pathlib import Path
from typing import Any


# ── Where all runs are stored ────────────────────────────────────────────────
RESULTS_ROOT = Path(__file__).parent / "results"


class ExperimentLogger:
    """Saves config, per-epoch metrics, and final test results for one run."""

    def __init__(self, experiment_name: str, config: dict[str, Any]):
        """
        Parameters
        ----------
        experiment_name : str
            Short human-readable label, e.g. "v2g_clean_only" or
            "v2g_pgen_postprocessed". Spaces become underscores.
        config : dict
            Anything you want to record: model architecture, dataset path,
            hyperparameters, data splits, augmentation flags, notes, etc.
        """
        self.experiment_name = experiment_name.replace(" ", "_")
        self.config = config

        # Unique run ID: timestamp + name
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{ts}__{self.experiment_name}"
        self.start_time = time.time()

        # Create run directory
        self.run_dir = RESULTS_ROOT / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Paths
        self._config_path  = self.run_dir / "config.json"
        self._epochs_path  = self.run_dir / "epochs.csv"
        self._summary_path = self.run_dir / "summary.json"
        self._cm_path      = self.run_dir / "confusion_matrix.npy"

        # Save config immediately so it's there even if training crashes
        self._save_config()

        # Open epoch CSV
        self._epoch_file = open(self._epochs_path, "w", newline="")
        self._epoch_writer = None  # created on first log_epoch call

        print(f"[ExperimentLogger] Run started: {self.run_id}")
        print(f"[ExperimentLogger] Saving to:   {self.run_dir}")

    # ── Per-epoch logging ────────────────────────────────────────────────────

    def log_epoch(self, epoch: int, **metrics):
        """
        Call once per epoch with whatever numeric metrics you have.

        Example
        -------
        logger.log_epoch(
            epoch=5,
            train_loss=0.82,
            val_loss=0.74,
            val_acc=0.61,
            val_top5_acc=0.89,
        )
        """
        row = {"epoch": epoch, **metrics}

        if self._epoch_writer is None:
            self._epoch_writer = csv.DictWriter(
                self._epoch_file, fieldnames=list(row.keys())
            )
            self._epoch_writer.writeheader()

        self._epoch_writer.writerow(row)
        self._epoch_file.flush()

        # Pretty-print to console
        metric_str = "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in row.items())
        print(f"[ExperimentLogger] {metric_str}")

    # ── Finish run ───────────────────────────────────────────────────────────

    def finish(self, confusion_matrix=None, class_names: list[str] | None = None, **test_metrics):
        """
        Call at the end of training with final test-set metrics.

        Parameters
        ----------
        confusion_matrix : array-like, optional
            NumPy array (N×N). Saved as .npy and also as a readable CSV.
        class_names : list[str], optional
            Gloss labels in the same order as confusion_matrix axes.
        **test_metrics
            Any scalar metrics: test_acc=0.55, test_top5_acc=0.84, bleu=12.3, etc.
        """
        self._epoch_file.close()

        elapsed = time.time() - self.start_time
        summary = {
            "run_id":          self.run_id,
            "experiment_name": self.experiment_name,
            "duration_seconds": round(elapsed, 1),
            "duration_human":   _fmt_duration(elapsed),
            **test_metrics,
        }

        with open(self._summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Save confusion matrix
        if confusion_matrix is not None:
            cm = np.array(confusion_matrix)
            np.save(self._cm_path, cm)

            # Human-readable CSV version
            cm_csv_path = self.run_dir / "confusion_matrix.csv"
            headers = class_names if class_names else [str(i) for i in range(cm.shape[0])]
            with open(cm_csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["pred→ / true↓"] + headers)
                for label, row in zip(headers, cm):
                    w.writerow([label] + list(row))

        print(f"\n[ExperimentLogger] Run complete: {self.run_id}")
        print(f"[ExperimentLogger] Duration:     {_fmt_duration(elapsed)}")
        for k, v in test_metrics.items():
            print(f"[ExperimentLogger]   {k}: {v}")
        print(f"[ExperimentLogger] Results saved to: {self.run_dir}\n")

    # ── Internal ─────────────────────────────────────────────────────────────

    def _save_config(self):
        payload = {
            "run_id":          self.run_id,
            "experiment_name": self.experiment_name,
            "timestamp":       datetime.datetime.now().isoformat(),
            "config":          self.config,
        }
        with open(self._config_path, "w") as f:
            json.dump(payload, f, indent=2)


# ── Utility: compare all runs ─────────────────────────────────────────────────

def load_all_summaries(results_root: str | Path = RESULTS_ROOT) -> list[dict]:
    """
    Return a list of summary dicts for every completed run, sorted newest first.
    Useful for quick comparison in a notebook.

    Example
    -------
    from experiment_logger import load_all_summaries
    import pandas as pd

    df = pd.DataFrame(load_all_summaries())
    print(df[["experiment_name", "test_acc", "test_top5_acc", "duration_human"]])
    """
    root = Path(results_root)
    summaries = []
    for summary_file in sorted(root.glob("*/summary.json"), reverse=True):
        with open(summary_file) as f:
            data = json.load(f)
        # Attach config fields for easy filtering
        config_file = summary_file.parent / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                cfg = json.load(f).get("config", {})
            data["_config"] = cfg
        summaries.append(data)
    return summaries


def load_epoch_curves(run_id: str, results_root: str | Path = RESULTS_ROOT) -> list[dict]:
    """Return per-epoch rows for a given run_id as a list of dicts."""
    path = Path(results_root) / run_id / "epochs.csv"
    if not path.exists():
        raise FileNotFoundError(f"No epoch log found at {path}")
    with open(path) as f:
        return list(csv.DictReader(f))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"