"""
analyse_results.py
Reads inference_results.csv and label_map.json to produce:
  - Per-class precision, recall, F1 sorted by F1
  - Top N best and worst performing glosses
  - Confusion summary (most common false positives)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support

# ── Config ─────────────────────────────────────────────────────
RESULTS_CSV    = "./video2gloss_model/inference_results.csv"
LABEL_MAP_PATH = "./video2gloss_model/label_map.json"
TOP_N          = 20   # how many best/worst classes to show
# ───────────────────────────────────────────────────────────────

def main():
    # Load label map
    with open(LABEL_MAP_PATH) as f:
        label_map = json.load(f)
    idx_to_gloss = {v: k for k, v in label_map.items()}
    n_classes = len(label_map)
    print(f"Vocabulary size: {n_classes} glosses\n")

    # Load results
    df = pd.read_csv(RESULTS_CSV)
    print(f"Test clips: {len(df)}\n")

    # Parse predictions and ground truth into binary matrices
    def parse_glosses(s, label_map):
        vec = np.zeros(n_classes, dtype=int)
        if pd.isna(s) or str(s).strip() == "":
            return vec
        for tok in str(s).strip().split():
            # Strip confidence scores e.g. "HELLO(0.91)" -> "HELLO"
            tok = tok.split("(")[0]
            if tok in label_map:
                vec[label_map[tok]] = 1
        return vec

    y_pred = np.vstack([
        parse_glosses(row["predicted_glosses"], label_map)
        for _, row in df.iterrows()
    ])
    y_true = np.vstack([
        parse_glosses(row["ground_truth"], label_map)
        for _, row in df.iterrows()
    ])

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    results = pd.DataFrame({
        "gloss":     [idx_to_gloss[i] for i in range(n_classes)],
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "support":   support.astype(int),   # true positives in test set
        "predicted": y_pred.sum(axis=0).astype(int),
    })

    # Save full per-class results
    out_path = "./video2gloss_model/perclass_results.csv"
    results.sort_values("f1", ascending=False).to_csv(out_path, index=False)
    print(f"Full per-class results saved to: {out_path}\n")

    # ── Summary stats ──────────────────────────────────────────
    has_support = results[results["support"] > 0]
    print(f"Classes with at least 1 true example in test set: {len(has_support)}")
    print(f"Classes with F1 > 0:    {(results['f1'] > 0).sum()}")
    print(f"Classes with F1 > 0.1:  {(results['f1'] > 0.1).sum()}")
    print(f"Classes with F1 > 0.3:  {(results['f1'] > 0.3).sum()}")
    print(f"Classes with F1 > 0.5:  {(results['f1'] > 0.5).sum()}")
    print(f"\nMacro F1 (all classes):          {f1.mean():.4f}")
    print(f"Macro F1 (classes with support): {has_support['f1'].mean():.4f}")
    print(f"Micro F1:                        "
          f"{2 * (y_pred * y_true).sum() / (y_pred.sum() + y_true.sum() + 1e-9):.4f}")

    # ── Top N best performing ──────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"TOP {TOP_N} BEST PERFORMING GLOSSES")
    print(f"{'─'*65}")
    print(f"{'Gloss':<25} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Support':>8}")
    print(f"{'─'*65}")
    top = results.sort_values("f1", ascending=False).head(TOP_N)
    for _, row in top.iterrows():
        print(f"{row['gloss']:<25} {row['f1']:>6.3f} {row['precision']:>6.3f} "
              f"{row['recall']:>6.3f} {int(row['support']):>8}")

    # ── Top N worst performing (with support > 0) ──────────────
    print(f"\n{'─'*65}")
    print(f"TOP {TOP_N} WORST PERFORMING GLOSSES (with test examples)")
    print(f"{'─'*65}")
    print(f"{'Gloss':<25} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Support':>8}")
    print(f"{'─'*65}")
    worst = has_support.sort_values("f1").head(TOP_N)
    for _, row in worst.iterrows():
        print(f"{row['gloss']:<25} {row['f1']:>6.3f} {row['precision']:>6.3f} "
              f"{row['recall']:>6.3f} {int(row['support']):>8}")

    # ── Most over-predicted (false positives) ──────────────────
    print(f"\n{'─'*65}")
    print(f"MOST OVER-PREDICTED GLOSSES (predicted >> actual)")
    print(f"{'─'*65}")
    print(f"{'Gloss':<25} {'Predicted':>10} {'Actual':>8} {'Ratio':>7}")
    print(f"{'─'*65}")
    results["ratio"] = results["predicted"] / (results["support"] + 1e-6)
    over = results[results["predicted"] > 5].sort_values("ratio", ascending=False).head(TOP_N)
    for _, row in over.iterrows():
        print(f"{row['gloss']:<25} {int(row['predicted']):>10} "
              f"{int(row['support']):>8} {row['ratio']:>7.1f}x")

    # ── Sample predictions ─────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"SAMPLE PREDICTIONS (first 10 test clips)")
    print(f"{'─'*65}")
    for _, row in df.head(10).iterrows():
        print(f"Clip:  {row['clip']}")
        print(f"Pred:  {row['predicted_with_confidence']}")
        print(f"GT:    {row['ground_truth']}")
        print()


if __name__ == "__main__":
    main()