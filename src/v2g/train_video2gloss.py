"""
train_video2gloss.py

Video-to-gloss multi-label classifier.
Input:  .npy feature files (frames × 258) from MediaPipe Holistic
Output: which glosses appear in the clip + confidence scores

Architecture: frame-level linear projection → Transformer encoder
              → mean pool → multi-label sigmoid head

Usage:
    python src/v2g/train_video2gloss.py --config experiments/configs/exp01_clean_only.yaml
    python src/v2g/train_video2gloss.py --config experiments/configs/exp01_clean_only.yaml --infer
"""

import argparse
import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

# Allow imports from repo root (for experiment_logger)
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from experiment_logger import ExperimentLogger


# ── Config loading ────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Resolve all paths relative to repo root
    data = cfg["data"]
    for key in ["auslan_daily_manifest", "annotated_manifest",
                "auslan_daily_features_dir", "annotated_features_dir"]:
        if key in data and data[key]:
            data[key] = str(REPO_ROOT / data[key])
    cfg["output"]["results_dir"] = str(REPO_ROOT / cfg["output"]["results_dir"])
    return cfg


# ── Manifest loading ──────────────────────────────────────────────────────────

def load_combined_manifest(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build train / dev / test DataFrames according to the experiment config.

    Returns: (train_df, dev_df, test_df)

    Experiment modes (set in config):
      use_auslan_daily           – include Auslan Daily clips
      use_annotated_clips        – include hand-annotated clean clips
      filter_to_manual_glosses   – restrict Auslan Daily to glosses that
                                   appear in the annotated (manual) clip set
      clean_clips_as_test        – use non-augmented annotated clips as the
                                   test split (instead of Auslan Daily "test")
    """
    data_cfg = cfg["data"]
    gloss_col = data_cfg["gloss_col"]
    split_col = data_cfg["split_col"]

    frames = []

    # ── Auslan Daily ──────────────────────────────────────────────────────────
    if data_cfg.get("use_auslan_daily", False):
        manifest = data_cfg["auslan_daily_manifest"]
        if manifest.endswith((".xlsx", ".xls")):
            df_daily = pd.read_excel(manifest)
        else:
            # Sniff separator — Auslan Daily uses ";" but fall back to ","
            with open(manifest) as _f:
                first_line = _f.readline()
            sep = ";" if first_line.count(";") > first_line.count(",") else ","
            df_daily = pd.read_csv(manifest, sep=sep)

        df_daily = df_daily.rename(columns={"Video_Clip_Name": "clip_id"})
        df_daily = df_daily[["clip_id", gloss_col, split_col]].copy()
        df_daily["source"] = "auslan_daily"
        frames.append(df_daily)
        print(f"  Auslan Daily clips:          {len(df_daily)}")

    # ── Annotated (clean) clips ───────────────────────────────────────────────
    ann_df = None
    if data_cfg.get("use_annotated_clips", False) or data_cfg.get("clean_clips_as_test", False):
        ann = pd.read_excel(data_cfg["annotated_manifest"])
        ann = ann[ann["status"] == "saved"].copy()
        ann["clip_id"] = ann["output_file"].apply(lambda f: Path(str(f)).stem)
        ann = ann.rename(columns={"gloss": gloss_col})
        ann["source"] = "annotated"
        ann_orig = ann[["clip_id", gloss_col, "source"]].copy()

        # ── Decide test split strategy for originals ──────────────────────────
        strategy = data_cfg.get("test_split_strategy", "all_originals")
        # "all_originals"  — every non-augmented clip → test  (default)
        # "stratified"     — per-gloss holdout of `test_fraction` of originals

        if not data_cfg.get("clean_clips_as_test", False):
            # Not using clean clips as test at all — everything goes to train
            ann_orig[split_col] = "train"

        elif strategy == "stratified":
            import math
            test_fraction = data_cfg.get("test_fraction", 0.2)
            rng = np.random.default_rng(seed=data_cfg.get("split_seed", 42))
            ann_orig[split_col] = "train"   # default

            for gloss_val, group in ann_orig.groupby(gloss_col):
                n_test = max(1, math.floor(len(group) * test_fraction))
                if len(group) < 3:
                    # Too few clips to safely hold any out — keep all in train
                    print(f"  ⚠️  {gloss_val}: only {len(group)} clip(s) — skipping test holdout")
                    continue
                test_idx = rng.choice(group.index, size=n_test, replace=False)
                ann_orig.loc[test_idx, split_col] = "test"

            n_test  = (ann_orig[split_col] == "test").sum()
            n_train = (ann_orig[split_col] == "train").sum()
            print(f"  Stratified split ({test_fraction*100:.0f}% test): "
                  f"{n_train} train originals, {n_test} test originals")

        else:  # "all_originals" — original behaviour
            ann_orig[split_col] = "test"

        # ── Augmented variants — always train ─────────────────────────────────
        aug_suffixes = {
            "mirror", "slow", "fast", "crop1", "crop2",
            "noise1", "noise2", "mirror_slow", "mirror_fast"
        }
        ann_features_dir = Path(data_cfg["annotated_features_dir"])
        aug_rows = []
        # Only augment clips that ended up in train (never augment test clips)
        train_originals = ann_orig[ann_orig[split_col] == "train"]
        for _, row in train_originals.iterrows():
            for suffix in aug_suffixes:
                aug_id = f"{row['clip_id']}_{suffix}"
                if (ann_features_dir / f"{aug_id}.npy").exists():
                    aug_rows.append({
                        "clip_id":  aug_id,
                        gloss_col:  row[gloss_col],
                        split_col:  "train",
                        "source":   "augmented",
                    })

        aug_df = pd.DataFrame(aug_rows) if aug_rows else pd.DataFrame(
            columns=["clip_id", gloss_col, split_col, "source"]
        )

        ann_df = pd.concat([ann_orig, aug_df], ignore_index=True)
        print(f"  Annotated originals:         {len(ann_orig)}")
        print(f"  Augmented variants:          {len(aug_df)}")

        if data_cfg.get("use_annotated_clips", False):
            frames.append(ann_df)

    # ── Combine ───────────────────────────────────────────────────────────────
    if not frames and not data_cfg.get("clean_clips_as_test", False):
        raise ValueError("No data sources enabled in config.")

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["clip_id", gloss_col, split_col, "source"]
    )

    # ── Filter Auslan Daily to manual-clip glosses only ───────────────────────
    # Reads the annotated manifest directly for the authoritative gloss list,
    # regardless of whether annotated clips are in the training set.
    # Only "auslan_daily" rows are filtered — clean clip rows are never dropped.
    if data_cfg.get("filter_to_manual_glosses", False):
        ann_for_filter = pd.read_excel(data_cfg["annotated_manifest"])
        ann_for_filter = ann_for_filter[ann_for_filter["status"] == "saved"]
        manual_glosses = set(ann_for_filter["gloss"].dropna().str.strip().unique())
        print(f"  Manual gloss vocabulary ({len(manual_glosses)}): {sorted(manual_glosses)}")

        before = len(combined)
        is_daily = combined["source"] == "auslan_daily"
        daily_keep = combined[is_daily][
            combined[is_daily][gloss_col].apply(
                lambda g: any(tok in manual_glosses
                              for tok in str(g).strip().split()) if pd.notna(g) else False
            )
        ]
        combined = pd.concat(
            [daily_keep, combined[~is_daily]], ignore_index=True
        )
        print(f"  Filtered Auslan Daily to manual glosses: {before} → {len(combined)} rows")

    # ── Override test split with clean clips ──────────────────────────────────
    if data_cfg.get("clean_clips_as_test", False) and ann_df is not None:
        # Remove any existing "test" rows from combined (from Auslan Daily split)
        combined = combined[combined[split_col] != "test"].copy()
        # Add annotated originals as test
        test_rows = ann_df[ann_df[split_col] == "test"].copy()
        combined = pd.concat([combined, test_rows], ignore_index=True)

    combined = combined[combined[gloss_col].notna() & (combined[gloss_col].str.strip() != "")]

    train_df = combined[combined[split_col] == "train"].copy()
    dev_df   = combined[combined[split_col] == "dev"].copy()
    test_df  = combined[combined[split_col] == "test"].copy()

    # If no dev split exists (e.g. clean-only exp), carve 10% out of train.
    # Only sample from non-augmented clips so dev mirrors real distribution.
    if len(dev_df) == 0 and len(train_df) > 0:
        non_aug = train_df[train_df["source"] == "annotated"]
        if len(non_aug) >= 10:
            dev_sample = non_aug.groupby(gloss_col, group_keys=False).apply(
                lambda g: g.sample(frac=0.10, random_state=42) if len(g) >= 5 else g.iloc[:0]
            )
            dev_df   = dev_sample.copy()
            train_df = train_df.drop(dev_sample.index).copy()
            print(f"  ℹ️  No dev split found — carved {len(dev_df)} non-augmented clips from train as dev.")
        else:
            print(f"  ⚠️  No dev split and not enough non-augmented clips to carve one — training without dev eval.")

    print(f"  Combined total:              {len(combined)}")
    print(f"  Splits — train: {len(train_df)}  dev: {len(dev_df)}  test: {len(test_df)}")

    return train_df, dev_df, test_df


# ── Data ──────────────────────────────────────────────────────────────────────

def parse_glosses(gloss_str: str) -> list:
    if pd.isna(gloss_str) or str(gloss_str).strip() == "":
        return []
    return str(gloss_str).strip().split()


def build_label_map(df: pd.DataFrame, gloss_col: str, min_freq: int) -> dict:
    from collections import Counter
    counter = Counter()
    for g in df[gloss_col].dropna():
        counter.update(parse_glosses(g))
    vocab = sorted(g for g, c in counter.items() if c >= min_freq)
    label_map = {g: i for i, g in enumerate(vocab)}
    print(f"Vocabulary: {len(label_map)} glosses (min_freq={min_freq})")
    return label_map


class GlossDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_map: dict, cfg: dict):
        data_cfg  = cfg["data"]
        model_cfg = cfg["model"]

        self.label_map  = label_map
        self.n_classes  = len(label_map)
        self.max_frames = model_cfg["max_frames"]
        self.feat_dirs  = [
            Path(data_cfg["auslan_daily_features_dir"]),
            Path(data_cfg["annotated_features_dir"]),
        ]
        self.records = []
        missing = 0

        for _, row in df.iterrows():
            clip_id = row["clip_id"]
            npy_path = None
            for d in self.feat_dirs:
                candidate = d / f"{clip_id}.npy"
                if candidate.exists():
                    npy_path = candidate
                    break
            if npy_path is None:
                missing += 1
                continue

            glosses = [g for g in parse_glosses(row[data_cfg["gloss_col"]])
                       if g in label_map]
            if not glosses:
                continue

            self.records.append((npy_path, glosses, clip_id))

        if missing:
            print(f"  Warning: {missing} clips missing feature files — skipped.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        npy_path, glosses, clip_id = self.records[idx]

        frames = np.load(npy_path).astype(np.float32)
        if len(frames) > self.max_frames:
            indices = np.linspace(0, len(frames) - 1, self.max_frames, dtype=int)
            frames = frames[indices]

        frames_tensor = torch.from_numpy(frames)

        label = torch.zeros(self.n_classes, dtype=torch.float32)
        for g in glosses:
            label[self.label_map[g]] = 1.0

        return frames_tensor, label, clip_id


def collate_fn(batch):
    frames_list, labels, names = zip(*batch)
    padded  = pad_sequence(frames_list, batch_first=True)
    lengths = torch.tensor([f.shape[0] for f in frames_list])
    max_len = padded.shape[1]
    mask    = torch.arange(max_len).unsqueeze(0) >= lengths.unsqueeze(1)
    labels  = torch.stack(labels)
    return padded, mask, labels, list(names)


# ── Model ─────────────────────────────────────────────────────────────────────

class Video2GlossTransformer(nn.Module):
    def __init__(self, cfg: dict, n_classes: int):
        super().__init__()
        m = cfg["model"]
        feat_dim  = m["frame_feat_dim"]
        proj_dim  = m["proj_dim"]
        n_heads   = m["n_heads"]
        n_layers  = m["n_layers"]
        dropout   = m["dropout"]
        max_frames = m["max_frames"]

        self.max_frames = max_frames
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pos_emb = nn.Embedding(max_frames + 1, proj_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=n_heads,
            dim_feedforward=proj_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim // 2, n_classes),
        )

    def forward(self, x, key_padding_mask):
        B, T, _ = x.shape
        x = self.input_proj(x)
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        positions = positions.clamp(max=self.max_frames)
        x = x + self.pos_emb(positions)
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        mask_float = (~key_padding_mask).float().unsqueeze(-1)
        x = (x * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        return self.classifier(x)


# ── Training / Evaluation ─────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for frames, mask, labels, _ in tqdm(loader, desc="  train", leave=False):
        frames, mask, labels = frames.to(device), mask.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(frames, mask)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for frames, mask, labels, _ in tqdm(loader, desc="  eval ", leave=False):
        frames, mask, labels = frames.to(device), mask.to(device), labels.to(device)
        logits = model(frames, mask)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        preds = (torch.sigmoid(logits) >= threshold).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())
    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    f1_micro = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader), f1_micro, f1_macro, all_preds, all_labels


# ── Results saving ────────────────────────────────────────────────────────────

def save_per_class_results(all_labels, all_preds, label_map, run_dir: Path):
    """Save per-class precision, recall, F1, support to CSV."""
    idx_to_gloss = {v: k for k, v in label_map.items()}
    report = classification_report(
        all_labels, all_preds,
        target_names=[idx_to_gloss[i] for i in range(len(label_map))],
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for gloss, metrics in report.items():
        if isinstance(metrics, dict):
            rows.append({"gloss": gloss, **metrics})
    pd.DataFrame(rows).to_csv(run_dir / "perclass_results.csv", index=False)


def save_confusion_matrix(all_labels, all_preds, label_map, run_dir: Path):
    """
    For multi-label: save an N×N co-occurrence confusion matrix.
    Entry [i,j] = number of clips where gloss i was true AND gloss j was predicted.
    The diagonal is true positives per class.
    Also saves as .npy for plotting.
    """
    n = len(label_map)
    cm = np.zeros((n, n), dtype=int)
    for true_row, pred_row in zip(all_labels, all_preds):
        true_idx = np.where(true_row)[0]
        pred_idx = np.where(pred_row)[0]
        for ti in true_idx:
            for pi in pred_idx:
                cm[ti, pi] += 1

    np.save(run_dir / "confusion_matrix.npy", cm)

    idx_to_gloss = {v: k for k, v in label_map.items()}
    headers = [idx_to_gloss[i] for i in range(n)]
    rows_out = [["true↓ / pred→"] + headers]
    for i, row in enumerate(cm):
        rows_out.append([headers[i]] + list(row))

    import csv
    with open(run_dir / "confusion_matrix.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows_out)


def save_inference_results(results, run_dir: Path):
    rows = []
    for r in results:
        preds_str = " ".join(f"{g}({c:.2f})" for g, c in r["predictions"])
        rows.append({
            "clip":                      r["clip"],
            "predicted_glosses":         " ".join(g for g, _ in r["predictions"]),
            "predicted_with_confidence": preds_str,
            "ground_truth":              " ".join(r["ground_truth"]),
            "correct": set(g for g, _ in r["predictions"]) == set(r["ground_truth"]),
        })
    pd.DataFrame(rows).to_csv(run_dir / "inference_results.csv", index=False)


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer(model, loader, label_map, device, threshold):
    model.eval()
    idx_to_gloss = {v: k for k, v in label_map.items()}
    results = []
    for frames, mask, labels, names in tqdm(loader, desc="  infer"):
        frames, mask = frames.to(device), mask.to(device)
        logits = model(frames, mask)
        probs  = torch.sigmoid(logits).cpu().numpy()
        for clip_name, prob_vec, label_vec in zip(names, probs, labels.numpy()):
            pred_idx = np.where(prob_vec >= threshold)[0]
            predictions = sorted(
                [(idx_to_gloss[i], float(prob_vec[i])) for i in pred_idx],
                key=lambda x: -x[1]
            )
            gt_glosses = [idx_to_gloss[i] for i in np.where(label_vec == 1)[0]]
            results.append({
                "clip":         clip_name,
                "predictions":  predictions,
                "ground_truth": gt_glosses,
            })
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--infer",  action="store_true", help="Run inference using saved model")
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_name = cfg["experiment_name"]
    t_cfg    = cfg["training"]
    m_cfg    = cfg["model"]
    out_cfg  = cfg["output"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"Device:     {device}")
    print(f"{'='*60}\n")

    print("Loading manifests...")
    train_df, dev_df, test_df = load_combined_manifest(cfg)

    # ── Inference mode ────────────────────────────────────────────────────────
    if args.infer:
        # Find the most recent results dir for this experiment
        results_root = Path(out_cfg["results_dir"])
        matching = sorted(results_root.glob(f"*__{exp_name}"), reverse=True)
        if not matching:
            raise FileNotFoundError(f"No saved results found for experiment '{exp_name}'")
        run_dir = matching[0]
        with open(run_dir / "label_map.json") as f:
            label_map = json.load(f)

        model = Video2GlossTransformer(cfg, len(label_map)).to(device)
        model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))

        test_ds     = GlossDataset(test_df, label_map, cfg)
        test_loader = DataLoader(test_ds, batch_size=t_cfg["batch_size"],
                                 collate_fn=collate_fn, num_workers=0)
        results = infer(model, test_loader, label_map, device, t_cfg["confidence_threshold"])
        save_inference_results(results, run_dir)
        print(f"Inference results saved to: {run_dir}/inference_results.csv")
        return

    # ── Training mode ─────────────────────────────────────────────────────────
    label_map = build_label_map(train_df, cfg["data"]["gloss_col"], t_cfg["min_gloss_freq"])

    # Start experiment logger
    logger = ExperimentLogger(
        experiment_name=exp_name,
        config={
            "description":  cfg.get("description", ""),
            "data":         cfg["data"],
            "model":        cfg["model"],
            "training":     cfg["training"],
            "n_classes":    len(label_map),
        }
    )
    run_dir = logger.run_dir

    # Save label map into run dir
    with open(run_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    train_ds = GlossDataset(train_df, label_map, cfg)
    dev_ds   = GlossDataset(dev_df,   label_map, cfg)
    test_ds  = GlossDataset(test_df,  label_map, cfg)
    print(f"Dataset sizes — train: {len(train_ds)}  dev: {len(dev_ds)}  test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=t_cfg["batch_size"], shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    dev_loader   = DataLoader(dev_ds,   batch_size=t_cfg["batch_size"],
                              collate_fn=collate_fn, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=t_cfg["batch_size"],
                              collate_fn=collate_fn, num_workers=0)

    model = Video2GlossTransformer(cfg, len(label_map)).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Positive weight for class imbalance
    label_counts = np.zeros(len(label_map))
    for _, labels, _ in train_ds:
        label_counts += labels.numpy()
    pos_weight = torch.tensor(
        np.clip((len(train_ds) - label_counts) / (label_counts + 1e-6), 1.0, 20.0),
        dtype=torch.float32,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(t_cfg["lr"]), weight_decay=float(t_cfg["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_cfg["epochs"]
    )

    best_f1    = 0.0
    best_epoch = 0
    best_preds = None
    best_labels = None

    print(f"\nTraining for {t_cfg['epochs']} epochs...\n")
    for epoch in range(1, t_cfg["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        dev_loss, dev_f1_micro, dev_f1_macro, dev_preds, dev_labels = evaluate(
            model, dev_loader, criterion, device, t_cfg["confidence_threshold"]
        )
        scheduler.step()

        print(f"Epoch {epoch:02d}/{t_cfg['epochs']}  "
              f"train_loss={train_loss:.4f}  "
              f"dev_loss={dev_loss:.4f}  "
              f"dev_f1_micro={dev_f1_micro:.4f}  "
              f"dev_f1_macro={dev_f1_macro:.4f}")

        logger.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            dev_loss=dev_loss,
            dev_f1_micro=dev_f1_micro,
            dev_f1_macro=dev_f1_macro,
        )

        if dev_f1_macro > best_f1:
            best_f1     = dev_f1_macro
            best_epoch  = epoch
            best_preds  = dev_preds
            best_labels = dev_labels
            torch.save(model.state_dict(), run_dir / "best_model.pt")
            print(f"  ✓ New best model saved (f1_macro={best_f1:.4f})")

    print(f"\nBest model: epoch {best_epoch}, dev f1_macro={best_f1:.4f}")

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    test_loss, test_f1_micro, test_f1_macro, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device, t_cfg["confidence_threshold"]
    )
    print(f"Test — loss={test_loss:.4f}  "
          f"f1_micro={test_f1_micro:.4f}  "
          f"f1_macro={test_f1_macro:.4f}")

    # Save all results artifacts
    save_per_class_results(test_labels, test_preds, label_map, run_dir)
    save_confusion_matrix(test_labels, test_preds, label_map, run_dir)

    results = infer(model, test_loader, label_map, device, t_cfg["confidence_threshold"])
    save_inference_results(results, run_dir)

    logger.finish(
        test_loss=round(test_loss, 4),
        test_f1_micro=round(test_f1_micro, 4),
        test_f1_macro=round(test_f1_macro, 4),
        best_dev_f1_macro=round(best_f1, 4),
        best_epoch=best_epoch,
        n_classes=len(label_map),
        n_train=len(train_ds),
        n_dev=len(dev_ds),
        n_test=len(test_ds),
    )


if __name__ == "__main__":
    main()