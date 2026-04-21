"""
train_video2gloss.py

Video-to-gloss multi-label classifier.
Input:  .npy feature files (frames × 225) from MediaPipe Holistic
Output: which glosses appear in the clip + confidence scores

Architecture: frame-level linear projection → Transformer encoder
              → mean pool → multi-label sigmoid head

Usage:
    python train_video2gloss.py            # train
    python train_video2gloss.py --infer    # run inference on test set
"""

import argparse
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────
MANIFEST_CSV   = "/Users/hamishdawson/Desktop/Thesis/Auslan-Daily_Communication_with_gloss_fixed.xlsx"
FEATURES_DIR   = Path("/Users/hamishdawson/Desktop/Thesis/Thesis-Auslan-Translator/features")
GLOSS_COL      = "gloss"       # column in manifest with space-separated glosses
SPLIT_COL      = "Split"       # column with train/dev/test
CLIP_COL       = "Video_Clip_Name"

OUT_DIR        = "./video2gloss_model"
LABEL_MAP_PATH = os.path.join(OUT_DIR, "label_map.json")

# Model hyperparameters
FRAME_FEAT_DIM  = 258    
PROJ_DIM        = 128      # project frames to this dim before transformer
N_HEADS         = 4
N_LAYERS        = 3
DROPOUT         = 0.2
MAX_FRAMES      = 300      # clips longer than this are trimmed

# Training
BATCH_SIZE      = 32
EPOCHS          = 60
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
MIN_GLOSS_FREQ  = 25        # ignore glosses that appear in fewer than N clips
CONFIDENCE_THRESHOLD = 0.35  # sigmoid threshold for positive prediction
# ────────────────────────────────────────────────────────────────────────────


# ── Data ─────────────────────────────────────────────────────────────────────

def parse_glosses(gloss_str: str) -> list[str]:
    """Split a gloss string into individual gloss tokens."""
    if pd.isna(gloss_str) or str(gloss_str).strip() == "":
        return []
    return str(gloss_str).strip().split()


def build_label_map(df: pd.DataFrame, min_freq: int) -> dict:
    """Build gloss→index map, filtering rare glosses."""
    from collections import Counter
    counter = Counter()
    for g in df[GLOSS_COL].dropna():
        counter.update(parse_glosses(g))
    vocab = sorted(g for g, c in counter.items() if c >= min_freq)
    label_map = {g: i for i, g in enumerate(vocab)}
    print(f"Vocabulary: {len(label_map)} glosses (min_freq={min_freq})")
    return label_map


class GlossDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_map: dict, max_frames: int):
        self.records   = []
        self.label_map = label_map
        self.n_classes = len(label_map)
        self.max_frames = max_frames

        missing = 0
        for _, row in df.iterrows():
            npy_path = FEATURES_DIR / f"{row[CLIP_COL]}.npy"
            if not npy_path.exists():
                missing += 1
                continue
            glosses = [g for g in parse_glosses(row[GLOSS_COL])
                       if g in label_map]
            if not glosses:
                continue
            self.records.append((npy_path, glosses, row[CLIP_COL]))

        if missing:
            print(f"  Warning: {missing} clips missing feature files — skipped.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        npy_path, glosses, clip_name = self.records[idx]

        # Load and trim/pad frames
        frames = np.load(npy_path).astype(np.float32)   # (T, 225)
        if len(frames) > self.max_frames:
            # Uniformly sample max_frames from the clip
            indices = np.linspace(0, len(frames) - 1, self.max_frames, dtype=int)
            frames = frames[indices]

        frames_tensor = torch.from_numpy(frames)   # (T, 225)

        # Multi-hot label vector
        label = torch.zeros(self.n_classes, dtype=torch.float32)
        for g in glosses:
            label[self.label_map[g]] = 1.0

        return frames_tensor, label, clip_name


def collate_fn(batch):
    """Pad variable-length frame sequences to the same length."""
    frames_list, labels, names = zip(*batch)
    # pad_sequence expects list of (T, D) tensors → (T_max, B, D)
    padded = pad_sequence(frames_list, batch_first=True)   # (B, T_max, D)
    # Create padding mask: True where padded (i.e., not real data)
    lengths = torch.tensor([f.shape[0] for f in frames_list])
    max_len = padded.shape[1]
    mask = torch.arange(max_len).unsqueeze(0) >= lengths.unsqueeze(1)  # (B, T_max)
    labels = torch.stack(labels)
    return padded, mask, labels, list(names)


# ── Model ─────────────────────────────────────────────────────────────────────

class Video2GlossTransformer(nn.Module):
    def __init__(self, feat_dim: int, proj_dim: int, n_heads: int,
                 n_layers: int, n_classes: int, dropout: float):
        super().__init__()

        # Project raw MediaPipe features to model dimension
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Positional encoding (learned)
        self.pos_emb = nn.Embedding(MAX_FRAMES + 1, proj_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=n_heads,
            dim_feedforward=proj_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-norm: more stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Multi-label classification head
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim // 2, n_classes),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor):
        """
        x:                (B, T, feat_dim)
        key_padding_mask: (B, T) — True for padded positions
        returns:          (B, n_classes) logits
        """
        B, T, _ = x.shape

        x = self.input_proj(x)   # (B, T, proj_dim)

        # Add positional embeddings
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        positions = positions.clamp(max=MAX_FRAMES)
        x = x + self.pos_emb(positions)

        # Transformer (mask padded frames)
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        # Mean pool over real (non-padded) frames only
        mask_float = (~key_padding_mask).float().unsqueeze(-1)   # (B, T, 1)
        x = (x * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)

        return self.classifier(x)   # (B, n_classes)


# ── Training ──────────────────────────────────────────────────────────────────

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
def evaluate(model, loader, criterion, device, threshold: float):
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
    return total_loss / len(loader), f1_micro, f1_macro


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer(model, loader, label_map: dict, device, threshold: float):
    """Run inference and return predictions with confidence scores."""
    model.eval()
    idx_to_gloss = {v: k for k, v in label_map.items()}
    results = []

    for frames, mask, labels, names in tqdm(loader, desc="  infer"):
        frames, mask = frames.to(device), mask.to(device)
        logits = model(frames, mask)
        probs  = torch.sigmoid(logits).cpu().numpy()

        for clip_name, prob_vec, label_vec in zip(names, probs, labels.numpy()):
            # All predictions above threshold, sorted by confidence
            pred_indices = np.where(prob_vec >= threshold)[0]
            predictions  = sorted(
                [(idx_to_gloss[i], float(prob_vec[i])) for i in pred_indices],
                key=lambda x: -x[1]
            )
            # Ground truth glosses
            gt_glosses = [idx_to_gloss[i] for i in np.where(label_vec == 1)[0]]

            results.append({
                "clip":        clip_name,
                "predictions": predictions,
                "ground_truth": gt_glosses,
            })

    return results


def print_sample_predictions(results: list, n: int = 10):
    print(f"\n{'─'*70}")
    print(f"{'CLIP':<30} {'PREDICTIONS (gloss: confidence)'}")
    print(f"{'─'*70}")
    for r in results[:n]:
        preds_str = "  ".join(
            f"{g}:{c:.2f}" for g, c in r["predictions"]
        ) or "(none)"
        gt_str = " ".join(r["ground_truth"]) or "(none)"
        print(f"{r['clip']:<30} {preds_str}")
        print(f"{'':30} GT: {gt_str}")
        print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(run_infer: bool = False):
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load manifest — supports .xlsx or .csv
    if MANIFEST_CSV.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(MANIFEST_CSV)
    else:
        df = pd.read_csv(MANIFEST_CSV)
    df = df[df[GLOSS_COL].notna() & (df[GLOSS_COL].str.strip() != "")]
    print(f"Manifest rows with glosses: {len(df)}")

    if run_infer:
        # ── Inference mode ──────────────────────────────────────────────────
        with open(LABEL_MAP_PATH) as f:
            label_map = json.load(f)
        n_classes = len(label_map)

        model = Video2GlossTransformer(
            FRAME_FEAT_DIM, PROJ_DIM, N_HEADS, N_LAYERS, n_classes, DROPOUT
        ).to(device)
        ckpt = torch.load(os.path.join(OUT_DIR, "best_model.pt"), map_location=device)
        model.load_state_dict(ckpt)

        test_df  = df[df[SPLIT_COL] == "test"]
        test_ds  = GlossDataset(test_df, label_map, MAX_FRAMES)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                                 collate_fn=collate_fn, num_workers=0)

        results = infer(model, test_loader, label_map, device, CONFIDENCE_THRESHOLD)
        print_sample_predictions(results, n=20)

        # Save full results to CSV
        rows = []
        for r in results:
            preds_str = " ".join(f"{g}({c:.2f})" for g, c in r["predictions"])
            rows.append({
                "clip": r["clip"],
                "predicted_glosses": " ".join(g for g, _ in r["predictions"]),
                "predicted_with_confidence": preds_str,
                "ground_truth": " ".join(r["ground_truth"]),
            })
        out_csv = os.path.join(OUT_DIR, "inference_results.csv")
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"\nFull results saved to: {out_csv}")
        return

    # ── Training mode ───────────────────────────────────────────────────────
    train_df = df[df[SPLIT_COL] == "train"]
    dev_df   = df[df[SPLIT_COL] == "dev"]
    test_df  = df[df[SPLIT_COL] == "test"]
    print(f"Splits — train: {len(train_df)}  dev: {len(dev_df)}  test: {len(test_df)}")

    label_map = build_label_map(train_df, MIN_GLOSS_FREQ)
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"Label map saved to: {LABEL_MAP_PATH}")

    train_ds = GlossDataset(train_df, label_map, MAX_FRAMES)
    dev_ds   = GlossDataset(dev_df,   label_map, MAX_FRAMES)
    test_ds  = GlossDataset(test_df,  label_map, MAX_FRAMES)
    print(f"Dataset sizes — train: {len(train_ds)}  dev: {len(dev_ds)}  test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    dev_loader   = DataLoader(dev_ds,   batch_size=BATCH_SIZE,
                              collate_fn=collate_fn, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              collate_fn=collate_fn, num_workers=0)

    n_classes = len(label_map)
    model = Video2GlossTransformer(
        FRAME_FEAT_DIM, PROJ_DIM, N_HEADS, N_LAYERS, n_classes, DROPOUT
    ).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Positive weight: upweights rare glosses so the model doesn't ignore them
    # Computed as (neg_count / pos_count) per class, clipped to [1, 20]
    label_counts = np.zeros(n_classes)
    for _, labels, _ in train_ds:
        label_counts += labels.numpy()
    pos_weight = torch.tensor(
        np.clip((len(train_ds) - label_counts) / (label_counts + 1e-6), 1.0, 20.0),
        dtype=torch.float32
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_f1    = 0.0
    best_epoch = 0

    print(f"\nTraining for {EPOCHS} epochs...\n")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        dev_loss, dev_f1_micro, dev_f1_macro = evaluate(
            model, dev_loader, criterion, device, CONFIDENCE_THRESHOLD
        )
        scheduler.step()

        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  "
              f"dev_loss={dev_loss:.4f}  "
              f"dev_f1_micro={dev_f1_micro:.4f}  "
              f"dev_f1_macro={dev_f1_macro:.4f}")

        if dev_f1_macro > best_f1:
            best_f1    = dev_f1_macro
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model.pt"))
            print(f"  ✓ New best model saved (f1_macro={best_f1:.4f})")

    print(f"\nBest model: epoch {best_epoch}, dev f1_macro={best_f1:.4f}")

    # Final test evaluation with best model
    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best_model.pt"),
                                     map_location=device))
    test_loss, test_f1_micro, test_f1_macro = evaluate(
        model, test_loader, criterion, device, CONFIDENCE_THRESHOLD
    )
    print(f"Test — loss={test_loss:.4f}  "
          f"f1_micro={test_f1_micro:.4f}  "
          f"f1_macro={test_f1_macro:.4f}")

    # Show sample predictions
    results = infer(model, test_loader, label_map, device, CONFIDENCE_THRESHOLD)
    print_sample_predictions(results, n=10)

    # Save inference results
    rows = []
    for r in results:
        preds_str = " ".join(f"{g}({c:.2f})" for g, c in r["predictions"])
        rows.append({
            "clip": r["clip"],
            "predicted_glosses": " ".join(g for g, _ in r["predictions"]),
            "predicted_with_confidence": preds_str,
            "ground_truth": " ".join(r["ground_truth"]),
        })
    out_csv = os.path.join(OUT_DIR, "inference_results.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nInference results saved to: {out_csv}")
    print(f"Model saved to: {OUT_DIR}/best_model.pt")
    print(f"Label map saved to: {LABEL_MAP_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer", action="store_true",
                        help="Run inference on test set using saved model")
    args = parser.parse_args()
    main(run_infer=args.infer)