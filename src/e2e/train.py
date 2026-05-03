"""
Training script for the Auslan → English Transformer.

Usage:
    python train.py \
        --manifest AuslanDaily_Communication.xlsx \
        --features_dir features/ \
        --output_dir checkpoints/ \
        [--epochs 50] [--batch_size 32] [--lr 1e-4] [--device cuda]
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.g2t.dataset import AuslanDataset, Vocabulary, collate_fn, BOS_IDX, EOS_IDX, PAD_IDX
from src.g2t.model import AuslanTransformer
from evaluate import compute_bleu


# ---------------------------------------------------------------------------
# Label Smoothing Loss
# ---------------------------------------------------------------------------
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B*T, V)
        targets : (B*T,)
        """
        log_probs = torch.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        smooth = -log_probs.mean(dim=-1)
        loss = self.confidence * nll + self.smoothing * smooth

        # Mask padding
        pad_mask = targets != self.pad_idx
        loss = (loss * pad_mask).sum() / pad_mask.sum()
        return loss


# ---------------------------------------------------------------------------
# Warmup + Cosine LR Scheduler (Transformer-style)
# ---------------------------------------------------------------------------
class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-6):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch + 1
        if step < self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            scale = max(self.min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return [base_lr * scale for base_lr in self.base_lrs]


# ---------------------------------------------------------------------------
# Training and validation loops
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, scheduler, criterion, device, grad_clip=1.0):
    model.train()
    total_loss, total_tokens = 0.0, 0

    for batch in loader:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        src_mask = batch["src_key_padding_mask"].to(device)
        tgt_mask = batch["tgt_key_padding_mask"].to(device)

        # Decoder input: all but last token; target: all but first (BOS)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        tgt_in_mask = tgt_mask[:, :-1]

        logits = model(src, tgt_in, src_mask, tgt_in_mask)  # (B, T-1, V)

        B, T, V = logits.shape
        loss = criterion(logits.reshape(B * T, V), tgt_out.reshape(B * T))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        n_tokens = (tgt_out != PAD_IDX).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / max(1, total_tokens)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    hypotheses, references = [], []

    for batch in loader:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        src_mask = batch["src_key_padding_mask"].to(device)
        tgt_mask = batch["tgt_key_padding_mask"].to(device)

        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        tgt_in_mask = tgt_mask[:, :-1]

        logits = model(src, tgt_in, src_mask, tgt_in_mask)
        B, T, V = logits.shape
        loss = criterion(logits.reshape(B * T, V), tgt_out.reshape(B * T))

        n_tokens = (tgt_out != PAD_IDX).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

        # Greedy decode for BLEU
        preds = model.greedy_decode(src, src_mask, BOS_IDX, EOS_IDX)
        for pred_ids, ref_ids in zip(preds, tgt_out.tolist()):
            ref_ids = [t for t in ref_ids if t != PAD_IDX and t != EOS_IDX]
            hypotheses.append(pred_ids)
            references.append([ref_ids])

    bleu = compute_bleu(hypotheses, references)
    return total_loss / max(1, total_tokens), bleu


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="AuslanDaily_Communication.xlsx")
    p.add_argument("--features_dir", default="features")
    p.add_argument("--output_dir", default="checkpoints")
    p.add_argument("--vocab_path", default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--enc_layers", type=int, default=4)
    p.add_argument("--dec_layers", type=int, default=4)
    p.add_argument("--dim_ff", type=int, default=1024)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--eval_every", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load manifest
    # ------------------------------------------------------------------
    manifest = pd.read_csv(args.manifest, sep=';')
    features_dir = Path(args.features_dir)

    # ------------------------------------------------------------------
    # Build / load vocabulary
    # ------------------------------------------------------------------
    vocab_path = args.vocab_path or str(output_dir / "vocab.json")
    if Path(vocab_path).exists():
        vocab = Vocabulary.load(vocab_path)
        print(f"Loaded vocab ({len(vocab)} tokens) from {vocab_path}")
    else:
        train_sents = manifest[manifest["Split"] == "train"]["Subtitle"].tolist()
        vocab = Vocabulary.build(train_sents, min_freq=1)
        vocab.save(vocab_path)
        print(f"Built vocab with {len(vocab)} tokens → {vocab_path}")

    # ------------------------------------------------------------------
    # Datasets & loaders
    # ------------------------------------------------------------------
    train_ds = AuslanDataset(manifest, features_dir, vocab, split="train")
    dev_ds = AuslanDataset(manifest, features_dir, vocab, split="dev")

    # Compute or load normalisation stats
    norm_mean_path = output_dir / "norm_mean.npy"
    norm_std_path = output_dir / "norm_std.npy"
    if norm_mean_path.exists():
        train_ds.mean = np.load(norm_mean_path)
        train_ds.std = np.load(norm_std_path)
        dev_ds.mean = train_ds.mean
        dev_ds.std = train_ds.std
        print("Loaded normalisation stats.")
    else:
        print("Computing normalisation stats (first run only)...")
        train_ds.mean, train_ds.std = train_ds.compute_norm_stats()
        np.save(norm_mean_path, train_ds.mean)
        np.save(norm_std_path, train_ds.std)
        dev_ds.mean, dev_ds.std = train_ds.mean, train_ds.std
        print("Normalisation stats saved.")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = AuslanTransformer(
        vocab_size=len(vocab),
        input_dim=258,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_ff,
        dropout=args.dropout,
        pad_idx=PAD_IDX,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Optimiser / scheduler / loss
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98)
    )
    total_steps = args.epochs * len(train_loader)
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_steps, total_steps)
    criterion = LabelSmoothingCrossEntropy(len(vocab), PAD_IDX, args.label_smoothing)

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_epoch = 0
    best_bleu = 0.0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_bleu = ckpt.get("best_bleu", 0.0)
        print(f"Resumed from epoch {start_epoch}, best BLEU={best_bleu:.2f}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    history = []
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, args.grad_clip
        )
        elapsed = time.time() - t0

        log = {"epoch": epoch, "train_loss": train_loss, "train_ppl": math.exp(train_loss)}

        if (epoch + 1) % args.eval_every == 0:
            dev_loss, dev_bleu = eval_epoch(model, dev_loader, criterion, device)
            log.update({"dev_loss": dev_loss, "dev_ppl": math.exp(dev_loss), "dev_bleu": dev_bleu})
            print(
                f"Epoch {epoch+1:3d}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"dev_loss={dev_loss:.4f} | "
                f"dev_BLEU={dev_bleu:.2f} | "
                f"{elapsed:.1f}s"
            )

            if dev_bleu > best_bleu:
                best_bleu = dev_bleu
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_bleu": best_bleu,
                        "vocab_path": vocab_path,
                        "args": vars(args),
                    },
                    output_dir / "best_model.pt",
                )
                print(f"  ✓ New best BLEU={best_bleu:.2f} — checkpoint saved.")
        else:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | train_loss={train_loss:.4f} | {elapsed:.1f}s")

        # Save last checkpoint every epoch
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_bleu": best_bleu,
                "vocab_path": vocab_path,
                "args": vars(args),
            },
            output_dir / "last_model.pt",
        )
        history.append(log)

        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best dev BLEU: {best_bleu:.2f}")


if __name__ == "__main__":
    main()