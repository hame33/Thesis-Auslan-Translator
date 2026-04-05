"""
Evaluation utilities for Auslan → English translation.

- compute_bleu  : corpus-level BLEU-4
- evaluate_test : run model on test split, print & save results
- Standalone    : python evaluate.py --checkpoint checkpoints/best_model.pt
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import AuslanDataset, Vocabulary, collate_fn, BOS_IDX, EOS_IDX, PAD_IDX


# ---------------------------------------------------------------------------
# BLEU Implementation
# ---------------------------------------------------------------------------
def _ngram_counts(tokens: list, n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _clip_count(hyp: list, refs: list[list], n: int) -> tuple[int, int]:
    hyp_counts = _ngram_counts(hyp, n)
    max_ref_counts: Counter = Counter()
    for ref in refs:
        max_ref_counts |= _ngram_counts(ref, n)
    clipped = {ng: min(cnt, max_ref_counts[ng]) for ng, cnt in hyp_counts.items()}
    return sum(clipped.values()), max(0, len(hyp) - n + 1)


def compute_bleu(
    hypotheses: list[list[int]],
    references: list[list[list[int]]],
    max_n: int = 4,
    smooth: bool = True,
) -> float:
    """
    Corpus-level BLEU-4.
    hypotheses : list of token-id lists
    references : list of [list of token-id lists]  (one or more refs per sample)
    """
    assert len(hypotheses) == len(references)

    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    hyp_len = 0
    ref_len = 0

    for hyp, refs in zip(hypotheses, references):
        hyp_len += len(hyp)
        ref_len += min(refs, key=lambda r: abs(len(r) - len(hyp))).__len__()
        for n in range(1, max_n + 1):
            c, t = _clip_count(hyp, refs, n)
            clipped_counts[n - 1] += c
            total_counts[n - 1] += t

    log_bp = min(0.0, 1.0 - ref_len / max(1, hyp_len))
    bp = math.exp(log_bp)

    log_prec = 0.0
    for n in range(max_n):
        if smooth:
            log_prec += math.log((clipped_counts[n] + 1) / max(1, total_counts[n] + 1))
        else:
            if clipped_counts[n] == 0:
                return 0.0
            log_prec += math.log(clipped_counts[n] / max(1, total_counts[n]))

    bleu = bp * math.exp(log_prec / max_n) * 100
    return bleu


# ---------------------------------------------------------------------------
# Full test-set evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_test(
    model,
    test_loader: DataLoader,
    vocab: Vocabulary,
    device: torch.device,
    beam_size: int = 4,
    output_path: str | None = None,
) -> dict:
    model.eval()
    hypotheses, references = [], []
    results = []

    for batch in test_loader:
        src = batch["src"].to(device)
        src_mask = batch["src_key_padding_mask"].to(device)
        tgt = batch["tgt"]  # keep on CPU for reference extraction
        clip_names = batch["clip_names"]

        if beam_size > 1:
            # Beam search one sample at a time
            preds = []
            for i in range(src.size(0)):
                s = src[i : i + 1]
                sm = src_mask[i : i + 1] if src_mask is not None else None
                pred = model.beam_decode(s, sm, BOS_IDX, EOS_IDX, beam_size=beam_size)
                preds.extend(pred)
        else:
            preds = model.greedy_decode(src, src_mask, BOS_IDX, EOS_IDX)

        for i, (pred_ids, clip) in enumerate(zip(preds, clip_names)):
            ref_ids = tgt[i, 1:].tolist()
            ref_ids = [t for t in ref_ids if t not in (PAD_IDX, EOS_IDX)]
            hypotheses.append(pred_ids)
            references.append([ref_ids])

            hyp_text = vocab.decode(pred_ids)
            ref_text = vocab.decode(ref_ids)
            results.append(
                {"clip": clip, "reference": ref_text, "hypothesis": hyp_text}
            )

    bleu = compute_bleu(hypotheses, references)

    print(f"\n{'='*60}")
    print(f"Test BLEU-4: {bleu:.2f}")
    print(f"{'='*60}")
    print("\nSample predictions:")
    for r in results[:10]:
        print(f"  Clip : {r['clip']}")
        print(f"  REF  : {r['reference']}")
        print(f"  HYP  : {r['hypothesis']}")
        print()

    if output_path:
        with open(output_path, "w") as f:
            json.dump({"bleu": bleu, "results": results}, f, indent=2)
        print(f"Results saved to {output_path}")

    return {"bleu": bleu, "results": results}


# ---------------------------------------------------------------------------
# Standalone inference
# ---------------------------------------------------------------------------
def translate_single(
    npy_path: str,
    model,
    vocab: Vocabulary,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    device: torch.device,
    beam_size: int = 4,
) -> str:
    """Translate a single .npy feature file to English."""
    features = np.load(npy_path).astype(np.float32)
    features = (features - norm_mean) / norm_std
    src = torch.from_numpy(features).unsqueeze(0).to(device)  # (1, T, 258)

    if beam_size > 1:
        token_ids = model.beam_decode(src, None, BOS_IDX, EOS_IDX, beam_size=beam_size)[0]
    else:
        token_ids = model.greedy_decode(src, None, BOS_IDX, EOS_IDX)[0]

    return vocab.decode(token_ids)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", default="AuslanDaily_Communication.xlsx")
    p.add_argument("--features_dir", default="features")
    p.add_argument("--split", default="test")
    p.add_argument("--beam_size", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--output", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    from auslan_transformer.model import AuslanTransformer

    args = parse_args()
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_args = ckpt["args"]

    vocab = Vocabulary.load(ckpt["vocab_path"])
    print(f"Vocabulary size: {len(vocab)}")

    model = AuslanTransformer(
        vocab_size=len(vocab),
        d_model=saved_args["d_model"],
        nhead=saved_args["nhead"],
        num_encoder_layers=saved_args["enc_layers"],
        num_decoder_layers=saved_args["dec_layers"],
        dim_feedforward=saved_args["dim_ff"],
        dropout=0.0,
        pad_idx=PAD_IDX,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Model loaded from {args.checkpoint}")

    manifest = pd.read_excel(args.manifest)
    features_dir = Path(args.features_dir)
    output_dir = Path(args.checkpoint).parent

    norm_mean = np.load(output_dir / "norm_mean.npy")
    norm_std = np.load(output_dir / "norm_std.npy")

    test_ds = AuslanDataset(manifest, features_dir, vocab, split=args.split)
    test_ds.mean, test_ds.std = norm_mean, norm_std

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    evaluate_test(
        model, test_loader, vocab, device,
        beam_size=args.beam_size,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()