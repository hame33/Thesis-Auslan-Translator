"""
Dataset and Vocabulary utilities for Auslan → English translation.
"""

import re
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3


class Vocabulary:
    def __init__(self):
        self.token2idx: dict[str, int] = {}
        self.idx2token: dict[int, str] = {}

    # ------------------------------------------------------------------
    @classmethod
    def build(cls, sentences: list[str], min_freq: int = 1) -> "Vocabulary":
        vocab = cls()
        counter: Counter = Counter()
        for sent in sentences:
            counter.update(_tokenize(sent))

        tokens = SPECIAL_TOKENS + [
            tok for tok, cnt in counter.most_common() if cnt >= min_freq
        ]
        vocab.token2idx = {tok: i for i, tok in enumerate(tokens)}
        vocab.idx2token = {i: tok for tok, i in vocab.token2idx.items()}
        return vocab

    # ------------------------------------------------------------------
    def encode(self, sentence: str) -> list[int]:
        return [self.token2idx.get(t, UNK_IDX) for t in _tokenize(sentence)]

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        tokens = [self.idx2token.get(i, "<unk>") for i in ids]
        if skip_special:
            tokens = [t for t in tokens if t not in SPECIAL_TOKENS]
        return " ".join(tokens)

    def __len__(self) -> int:
        return len(self.token2idx)

    def save(self, path: str | Path):
        with open(path, "w") as f:
            json.dump(self.token2idx, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        vocab = cls()
        with open(path) as f:
            vocab.token2idx = json.load(f)
        vocab.idx2token = {int(i): tok for tok, i in vocab.token2idx.items()}
        return vocab


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip trailing punctuation, split on whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)
    return text.split()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class AuslanDataset(Dataset):
    """
    Loads pre-extracted .npy feature files and their English translations.

    Each sample:
        features : FloatTensor (T, 258) — variable length
        target   : LongTensor  (S,)     — BOS + token ids + EOS
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        features_dir: str | Path,
        vocab: Vocabulary,
        split: str = "train",
        max_src_len: int = 1000,
        normalise: bool = True,
        mean_path: str | Path | None = None,
        std_path: str | Path | None = None,
    ):
        features_dir = Path(features_dir)
        sub = manifest[manifest["Split"] == split].reset_index(drop=True)

        # Keep only rows where the .npy file actually exists
        mask = sub["Video_Clip_Name"].apply(
            lambda n: (features_dir / f"{n}.npy").exists()
        )
        self.df = sub[mask].reset_index(drop=True)
        self.features_dir = features_dir
        self.vocab = vocab
        self.max_src_len = max_src_len
        self.normalise = normalise

        # Load or compute normalisation stats
        if normalise:
            if mean_path and Path(mean_path).exists():
                self.mean = np.load(mean_path)
                self.std = np.load(std_path)
            else:
                self.mean, self.std = None, None  # will be computed lazily

        print(f"[{split}] {len(self.df)} samples loaded.")

    # ------------------------------------------------------------------
    def compute_norm_stats(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute mean/std across ALL frames in the dataset (call once)."""
        all_frames = []
        for i in range(len(self.df)):
            path = self.features_dir / f"{self.df.loc[i, 'Video_Clip_Name']}.npy"
            arr = np.load(path)
            all_frames.append(arr)
        data = np.concatenate(all_frames, axis=0)
        mean = data.mean(axis=0).astype(np.float32)
        std = (data.std(axis=0) + 1e-6).astype(np.float32)
        return mean, std

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        feat_path = self.features_dir / f"{row['Video_Clip_Name']}.npy"
        features = np.load(feat_path).astype(np.float32)

        # Truncate very long sequences
        if len(features) > self.max_src_len:
            features = features[: self.max_src_len]

        # Normalise
        if self.normalise and self.mean is not None:
            features = (features - self.mean) / self.std

        src = torch.from_numpy(features)  # (T, 258)

        # Encode target
        token_ids = self.vocab.encode(row["Subtitle"])
        tgt = torch.tensor(
            [BOS_IDX] + token_ids + [EOS_IDX], dtype=torch.long
        )  # (S+2,)

        return {"src": src, "tgt": tgt, "clip_name": row["Video_Clip_Name"]}


# ---------------------------------------------------------------------------
# Collate function  (handles variable-length sequences)
# ---------------------------------------------------------------------------
def collate_fn(batch: list[dict]) -> dict:
    srcs = [item["src"] for item in batch]
    tgts = [item["tgt"] for item in batch]

    # Pad sources
    src_padded = pad_sequence(srcs, batch_first=True, padding_value=0.0)
    src_lengths = torch.tensor([s.size(0) for s in srcs])
    src_mask = ~(
        torch.arange(src_padded.size(1)).unsqueeze(0) < src_lengths.unsqueeze(1)
    )  # True = padding

    # Pad targets
    tgt_padded = pad_sequence(tgts, batch_first=True, padding_value=PAD_IDX)
    tgt_lengths = torch.tensor([t.size(0) for t in tgts])
    tgt_mask = ~(
        torch.arange(tgt_padded.size(1)).unsqueeze(0) < tgt_lengths.unsqueeze(1)
    )

    return {
        "src": src_padded,              # (B, T_src, 258)
        "tgt": tgt_padded,              # (B, T_tgt)
        "src_key_padding_mask": src_mask,   # (B, T_src) bool
        "tgt_key_padding_mask": tgt_mask,   # (B, T_tgt) bool
        "clip_names": [item["clip_name"] for item in batch],
    }