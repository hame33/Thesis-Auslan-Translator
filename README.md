# Auslan → English Sign Language Translation Transformer

A sequence-to-sequence Transformer that translates **Australian Sign Language (Auslan)** video clips into English text, using MediaPipe keypoint features as input.

---

## Architecture Overview

```
Input (.npy)           Encoder                   Decoder               Output
(N, 258) ──► FeatureProjection ──► TransformerEncoder ──► TransformerDecoder ──► English text
             (258 → d_model)       (4 layers, 8 heads)    (4 layers, 8 heads)
             + PositionalEncoding                          + Token Embedding
                                                          + PositionalEncoding
```

### Feature vector (258-d per frame)
| Segment       | Landmarks | Values each | Total |
|---------------|-----------|-------------|-------|
| Pose          | 33        | x, y, z, visibility | 132 |
| Left hand     | 21        | x, y, z     | 63   |
| Right hand    | 21        | x, y, z     | 63   |
| **Total**     |           |             | **258** |

---

## Project Structure

```
auslan_transformer/
├── model.py          # AuslanTransformer (encoder + decoder)
├── dataset.py        # Vocabulary, AuslanDataset, collate_fn
├── train.py          # Training loop with warmup-cosine LR, label smoothing
├── evaluate.py       # BLEU-4, test evaluation, single-file inference
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare your data
```
project/
├── AuslanDaily_Communication.xlsx   # manifest with Subtitle + Split columns
├── features/                        # 6817 .npy files of shape (N, 258)
│   ├── video_1_0.npy
│   ├── video_1_1.npy
│   └── ...
└── auslan_transformer/              # this repo
```

### 3. Train
```bash
cd auslan_transformer

python train.py \
    --manifest ../AuslanDaily_Communication.xlsx \
    --features_dir ../features \
    --output_dir checkpoints \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4 \
    --d_model 256 \
    --nhead 8 \
    --enc_layers 4 \
    --dec_layers 4 \
    --device cuda
```

On first run the script will:
1. Build and save a vocabulary (`checkpoints/vocab.json`)
2. Compute and save normalisation statistics (`checkpoints/norm_mean.npy`, `norm_std.npy`)
3. Train for N epochs, saving `last_model.pt` and `best_model.pt` (by dev BLEU)
4. Write a `history.json` with per-epoch metrics

### 4. Evaluate on test set
```bash
python evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --manifest ../AuslanDaily_Communication.xlsx \
    --features_dir ../features \
    --split test \
    --beam_size 4 \
    --output test_results.json
```

### 5. Translate a single clip
```python
import torch, numpy as np
from model import AuslanTransformer
from dataset import Vocabulary, PAD_IDX
from evaluate import translate_single

device = torch.device("cuda")
ckpt = torch.load("checkpoints/best_model.pt", map_location=device)
vocab = Vocabulary.load(ckpt["vocab_path"])

model = AuslanTransformer(
    vocab_size=len(vocab),
    d_model=256, nhead=8,
    num_encoder_layers=4, num_decoder_layers=4,
    dim_feedforward=1024, dropout=0.0, pad_idx=PAD_IDX,
).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

norm_mean = np.load("checkpoints/norm_mean.npy")
norm_std  = np.load("checkpoints/norm_std.npy")

text = translate_single(
    "features/video_1_0.npy", model, vocab,
    norm_mean, norm_std, device, beam_size=4
)
print(text)  # e.g. "hello ."
```

---

## Hyperparameter Reference

| Argument          | Default | Description |
|-------------------|---------|-------------|
| `--d_model`       | 256     | Transformer hidden dimension |
| `--nhead`         | 8       | Attention heads |
| `--enc_layers`    | 4       | Encoder layers |
| `--dec_layers`    | 4       | Decoder layers |
| `--dim_ff`        | 1024    | Feedforward dimension |
| `--dropout`       | 0.1     | Dropout rate |
| `--lr`            | 1e-4    | Peak learning rate |
| `--warmup_steps`  | 500     | LR warmup steps |
| `--label_smoothing` | 0.1  | Label smoothing ε |
| `--grad_clip`     | 1.0     | Gradient clipping norm |
| `--epochs`        | 50      | Training epochs |
| `--batch_size`    | 32      | Samples per batch |

---

## Decoding Options

| Method | Flag | Notes |
|--------|------|-------|
| Greedy | `--beam_size 1` | Fastest, lower BLEU |
| Beam search | `--beam_size 4` | Best quality (default) |

Length penalty α = 0.6 (adjustable in `model.beam_decode`).

---

## Training Tips

- **GPU memory**: On a 12 GB GPU with `batch_size=32` and long sequences, you may need to reduce `--batch_size` to 16 or 8.
- **Larger model**: Try `--d_model 512 --nhead 8 --enc_layers 6 --dec_layers 6 --dim_ff 2048` if you have a strong GPU.
- **Resume training**: `--resume checkpoints/last_model.pt`
- **Longer warmup**: If loss spikes early, increase `--warmup_steps` to 1000–2000.
- **Data augmentation**: Consider adding temporal jitter or small Gaussian noise to features during training.

---