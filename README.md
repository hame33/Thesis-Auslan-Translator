# Auslan → English Sign Language Translator

A computer vision pipeline for translating **Australian Sign Language (Auslan)**
video into English text, using MediaPipe keypoint features as input. The system
implements a two-stage **Video-to-Gloss (V2G)** and **Gloss-to-Text (G2T)**
pipeline, alongside an end-to-end (E2E) baseline transformer.

This repository accompanies the thesis:
> *Dynamic Auslan Sign-Language Translator Using Computer Vision*
> Hamish Dawson, University of Sydney (2026)
> Supervisor: Prof. Mitch Bryson

---

## Results Summary

| Component | Model | Test Metric |
|-----------|-------|-------------|
| E2E Transformer | Encoder-decoder transformer | BLEU-4: 3.39 (dev) |
| V2G Classifier (exp01) | Transformer encoder + classifier | F1 macro: 0.509 |
| V2G + NON_DETECTION | Transformer encoder + 11 classes | F1 macro: 0.494 |
| G2T Translator (Claude PGen) | T5-small fine-tuned | SacreBLEU: 41.14 |

---

## Repository Structure

```
├── src/
│   ├── e2e/                          # End-to-end transformer (baseline)
│   │   ├── model.py                  # AuslanTransformer (encoder-decoder)
│   │   ├── dataset.py                # Vocabulary, AuslanDataset, collate_fn
│   │   ├── train.py                  # Training loop
│   │   └── evaluate.py               # BLEU-4 evaluation and inference
│   ├── v2g/                          # Video-to-gloss classifier
│   │   ├── train_video2gloss.py      # Transformer encoder + classifier head
│   │   └── sliding_windows_demo.py   # Real-time webcam inference demo
│   ├── g2t/                          # Gloss-to-text translator
│   │   ├── train_gloss2text.py       # T5-small fine-tuning
│   │   └── infer_gloss2text.py       # Inference on gloss sequences
│   ├── features/                     # MediaPipe feature extraction
│   │   ├── extract_features.py       # Single clip extraction
│   │   └── extract_all_features.py   # Batch extraction
│   └── preprocessing/                # Dataset construction tools
│       ├── annotator.py              # Manual gloss clip annotation tool
│       ├── augment_poses.py          # Mirror, speed, frame-drop augmentation
│       ├── clip_filtering.py         # PGen-based clip filtering
│       ├── collect_resting_poses.py  # NON_DETECTION webcam recorder
│       ├── non_detection_sampler.py  # Off-cut clip sampler
│       └── validator.py              # Dataset validation utilities
├── backtranslation/
│   ├── backtranslate.py              # Claude Haiku PGen gloss generation
│   ├── BacktranslationClaude/        # Claude PGen TSV datasets (14,041 pairs)
│   └── BacktranslationGPT5/          # GPT-5 PGen TSV datasets (~2,000 pairs)
├── experiments/
│   ├── configs/                      # YAML configs for all V2G experiments
│   │   ├── exp01_clean_only.yaml
│   │   ├── exp02_auslan_daily_manual_glosses.yaml
│   │   ├── exp03_full_auslan_daily.yaml
│   │   ├── exp04_manual_glosses_plus_clean.yaml
│   │   ├── exp05_full_auslan_daily_plus_clean.yaml
│   │   ├── exp_clean_with_nondet.yaml
│   │   └── exp_clean_with_nondet_v3.yaml
│   ├── run_all.py                    # Run all experiments sequentially
│   ├── dry_run.py                    # Validate config without training
│   └── analyse_results.py            # Aggregate and compare results
├── experiment_logger.py              # Structured experiment logging
├── confusion_matrices_vis.py         # Confusion matrix visualisation
├── auslan_translator.yml             # Conda environment
└── data/                             # Manifests and feature files (not tracked)
```

---

## Environment Setup

```bash
conda env create -f environment.yml
conda activate auslan_translator
```

Key dependencies:
- Python 3.10
- PyTorch 2.2.2
- Transformers 4.40.2 + T5 / SentencePiece
- MediaPipe 0.10.11
- SacreBLEU 2.4.0
- scikit-learn 1.4.2

> **Note:** `environment.yml` uses CPU-only PyTorch wheels for Mac Intel.
> If you are on Apple Silicon, replace `torch==2.2.2` with the MPS-compatible
> build from https://pytorch.org/get-started/locally/

---

## Data

This project uses the **Auslan Daily Dataset (Communication sub-dataset)**:
> Shen et al., *Auslan-Daily: Australian Sign Language Translation for Daily
> Communication and News*, NeurIPS 2023.
> https://uq-cvlab.github.io/Auslan-Daily-Dataset/

Place manifests in `data/manifests/` and extracted features in
`data/auslan_daily_features/` and `data/gloss_clips_features/`.

---

## Usage

### 0. Download the Auslan Daily Dataset

Request access and download the Communication sub-dataset from the official source:

> https://uq-cvlab.github.io/Auslan-Daily-Dataset/docs/en/dataset-download

'''
Once downloaded, place the manifest and video clips as follows:
data/
├── manifests/
│   └── AuslanDaily_Communication.csv
└── clips/
├── video_1_0.mp4
├── video_1_1.mp4
└── ...
'''

### 1. Extract MediaPipe features

```bash
python src/features/extract_all_features.py \
    --manifest data/manifests/AuslanDaily_Communication.csv \
    --clips_dir data/clips/ \
    --output_dir data/auslan_daily_features/
```

### 2. Generate PGen gloss annotations (Claude Haiku)

```bash
python backtranslation/backtranslate.py
# Output: backtranslation/BacktranslationClaude/
```

Set your `ANTHROPIC_API_KEY` environment variable before running.

### 3. Train V2G classifier

```bash
python src/v2g/train_video2gloss.py \
    --config experiments/configs/exp01_clean_only.yaml
```

Run all experiments:

```bash
python experiments/run_all.py
```

### 4. Run sliding window demo

```bash
python src/v2g/sliding_windows_demo.py \
    --model results/<run_id>/best_model.pt \
    --label-map results/<run_id>/label_map.json \
    --config experiments/configs/exp_clean_with_nondet_v3.yaml
```

### 5. Train G2T translator

```bash
conda activate text2gloss
python src/g2t/train_gloss2text.py
# Edit TRAIN_PATH, DEV_PATH, TEST_PATH in script before running
```

### 6. Run G2T inference

```bash
python src/g2t/infer_gloss2text.py
```

### 7. Train E2E baseline

```bash
PYTHONPATH=. python src/e2e/train.py \
    --manifest data/manifests/AuslanDaily_Communication.xlsx \
    --features_dir data/auslan_daily_features/ \
    --output_dir src/e2e/checkpoints/ \
    --epochs 50 --batch_size 32
```

---

## Annotation Tools

Two tools were developed for dataset construction:

**Gloss annotation tool** — manually clip and verify individual sign instances:

```bash
python src/preprocessing/annotator.py
```

**Resting pose recorder** — capture NON_DETECTION training clips via webcam:

```bash
python src/preprocessing/collect_resting_poses.py
```

---

## Experiments

All V2G experiments are defined by YAML configs in `experiments/configs/`.

| Experiment | Training Data | Test F1 Macro |
|------------|--------------|---------------|
| exp01 | 530 verified clips + augmentation | **0.509** |
| exp02 | Auslan Daily (filtered to 10 glosses) | 0.010 |
| exp03 | Full Auslan Daily | 0.027 |
| exp04 | Filtered Auslan Daily + clean clips | 0.006 |
| exp05 | Full Auslan Daily + clean clips | 0.024 |
| nondet_v3 | exp01 + NON_DETECTION class | 0.494 |

---

## Citation

If you use this code or the synthetic gloss dataset, please cite:

```bibtex
@thesis{dawson2026auslan,
  author  = {Hamish Dawson},
  title   = {Dynamic Auslan Sign-Language Translator Using Computer Vision},
  school  = {University of Sydney},
  year    = {2026}
}
```

---

## License

See `LICENSE`.