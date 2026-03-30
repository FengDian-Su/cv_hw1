# NYCU Visual Recognition using Deep Learning 2026 HW1

- **Student ID**: 314551055
- **Name**: Su Feng-Dian

---

## Introduction

This repository contains the implementation for HW1: Image Classification.
The task is to classify RGB images into 100 categories given a training/validation
set of 21,024 images and a test set of 2,344 images.

The approach uses **SE-ResNet-101** — a ResNet-101 backbone with
Squeeze-and-Excitation (SE) modules injected into every Bottleneck block,
providing channel-wise attention. Training combines MixUp, AutoAugment,
Label Smoothing, and a Warmup Cosine learning rate schedule to reduce
overfitting. At inference time, Test Time Augmentation (TTA) averages
predictions over six crop-and-flip variants for improved robustness.

**Public leaderboard score: 0.96**

---

## Environment Setup

```bash
# Create and activate a conda environment
conda create -n cv_hw1 python=3.10
conda activate cv_hw1

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pandas matplotlib pillow
```

---

## Repository Structure

```
cv_hw1/
├── code/
│   ├── dataset_v2.py        # Dataset, transforms, DataLoader factory
│   ├── model.py             # SE-ResNet-101 definition
│   ├── train_0323_v2.py     # Training script
│   ├── inference.py         # Inference with TTA
│   └── plot_results.py      # Plot training curves from log CSV
├── data/
│   ├── train/               # Training images (class folders 0–99)
│   ├── val/                 # Validation images (class folders 0–99)
│   └── test/                # Test images (flat folder)
├── model/                   # Saved checkpoints and training logs (auto-created)
├── figures/                 # Output figures (auto-created)
└── README.md
```

---

## Usage

All commands should be run from inside the `code/` directory:

```bash
cd code
```

### Training

```bash
python train_0323_v2.py \
    --data_root ../data \
    --model_dir ../model \
    --epochs 60 \
    --batch_size 64 \
    --lr 3e-4 \
    --weight_decay 1e-4 \
    --warmup_epochs 5 \
    --smoothing 0.05 \
    --mixup_alpha 0.2 \
    --num_workers 8 \
    --gpu 0 \
    --run_name se_resnet101_v2
```

The best checkpoint and training log will be saved to:
- `model/se_resnet101_v2_best.pt`
- `model/se_resnet101_v2_log.csv`

### Plot Training Curves (Optional)

```bash
python plot_results.py \
    --log ../model/se_resnet101_v2_log.csv \
    --out_dir ../figures
```

### Inference

```bash
python inference.py \
    --checkpoint ../model/se_resnet101_v2_best.pt \
    --data_root ../data \
    --output ../prediction.csv \
    --batch_size 64 \
    --num_workers 8 \
    --gpu 0
```

TTA is enabled by default. Add `--no_tta` to disable it (faster but less accurate).
