"""
train_0323_v2.py
Main training script for SE-ResNet-101 image classification.

v2 changes over v1:
    - MixUp augmentation in training loop.
      MixUp randomly interpolates two images and their labels:
          x_mix = λ·x_i + (1-λ)·x_j
          y_mix = λ·y_i + (1-λ)·y_j
      where λ ~ Beta(alpha, alpha).
      This forces the model to behave linearly between training examples,
      significantly reducing overconfidence and improving generalisation.
    - Updated AMP API to suppress FutureWarning.
    - AutoAugment added in dataset pipeline (see dataset_v2.py).
    - Backbone upgraded from ResNet-50 to ResNet-101.
"""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from dataset_v2 import build_dataloaders
from model import build_model, count_parameters


# ── Loss ───────────────────────────────────────────────────────────────────────

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1) -> None:
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_prob = nn.functional.log_softmax(logits, dim=1)
        nll = -log_prob.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        smooth = -log_prob.mean(dim=1)
        loss = (1 - self.smoothing) * nll + self.smoothing * smooth
        return loss.mean()


class MixUpCrossEntropy(nn.Module):
    """
    Cross-entropy loss compatible with MixUp soft labels.
    Targets are one-hot mixed vectors instead of hard integer labels.
    """

    def __init__(self, smoothing: float = 0.1) -> None:
        super().__init__()
        self.smoothing = smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets_a: torch.Tensor,
        targets_b: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        log_prob = nn.functional.log_softmax(logits, dim=1)

        def smooth_nll(log_p, y):
            nll = -log_p.gather(dim=1, index=y.unsqueeze(1)).squeeze(1)
            smooth = -log_p.mean(dim=1)
            return (1 - self.smoothing) * nll + self.smoothing * smooth

        loss = lam * smooth_nll(log_prob, targets_a) + \
               (1 - lam) * smooth_nll(log_prob, targets_b)
        return loss.mean()


# ── MixUp ──────────────────────────────────────────────────────────────────────

def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.4,
    device: torch.device = torch.device("cpu"),
):
    """
    Apply MixUp to a batch.

    Args:
        x:     image batch  (B, C, H, W)
        y:     label batch  (B,)
        alpha: Beta distribution parameter (higher = more mixing)

    Returns:
        mixed_x, y_a, y_b, lambda
    """
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# ── LR Schedule ────────────────────────────────────────────────────────────────

def get_lr(optimizer: torch.optim.Optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


def warmup_cosine_schedule(
    optimizer,
    warmup_epochs: int,
    total_epochs: int,
    base_lr: float,
    min_lr: float = 1e-6,
):
    import math

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return (min_lr + cosine * (base_lr - min_lr)) / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Train / Val ────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion: MixUpCrossEntropy,
    scaler: GradScaler,
    device: torch.device,
    mixup_alpha: float = 0.4,
) -> tuple[float, float]:
    model.train()
    total_loss = correct = total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        # apply mixup
        mixed_imgs, targets_a, targets_b, lam = mixup_data(
            imgs, labels, alpha=mixup_alpha, device=device
        )

        optimizer.zero_grad()
        with autocast(device_type="cuda"):
            logits = model(mixed_imgs)
            loss = criterion(logits, targets_a, targets_b, lam)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        # accuracy: use hard labels for monitoring
        preds = logits.argmax(dim=1)
        correct += (lam * (preds == targets_a).float() +
                    (1 - lam) * (preds == targets_b).float()).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
) -> tuple[float, float]:
    """Val uses standard cross-entropy (no mixup)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = correct = total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast(device_type="cuda"):
            logits = model(imgs)
            loss = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train SE-ResNet-101 v2 with MixUp")
    parser.add_argument("--data_root",       type=str,   default="data")
    parser.add_argument("--model_dir",       type=str,   default="model")
    parser.add_argument("--epochs",          type=int,   default=60)
    parser.add_argument("--batch_size",      type=int,   default=64)
    parser.add_argument("--lr",              type=float, default=3e-4)
    parser.add_argument("--weight_decay",    type=float, default=1e-4)
    parser.add_argument("--warmup_epochs",   type=int,   default=5)
    parser.add_argument("--smoothing",       type=float, default=0.05)
    parser.add_argument("--mixup_alpha",     type=float, default=0.2)
    parser.add_argument("--num_workers",     type=int,   default=8)
    parser.add_argument("--gpu",             type=int,   default=0)
    parser.add_argument("--merge_train_val", action="store_true")
    parser.add_argument("--run_name",        type=str,   default="se_resnet101_v2")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_path  = model_dir / f"{args.run_name}_log.csv"
    best_ckpt = model_dir / f"{args.run_name}_best.pt"

    loaders, class_to_idx = build_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        merge_train_val=args.merge_train_val,
    )
    print(f"Train batches: {len(loaders['train'])}  "
          f"Val batches: {len(loaders['val'])}")

    model = build_model(num_classes=len(class_to_idx)).to(device)

    print(f"Parameters: {count_parameters(model) / 1e6:.2f} M")

    criterion = MixUpCrossEntropy(smoothing=args.smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = warmup_cosine_schedule(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        base_lr=args.lr,
    )
    scaler = GradScaler(device="cuda")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc",
                         "val_loss", "val_acc", "lr"])

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], optimizer, criterion,
            scaler, device, mixup_alpha=args.mixup_alpha,
        )
        val_loss, val_acc = evaluate(model, loaders["val"], device)
        scheduler.step()
        current_lr = get_lr(optimizer)
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"lr={current_lr:.6f}  time={elapsed:.1f}s"
        )

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.4f}",
                             f"{val_loss:.4f}", f"{val_acc:.4f}",
                             f"{current_lr:.6f}"])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "class_to_idx": class_to_idx,
            }, best_ckpt)
            print(f"  → Best model saved (val_acc={val_acc:.4f})")

    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")
    print(f"Checkpoint : {best_ckpt}")
    print(f"Log        : {log_path}")


if __name__ == "__main__":
    main()
