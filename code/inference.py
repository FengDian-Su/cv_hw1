"""
inference.py
Run inference on the test set and produce prediction.csv for submission.

Test Time Augmentation (TTA):
    Instead of predicting from a single centre-crop, we apply N different
    transforms (flips, crops) and average the softmax probabilities.
    This typically improves accuracy by 0.5–1.5% at zero training cost.
"""

import argparse
import csv
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.amp import autocast

from dataset_v2 import TestDataset, IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE
from model import build_model


# ── TTA transforms ─────────────────────────────────────────────────────────────

def get_tta_transforms(input_size: int = INPUT_SIZE):
    """
    Returns a list of deterministic transforms for TTA.
    Each image is passed through all transforms; probabilities are averaged.
    """
    resize = T.Resize(int(input_size * 256 / 224))
    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    return [
        # 1. centre crop (standard eval)
        T.Compose([resize, T.CenterCrop(input_size), T.ToTensor(), normalize]),
        # 2. horizontal flip + centre crop
        T.Compose([resize, T.CenterCrop(input_size),
                   T.RandomHorizontalFlip(p=1.0), T.ToTensor(), normalize]),
        # 3. top-left crop
        T.Compose([resize, T.FiveCrop(input_size),
                   T.Lambda(lambda crops: crops[0]),  # top-left
                   T.ToTensor(), normalize]),
        # 4. top-right crop
        T.Compose([resize, T.FiveCrop(input_size),
                   T.Lambda(lambda crops: crops[1]),  # top-right
                   T.ToTensor(), normalize]),
        # 5. bottom-left crop
        T.Compose([resize, T.FiveCrop(input_size),
                   T.Lambda(lambda crops: crops[2]),
                   T.ToTensor(), normalize]),
        # 6. bottom-right crop
        T.Compose([resize, T.FiveCrop(input_size),
                   T.Lambda(lambda crops: crops[3]),
                   T.ToTensor(), normalize]),
    ]


# ── Inference ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_with_tta(
    model: torch.nn.Module,
    test_root: str,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 8,
    tta: bool = True,
) -> tuple[list[str], list[int]]:
    """
    Run inference (with optional TTA) on the test set.

    Returns:
        filenames : list of image filenames (e.g. 'abc123.jpg')
        predictions : list of predicted class indices
    """
    model.eval()

    if tta:
        transforms = get_tta_transforms()
    else:
        transforms = [
            T.Compose([
                T.Resize(int(INPUT_SIZE * 256 / 224)),
                T.CenterCrop(INPUT_SIZE),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        ]

    # collect per-transform softmax probabilities
    all_probs = None   # shape: (N_images, N_classes)
    all_fnames = None

    for t_idx, transform in enumerate(transforms):
        dataset = TestDataset(root=test_root, transform=transform)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

        probs_list = []
        fnames_list = []

        for imgs, fnames in loader:
            imgs = imgs.to(device)
            with autocast(device_type="cuda"):
                logits = model(imgs)
            probs = F.softmax(logits.float(), dim=1).cpu()
            probs_list.append(probs)
            if t_idx == 0:
                fnames_list.extend(fnames)

        transform_probs = torch.cat(probs_list, dim=0)   # (N, 100)

        if all_probs is None:
            all_probs = transform_probs
            all_fnames = fnames_list
        else:
            all_probs += transform_probs

        print(f"  TTA step {t_idx + 1}/{len(transforms)} done")

    # average over TTA steps → argmax
    all_probs /= len(transforms)
    predictions = all_probs.argmax(dim=1).tolist()

    return all_fnames, predictions


# ── Write CSV ──────────────────────────────────────────────────────────────────

def write_prediction_csv(
    filenames: list[str],
    predictions: list[int],
    output_path: str = "prediction.csv",
) -> None:
    """
    Write prediction.csv in the format expected by CodaBench.

    Format:
        image_name,pred_label
        abc123.jpg,42
        ...
    """
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "pred_label"])
        for fname, pred in zip(filenames, predictions):
            writer.writerow([fname.replace(".jpg", ""), pred])
    print(f"Saved {len(predictions)} predictions → {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference with TTA")
    parser.add_argument("--checkpoint",  type=str, required=True,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--data_root",   type=str, default="data")
    parser.add_argument("--output",      type=str, default="prediction.csv")
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu",         type=int, default=0)
    parser.add_argument("--no_tta",      action="store_true",
                        help="Disable TTA (faster but less accurate)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    class_to_idx = ckpt["class_to_idx"]
    num_classes = len(class_to_idx)
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}  "
          f"(val_acc={ckpt['val_acc']:.4f})  classes={num_classes}")

    # build model and load weights
    model = build_model(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # run inference
    test_root = os.path.join(args.data_root, "test")
    use_tta = not args.no_tta
    print(f"TTA: {'enabled' if use_tta else 'disabled'}")

    filenames, predictions = predict_with_tta(
        model=model,
        test_root=test_root,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tta=use_tta,
    )

    write_prediction_csv(filenames, predictions, output_path=args.output)


if __name__ == "__main__":
    main()
