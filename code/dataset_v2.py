# """
# dataset_v2.py
# Handles data loading for the image classification task.

# Key fix: folder names are numeric strings (0–99). Python's default
# os.listdir / ImageFolder sorts them lexicographically (0,1,10,11,...),
# which scrambles the label mapping. We explicitly sort by integer value.

# v2 changes:
#     - Stronger augmentation: added AutoAugment policy
#     - Added RandAugment as alternative
# """

# import os
# from pathlib import Path
# from typing import Callable, Optional, Tuple, List

# from PIL import Image
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as T


# # ── Label mapping ──────────────────────────────────────────────────────────────

# def get_class_to_idx(root: str) -> dict:
#     """
#     Return {class_name: label_idx} sorted numerically.
#     """
#     classes = sorted(
#         [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))],
#         key=lambda x: int(x),
#     )
#     return {cls: idx for idx, cls in enumerate(classes)}


# # ── Dataset classes ────────────────────────────────────────────────────────────

# class ClassificationDataset(Dataset):
#     def __init__(
#         self,
#         root: str,
#         class_to_idx: dict,
#         transform: Optional[Callable] = None,
#     ) -> None:
#         self.root = Path(root)
#         self.class_to_idx = class_to_idx
#         self.transform = transform
#         self.samples: List[Tuple[Path, int]] = self._load_samples()

#     def _load_samples(self) -> List[Tuple[Path, int]]:
#         samples = []
#         for cls_name, label in self.class_to_idx.items():
#             cls_dir = self.root / cls_name
#             if not cls_dir.is_dir():
#                 continue
#             for img_path in sorted(cls_dir.iterdir()):
#                 if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
#                     samples.append((img_path, label))
#         return samples

#     def __len__(self) -> int:
#         return len(self.samples)

#     def __getitem__(self, idx: int) -> Tuple:
#         img_path, label = self.samples[idx]
#         image = Image.open(img_path).convert("RGB")
#         if self.transform:
#             image = self.transform(image)
#         return image, label


# class TestDataset(Dataset):
#     def __init__(self, root: str, transform: Optional[Callable] = None) -> None:
#         self.root = Path(root)
#         self.transform = transform
#         self.img_paths: List[Path] = sorted(
#             [p for p in self.root.iterdir()
#              if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
#         )

#     def __len__(self) -> int:
#         return len(self.img_paths)

#     def __getitem__(self, idx: int) -> Tuple:
#         img_path = self.img_paths[idx]
#         image = Image.open(img_path).convert("RGB")
#         if self.transform:
#             image = self.transform(image)
#         return image, img_path.name


# # ── Transforms ─────────────────────────────────────────────────────────────────

# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD  = [0.229, 0.224, 0.225]
# INPUT_SIZE = 448


# def get_train_transform(input_size: int = INPUT_SIZE) -> T.Compose:
#     """
#     Strong augmentation pipeline.

#     Changes from v1:
#         - Added AutoAugment (ImageNet policy): automatically applies a
#           learned sequence of augmentations that has been shown to improve
#           generalisation on ImageNet-like datasets.
#         - Kept ColorJitter, RandomErasing as additional regularisation.
#         - Removed RandomGrayscale (redundant with AutoAugment).
#     """
#     return T.Compose([
#         T.RandomResizedCrop(input_size, scale=(0.7, 1.0)),  # ← 關鍵：不要裁太小
#         T.RandomHorizontalFlip(),

#         # ❌ 移除 VerticalFlip（不合理）
#         # ❌ 移除 ColorJitter（先降低正則）

#         T.AutoAugment(policy=T.AutoAugmentPolicy.IMAGENET),

#         T.ToTensor(),
#         T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),

#         T.RandomErasing(p=0.25, scale=(0.02, 0.2)),
#     ])


# def get_val_transform(input_size: int = INPUT_SIZE) -> T.Compose:
#     return T.Compose([
#         T.Resize(int(input_size * 256 / 224)),
#         T.CenterCrop(input_size),
#         T.ToTensor(),
#         T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
#     ])


# # ── DataLoader factory ─────────────────────────────────────────────────────────

# def build_dataloaders(
#     data_root: str = "data",
#     batch_size: int = 64,
#     num_workers: int = 8,
#     merge_train_val: bool = False,
# ) -> dict:
#     class_to_idx = get_class_to_idx(os.path.join(data_root, "train"))

#     train_ds = ClassificationDataset(
#         root=os.path.join(data_root, "train"),
#         class_to_idx=class_to_idx,
#         transform=get_train_transform(),
#     )

#     if merge_train_val:
#         val_ds_as_train = ClassificationDataset(
#             root=os.path.join(data_root, "val"),
#             class_to_idx=class_to_idx,
#             transform=get_train_transform(),
#         )
#         from torch.utils.data import ConcatDataset
#         train_ds = ConcatDataset([train_ds, val_ds_as_train])

#     val_ds = ClassificationDataset(
#         root=os.path.join(data_root, "val"),
#         class_to_idx=class_to_idx,
#         transform=get_val_transform(),
#     )

#     test_ds = TestDataset(
#         root=os.path.join(data_root, "test"),
#         transform=get_val_transform(),
#     )

#     loaders = {
#         "train": DataLoader(
#             train_ds, batch_size=batch_size, shuffle=True,
#             num_workers=num_workers, pin_memory=True, drop_last=True,
#         ),
#         "val": DataLoader(
#             val_ds, batch_size=batch_size, shuffle=False,
#             num_workers=num_workers, pin_memory=True,
#         ),
#         "test": DataLoader(
#             test_ds, batch_size=batch_size, shuffle=False,
#             num_workers=num_workers, pin_memory=True,
#         ),
#     }
#     return loaders, class_to_idx


# if __name__ == "__main__":
#     loaders, c2i = build_dataloaders(data_root="data", batch_size=8)
#     print(f"Classes: {len(c2i)}  |  First 5: {list(c2i.items())[:5]}")
#     print(f"Train batches: {len(loaders['train'])}")
#     imgs, labels = next(iter(loaders["train"]))
#     print(f"Image shape: {imgs.shape}  Labels: {labels.tolist()}")

# ====================================================================================

"""
dataset_v2.py
Handles data loading for the image classification task.

Key fix: folder names are numeric strings (0-99). Python's default
os.listdir / ImageFolder sorts them lexicographically (0, 1, 10, 11, ...),
which scrambles the label mapping. We explicitly sort by integer value.

v2 changes over v1:
    - Stronger augmentation: replaced ColorJitter with AutoAugment
      (ImageNet policy), which automatically applies a learned sequence
      of augmentations proven effective on ImageNet-like datasets.
    - Removed RandomGrayscale and RandomVerticalFlip (not appropriate
      for this dataset).
    - Added RandomErasing for additional regularisation.
    - Input resolution increased from 224 to 448.
"""

import os
from pathlib import Path
from typing import Callable, Optional, Tuple, List

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# ── Label mapping ──────────────────────────────────────────────────────────────

def get_class_to_idx(root: str) -> dict:
    """
    Return {class_name: label_idx} sorted numerically.
    """
    classes = sorted(
        [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))],
        key=lambda x: int(x),
    )
    return {cls: idx for idx, cls in enumerate(classes)}


# ── Dataset classes ────────────────────────────────────────────────────────────

class ClassificationDataset(Dataset):
    def __init__(
        self,
        root: str,
        class_to_idx: dict,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = self._load_samples()

    def _load_samples(self) -> List[Tuple[Path, int]]:
        samples = []
        for cls_name, label in self.class_to_idx.items():
            cls_dir = self.root / cls_name
            if not cls_dir.is_dir():
                continue
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    samples.append((img_path, label))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class TestDataset(Dataset):
    def __init__(self, root: str, transform: Optional[Callable] = None) -> None:
        self.root = Path(root)
        self.transform = transform
        self.img_paths: List[Path] = sorted(
            [p for p in self.root.iterdir()
             if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple:
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path.name


# ── Transforms ─────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
INPUT_SIZE = 448


def get_train_transform(input_size: int = INPUT_SIZE) -> T.Compose:
    """
    Strong augmentation pipeline for training.

    Steps:
        1. RandomResizedCrop: crop scale restricted to (0.7, 1.0) to retain
           sufficient semantic content in every crop.
        2. RandomHorizontalFlip: basic geometric augmentation.
        3. AutoAugment (ImageNet policy): data-driven augmentation strategy
           learned via reinforcement learning on ImageNet.
        4. ToTensor + Normalize: standard ImageNet normalisation.
        5. RandomErasing: randomly masks a rectangular region to encourage
           robust feature learning beyond single salient regions.
    """
    return T.Compose([
        T.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.AutoAugment(policy=T.AutoAugmentPolicy.IMAGENET),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        T.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])


def get_val_transform(input_size: int = INPUT_SIZE) -> T.Compose:
    """
    Deterministic preprocessing for validation and test sets.
    No random operations to ensure reproducible evaluation.
    """
    return T.Compose([
        T.Resize(int(input_size * 256 / 224)),
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ── DataLoader factory ─────────────────────────────────────────────────────────

def build_dataloaders(
    data_root: str = "data",
    batch_size: int = 64,
    num_workers: int = 8,
    merge_train_val: bool = False,
) -> dict:
    class_to_idx = get_class_to_idx(os.path.join(data_root, "train"))

    train_ds = ClassificationDataset(
        root=os.path.join(data_root, "train"),
        class_to_idx=class_to_idx,
        transform=get_train_transform(),
    )

    if merge_train_val:
        val_ds_as_train = ClassificationDataset(
            root=os.path.join(data_root, "val"),
            class_to_idx=class_to_idx,
            transform=get_train_transform(),
        )
        from torch.utils.data import ConcatDataset
        train_ds = ConcatDataset([train_ds, val_ds_as_train])

    val_ds = ClassificationDataset(
        root=os.path.join(data_root, "val"),
        class_to_idx=class_to_idx,
        transform=get_val_transform(),
    )

    test_ds = TestDataset(
        root=os.path.join(data_root, "test"),
        transform=get_val_transform(),
    )

    loaders = {
        "train": DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        ),
        "val": DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
        "test": DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
    }
    return loaders, class_to_idx


if __name__ == "__main__":
    loaders, c2i = build_dataloaders(data_root="data", batch_size=8)
    print(f"Classes: {len(c2i)}  |  First 5: {list(c2i.items())[:5]}")
    print(f"Train batches: {len(loaders['train'])}")
    imgs, labels = next(iter(loaders["train"]))
    print(f"Image shape: {imgs.shape}  Labels: {labels.tolist()}")