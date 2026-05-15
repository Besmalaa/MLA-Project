"""
Data utilities: load DermaMNIST and split into sequential CL tasks.

DermaMNIST has 7 classes:
  0: Melanocytic nevi
  1: Melanoma
  2: Benign keratosis-like lesions
  3: Basal cell carcinoma
  4: Actinic keratoses
  5: Vascular lesions
  6: Dermatofibroma

Task split (3 tasks, class-incremental):
  Task 0: classes [0, 1, 2]
  Task 1: classes [3, 4]
  Task 2: classes [5, 6]
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

try:
    import medmnist
    from medmnist import DermaMNIST
    MEDMNIST_AVAILABLE = True
except ImportError:
    MEDMNIST_AVAILABLE = False

# ---------------------------------------------------------------------------
# Default task configuration
# ---------------------------------------------------------------------------
TASK_CLASSES = [
    [0, 1, 2],   # Task 0 — 3 classes
    [3, 4],      # Task 1 — 2 classes
    [5, 6],      # Task 2 — 2 classes
]

CLASS_NAMES = [
    "Melanocytic nevi",
    "Melanoma",
    "Benign keratosis",
    "Basal cell carcinoma",
    "Actinic keratoses",
    "Vascular lesions",
    "Dermatofibroma",
]

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
TRAIN_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.Normalize(mean=[0.763, 0.546, 0.570],
                         std=[0.141, 0.152, 0.169]),
])

TEST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.763, 0.546, 0.570],
                         std=[0.141, 0.152, 0.169]),
])


# ---------------------------------------------------------------------------
# Task-filtered dataset wrapper
# ---------------------------------------------------------------------------
class TaskDataset(Dataset):
    """Wraps a MedMNIST dataset and filters to a subset of classes."""

    def __init__(self, base_dataset, task_classes: list[int],
                 remap_labels: bool = True):
        self.base = base_dataset
        self.task_classes = task_classes
        self.remap_labels = remap_labels

        # Build label mapping: original class -> local index (0, 1, 2, ...)
        self.label_map = {c: i for i, c in enumerate(task_classes)}

        # Filter indices
        self.indices = []
        for idx in range(len(base_dataset)):
            label = int(base_dataset[idx][1])
            if label in self.label_map:
                self.indices.append(idx)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.base[self.indices[idx]]
        label = int(label)
        if self.remap_labels:
            label = self.label_map[label]
        return img, label

    @property
    def num_classes(self):
        return len(self.task_classes)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_task_dataloaders(
    task_id: int,
    batch_size: int = 64,
    num_workers: int = 0,
    data_root: str = "./data",
    remap_labels: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader) for a given task."""
    assert MEDMNIST_AVAILABLE, "pip install medmnist"

    task_classes = TASK_CLASSES[task_id]

    train_base = DermaMNIST(split="train", transform=TRAIN_TRANSFORM,
                            download=True, root=data_root)
    test_base = DermaMNIST(split="test", transform=TEST_TRANSFORM,
                           download=True, root=data_root)

    train_ds = TaskDataset(train_base, task_classes, remap_labels)
    test_ds = TaskDataset(test_base, task_classes, remap_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=False)
    return train_loader, test_loader


def get_all_tasks_dataloaders(
    batch_size: int = 64,
    num_workers: int = 0,
    data_root: str = "./data",
) -> list[tuple[DataLoader, DataLoader]]:
    """Return list of (train_loader, test_loader) for all tasks."""
    return [
        get_task_dataloaders(t, batch_size, num_workers, data_root)
        for t in range(len(TASK_CLASSES))
    ]


def get_full_dataset_loader(
    batch_size: int = 64,
    num_workers: int = 0,
    data_root: str = "./data",
) -> tuple[DataLoader, DataLoader]:
    """Full DermaMNIST (all 7 classes) — used by Joint Training."""
    assert MEDMNIST_AVAILABLE, "pip install medmnist"
    train_base = DermaMNIST(split="train", transform=TRAIN_TRANSFORM,
                            download=True, root=data_root)
    test_base = DermaMNIST(split="test", transform=TEST_TRANSFORM,
                           download=True, root=data_root)

    # Squeeze label tensor to scalar
    class SqueezeLabel(Dataset):
        def __init__(self, ds):
            self.ds = ds
        def __len__(self): return len(self.ds)
        def __getitem__(self, i):
            img, lbl = self.ds[i]
            return img, int(lbl)

    train_loader = DataLoader(SqueezeLabel(train_base), batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(SqueezeLabel(test_base), batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    return train_loader, test_loader