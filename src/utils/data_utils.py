"""
Data utilities for DermaMNIST Continual Learning.

DermaMNIST has 7 classes (skin disease types):
    0: actinic keratoses
    1: basal cell carcinoma
    2: benign keratosis-like lesions
    3: dermatofibroma
    4: melanoma
    5: melanocytic nevi
    6: vascular lesions

We split into 3 sequential tasks for class-incremental learning:
    Task 0: classes [0, 1, 2]   — 3 classes
    Task 1: classes [3, 4]      — 2 classes
    Task 2: classes [5, 6]      — 2 classes
"""

from typing import List, Tuple, Dict
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision.transforms as transforms


# ── Class names ───────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Actinic keratoses",
    "Basal cell carcinoma",
    "Benign keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic nevi",
    "Vascular lesions",
]

# ── Task definition ────────────────────────────────────────────────────────────
TASK_CLASSES: List[List[int]] = [
    [0, 1, 2],   # Task 0
    [3, 4],      # Task 1
    [5, 6],      # Task 2
]

NUM_TASKS = len(TASK_CLASSES)


# ── Transforms ────────────────────────────────────────────────────────────────
TRAIN_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.Normalize(mean=[0.763, 0.546, 0.570],
                         std=[0.141, 0.152, 0.169]),
])

TEST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.763, 0.546, 0.570],
                         std=[0.141, 0.152, 0.169]),
])


def load_dermamnist(data_dir: str = "./data"):
    """
    Download and load DermaMNIST via the medmnist package.
    Returns train and test MedMNIST dataset objects.
    """
    try:
        import medmnist
        from medmnist import DermaMNIST
    except ImportError:
        raise ImportError("Install medmnist: pip install medmnist")

    train_dataset = DermaMNIST(
        split="train",
        transform=TRAIN_TRANSFORM,
        download=True,
        root=data_dir,
    )
    test_dataset = DermaMNIST(
        split="test",
        transform=TEST_TRANSFORM,
        download=True,
        root=data_dir,
    )
    return train_dataset, test_dataset


def split_into_tasks(
    dataset,
    task_classes: List[List[int]] = None,
) -> List[Subset]:
    """
    Split a DermaMNIST dataset into task-specific subsets.

    Args:
        dataset: MedMNIST dataset object
        task_classes: list of class lists per task

    Returns:
        List of torch Subset objects, one per task
    """
    if task_classes is None:
        task_classes = TASK_CLASSES

    # MedMNIST labels are shape (N, 1)
    labels = dataset.labels.squeeze()

    task_subsets = []
    for classes in task_classes:
        mask = np.isin(labels, classes)
        indices = np.where(mask)[0].tolist()
        task_subsets.append(Subset(dataset, indices))

    return task_subsets


def get_task_loaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    num_workers: int = 2,
) -> Tuple[List[DataLoader], List[DataLoader]]:
    """
    Returns (train_loaders, test_loaders) — one per task.
    """
    train_ds, test_ds = load_dermamnist(data_dir)
    train_tasks = split_into_tasks(train_ds)
    test_tasks = split_into_tasks(test_ds)

    train_loaders = [
        DataLoader(t, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=True)
        for t in train_tasks
    ]
    test_loaders = [
        DataLoader(t, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=True)
        for t in test_tasks
    ]
    return train_loaders, test_loaders


def get_joint_loader(
    data_dir: str = "./data",
    batch_size: int = 64,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns a single (train_loader, test_loader) for all 7 classes.
    Used for Joint Training upper-bound.
    """
    train_ds, test_ds = load_dermamnist(data_dir)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=True)
    return train_loader, test_loader


def remap_labels(labels: torch.Tensor, task_id: int) -> torch.Tensor:
    """
    Remap global class indices to local task class indices.

    E.g. Task 1 has global classes [3, 4].
    Global label 3 -> local label 0, global 4 -> local 1.
    """
    classes = TASK_CLASSES[task_id]
    remapped = torch.zeros_like(labels)
    for local_idx, global_idx in enumerate(classes):
        remapped[labels == global_idx] = local_idx
    return remapped


def get_cumulative_num_classes(up_to_task: int) -> int:
    """Total number of classes seen after completing task `up_to_task`."""
    return sum(len(TASK_CLASSES[t]) for t in range(up_to_task + 1))


def get_task_class_offsets() -> List[int]:
    """
    Returns list of class offsets per task.
    E.g. [0, 3, 5] meaning:
        Task 0 classes start at output index 0
        Task 1 classes start at output index 3
        Task 2 classes start at output index 5
    """
    offsets = []
    cumulative = 0
    for classes in TASK_CLASSES:
        offsets.append(cumulative)
        cumulative += len(classes)
    return offsets
