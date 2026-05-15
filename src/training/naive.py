"""
Naive Fine-tuning Baseline for Continual Learning.

The simplest approach: just train on each task sequentially with standard
cross-entropy, no forgetting protection at all.
This is the catastrophic forgetting baseline.
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Optional

from src.models.backbone import DermaCNN
from src.utils.data_utils import (
    TASK_CLASSES, get_task_class_offsets, get_cumulative_num_classes
)
from src.utils.metrics import accuracy_on_loader


def train_naive(
    train_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    epochs_per_task: int = 10,
    lr: float = 1e-3,
    device: torch.device = None,
    seed: int = 42,
    verbose: bool = True,
) -> tuple:
    """
    Naive sequential fine-tuning.

    Args:
        train_loaders: list of DataLoaders, one per task
        test_loaders: list of DataLoaders for evaluation
        epochs_per_task: number of epochs per task
        lr: learning rate
        device: computation device
        seed: random seed for reproducibility
        verbose: print progress

    Returns:
        (model_snapshots, accuracy_matrix)
        - model_snapshots[i] = model after training task i
        - accuracy_matrix[i][j] = accuracy on task j after task i
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    T = len(train_loaders)
    offsets = get_task_class_offsets()

    model = DermaCNN(num_classes=0).to(device)
    model_snapshots = []
    accuracy_rows = []

    for task_id in range(T):
        # Expand head for new task's classes
        new_classes = len(TASK_CLASSES[task_id])
        model.expand_classifier(new_classes)
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        criterion = nn.CrossEntropyLoss()

        if verbose:
            print(f"\n[Naive] Training Task {task_id} ({new_classes} new classes) ...")

        for epoch in range(epochs_per_task):
            model.train()
            total_loss, correct, total = 0.0, 0, 0
            loader = train_loaders[task_id]
            pbar = tqdm(loader, desc=f"  Epoch {epoch+1}/{epochs_per_task}",
                        leave=False, disable=not verbose)

            for images, labels in pbar:
                images = images.to(device)
                labels = labels.squeeze().long().to(device)

                # Map global labels to local task indices for loss
                local_labels = torch.zeros_like(labels)
                for local_idx, global_idx in enumerate(TASK_CLASSES[task_id]):
                    local_labels[labels == global_idx] = local_idx

                optimizer.zero_grad()
                outputs = model(images)
                # Only use this task's slice of the logits
                offset = offsets[task_id]
                task_logits = outputs[:, offset: offset + new_classes]
                loss = criterion(task_logits, local_labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * images.size(0)
                preds = task_logits.argmax(dim=1)
                correct += (preds == local_labels).sum().item()
                total += images.size(0)
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            scheduler.step()
            if verbose:
                print(f"    Epoch {epoch+1}: loss={total_loss/total:.4f}  "
                      f"acc={100*correct/total:.1f}%")

        # Snapshot model state
        model_snapshots.append(copy.deepcopy(model))

        # Evaluate on all tasks seen so far
        row = {}
        for j in range(task_id + 1):
            nc = len(TASK_CLASSES[j])
            acc = accuracy_on_loader(
                model, test_loaders[j], task_id=j, device=device,
                class_offset=offsets[j], num_classes_task=nc
            )
            row[j] = acc
            if verbose:
                print(f"  Task {j} test acc: {100*acc:.2f}%")
        accuracy_rows.append(row)

    # Build accuracy matrix
    import numpy as np
    R = np.zeros((T, T))
    for i, row in enumerate(accuracy_rows):
        for j, acc in row.items():
            R[i, j] = acc

    return model_snapshots, R
