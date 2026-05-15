"""
Elastic Weight Consolidation (EWC)
Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks", PNAS 2017.

Key idea: After training on task t, compute the Fisher information matrix
to measure parameter importance. For future tasks, add a penalty term that
prevents important weights from changing too much.

EWC loss = CE(current task) + λ * Σ_i F_i * (θ_i - θ*_i)²

where:
    F_i   = Fisher information (importance) of parameter i
    θ*_i  = optimal parameters after task t
    λ     = EWC regularization strength
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import numpy as np

from src.models.backbone import DermaCNN
from src.utils.data_utils import TASK_CLASSES, get_task_class_offsets
from src.utils.metrics import accuracy_on_loader


def compute_fisher_matrix(
    model: DermaCNN,
    loader: DataLoader,
    task_id: int,
    device: torch.device,
    n_samples: int = 200,
) -> Dict[str, torch.Tensor]:
    """
    Compute diagonal Fisher information matrix via empirical Fisher.

    F_i ≈ E[(∂ log p(y|x,θ) / ∂θ_i)²]

    Approximated by averaging squared gradients over a sample of data.
    """
    offsets = get_task_class_offsets()
    offset = offsets[task_id]
    n_classes = len(TASK_CLASSES[task_id])

    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}

    count = 0
    for images, labels in loader:
        if count >= n_samples:
            break
        images = images.to(device)
        labels = labels.squeeze().long().to(device)

        # Remap to local indices
        local_labels = torch.zeros_like(labels)
        for local_idx, global_idx in enumerate(TASK_CLASSES[task_id]):
            local_labels[labels == global_idx] = local_idx

        model.zero_grad()
        outputs = model(images)
        task_logits = outputs[:, offset: offset + n_classes]

        log_probs = nn.functional.log_softmax(task_logits, dim=1)
        # Sample from the model's distribution (empirical Fisher)
        sampled_labels = torch.multinomial(
            torch.exp(log_probs), num_samples=1
        ).squeeze()

        loss = nn.functional.nll_loss(log_probs, sampled_labels)
        loss.backward()

        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.data.pow(2) * images.size(0)

        count += images.size(0)

    # Normalize
    for n in fisher:
        fisher[n] /= count

    return fisher


class EWCTrainer:
    """
    EWC Trainer that accumulates Fisher matrices and optimal parameters
    across tasks.
    """

    def __init__(self, ewc_lambda: float = 5000.0):
        self.ewc_lambda = ewc_lambda
        self.fishers: List[Dict[str, torch.Tensor]] = []
        self.optima: List[Dict[str, torch.Tensor]] = []

    def ewc_penalty(self, model: DermaCNN) -> torch.Tensor:
        """
        Compute EWC penalty term.
        Returns scalar penalty summed over all previous tasks.
        """
        penalty = torch.tensor(0.0, device=next(model.parameters()).device)
        for fisher, optimum in zip(self.fishers, self.optima):
            for n, p in model.named_parameters():
                if n in fisher:
                    _fi = fisher[n].to(p.device)
                    _opt = optimum[n].to(p.device)
                    penalty += (_fi * (p - _opt).pow(2)).sum()
        return self.ewc_lambda * penalty

    def update(
        self,
        model: DermaCNN,
        loader: DataLoader,
        task_id: int,
        device: torch.device,
        n_samples: int = 200,
    ) -> None:
        """After training task `task_id`, store Fisher and optimal params."""
        fisher = compute_fisher_matrix(model, loader, task_id, device, n_samples)
        self.fishers.append(fisher)
        self.optima.append({n: p.data.clone() for n, p in model.named_parameters()})


def train_ewc(
    train_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    epochs_per_task: int = 10,
    lr: float = 1e-3,
    ewc_lambda: float = 5000.0,
    device: torch.device = None,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[List, np.ndarray]:
    """
    EWC sequential training.

    Args:
        train_loaders: one DataLoader per task
        test_loaders: one DataLoader per task
        epochs_per_task: training epochs per task
        lr: learning rate
        ewc_lambda: EWC regularization strength (higher = less forgetting, less plasticity)
        device: torch device
        seed: reproducibility seed
        verbose: print progress

    Returns:
        (model_snapshots, accuracy_matrix R)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    T = len(train_loaders)
    offsets = get_task_class_offsets()
    ewc = EWCTrainer(ewc_lambda=ewc_lambda)

    model = DermaCNN(num_classes=0).to(device)
    model_snapshots = []
    accuracy_rows = []

    for task_id in range(T):
        new_classes = len(TASK_CLASSES[task_id])
        model.expand_classifier(new_classes)
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        criterion = nn.CrossEntropyLoss()

        if verbose:
            print(f"\n[EWC λ={ewc_lambda}] Training Task {task_id} ...")

        for epoch in range(epochs_per_task):
            model.train()
            total_loss, ce_loss_sum, ewc_loss_sum = 0.0, 0.0, 0.0
            correct, total = 0, 0

            pbar = tqdm(train_loaders[task_id],
                        desc=f"  Epoch {epoch+1}/{epochs_per_task}",
                        leave=False, disable=not verbose)

            for images, labels in pbar:
                images = images.to(device)
                labels = labels.squeeze().long().to(device)

                local_labels = torch.zeros_like(labels)
                for local_idx, global_idx in enumerate(TASK_CLASSES[task_id]):
                    local_labels[labels == global_idx] = local_idx

                optimizer.zero_grad()
                outputs = model(images)
                offset = offsets[task_id]
                task_logits = outputs[:, offset: offset + new_classes]

                ce_loss = criterion(task_logits, local_labels)
                penalty = ewc.ewc_penalty(model)
                loss = ce_loss + penalty

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * images.size(0)
                ce_loss_sum += ce_loss.item() * images.size(0)
                ewc_loss_sum += penalty.item() * images.size(0)
                preds = task_logits.argmax(dim=1)
                correct += (preds == local_labels).sum().item()
                total += images.size(0)
                pbar.set_postfix(ce=f"{ce_loss.item():.3f}",
                                 ewc=f"{penalty.item():.3f}")

            scheduler.step()
            if verbose:
                print(f"    Epoch {epoch+1}: loss={total_loss/total:.4f}  "
                      f"CE={ce_loss_sum/total:.4f}  EWC={ewc_loss_sum/total:.4f}  "
                      f"acc={100*correct/total:.1f}%")

        # Update EWC with Fisher of this task
        ewc.update(model, train_loaders[task_id], task_id, device)

        model_snapshots.append(copy.deepcopy(model))

        row = {}
        for j in range(task_id + 1):
            nc = len(TASK_CLASSES[j])
            acc = accuracy_on_loader(
                model, test_loaders[j], task_id=j, device=device,
                class_offset=offsets[j], num_classes_task=nc,
            )
            row[j] = acc
            if verbose:
                print(f"  Task {j} test acc: {100*acc:.2f}%")
        accuracy_rows.append(row)

    R = np.zeros((T, T))
    for i, row in enumerate(accuracy_rows):
        for j, acc in row.items():
            R[i, j] = acc

    return model_snapshots, R
