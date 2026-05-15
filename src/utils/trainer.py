"""
Generic training loop used by all CL methods.
Each method can override `before_task`, `after_task`, and `compute_loss`.
"""
from __future__ import annotations

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class BaseTrainer:
    """
    Base class for all continual learning trainers.
    Subclasses override `compute_loss` to add regularization terms.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.model = model
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = self._build_optimizer()
        self.criterion = nn.CrossEntropyLoss()

    def _build_optimizer(self):
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def before_task(self, task_id: int, train_loader: DataLoader):
        """Called before training on a new task. Override to prepare state."""
        # Reset optimizer for each new task
        self.optimizer = self._build_optimizer()

    def after_task(self, task_id: int, train_loader: DataLoader):
        """Called after training on a task. Override to consolidate knowledge."""
        pass

    def compute_loss(self, imgs: torch.Tensor, labels: torch.Tensor,
                     task_id: int) -> torch.Tensor:
        """Override to add regularization. Default = CE loss."""
        logits = self.model(imgs)
        return self.criterion(logits, labels)

    def train_epoch(self, task_id: int, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n = 0
        for imgs, labels in loader:
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            loss = self.compute_loss(imgs, labels, task_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
        return total_loss / n if n > 0 else 0.0

    def fit(
        self,
        task_id: int,
        train_loader: DataLoader,
        num_epochs: int = 10,
        verbose: bool = True,
    ) -> list[float]:
        self.before_task(task_id, train_loader)
        losses = []
        for epoch in range(1, num_epochs + 1):
            loss = self.train_epoch(task_id, train_loader)
            losses.append(loss)
            if verbose:
                print(f"  [Task {task_id}] Epoch {epoch:02d}/{num_epochs}  "
                      f"Loss: {loss:.4f}")
        self.after_task(task_id, train_loader)
        return losses
