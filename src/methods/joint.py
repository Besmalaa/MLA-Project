"""
Joint Training — Oracle upper bound.

Trains on all tasks simultaneously (offline). This represents the best
possible performance since the model sees all data at once.
It is NOT a continual learning method — it is the performance ceiling.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.trainer import BaseTrainer


class JointTrainer(BaseTrainer):
    """
    Joint Training on the full dataset (all 7 DermaMNIST classes at once).
    Upper bound for any continual learning approach.
    """
    name = "Joint Training"

    def __init__(self, model: nn.Module, device: str = "cpu",
                 lr: float = 1e-3, weight_decay: float = 1e-4):
        super().__init__(model, device, lr, weight_decay)

    def compute_loss(self, imgs: torch.Tensor, labels: torch.Tensor,
                     task_id: int) -> torch.Tensor:
        logits = self.model(imgs)
        return self.criterion(logits, labels)
