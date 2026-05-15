"""
Learning without Forgetting (LwF) — Li & Hoiem, 2017.

LwF uses knowledge distillation: the outputs of the old model on new task data
serve as soft targets to preserve old task knowledge:

  L_LwF = L_CE(new_task) + alpha * L_KD(old_logits, new_logits)

where L_KD is KL divergence between old and new model outputs (at temperature T).

Reference: "Learning without Forgetting"
           Li & Hoiem, TPAMI 2017.
"""
from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils.trainer import BaseTrainer


class LwFTrainer(BaseTrainer):
    """
    LwF continual learner using knowledge distillation.

    Parameters
    ----------
    alpha : float
        Distillation loss weight (balance between plasticity and stability).
    temperature : float
        Softmax temperature for distillation (higher = softer distribution).
    """
    name = "LwF"

    def __init__(self, model: nn.Module, device: str = "cpu",
                 lr: float = 1e-3, weight_decay: float = 1e-4,
                 alpha: float = 1.0, temperature: float = 2.0):
        super().__init__(model, device, lr, weight_decay)
        self.alpha = alpha
        self.temperature = temperature
        self._old_model: nn.Module | None = None

    def before_task(self, task_id: int, train_loader: DataLoader):
        """Save snapshot of model before training on new task."""
        super().before_task(task_id, train_loader)
        if task_id > 0:
            # Deep-copy model to use as teacher
            self._old_model = copy.deepcopy(self.model)
            self._old_model.eval()
            for p in self._old_model.parameters():
                p.requires_grad_(False)

    def _distillation_loss(self, imgs: torch.Tensor,
                           new_logits: torch.Tensor) -> torch.Tensor:
        """KL divergence between old and new model outputs."""
        T = self.temperature
        with torch.no_grad():
            old_logits = self._old_model(imgs)
        old_soft = F.softmax(old_logits / T, dim=1)
        new_log_soft = F.log_softmax(new_logits / T, dim=1)
        # KL(old || new) * T^2 (standard LwF formulation)
        return F.kl_div(new_log_soft, old_soft, reduction="batchmean") * (T ** 2)

    def compute_loss(self, imgs: torch.Tensor, labels: torch.Tensor,
                     task_id: int) -> torch.Tensor:
        logits = self.model(imgs)
        ce_loss = self.criterion(logits, labels)

        if self._old_model is None or task_id == 0:
            return ce_loss

        kd_loss = self._distillation_loss(imgs, logits)
        return ce_loss + self.alpha * kd_loss
