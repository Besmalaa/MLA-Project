"""
Elastic Weight Consolidation (EWC) — Kirkpatrick et al., 2017.

EWC adds a quadratic penalty to prevent parameters important for
previous tasks from changing too much:

  L_EWC = L_CE(task_t) + (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2

where:
  F_i = Fisher Information diagonal estimate for parameter i
  theta*_i = parameter value after training on the previous task
  lambda = regularization strength (importance of previous tasks)

Reference: "Overcoming catastrophic forgetting in neural networks"
           Kirkpatrick et al., PNAS 2017.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.trainer import BaseTrainer


class EWCTrainer(BaseTrainer):
    """
    EWC continual learner.

    Parameters
    ----------
    ewc_lambda : float
        Regularization strength. Higher = more protection against forgetting.
        Typical range: 100–10000.
    """
    name = "EWC"

    def __init__(self, model: nn.Module, device: str = "cpu",
                 lr: float = 1e-3, weight_decay: float = 1e-4,
                 ewc_lambda: float = 5000.0):
        super().__init__(model, device, lr, weight_decay)
        self.ewc_lambda = ewc_lambda

        # Accumulated Fisher information and optimal parameters per task
        # stored as lists (one entry per completed task)
        self._fisher_diags: list[dict[str, torch.Tensor]] = []
        self._optimal_params: list[dict[str, torch.Tensor]] = []

    # ------------------------------------------------------------------
    # Fisher diagonal estimation via empirical Fisher
    # ------------------------------------------------------------------
    def _estimate_fisher(self, loader: DataLoader) -> dict[str, torch.Tensor]:
        """
        Estimate diagonal Fisher Information Matrix via squared gradients.
        Uses the training set of the just-completed task.
        """
        self.model.eval()
        fisher = {n: torch.zeros_like(p)
                  for n, p in self.model.named_parameters() if p.requires_grad}

        n_samples = 0
        for imgs, labels in loader:
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            self.model.zero_grad()
            with torch.enable_grad():
                logits = self.model(imgs)
                log_probs = torch.log_softmax(logits, dim=1)
                # Use labels (empirical Fisher)
                loss = nn.functional.nll_loss(log_probs, labels)
                loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += (p.grad.detach() ** 2) * imgs.size(0)

            n_samples += imgs.size(0)

        # Normalize
        if n_samples > 0:
            for n in fisher:
                fisher[n] /= n_samples

        return fisher

    def _store_optimal_params(self) -> dict[str, torch.Tensor]:
        return {n: p.detach().clone()
                for n, p in self.model.named_parameters() if p.requires_grad}

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def after_task(self, task_id: int, train_loader: DataLoader):
        """After each task: compute and store Fisher + optimal params."""
        print(f"  [EWC] Computing Fisher for task {task_id}...")
        fisher = self._estimate_fisher(train_loader)
        opt_params = self._store_optimal_params()
        self._fisher_diags.append(fisher)
        self._optimal_params.append(opt_params)

    # ------------------------------------------------------------------
    # Loss with EWC penalty
    # ------------------------------------------------------------------
    def compute_loss(self, imgs: torch.Tensor, labels: torch.Tensor,
                     task_id: int) -> torch.Tensor:
        logits = self.model(imgs)
        ce_loss = self.criterion(logits, labels)

        ewc_loss = torch.tensor(0.0, device=self.device)
        for fisher, opt in zip(self._fisher_diags, self._optimal_params):
            for n, p in self.model.named_parameters():
                if n in fisher and p.requires_grad:
                    ewc_loss += (fisher[n] * (p - opt[n]).pow(2)).sum()

        return ce_loss + (self.ewc_lambda / 2.0) * ewc_loss
