"""
Experience Replay — Memory-based continual learning.

Maintains a fixed-size episodic memory buffer of past task examples.
During training on new tasks, randomly replays stored examples alongside
new task data to preserve past knowledge.

Strategy: reservoir sampling ensures uniform random coverage across tasks.

Reference: "Continual Learning with Deep Generative Replay"
           (basis: iCaRL, GEM, reservoir replay)
"""
from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.trainer import BaseTrainer


class ReplayBuffer:
    """
    Fixed-size episodic memory buffer using reservoir sampling.

    Parameters
    ----------
    capacity : int
        Maximum number of samples to store (total, across all tasks).
    """

    def __init__(self, capacity: int = 500):
        self.capacity = capacity
        self.images: list[torch.Tensor] = []
        self.labels: list[int] = []
        self._n_seen = 0  # total samples seen (for reservoir sampling)

    def add_from_loader(self, loader: DataLoader, device: str = "cpu",
                        max_per_task: int | None = None):
        """
        Add samples from a DataLoader to the buffer using reservoir sampling.
        Keeps at most `max_per_task` from each task if specified.
        """
        new_imgs: list[torch.Tensor] = []
        new_labels: list[int] = []

        for imgs, labels in loader:
            for img, lbl in zip(imgs, labels):
                new_imgs.append(img.cpu())
                new_labels.append(int(lbl))

        # If max_per_task is set, randomly subsample
        if max_per_task and len(new_imgs) > max_per_task:
            idx = random.sample(range(len(new_imgs)), max_per_task)
            new_imgs = [new_imgs[i] for i in idx]
            new_labels = [new_labels[i] for i in idx]

        # Reservoir sampling into the buffer
        for img, lbl in zip(new_imgs, new_labels):
            self._n_seen += 1
            if len(self.images) < self.capacity:
                self.images.append(img)
                self.labels.append(lbl)
            else:
                j = random.randint(0, self._n_seen - 1)
                if j < self.capacity:
                    self.images[j] = img
                    self.labels[j] = lbl

    def sample(self, n: int, device: str = "cpu"):
        """Return a random batch of (imgs, labels) from the buffer."""
        if len(self.images) == 0:
            return None, None
        n = min(n, len(self.images))
        idx = random.sample(range(len(self.images)), n)
        imgs = torch.stack([self.images[i] for i in idx]).to(device)
        labels = torch.tensor([self.labels[i] for i in idx],
                              dtype=torch.long, device=device)
        return imgs, labels

    def __len__(self):
        return len(self.images)


class ReplayTrainer(BaseTrainer):
    """
    Experience Replay continual learner.

    Parameters
    ----------
    memory_size : int
        Total buffer capacity (shared across all tasks).
    replay_batch_size : int
        Number of replayed samples mixed into each training batch.
    """
    name = "Experience Replay"

    def __init__(self, model: nn.Module, device: str = "cpu",
                 lr: float = 1e-3, weight_decay: float = 1e-4,
                 memory_size: int = 500, replay_batch_size: int = 32):
        super().__init__(model, device, lr, weight_decay)
        self.memory_size = memory_size
        self.replay_batch_size = replay_batch_size
        self.buffer = ReplayBuffer(capacity=memory_size)

    def after_task(self, task_id: int, train_loader: DataLoader):
        """Store samples from the completed task into the replay buffer."""
        max_per_task = self.memory_size // (task_id + 1)
        print(f"  [Replay] Storing up to {max_per_task} samples from task {task_id}...")
        # Re-balance: shrink existing entries proportionally
        if task_id > 0:
            per_old = self.memory_size // (task_id + 1)
            # Simple truncation of buffer to keep uniform distribution
            if len(self.buffer.images) > per_old * task_id:
                self.buffer.images = self.buffer.images[:per_old * task_id]
                self.buffer.labels = self.buffer.labels[:per_old * task_id]

        self.buffer.add_from_loader(train_loader, device=self.device,
                                    max_per_task=max_per_task)
        print(f"  [Replay] Buffer size: {len(self.buffer)}")

    def compute_loss(self, imgs: torch.Tensor, labels: torch.Tensor,
                     task_id: int) -> torch.Tensor:
        # Current task CE loss
        logits = self.model(imgs)
        ce_loss = self.criterion(logits, labels)

        if task_id == 0 or len(self.buffer) == 0:
            return ce_loss

        # Replay loss from memory buffer
        replay_imgs, replay_labels = self.buffer.sample(
            self.replay_batch_size, device=self.device)
        if replay_imgs is not None:
            replay_logits = self.model(replay_imgs)
            replay_loss = self.criterion(replay_logits, replay_labels)
            return (ce_loss + replay_loss) / 2.0

        return ce_loss
