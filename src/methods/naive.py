"""
Naive Fine-tuning — Baseline for catastrophic forgetting.

Simply trains on each task sequentially without any protection
against forgetting previously learned tasks. This is the lower bound:
it shows maximum forgetting.
"""
from src.utils.trainer import BaseTrainer


class NaiveTrainer(BaseTrainer):
    """
    Standard SGD/Adam fine-tuning with no continual learning strategy.
    Expected to show severe catastrophic forgetting.
    """
    name = "Naive Fine-tuning"

    def __init__(self, model, device="cpu", lr=1e-3, weight_decay=1e-4):
        super().__init__(model, device, lr, weight_decay)
