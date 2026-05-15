"""
Continual Learning evaluation metrics.

Key metrics:
  - Final Average Accuracy (FAA): mean accuracy over all tasks after full CL training
  - Backward Transfer (BWT): measures forgetting
      BWT = 1/(T-1) * sum_{t=1}^{T-1} (R_{T,t} - R_{t,t})
      Negative BWT = forgetting, Positive = positive backward transfer
  - Per-task accuracy matrix R[i][j] = acc on task j after training on task i
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader


def evaluate(model: torch.nn.Module,
             loader: DataLoader,
             device: str = "cpu") -> float:
    """Return accuracy (0–1) of model on a dataloader."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def compute_cl_metrics(acc_matrix: np.ndarray) -> dict:
    """
    Compute CL metrics from the accuracy matrix.

    Parameters
    ----------
    acc_matrix : np.ndarray, shape (T, T)
        acc_matrix[i, j] = accuracy on task j after training on tasks 0..i.

    Returns
    -------
    dict with keys: faa, bwt, per_task_final
    """
    T = acc_matrix.shape[0]

    # Final Average Accuracy — last row
    faa = float(np.mean(acc_matrix[T - 1, :]))

    # Backward Transfer
    if T > 1:
        bwt = float(np.mean([
            acc_matrix[T - 1, t] - acc_matrix[t, t]
            for t in range(T - 1)
        ]))
    else:
        bwt = 0.0

    per_task_final = acc_matrix[T - 1, :].tolist()

    return {"faa": faa, "bwt": bwt, "per_task_final": per_task_final}


def print_metrics_table(results: dict[str, dict]) -> None:
    """Pretty-print a table comparing methods."""
    print("\n" + "=" * 60)
    print(f"{'Method':<22} {'FAA':>8} {'BWT':>10}")
    print("-" * 60)
    for method, m in results.items():
        faa = f"{m['faa']*100:.1f}%"
        bwt = f"{m['bwt']*100:+.1f}%"
        print(f"{method:<22} {faa:>8} {bwt:>10}")
    print("=" * 60)
