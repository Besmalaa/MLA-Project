"""
Visualization utilities for continual learning experiments.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import List, Dict, Optional

TASK_NAMES = ["Task 0\n(Actinic, BCC, Benign)", "Task 1\n(Dermatofibroma, Melanoma)", "Task 2\n(Nevi, Vascular)"]
METHOD_COLORS = {
    "Naive": "#e74c3c",
    "EWC": "#3498db",
    "LwF": "#2ecc71",
    "Replay": "#f39c12",
    "Joint": "#9b59b6",
}


def plot_accuracy_matrix(
    R: np.ndarray,
    method_name: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of the accuracy matrix R[i,j].
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    T = R.shape[0]

    mask = np.zeros_like(R, dtype=bool)
    for i in range(T):
        for j in range(T):
            if j > i:
                mask[i, j] = True

    annot = np.where(mask, np.nan, R * 100)
    annot_str = np.where(mask, "", [[f"{v:.1f}%" for v in row] for row in R * 100])

    sns.heatmap(
        np.where(mask, np.nan, R * 100),
        annot=annot_str,
        fmt="",
        cmap="YlOrRd_r",
        vmin=0, vmax=100,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Accuracy (%)"},
    )

    ax.set_title(f"Accuracy Matrix — {method_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Test Task j", fontsize=11)
    ax.set_ylabel("After Training Task i", fontsize=11)
    ax.set_xticklabels([f"T{j}" for j in range(T)])
    ax.set_yticklabels([f"T{i}" for i in range(T)], rotation=0)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_forgetting_curves(
    all_results: Dict[str, dict],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Line plot: accuracy on Task 0 as training progresses through tasks.
    Shows catastrophic forgetting for each method.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for t in range(3):
        ax = axes[t]
        for method, res in all_results.items():
            R = np.array(res["accuracy_matrix"])
            T = R.shape[0]
            # Accuracy on task t after each subsequent task training
            accs = [R[i, t] * 100 for i in range(t, T)]
            x = list(range(t, T))
            color = METHOD_COLORS.get(method, "gray")
            ax.plot(x, accs, marker="o", label=method, color=color, linewidth=2)

        ax.set_title(f"Accuracy on Task {t}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Training step (task index)")
        if t == 0:
            ax.set_ylabel("Test Accuracy (%)")
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"After T{i}" for i in range(3)], fontsize=8)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle("Forgetting Curves per Task", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_final_comparison(
    all_results: Dict[str, dict],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart comparing FAA and BWT across all methods.
    """
    methods = list(all_results.keys())
    faas = [all_results[m]["final_avg_accuracy"] for m in methods]
    bwts = [all_results[m]["backward_transfer"] for m in methods]
    colors = [METHOD_COLORS.get(m, "gray") for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # FAA
    bars = axes[0].bar(methods, faas, color=colors, edgecolor="black", linewidth=0.8)
    axes[0].set_title("Final Average Accuracy (FAA)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[0].grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, faas):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # BWT
    bars2 = axes[1].bar(methods, bwts, color=colors, edgecolor="black", linewidth=0.8)
    axes[1].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[1].set_title("Backward Transfer (BWT) — Forgetting", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("BWT (%)")
    axes[1].grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars2, bwts):
        va = "bottom" if val >= 0 else "top"
        offset = 0.3 if val >= 0 else -0.3
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                     f"{val:.1f}%", ha="center", va=va, fontsize=10, fontweight="bold")

    plt.suptitle("Method Comparison: Continual Learning on DermaMNIST", fontsize=14,
                 fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_training_history(
    histories: Dict[str, List[float]],
    title: str = "Training Loss",
    ylabel: str = "Loss",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Generic training curve plotter."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, values in histories.items():
        ax.plot(values, label=label, linewidth=2)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
