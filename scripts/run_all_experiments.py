"""
run_all_experiments.py — Reproduce all main results for Topic 5.

Usage:
    python scripts/run_all_experiments.py [--epochs 10] [--batch_size 64] [--device cpu] [--data_root ./data]
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cnn import build_model
from src.methods.naive import NaiveTrainer
from src.methods.ewc import EWCTrainer
from src.methods.lwf import LwFTrainer
from src.methods.replay import ReplayTrainer
from src.methods.joint import JointTrainer
from src.utils.data import (
    get_all_tasks_dataloaders,
    get_full_dataset_loader,
    TASK_CLASSES,
    CLASS_NAMES,
)
from src.utils.metrics import evaluate, compute_cl_metrics, print_metrics_table

NUM_CLASSES = 7


def run_cl_experiment(trainer_cls, trainer_kwargs, all_train_loaders,
                      all_test_loaders, epochs, device):
    """Train a CL method sequentially on all tasks. Returns accuracy matrix."""
    model = build_model(num_classes=NUM_CLASSES, device=device)
    trainer = trainer_cls(model=model, device=device, **trainer_kwargs)
    T = len(TASK_CLASSES)
    acc_matrix = np.zeros((T, T))
    for task_id in range(T):
        print(f"\n{'='*50}")
        print(f"Task {task_id} | Classes: {[CLASS_NAMES[c] for c in TASK_CLASSES[task_id]]}")
        trainer.fit(task_id=task_id, train_loader=all_train_loaders[task_id],
                    num_epochs=epochs)
        for eval_task in range(T):
            acc = evaluate(model, all_test_loaders[eval_task], device=device)
            acc_matrix[task_id, eval_task] = acc
            if eval_task <= task_id:
                print(f"  Acc on Task {eval_task}: {acc*100:.1f}%")
    return acc_matrix


def run_joint_experiment(all_test_loaders, epochs, device, data_root):
    """Joint training on all 7 classes at once — oracle upper bound."""
    print(f"\n{'='*50}\nJoint Training (Upper Bound)\n{'='*50}")
    model = build_model(num_classes=NUM_CLASSES, device=device)
    trainer = JointTrainer(model=model, device=device)
    full_train_loader, _ = get_full_dataset_loader(
        batch_size=64, data_root=data_root
    )
    trainer.fit(task_id=0, train_loader=full_train_loader, num_epochs=epochs)
    T = len(TASK_CLASSES)
    acc_matrix = np.zeros((T, T))
    for eval_task in range(T):
        acc = evaluate(model, all_test_loaders[eval_task], device=device)
        acc_matrix[:, eval_task] = acc
        print(f"  Acc on Task {eval_task}: {acc*100:.1f}%")
    return acc_matrix


def plot_heatmaps(all_matrices, save_path):
    n = len(all_matrices)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, mat) in zip(axes, all_matrices.items()):
        sns.heatmap(
            mat * 100, ax=ax, annot=True, fmt=".1f",
            cmap="YlOrRd_r", vmin=0, vmax=100,
            xticklabels=[f"T{i}" for i in range(mat.shape[1])],
            yticklabels=[f"After T{i}" for i in range(mat.shape[0])],
        )
        ax.set_title(name, fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_forgetting(all_matrices, task_id, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
    for (name, mat), color in zip(all_matrices.items(), colors):
        T = mat.shape[0]
        ax.plot(range(T), [mat[i, task_id] * 100 for i in range(T)],
                marker="o", label=name, color=color, linewidth=2)
    ax.set_xlabel("Sequential Training Step")
    ax.set_ylabel(f"Acc on Task {task_id} (%)")
    ax.set_title(f"Catastrophic Forgetting on Task {task_id}")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_bar(results, save_path):
    methods = list(results.keys())
    faa = [results[m]["faa"] * 100 for m in methods]
    bwt = [results[m]["bwt"] * 100 for m in methods]
    x = np.arange(len(methods)); w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, faa, w, label="FAA (%)", color="#3498db")
    ax.bar(x + w / 2, bwt, w, label="BWT (%)", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("FAA & BWT by Method"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--device",     type=str,   default="cpu")
    parser.add_argument("--data_root",  type=str,   default="./data")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU"); device = "cpu"

    os.makedirs("results/plots", exist_ok=True)

    print(f"Device: {device} | Epochs/task: {args.epochs} | Batch: {args.batch_size}")

    # Load per-task dataloaders
    all_loaders = get_all_tasks_dataloaders(
        batch_size=args.batch_size, data_root=args.data_root
    )
    all_train_loaders = [tl for tl, _ in all_loaders]
    all_test_loaders  = [vl for _, vl in all_loaders]

    # Define experiments
    experiments = [
        ("Naive Fine-tuning",  NaiveTrainer,  {"lr": 1e-3, "weight_decay": 1e-4}),
        ("EWC",                EWCTrainer,    {"lr": 1e-3, "weight_decay": 1e-4, "ewc_lambda": 5000.0}),
        ("LwF",                LwFTrainer,    {"lr": 1e-3, "weight_decay": 1e-4, "alpha": 1.0, "temperature": 2.0}),
        ("Experience Replay",  ReplayTrainer, {"lr": 1e-3, "weight_decay": 1e-4, "memory_size": 500, "replay_batch_size": 32}),
    ]

    all_matrices = {}
    all_results  = {}

    # Run CL experiments
    for name, cls, kwargs in experiments:
        print(f"\n{'#'*60}\n# {name}\n{'#'*60}")
        mat = run_cl_experiment(
            cls, kwargs, all_train_loaders, all_test_loaders, args.epochs, device
        )
        all_matrices[name] = mat
        all_results[name]  = compute_cl_metrics(mat)
        np.save(f"results/accuracy_matrix_{name.replace(' ', '_')}.npy", mat)

    # Run Joint Training (upper bound)
    joint_mat = run_joint_experiment(
        all_test_loaders, args.epochs, device, args.data_root
    )
    all_matrices["Joint Training"] = joint_mat
    all_results["Joint Training"]  = compute_cl_metrics(joint_mat)
    np.save("results/accuracy_matrix_Joint_Training.npy", joint_mat)

    # Print summary table
    print_metrics_table(all_results)

    # Save CSV
    rows = [
        {"Method": m, "FAA": v["faa"], "BWT": v["bwt"],
         **{f"Task{t}_Acc": a for t, a in enumerate(v["per_task_final"])}}
        for m, v in all_results.items()
    ]
    pd.DataFrame(rows).to_csv("results/metrics_summary.csv", index=False)
    print("Saved: results/metrics_summary.csv")

    # Plots
    plot_heatmaps(all_matrices, "results/plots/accuracy_matrix_heatmaps.png")
    plot_forgetting(all_matrices, 0, "results/plots/forgetting_task0.png")
    plot_bar(all_results, "results/plots/faa_bwt_summary.png")

    print("\nAll experiments complete. Results saved in results/")


if __name__ == "__main__":
    main()