"""
Evaluation: metrics, confusion matrix, training curves, and saving results.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    matthews_corrcoef,
)
from sklearn.preprocessing import label_binarize


def compute_all_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    labels: Optional[List[int]] = None,
    average: str = "weighted",
) -> Dict[str, float]:
    """
    Compute accuracy, precision, recall, F1, and MCC.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        labels: Class indices (default [0,1,2,3]).
        average: "weighted", "macro", or "micro" for P/R/F1.

    Returns:
        Dict with keys: accuracy, precision, recall, f1, mcc.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if labels is None:
        labels = list(range(4))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=average, zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mcc": float(mcc),
    }


def plot_confusion_matrix(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    class_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Confusion Matrix",
) -> None:
    """
    Plot and optionally save confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Labels for classes (default positive, negative, neutral, conflict).
        save_path: Path to save figure.
        title: Plot title.
    """
    if class_names is None:
        class_names = ["positive", "negative", "neutral", "conflict"]
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    cm = confusion_matrix(y_true, y_pred)
    # Ensure we have rows/cols for all classes
    n = len(class_names)
    cm_full = np.zeros((n, n), dtype=int)
    for i in range(min(cm.shape[0], n)):
        for j in range(min(cm.shape[1], n)):
            cm_full[i, j] = cm[i, j]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_full,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Count"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_curves(
    train_losses: Optional[List[float]] = None,
    val_losses: Optional[List[float]] = None,
    train_metrics: Optional[Dict[str, List[float]]] = None,
    val_metrics: Optional[Dict[str, List[float]]] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot training/validation loss and optional metrics (e.g. accuracy, F1).

    Args:
        train_losses: Per-epoch training loss.
        val_losses: Per-epoch validation loss.
        train_metrics: Dict of metric name -> list of values.
        val_metrics: Dict of metric name -> list of values.
        save_path: Path to save figure.
    """
    n_plots = 0
    if train_losses is not None or val_losses is not None:
        n_plots += 1
    if train_metrics or val_metrics:
        n_plots += len(train_metrics or val_metrics or {})
    if n_plots == 0:
        return
    n_plots = max(n_plots, 1)
    fig, axes = plt.subplots(n_plots, 1, figsize=(8, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]
    idx = 0
    if train_losses is not None or val_losses is not None:
        ax = axes[idx]
        epochs = range(1, len(train_losses or val_losses or []) + 1)
        if train_losses:
            ax.plot(epochs, train_losses, label="Train loss", marker="o", markersize=3)
        if val_losses:
            ax.plot(epochs, val_losses, label="Val loss", marker="s", markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.set_title("Loss")
        idx += 1
    for name in (train_metrics or val_metrics or {}).keys():
        ax = axes[idx]
        t_vals = (train_metrics or {}).get(name, [])
        v_vals = (val_metrics or {}).get(name, [])
        epochs = range(1, max(len(t_vals), len(v_vals)) + 1)
        if t_vals:
            ax.plot(epochs[: len(t_vals)], t_vals, label=f"Train {name}", marker="o", markersize=3)
        if v_vals:
            ax.plot(epochs[: len(v_vals)], v_vals, label=f"Val {name}", marker="s", markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.legend()
        ax.set_title(name)
        idx += 1
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_metrics_table(
    results_list: List[Dict[str, Any]],
    save_path: Union[str, Path],
    append: bool = True,
) -> None:
    """
    Save or append metrics to a CSV table (e.g. results/metrics/all_results.csv).

    Each dict in results_list should have keys like model_name, accuracy, f1, etc.

    Args:
        results_list: List of dicts (e.g. [{"model": "baseline", "accuracy": 0.8, ...}]).
        save_path: CSV path.
        append: If True and file exists, append rows; else overwrite.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results_list)
    if append and save_path.exists():
        existing = pd.read_csv(save_path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(save_path, index=False)
