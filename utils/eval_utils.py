"""
eval_utils.py
-------------
Model evaluation: metrics, confusion matrix, classification report,
training-curve plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
from sklearn.metrics import roc_auc_score

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

CLASSES = ["Benign", "Malignant"]


# ── Training curves ───────────────────────────────────────────────────────────
def plot_training_curves(history, save_dir: str = "outputs") -> None:
    """Plot accuracy, loss, AUC side-by-side and save to disk."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Training History", fontsize=16, fontweight="bold")

    metrics = [
        ("accuracy",  "Accuracy"),
        ("loss",      "Loss"),
        ("auc",       "AUC-ROC"),
    ]

    for ax, (key, title) in zip(axes, metrics):
        ax.plot(history.history[key],        label="Train", linewidth=2)
        ax.plot(history.history[f"val_{key}"], label="Val",  linewidth=2, linestyle="--")
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_dir, "training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[eval_utils] Saved → {out}")


# ── Confusion matrix ─────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, save_dir: str = "outputs") -> None:
    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
        linewidths=0.5, linecolor="gray",
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[eval_utils] Saved → {out}")


# ── ROC curve ────────────────────────────────────────────────────────────────
def plot_roc_curve(y_true, y_prob, save_dir: str = "outputs") -> None:
    os.makedirs(save_dir, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2,
            label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_dir, "roc_curve.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[eval_utils] Saved → {out}")


# ── Full evaluation report ────────────────────────────────────────────────────
def full_evaluation(model, test_gen,
                    threshold: float = 0.3,
                    save_dir: str = "outputs") -> dict:
    """
    Run complete evaluation on test_gen.
    Returns dict with accuracy, precision, recall, f1, auc.
    """
    print("[eval_utils] Running inference on test set …")
    y_prob = model.predict(test_gen, verbose=1).ravel()
    y_pred = (y_prob >= threshold).astype(int)
    y_true = test_gen.labels

    print("\n" + "=" * 55)
    print("CLASSIFICATION REPORT")
    print("=" * 55)
    print(classification_report(y_true, y_pred, target_names=CLASSES))
    report = classification_report(y_true, y_pred, target_names=CLASSES)

    with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    plot_confusion_matrix(y_true, y_pred, save_dir)
    plot_roc_curve(y_true, y_prob, save_dir)

    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    results = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred,zero_division=0),
        "recall":    recall_score(y_true, y_pred,zero_division=0),
        "f1":        f1_score(y_true, y_pred,zero_division=0),
        "auc":       roc_auc_score(y_true, y_prob),
    }
    print("\nSummary:", results)
    return results
