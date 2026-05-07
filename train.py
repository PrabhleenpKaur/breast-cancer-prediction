"""
train.py
--------
End-to-end training pipeline for breast cancer detection.

Run:
    python train.py --data_dir data/split --backbone efficientnet --epochs 30
"""

import argparse
import os
import json

import tensorflow as tf

from utils.data_utils  import get_data_generators
from utils.model_utils import build_model, unfreeze_top_layers, model_summary
from utils.eval_utils  import plot_training_curves, full_evaluation


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Breast Cancer Detection – Training")
    p.add_argument("--data_dir",    default="data/split",   help="Root of split dataset")
    p.add_argument("--backbone",    default="efficientnet", choices=["efficientnet","mobilenet","resnet50"])
    p.add_argument("--epochs",      type=int, default=30)
    p.add_argument("--fine_tune_epochs", type=int, default=15)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--fine_tune_lr",type=float, default=1e-5)
    p.add_argument("--dropout",     type=float, default=0.4)
    p.add_argument("--model_dir",   default="model")
    p.add_argument("--output_dir",  default="outputs")
    return p.parse_args()


# ── Callbacks ─────────────────────────────────────────────────────────────────
def get_callbacks(model_dir: str):
    os.makedirs(model_dir, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, "best_model.keras"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(model_dir, "training_log.csv")
        ),
    ]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # GPU config
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"[train] GPU detected: {gpus[0].name}")
    else:
        print("[train] No GPU found – training on CPU")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_gen, val_gen, test_gen = get_data_generators(args.data_dir)
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    classes = train_gen.classes

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(classes),
        y=classes
    )

    class_weights = dict(enumerate(weights))

    print("Class weights:", class_weights)
    print(f"\nClass indices : {train_gen.class_indices}")
    print(f"Train samples : {train_gen.samples}")
    print(f"Val   samples : {val_gen.samples}")
    print(f"Test  samples : {test_gen.samples}\n")

    # ── Phase 1 : Head only ───────────────────────────────────────────────────
    print("=" * 55)
    print("PHASE 1 — Training classification head (base frozen)")
    print("=" * 55)
    model = build_model(
        backbone=args.backbone,
        dropout_rate=args.dropout,
        learning_rate=args.lr,
    )
    model_summary(model)

    history1 = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=get_callbacks(args.model_dir),
        class_weight=class_weights
    )

    # ── Phase 2 : Fine-tuning ─────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("PHASE 2 — Fine-tuning top layers")
    print("=" * 55)
    model = unfreeze_top_layers(model, n_layers=20, fine_tune_lr=args.fine_tune_lr)

    history2 = model.fit(
        train_gen,
        epochs=args.fine_tune_epochs,
        validation_data=val_gen,
        callbacks=get_callbacks(args.model_dir),
        class_weight= class_weights,
        
    )

    # ── Merge histories ───────────────────────────────────────────────────────
    combined = _merge_histories(history1, history2)
    plot_training_curves(combined, save_dir=args.output_dir)

    # ── Evaluation ────────────────────────────────────────────────────────────
    best_model = tf.keras.models.load_model(
        os.path.join(args.model_dir, "best_model.keras")
    )
    results = full_evaluation(best_model, test_gen, save_dir=args.output_dir)

    # Save metrics
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[train] All outputs saved to ./{args.output_dir}/")
    print("[train] Training complete ✓")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _merge_histories(h1, h2):
    """Merge two Keras History objects into a single object."""
    class MergedHistory:
        pass
    m = MergedHistory()
    m.history = {}
    for k in h1.history:
        m.history[k] = h1.history[k] + h2.history.get(k, [])
    return m


if __name__ == "__main__":
    main()
