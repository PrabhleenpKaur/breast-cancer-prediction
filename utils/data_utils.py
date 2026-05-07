"""
data_utils.py
-------------
Handles all data loading, augmentation, and preprocessing for
the breast cancer detection pipeline.

Dataset: BreaKHis / BreastMNIST or local directory in format:
    data/
      train/
        benign/   *.png
        malignant/ *.png
      val/  ...
      test/ ...
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split
import shutil
import pathlib

# ── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
SEED       = 42


# ── Augmentation generators ───────────────────────────────────────────────────
def get_data_generators(data_dir: str):
    """
    Returns (train_gen, val_gen, test_gen) Keras generators from a
    directory structured as:
        data_dir/train/benign / malignant
        data_dir/val/  benign / malignant
        data_dir/test/ benign / malignant
    """
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode="nearest",
    )

    val_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

    def _flow(datagen, subset, shuffle):
        return datagen.flow_from_directory(
            os.path.join(data_dir, subset),
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="binary",
            shuffle=shuffle,
            seed=SEED,
        )

    train_gen = _flow(train_datagen,    "train", shuffle=True)
    val_gen   = _flow(val_test_datagen, "val",   shuffle=False)
    test_gen  = _flow(val_test_datagen, "test",  shuffle=False)

    return train_gen, val_gen, test_gen


# ── Helper: split flat directory into train / val / test ─────────────────────
def split_dataset(src_dir: str, dest_dir: str,
                  train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Given src_dir with sub-folders per class, create a train/val/test
    split inside dest_dir.

    src_dir/
        benign/    *.png
        malignant/ *.png
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    classes = [d for d in os.listdir(src_dir)
               if os.path.isdir(os.path.join(src_dir, d))]

    for cls in classes:
        files = list(pathlib.Path(src_dir, cls).glob("*.*"))
        np.random.seed(SEED)
        np.random.shuffle(files)

        n      = len(files)
        n_tr   = int(n * train_ratio)
        n_val  = int(n * val_ratio)

        splits = {
            "train": files[:n_tr],
            "val":   files[n_tr : n_tr + n_val],
            "test":  files[n_tr + n_val:],
        }

        for split, paths in splits.items():
            out = pathlib.Path(dest_dir, split, cls)
            out.mkdir(parents=True, exist_ok=True)
            for p in paths:
                shutil.copy(p, out / p.name)

    print(f"[data_utils] Dataset split complete → {dest_dir}")


# ── Single-image preprocessor (used in inference) ────────────────────────────
def preprocess_image(image_path: str) -> np.ndarray:
    """Load, resize, normalise a single image → (1, 224, 224, 3) array."""
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)
