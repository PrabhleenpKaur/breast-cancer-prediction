"""
prepare_data.py
---------------
Downloads BreastMNIST (medmnist) and organises it into the
train/val/test folder structure expected by train.py.

Install dependency:  pip install medmnist

Run:
    python prepare_data.py --output_dir data/split
"""

import argparse
import os
import numpy as np
from PIL import Image
import pathlib
import logging

LABEL_MAP = {0: "benign", 1: "malignant"}


def save_dataset_split(images, labels, split: str, output_dir: str):
    """Save numpy arrays as individual PNG files in class sub-folders."""
    saved = 0
    for idx, (img_arr, lbl) in enumerate(zip(images, labels)):
        cls  = LABEL_MAP[int(lbl[0])]
        dest = pathlib.Path(output_dir, split, cls)
        dest.mkdir(parents=True, exist_ok=True)
        filepath = dest / f"{split}_{idx:05d}.png"
        img = Image.fromarray(img_arr.squeeze())  # (28,28) or (28,28,3)
        if img.mode != "RGB":
            img = img.convert("RGB")
        if len(images) != len(labels):
            raise ValueError("Mismatch between images and labels")
        img.save(filepath)
        saved += 1
    print(f"[prepare_data] {split:5s} → {saved:5d} images saved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/split")
    args = parser.parse_args()

    try:
        import medmnist
        from medmnist import BreastMNIST
    except ImportError:
        print("ERROR: medmnist not installed.\n  pip install medmnist")
        return
    
    if pathlib.Path(args.output_dir).exists():
        print(f"[WARNING] {args.output_dir} already exists. Files may be overwritten.")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    logging.info("Downloading BreastMNIST...")
    #print("[prepare_data] Downloading BreastMNIST …")
    train_ds = BreastMNIST(split="train", download=True, size=224)
    val_ds   = BreastMNIST(split="val",   download=True, size=224)
    test_ds  = BreastMNIST(split="test",  download=True, size=224)

    for ds, split in [(train_ds, "train"), (val_ds, "val"), (test_ds, "test")]:
        imgs   = ds.imgs
        labels = ds.labels
        save_dataset_split(imgs, labels, split, args.output_dir)

    print(f"\n[prepare_data] Dataset ready at: {args.output_dir}/")
    print("Structure:")
    for split in ["train", "val", "test"]:
        for cls in ["benign", "malignant"]:
            p = pathlib.Path(args.output_dir, split, cls)
            n = len(list(p.glob("*.png"))) if p.exists() else 0
            print(f"  {split}/{cls}: {n} images")


if __name__ == "__main__":
    main()
