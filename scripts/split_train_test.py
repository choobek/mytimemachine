#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split a directory of images into train/test and copy to target folders.

Usage example:
  python scripts/split_train_test.py \
    --source_dir data/dataset_prep/060_filtered/extended_emb/valid \
    --train_dir data/train \
    --test_dir data/test \
    --test_ratio 0.10 --seed 42
"""
import argparse
import os
import random
import shutil
from pathlib import Path
from typing import List


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def unique_path(dst_dir: Path, name: str) -> Path:
    base, ext = os.path.splitext(name)
    out = dst_dir / name
    i = 1
    while out.exists():
        out = dst_dir / f"{base}_{i}{ext}"
        i += 1
    return out


def collect_images(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if is_image(p)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Split images into train/test and copy")
    ap.add_argument("--source_dir", type=Path, required=True,
                    help="Directory containing input images (recursively)")
    ap.add_argument("--train_dir", type=Path, required=True,
                    help="Output directory for training images")
    ap.add_argument("--test_dir", type=Path, required=True,
                    help="Output directory for test images")
    ap.add_argument("--test_ratio", type=float, default=0.10,
                    help="Fraction of images to allocate to test set (0-1)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    source_dir: Path = args.source_dir
    train_dir: Path = args.train_dir
    test_dir: Path = args.test_dir
    test_ratio: float = max(0.0, min(1.0, args.test_ratio))
    seed: int = args.seed

    if not source_dir.exists():
        raise SystemExit(f"[err] Source dir does not exist: {source_dir}")

    ensure_dir(train_dir)
    ensure_dir(test_dir)

    images = collect_images(source_dir)
    if not images:
        raise SystemExit("[err] No images found to split.")

    random.seed(seed)
    random.shuffle(images)

    n = len(images)
    num_test = max(1, int(round(n * test_ratio))) if n > 1 else 1
    num_train = max(0, n - num_test)

    train_imgs = images[:num_train]
    test_imgs = images[num_train:]

    for src in train_imgs:
        dst = unique_path(train_dir, src.name)
        shutil.copy2(src, dst)

    for src in test_imgs:
        dst = unique_path(test_dir, src.name)
        shutil.copy2(src, dst)

    print(f"Input images: {n}")
    print(f"Copied to train: {len(train_imgs)} -> {train_dir}")
    print(f"Copied to test:  {len(test_imgs)} -> {test_dir}")


if __name__ == "__main__":
    main()


