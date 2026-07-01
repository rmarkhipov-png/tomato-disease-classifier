"""Split raw dataset into train/val/test folders."""
from __future__ import annotations
import pathlib, shutil, random
from collections import defaultdict


def split_dataset(
    raw_dir: str | pathlib.Path,
    out_dir: str | pathlib.Path,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> None:
    raw_dir = pathlib.Path(raw_dir)
    out_dir = pathlib.Path(out_dir)
    random.seed(seed)

    for class_dir in sorted(raw_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        images = list(class_dir.glob("*.[jJpP][pPnN][gG]"))
        random.shuffle(images)
        n = len(images)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        splits = {
            "train": images[:n_train],
            "val": images[n_train: n_train + n_val],
            "test": images[n_train + n_val:],
        }
        for split, imgs in splits.items():
            dest = out_dir / split / class_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy2(img, dest / img.name)
    print(f"Dataset split into {out_dir}")


if __name__ == "__main__":
    import sys
    raw = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/splits"
    split_dataset(raw, out)
