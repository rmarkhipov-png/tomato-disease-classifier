"""Evaluation & metrics."""
from __future__ import annotations
import json, pathlib
import torch
from torch.cuda.amp import autocast
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from .config import TrainConfig, load_config
from .data import get_loader
from .model import build_model
from .labels import NUM_CLASSES, LABELS


def evaluate(cfg: TrainConfig | None = None, split: str = "test") -> None:
    if cfg is None:
        cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = get_loader(cfg.splits_dir, split, cfg.img_size, cfg.batch_size, cfg.num_workers)
    model = build_model(cfg.model_name, NUM_CLASSES).to(device)
    ckpt = pathlib.Path(cfg.out_dir) / cfg.project_name / cfg.version / "best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            with autocast(enabled=cfg.amp):
                out = model(imgs)
            all_preds.extend(out.argmax(1).cpu().tolist())
            all_labels.extend(labels.tolist())

    print(classification_report(all_labels, all_preds, target_names=LABELS))


if __name__ == "__main__":
    evaluate()
