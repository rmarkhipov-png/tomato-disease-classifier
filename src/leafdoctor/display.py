"""Visualization helpers."""
from __future__ import annotations
import pathlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from .labels import ID2LABEL


def show_prediction(image_path: str | pathlib.Path, result: dict, save_to: str | None = None) -> None:
    img = Image.open(image_path).convert("RGB")
    label = result["label"]
    conf = result["confidence"]
    color = "green" if "healthy" in label else "red"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img)
    axes[0].set_title(f"{label}\nConfidence: {conf:.1%}", color=color, fontsize=13)
    axes[0].axis("off")

    probs = result["probabilities"]
    labels_sorted = sorted(probs, key=probs.get, reverse=True)[:5]
    values = [probs[l] for l in labels_sorted]
    short_labels = [l.replace("Tomato___", "") for l in labels_sorted]
    bars = axes[1].barh(short_labels[::-1], values[::-1], color=color)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Probability")
    axes[1].set_title("Top-5 predictions")
    plt.tight_layout()

    if save_to:
        plt.savefig(save_to, dpi=150)
    plt.show()
