from __future__ import annotations
import tomllib
import pathlib
from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    data_dir: str = "data/raw"
    splits_dir: str = "data/splits"
    out_dir: str = "artifacts"
    project_name: str = "plant-leaf-tomato"
    version: str = "v0.1.0"
    model_name: str = "resnet18"
    img_size: int = 224
    epochs: int = 8
    freeze_backbone_epochs: int = 2
    batch_size: int = 64
    lr: float = 5e-4
    head_lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    amp: bool = True
    class_weights: bool = False
    seed: int = 42
    num_workers: int = 4


def load_config(path: str | pathlib.Path = "configs/train.toml") -> TrainConfig:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return TrainConfig(**{k: v for k, v in data.items() if k in TrainConfig.__dataclass_fields__})
