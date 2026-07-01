"""Model factory."""
from __future__ import annotations
import torch
import torch.nn as nn
from torchvision import models


def build_model(model_name: str = "resnet18", num_classes: int = 10) -> nn.Module:
    weights_map = {
        "resnet18": models.ResNet18_Weights.DEFAULT,
        "resnet50": models.ResNet50_Weights.DEFAULT,
        "efficientnet_b0": models.EfficientNet_B0_Weights.DEFAULT,
    }
    if model_name == "resnet18":
        m = models.resnet18(weights=weights_map[model_name])
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif model_name == "resnet50":
        m = models.resnet50(weights=weights_map[model_name])
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif model_name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=weights_map[model_name])
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return m


def freeze_backbone(model: nn.Module, model_name: str) -> None:
    for name, param in model.named_parameters():
        if model_name.startswith("resnet"):
            param.requires_grad = name.startswith("fc")
        elif model_name.startswith("efficientnet"):
            param.requires_grad = "classifier" in name


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True
