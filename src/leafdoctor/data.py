"""PyTorch datasets & transforms."""
from __future__ import annotations
import pathlib
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_transforms(img_size: int = 224, split: str = "train"):
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def get_loader(
    splits_dir: str | pathlib.Path,
    split: str,
    img_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4,
) -> DataLoader:
    ds = datasets.ImageFolder(
        str(pathlib.Path(splits_dir) / split),
        transform=get_transforms(img_size, split),
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
    )
