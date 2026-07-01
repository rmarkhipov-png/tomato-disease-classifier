"""Training loop."""
from __future__ import annotations
import pathlib, time, json
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from .config import TrainConfig, load_config
from .data import get_loader
from .model import build_model, freeze_backbone, unfreeze_all
from .labels import NUM_CLASSES


def train(cfg: TrainConfig | None = None) -> None:
    if cfg is None:
        cfg = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader = get_loader(cfg.splits_dir, "train", cfg.img_size, cfg.batch_size, cfg.num_workers)
    val_loader   = get_loader(cfg.splits_dir, "val",   cfg.img_size, cfg.batch_size, cfg.num_workers)

    model = build_model(cfg.model_name, NUM_CLASSES).to(device)
    freeze_backbone(model, cfg.model_name)

    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=cfg.amp)

    def make_optimizer(params):
        if cfg.optimizer == "adamw":
            return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        return torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    optimizer = make_optimizer(filter(lambda p: p.requires_grad, model.parameters()))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    out_dir = pathlib.Path(cfg.out_dir) / cfg.project_name / cfg.version
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_acc = 0.0

    for epoch in range(1, cfg.epochs + 1):
        if epoch == cfg.freeze_backbone_epochs + 1:
            unfreeze_all(model)
            optimizer = make_optimizer(model.parameters())
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.epochs - epoch + 1)
            print("Backbone unfrozen")

        model.train()
        total_loss, correct, total = 0.0, 0, 0
        t0 = time.time()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast(enabled=cfg.amp):
                out = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * imgs.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += imgs.size(0)
        scheduler.step()

        train_acc = correct / total
        val_acc, val_loss = _evaluate(model, val_loader, criterion, device, cfg.amp)
        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{cfg.epochs}  "
              f"loss={total_loss/total:.4f}  acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"time={elapsed:.1f}s")
        history.append({"epoch": epoch, "train_acc": train_acc, "val_acc": val_acc})

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), out_dir / "best.pt")

    torch.save(model.state_dict(), out_dir / "last.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Best val acc: {best_acc:.4f}  →  {out_dir}")


def _evaluate(model, loader, criterion, device, amp):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with autocast(enabled=amp):
                out = model(imgs)
                loss = criterion(out, labels)
            total_loss += loss.item() * imgs.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += imgs.size(0)
    return correct / total, total_loss / total


if __name__ == "__main__":
    train()
