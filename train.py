#!/usr/bin/env python3
"""
Train and evaluate LeNet-5 or ZFNet on CIFAR-10.

CIFAR-10 is a standard 32×32×10-class benchmark — realistic enough for coursework
while remaining trainable on CPU in reasonable time with modest epochs.
"""

from __future__ import annotations

import argparse
import os
import random
import time
import warnings
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

# TensorBoard import is optional; handle missing package gracefully so the
# script still runs when `tensorboard` isn't installed in the environment.
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional runtime dependency
    SummaryWriter = None

from models import LeNet5, ZFNet


warnings.filterwarnings(
    "ignore",
    message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
    category=Warning,
    module=r"torchvision\.datasets\.cifar",
)


def set_seed(seed: int) -> None:
    """Make runs reproducible for reports and debugging."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def build_dataloaders(
    data_dir: str, batch_size: int, num_workers: int, model: str = "zfnet"
) -> Tuple[DataLoader, DataLoader]:
    # Use MNIST for LeNet (28×28 grayscale) and CIFAR-10 for ZFNet (32×32 RGB)
    if model.lower() == "lenet":
        print(f"[data] loading MNIST from {data_dir} (download=True)")
        # MNIST: 28×28 grayscale images, resized to 32×32 to match LeNet's expected input
        train_tf = T.Compose(
            [
                T.Resize(32),
                T.RandomRotation(10),
                T.ToTensor(),
                T.Normalize(mean=(0.1307,), std=(0.3081,)),
            ]
        )
        test_tf = T.Compose(
            [
                T.Resize(32),
                T.ToTensor(),
                T.Normalize(mean=(0.1307,), std=(0.3081,)),
            ]
        )
        train_set = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=True, transform=train_tf
        )
        test_set = torchvision.datasets.MNIST(
            root=data_dir, train=False, download=True, transform=test_tf
        )
        print(f"[data] MNIST ready: train={len(train_set)} test={len(test_set)}")
    else:
        print(f"[data] loading CIFAR-10 from {data_dir} (download=True)")
        # CIFAR-10: 32×32 RGB images; light augmentation improves generalization slightly.
        train_tf = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.ToTensor(),
                T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
            ]
        )
        test_tf = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
            ]
        )
        train_set = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=train_tf
        )
        test_set = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=test_tf
        )
        print(f"[data] CIFAR-10 ready: train={len(train_set)} test={len(test_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"[data] dataloaders ready: train_batches={len(train_loader)} test_batches={len(test_loader)}")
    return train_loader, test_loader


def select_model(name: str) -> nn.Module:
    name_l = name.lower()
    if name_l == "lenet":
        # LeNet for MNIST: 1 input channel (grayscale)
        return LeNet5(num_classes=10, in_channels=1)
    if name_l == "zfnet":
        # ZFNet for CIFAR-10: 3 input channels (RGB)
        return ZFNet(num_classes=10, in_channels=3)
    raise ValueError(f"Unknown model '{name}'. Use 'lenet' or 'zfnet'.")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total += targets.numel()
    return total_correct / max(total, 1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int = 0,
    writer: SummaryWriter | None = None,
    log_interval: int = 100,
) -> Tuple[float, float]:
    """Returns (average loss, train accuracy)."""
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0
    for batch_idx, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batch_acc = accuracy(logits.detach(), targets)
        running_acc += batch_acc
        n_batches += 1

        if writer is not None and (batch_idx % log_interval == 0):
            global_step = epoch * len(loader) + batch_idx
            writer.add_scalar("train/batch_loss", loss.item(), global_step)
            writer.add_scalar("train/batch_acc", batch_acc, global_step)

    return running_loss / max(n_batches, 1), running_acc / max(n_batches, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LeNet-5 or ZFNet on CIFAR-10.")
    p.add_argument("--model", type=str, default="zfnet", choices=["lenet", "zfnet"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"])
    p.add_argument("--log-interval", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    dataset_name = "mnist" if args.model.lower() == "lenet" else "cifar10"
    print(f"starting {args.model} on {dataset_name} using {device}")

    train_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, model=args.model
    )

    model = select_model(args.model).to(device)
    print(f"[model] initialized {model.__class__.__name__} with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable params")
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # simple LR scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # TensorBoard writer (logs saved under checkpoint dir)
    try:
        writer = SummaryWriter(log_dir=os.path.join(args.checkpoint_dir, "runs"))
    except Exception:
        writer = None

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_path = os.path.join(args.checkpoint_dir, f"{args.model}_{dataset_name}_best.pt")
    print(f"[checkpoint] best model will be saved to {best_path}")

    best_acc = 0.0
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        print(f"[train] epoch {epoch}/{args.epochs} starting")
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            device,
            criterion,
            optimizer,
            epoch=epoch,
            writer=writer,
            log_interval=args.log_interval,
        )
        test_acc = evaluate(model, test_loader, device)
        # scheduler step after validation
        try:
            scheduler.step()
        except Exception:
            pass

        if writer is not None:
            writer.add_scalar("train/epoch_loss", train_loss, epoch)
            writer.add_scalar("train/epoch_acc", train_acc, epoch)
            writer.add_scalar("test/epoch_acc", test_acc, epoch)
        print(
            f"epoch {epoch:03d} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} "
            f"| test_acc {test_acc:.4f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "model": args.model,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": getattr(scheduler, "state_dict", lambda: None)(),
                    "test_acc": test_acc,
                    "epoch": epoch,
                },
                best_path,
            )

    elapsed = time.perf_counter() - t0
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"done in {elapsed:.1f}s | best_test_acc {best_acc:.4f} | params {params:,} | saved {best_path}")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
