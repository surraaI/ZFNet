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
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from models import LeNet5, ZFNet


def set_seed(seed: int) -> None:
    """Make runs reproducible for reports and debugging."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def build_dataloaders(
    data_dir: str, batch_size: int, num_workers: int
) -> Tuple[DataLoader, DataLoader]:
    # CIFAR-10 ships at 32×32; light augmentation improves generalization slightly.
    train_tf = T.Compose(
        [
            T.RandomHorizontalFlip(),
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
    return train_loader, test_loader


def select_model(name: str) -> nn.Module:
    name_l = name.lower()
    if name_l == "lenet":
        return LeNet5(num_classes=10, in_channels=3)
    if name_l == "zfnet":
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
) -> Tuple[float, float]:
    """Returns (average loss, train accuracy)."""
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy(logits.detach(), targets)
        n_batches += 1

    return running_loss / max(n_batches, 1), running_acc / max(n_batches, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LeNet-5 or ZFNet on CIFAR-10.")
    p.add_argument("--model", type=str, default="zfnet", choices=["lenet", "zfnet"])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    train_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = select_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_path = os.path.join(args.checkpoint_dir, f"{args.model}_cifar10_best.pt")

    best_acc = 0.0
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, device, criterion, optimizer)
        test_acc = evaluate(model, test_loader, device)
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
                    "test_acc": test_acc,
                    "epoch": epoch,
                },
                best_path,
            )

    elapsed = time.perf_counter() - t0
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"done in {elapsed:.1f}s | best_test_acc {best_acc:.4f} | params {params:,} | saved {best_path}")


if __name__ == "__main__":
    main()
