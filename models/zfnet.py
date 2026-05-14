"""
ZFNet (Zeiler & Fergus, 2013) — a refined AlexNet-style architecture proposed after
systematically visualizing what intermediate layers learn.

The *ImageNet* configuration uses larger images and specific stride/padding choices.
For CIFAR-10 (32×32), we keep the same **filter-count progression** (96 → 256 → 384 →
384 → 256) and **kernel-size story** (smaller first-layer kernels than AlexNet’s
11×11) while adjusting strides/padding so tensors remain well-defined.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ZFNet(nn.Module):
    """
    ZFNet-inspired CNN for 32×32 inputs (CIFAR-10).

    Compared to AlexNet (Krizhevsky et al., 2012), the key conceptual differences
    Zeiler & Fergus emphasized were:
    - **First layer**: smaller 7×7 (here 5×5 on tiny images) receptive fields with
      less aggressive downsampling than AlexNet’s 11×11 / stride 4 — finer
      preservation of low-level structure.
    - **Deeper stacks of 3×3 convolutions** in the mid network (same *idea* as the
      ImageNet model’s 3×3 stages), trading some parameter locality for compositional
      power.

    We use ReLU (like AlexNet/ZFNet practice) and local response normalization is
    omitted for simplicity (it is rarely used in modern PyTorch baselines).
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3, dropout: float = 0.5) -> None:
        super().__init__()

        # Layer 1: fewer spatial jumps than AlexNet’s first conv — more detail retained.
        self.conv1 = nn.Conv2d(in_channels, 96, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layer 2: wider (256) mid-level bank of filters.
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layers 3–5: repeated 3×3 conv blocks (384 / 384 / 256) as in the paper’s trunk.
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(384)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(384)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Adaptive pooling makes the classifier robust to small shape differences.
        self.head = nn.Sequential(
          nn.AdaptiveAvgPool2d((1, 1)),
          nn.Flatten(),
          nn.Dropout(p=dropout),
          nn.Linear(256, num_classes),
        )

        self.act = nn.ReLU(inplace=True)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
          nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
          if module.bias is not None:
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
          nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
          if module.bias is not None:
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
          nn.init.ones_(module.weight)
          nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.act(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        x = self.act(self.bn5(self.conv5(x)))
        x = self.pool3(x)
        return self.head(x)
