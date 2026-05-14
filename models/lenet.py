"""
LeNet-5 (LeCun et al., 1998) — a classic convolutional network for digit recognition.

The original paper used 32×32 grayscale MNIST-like inputs. Here we adapt the same
*depth pattern* (conv → pool → conv → pool → conv → FC → FC → logits) for RGB
CIFAR-10 (32×32×3) by changing the input channel count and keeping filter sizes
faithful to the original design intent.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LeNet5(nn.Module):
    """
    LeNet-5-style CNN for 32×32 images.

    Architectural intent (mapped to this implementation):
    - **C1**: convolutional feature extraction with small spatial kernels (5×5).
    - **S2**: spatial downsampling (max pooling replaces the original average /
      subsampling for simpler, stable training in modern frameworks).
    - **C3 / S4**: deeper conv + pool stack (original LeNet-5 used a sparse map
      connection table from C3; we use dense convolutions, which is the common
      modern simplification and still illustrates the same hierarchical idea).
    - **C5 / F6 / output**: high-level representation and classification layers.
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3) -> None:
        super().__init__()

        # C1: learn local edge/texture detectors over the full input field.
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=0)
        # S2: reduce spatial resolution; improves translation tolerance and cost.
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # C3: combine low-level features into richer patterns.
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # After two 5×5 convs and 2×2 pools on 32×32 RGB:
        # 32→28→14→10→5 feature map spatial size is 5×5 with these settings.
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)

        # Fully connected stages: global reasoning over the entire field.
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)

        # Use ReLU for better CIFAR performance (optionally keep tanh for historical fidelity)
        self.act = nn.ReLU(inplace=True)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1: first conv nonlinearity + pooling.
        x = self.pool1(self.act(self.conv1(x)))
        # Block 2: second conv nonlinearity + pooling.
        x = self.pool2(self.act(self.conv2(x)))
        # Block 3: conv that collapses remaining spatial dimensions to a vector.
        x = self.act(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        # Classifier head.
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x
