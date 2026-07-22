"""CNN classifier factory for the RGB baseline (and later the residual CNN).

Outputs a single logit; sigmoid(logit) = P(AI-generated).
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_model(arch: str = "resnet18", pretrained: bool = True, in_channels: int = 3) -> nn.Module:
    if arch != "resnet18":
        raise ValueError(f"Unsupported arch: {arch!r} (only resnet18 for now)")

    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    if in_channels != 3:
        model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model
