from __future__ import annotations

import torch.nn as nn
from torchvision import models


SUPPORTED_ARCHS = ("resnet18", "mobilenet_v3_small", "shufflenet_v2_x0_5")


def create_model(arch: str, num_classes: int, pretrained: bool) -> nn.Module:
    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if arch == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if arch == "shufflenet_v2_x0_5":
        weights = models.ShuffleNet_V2_X0_5_Weights.DEFAULT if pretrained else None
        model = models.shufflenet_v2_x0_5(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    raise ValueError(f"Unsupported arch '{arch}'. Supported: {', '.join(SUPPORTED_ARCHS)}")
