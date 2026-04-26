"""Edge-friendly backbone placeholders for MyTorch bridge."""

from dataclasses import dataclass


@dataclass
class BackboneConfig:
    name: str = "shironet_tiny"
    width_mult: float = 1.0
    num_classes: int = 10


def build_backbone(cfg: BackboneConfig) -> dict:
    """Return a serializable placeholder structure until MyTorch wiring is added."""
    return {
        "name": cfg.name,
        "width_mult": cfg.width_mult,
        "num_classes": cfg.num_classes,
    }
