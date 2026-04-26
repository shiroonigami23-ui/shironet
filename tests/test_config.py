import os

from src.config import load_config
from src.models.backbone import BackboneConfig, build_backbone
from src.training.adversarial import AdversarialHookConfig, build_adv_hook


def test_load_config_defaults():
    os.environ.pop("NEON_BRANCH", None)
    cfg = load_config()
    assert cfg.neon.branch == "main"
    assert cfg.ghost.mode in {"strict", "balanced", "rapid"}


def test_build_backbone_placeholder():
    model = build_backbone(BackboneConfig(num_classes=5))
    assert model["num_classes"] == 5


def test_adv_hook_placeholder():
    hook = build_adv_hook(AdversarialHookConfig(steps=7))
    assert hook["steps"] == 7
