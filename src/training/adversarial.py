"""Adversarial training hook placeholders."""

from dataclasses import dataclass


@dataclass
class AdversarialHookConfig:
    epsilon: float = 4 / 255
    alpha: float = 1 / 255
    steps: int = 4


def build_adv_hook(cfg: AdversarialHookConfig) -> dict:
    return {
        "epsilon": cfg.epsilon,
        "alpha": cfg.alpha,
        "steps": cfg.steps,
    }
