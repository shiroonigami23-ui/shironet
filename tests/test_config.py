import os
from src.config import load_config


def test_load_config_defaults():
    os.environ.pop("NEON_BRANCH", None)
    cfg = load_config()
    assert cfg.neon.branch == "main"
    assert cfg.ghost.mode in {"strict", "balanced", "rapid"}
