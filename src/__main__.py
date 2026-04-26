"""ShiroNet entrypoint placeholder."""

from src.config import load_config


if __name__ == "__main__":
    cfg = load_config()
    print("ShiroNet initialized")
    print(f"GHOST-REF-ZERO mode: {cfg.ghost.mode}")
