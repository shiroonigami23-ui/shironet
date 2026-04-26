"""Data processing and distillation logging hooks."""

from pathlib import Path
from typing import Iterable


def ensure_data_dirs(base_dir: str = "data") -> None:
    root = Path(base_dir)
    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / "distill_logs").mkdir(parents=True, exist_ok=True)


def normalize_labels(labels: Iterable[int]) -> list[int]:
    """Small helper for consistent label typing."""
    return [int(x) for x in labels]
