import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError


def _resolve_split(root: Path, names: list[str]) -> Path:
    def looks_like_class_root(path: Path) -> bool:
        if not path.is_dir():
            return False
        class_dirs = [d for d in path.iterdir() if d.is_dir()]
        if not class_dirs:
            return False
        return any(any(p.is_file() for p in d.iterdir()) for d in class_dirs)

    for name in names:
        p = root / name
        if looks_like_class_root(p):
            return p
        nested = p / name
        if looks_like_class_root(nested):
            return nested
    for child in root.iterdir():
        if not child.is_dir():
            continue
        for name in names:
            p = child / name
            if looks_like_class_root(p):
                return p
            nested = p / name
            if looks_like_class_root(nested):
                return nested
    raise FileNotFoundError(f"Could not find any split from {names} under {root}")


def _is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def _copy_valid_images(src_class_dir: Path, dst_class_dir: Path) -> tuple[int, int]:
    dst_class_dir.mkdir(parents=True, exist_ok=True)
    kept = 0
    dropped = 0
    for img_path in src_class_dir.rglob("*"):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        if _is_valid_image(img_path):
            shutil.copy2(img_path, dst_class_dir / img_path.name)
            kept += 1
        else:
            dropped += 1
    return kept, dropped


def _split_train_val(
    train_root: Path, val_root: Path, val_ratio: float, seed: int
) -> tuple[dict[str, int], dict[str, int]]:
    rng = random.Random(seed)
    train_counts: dict[str, int] = {}
    val_counts: dict[str, int] = {}

    for class_dir in sorted(train_root.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        files = [p for p in class_dir.iterdir() if p.is_file()]
        rng.shuffle(files)
        n_val = int(len(files) * val_ratio)
        val_files = files[:n_val]

        dst_val_class = val_root / class_name
        dst_val_class.mkdir(parents=True, exist_ok=True)
        for fp in val_files:
            shutil.move(str(fp), str(dst_val_class / fp.name))

        train_counts[class_name] = len(files) - n_val
        val_counts[class_name] = n_val

    return train_counts, val_counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean and prepare a Kaggle image dataset for training.")
    parser.add_argument("--input-root", required=True, help="Raw dataset root")
    parser.add_argument("--output-root", default="data/processed/intel_scenes", help="Processed dataset root")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio from train")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    args = parser.parse_args()

    in_root = Path(args.input_root)
    out_root = Path(args.output_root)

    train_src = _resolve_split(in_root, ["train", "seg_train", "training"])
    test_src = _resolve_split(in_root, ["test", "seg_test"])

    if out_root.exists():
        shutil.rmtree(out_root)
    train_dst = out_root / "train"
    val_dst = out_root / "val"
    test_dst = out_root / "test"
    train_dst.mkdir(parents=True, exist_ok=True)
    val_dst.mkdir(parents=True, exist_ok=True)
    test_dst.mkdir(parents=True, exist_ok=True)

    dropped_total = 0
    raw_train_counts: dict[str, int] = {}
    raw_test_counts: dict[str, int] = {}

    for class_dir in sorted(train_src.iterdir()):
        if not class_dir.is_dir():
            continue
        kept, dropped = _copy_valid_images(class_dir, train_dst / class_dir.name)
        raw_train_counts[class_dir.name] = kept
        dropped_total += dropped

    for class_dir in sorted(test_src.iterdir()):
        if not class_dir.is_dir():
            continue
        kept, dropped = _copy_valid_images(class_dir, test_dst / class_dir.name)
        raw_test_counts[class_dir.name] = kept
        dropped_total += dropped

    train_counts, val_counts = _split_train_val(train_dst, val_dst, args.val_ratio, args.seed)
    final_test_counts = {
        class_dir.name: len([p for p in class_dir.iterdir() if p.is_file()])
        for class_dir in sorted(test_dst.iterdir())
        if class_dir.is_dir()
    }

    report = {
        "input_root": str(in_root),
        "output_root": str(out_root),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "raw_train_counts": raw_train_counts,
        "raw_test_counts": raw_test_counts,
        "final_train_counts": train_counts,
        "final_val_counts": val_counts,
        "final_test_counts": final_test_counts,
        "dropped_corrupt_images": dropped_total,
    }
    report_path = out_root / "prepare_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Prepared dataset written to: {out_root}")
    print(f"Dropped corrupt images: {dropped_total}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
