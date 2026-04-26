import argparse
import os
import subprocess
from pathlib import Path


def has_kaggle_auth() -> bool:
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return True
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists()


def run() -> int:
    parser = argparse.ArgumentParser(
        description="Download and optionally unzip a Kaggle dataset."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Kaggle dataset slug in the form owner/dataset-name",
    )
    parser.add_argument(
        "--out",
        default="data/raw",
        help="Output directory for downloaded data (default: data/raw)",
    )
    parser.add_argument(
        "--no-unzip",
        action="store_true",
        help="Do not unzip after download",
    )
    args = parser.parse_args()

    if not has_kaggle_auth():
        raise SystemExit(
            "Kaggle auth not found. Set KAGGLE_USERNAME/KAGGLE_KEY or place kaggle.json in ~/.kaggle/"
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        args.dataset,
        "-p",
        str(out_dir),
    ]
    if not args.no_unzip:
        cmd.append("--unzip")

    print(f"Running: {' '.join(cmd)}")
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(run())
