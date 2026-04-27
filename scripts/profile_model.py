import argparse
import json
import time
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.factory import create_model


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def benchmark_latency(
    model: torch.nn.Module,
    device: torch.device,
    img_size: int,
    batch_size: int,
    warmup: int,
    steps: int,
) -> float:
    model.eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / steps


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile ShiroNet architectures for size and latency.")
    parser.add_argument("--arch", default="mobilenet_v3_small")
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--img-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--out", default="docs/assets/benchmark_kaggle_v1/profile.json")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = create_model(args.arch, num_classes=args.num_classes, pretrained=False).to(device)
    params = count_params(model)
    latency_ms = benchmark_latency(
        model=model,
        device=device,
        img_size=args.img_size,
        batch_size=args.batch_size,
        warmup=args.warmup,
        steps=args.steps,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "arch": args.arch,
        "num_classes": args.num_classes,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "device": str(device),
        "parameters": params,
        "latency_ms_per_batch": latency_ms,
        "latency_ms_per_image": latency_ms / max(args.batch_size, 1),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
