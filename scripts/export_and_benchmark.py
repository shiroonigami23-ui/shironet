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


def benchmark_latency(model: torch.nn.Module, device: torch.device, img_size: int, batch_size: int, warmup: int, steps: int) -> float:
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


def maybe_export_onnx(model: torch.nn.Module, out_path: Path, img_size: int) -> tuple[bool, str]:
    try:
        x = torch.randn(1, 3, img_size, img_size)
        torch.onnx.export(
            model.cpu(),
            x,
            str(out_path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export and benchmark ShiroNet checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default="docs/assets/optimization/export_benchmark")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    arch = ckpt.get("arch", "shufflenet_v2_x0_5")
    classes = ckpt["classes"]
    img_size = int(ckpt.get("img_size", 160))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = create_model(arch, num_classes=len(classes), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # FP32 baseline
    cpu = torch.device("cpu")
    fp32_cpu_ms = benchmark_latency(model.cpu(), cpu, img_size, args.batch_size, args.warmup, args.steps)

    fp32_gpu_ms = None
    if torch.cuda.is_available():
        gpu_model = create_model(arch, num_classes=len(classes), pretrained=False)
        gpu_model.load_state_dict(ckpt["model_state_dict"])
        gpu_model = gpu_model.cuda().eval()
        fp32_gpu_ms = benchmark_latency(gpu_model, torch.device("cuda"), img_size, args.batch_size, args.warmup, args.steps)

    # TorchScript
    ts_path = out_dir / f"{arch}_ts.pt"
    example = torch.randn(1, 3, img_size, img_size)
    scripted = torch.jit.trace(model.cpu(), example)
    scripted.save(str(ts_path))
    ts_cpu_ms = benchmark_latency(scripted, cpu, img_size, args.batch_size, args.warmup, args.steps)

    # Dynamic quantization (best effort; mainly affects Linear layers)
    q_model = torch.quantization.quantize_dynamic(model.cpu(), {torch.nn.Linear}, dtype=torch.qint8)
    q_path = out_dir / f"{arch}_dynamic_q.pt"
    torch.save(q_model.state_dict(), q_path)
    q_cpu_ms = benchmark_latency(q_model, cpu, img_size, args.batch_size, args.warmup, args.steps)

    # ONNX export
    onnx_path = out_dir / f"{arch}.onnx"
    onnx_ok, onnx_err = maybe_export_onnx(model, onnx_path, img_size)

    payload = {
        "arch": arch,
        "checkpoint": str(ckpt_path),
        "img_size": img_size,
        "batch_size": args.batch_size,
        "latency_ms": {
            "fp32_cpu": fp32_cpu_ms,
            "fp32_gpu": fp32_gpu_ms,
            "torchscript_cpu": ts_cpu_ms,
            "dynamic_int8_cpu": q_cpu_ms,
        },
        "artifacts": {
            "torchscript": str(ts_path),
            "onnx": str(onnx_path) if onnx_ok else None,
            "dynamic_quant_state_dict": str(q_path),
        },
        "onnx_export": {
            "success": onnx_ok,
            "error": onnx_err,
        },
    }

    json_path = out_dir / "export_benchmark.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        "# Export and Latency Benchmark",
        "",
        f"- Architecture: `{arch}`",
        f"- Checkpoint: `{ckpt_path}`",
        f"- Image size: `{img_size}`",
        f"- Batch size: `{args.batch_size}`",
        "",
        "| Variant | Latency (ms/image) |",
        "|---|---:|",
        f"| FP32 CPU | {fp32_cpu_ms:.4f} |",
        f"| FP32 GPU | {fp32_gpu_ms:.4f} |" if fp32_gpu_ms is not None else "| FP32 GPU | N/A |",
        f"| TorchScript CPU | {ts_cpu_ms:.4f} |",
        f"| Dynamic INT8 CPU | {q_cpu_ms:.4f} |",
        "",
        f"- ONNX export: `{'success' if onnx_ok else 'failed'}`",
    ]
    if not onnx_ok:
        md.append(f"- ONNX error: `{onnx_err}`")
    (out_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
