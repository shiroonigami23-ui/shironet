import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, models, transforms


def _resolve_split(root: Path, split: str) -> Path:
    direct = root / split
    if direct.exists():
        return direct
    for child in root.iterdir():
        candidate = child / split
        if child.is_dir() and candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find split '{split}' under {root}")


def _compute_metrics(confusion: torch.Tensor) -> tuple[list[dict], float]:
    num_classes = confusion.size(0)
    total = int(confusion.sum().item())
    correct = int(torch.diag(confusion).sum().item())
    accuracy = correct / total if total > 0 else 0.0

    rows: list[dict] = []
    for i in range(num_classes):
        tp = float(confusion[i, i].item())
        fp = float(confusion[:, i].sum().item() - tp)
        fn = float(confusion[i, :].sum().item() - tp)
        support = int(confusion[i, :].sum().item())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        rows.append(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    return rows, accuracy


def _plot_confusion(confusion: torch.Tensor, classes: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(confusion.numpy(), interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(classes)),
        yticks=range(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title="ShiroNet Intel Run2 Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = confusion.max().item() / 2 if confusion.numel() > 0 else 0.0
    for i in range(confusion.size(0)):
        for j in range(confusion.size(1)):
            val = int(confusion[i, j].item())
            ax.text(
                j,
                i,
                f"{val}",
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_precision_recall(rows: list[dict], classes: list[str], out_path: Path) -> None:
    x = list(range(len(classes)))
    precision = [r["precision"] for r in rows]
    recall = [r["recall"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.36
    ax.bar([i - width / 2 for i in x], precision, width=width, label="Precision")
    ax.bar([i + width / 2 for i in x], recall, width=width, label="Recall")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Class-wise Precision and Recall (Intel Run2)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate ShiroNet checkpoint and generate report assets.")
    parser.add_argument("--data-root", required=True, help="Processed dataset root containing train/val/test")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate (default: test)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--out-dir", default="docs/assets/intel_run2")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    classes = ckpt["classes"]
    img_size = int(ckpt.get("img_size", 160))
    data_root = Path(args.data_root)
    split_dir = _resolve_split(data_root, args.split)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tfm = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    ds = datasets.ImageFolder(split_dir, transform=tfm)
    if ds.classes != classes:
        raise SystemExit(f"Class mismatch: checkpoint={classes}, dataset={ds.classes}")

    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    confusion = torch.zeros((len(classes), len(classes)), dtype=torch.int64)
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            pred = logits.argmax(dim=1)
            for t, p in zip(y.view(-1), pred.view(-1)):
                confusion[int(t), int(p)] += 1

    class_rows, accuracy = _compute_metrics(confusion)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_json = out_dir / "eval_metrics.json"
    metrics_csv = out_dir / "class_metrics.csv"
    confusion_png = out_dir / "confusion_matrix.png"
    pr_png = out_dir / "precision_recall.png"
    report_md = out_dir / "report.md"

    payload = {
        "split": args.split,
        "num_samples": int(confusion.sum().item()),
        "accuracy": accuracy,
        "classes": classes,
        "class_metrics": class_rows,
        "confusion_matrix": confusion.tolist(),
    }
    metrics_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for cls, row in zip(classes, class_rows):
            writer.writerow([cls, f"{row['precision']:.6f}", f"{row['recall']:.6f}", f"{row['f1']:.6f}", row["support"]])

    _plot_confusion(confusion, classes, confusion_png)
    _plot_precision_recall(class_rows, classes, pr_png)

    lines = [
        "# Intel Run2 Evaluation",
        "",
        f"- Split: `{args.split}`",
        f"- Samples: `{int(confusion.sum().item())}`",
        f"- Accuracy: `{accuracy:.4f}`",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for cls, row in zip(classes, class_rows):
        lines.append(
            f"| {cls} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['support']} |"
        )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Evaluation complete. Accuracy={accuracy:.4f}")
    print(f"Artifacts written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
