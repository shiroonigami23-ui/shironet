import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def _resolve_split(root: Path, split: str) -> Path:
    alias = {
        "train": ["train", "training", "training_set"],
        "val": ["val", "valid", "validation", "test", "test_set"],
        "test": ["test", "test_set"],
    }
    names = alias.get(split, [split])
    for name in names:
        direct = root / name
        if direct.exists():
            return direct

    # Kaggle datasets are often extracted under a top-level folder.
    for child in root.iterdir():
        if not child.is_dir():
            continue
        for name in names:
            candidate = child / name
            if candidate.exists():
                return candidate

    raise FileNotFoundError(f"Could not find any of {names} under: {root}")


def _fgsm_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    adv = x.detach().clone().requires_grad_(True)
    logits = model(adv)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    perturbed = torch.clamp(adv + eps * adv.grad.sign(), 0.0, 1.0).detach()
    model.zero_grad(set_to_none=True)
    return perturbed


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += float(loss.item()) * x.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += x.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train ShiroNet on a real image dataset.")
    parser.add_argument("--data-root", default="data/raw", help="Dataset root directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--adv-eps", type=float, default=0.0, help="FGSM epsilon (0 disables adversarial step)")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-dir", default="models")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet-pretrained ResNet18 backbone")
    parser.add_argument(
        "--augment-profile",
        choices=["baseline", "shironet"],
        default="shironet",
        help="Augmentation profile: baseline (minimal) or shironet (strong)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    train_dir = _resolve_split(data_root, "train")
    val_dir = _resolve_split(data_root, "val") if (data_root / "val").exists() else _resolve_split(data_root, "test")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if args.augment_profile == "baseline":
        tfm_train = transforms.Compose(
            [
                transforms.Resize((args.img_size, args.img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        tfm_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(args.img_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=12),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                transforms.RandomErasing(p=0.15, scale=(0.02, 0.10), ratio=(0.3, 3.3)),
            ]
        )
    tfm_eval = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    train_ds = datasets.ImageFolder(train_dir, transform=tfm_train)
    val_ds = datasets.ImageFolder(val_dir, transform=tfm_eval)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = models.ResNet18_Weights.DEFAULT if args.pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / "train_metrics.jsonl"
    ckpt_path = save_dir / "shironet_resnet18.pt"

    print(f"Training on real data from: {train_dir}")
    print(f"Validation split from: {val_dir}")
    print(f"Classes: {train_ds.classes}")
    print(f"Device: {device}")
    print(f"Pretrained backbone: {bool(args.pretrained)}")
    print(f"Augmentation profile: {args.augment_profile}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_total = 0
        running_correct = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            if args.adv_eps > 0:
                optimizer.zero_grad(set_to_none=True)
                adv_x = _fgsm_attack(model, x, y, args.adv_eps)
                adv_logits = model(adv_x)
                adv_loss = loss_fn(adv_logits, y)
                adv_loss.backward()
                optimizer.step()

            running_loss += float(loss.item()) * x.size(0)
            pred = logits.argmax(dim=1)
            running_correct += int((pred == y).sum().item())
            running_total += x.size(0)

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)
        val_loss, val_acc = _evaluate(model, val_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "adv_eps": args.adv_eps,
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": train_ds.classes,
            "img_size": args.img_size,
            "arch": "resnet18",
        },
        ckpt_path,
    )
    print(f"Checkpoint saved: {ckpt_path}")
    print(f"Metrics log saved: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
