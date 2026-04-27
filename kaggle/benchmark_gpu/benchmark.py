import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_gpu_compatible_torch() -> None:
    if not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability(0)
    if major >= 7:
        return
    if os.getenv("SHIRONET_TORCH_REINSTALLED") == "1":
        return

    print("Detected legacy GPU capability. Installing compatible torch build for Kaggle GPU...")
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "numpy==1.26.4",
        "torch==2.2.2+cu118",
        "torchvision==0.17.2+cu118",
        "--index-url",
        "https://download.pytorch.org/whl/cu118",
    ]
    try:
        subprocess.check_call(cmd)
        os.environ["SHIRONET_TORCH_REINSTALLED"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except subprocess.CalledProcessError as exc:
        print(f"GPU-compatible torch install failed: {exc}. Falling back to CPU.")
        os.environ["SHIRONET_FORCE_CPU"] = "1"


def resolve_split(root: Path, names: list[str]) -> Path:
    def is_class_root(p: Path) -> bool:
        if not p.exists() or not p.is_dir():
            return False
        dirs = [d for d in p.iterdir() if d.is_dir()]
        if not dirs:
            return False
        return any(any(f.is_file() for f in d.iterdir()) for d in dirs)

    for name in names:
        p = root / name
        if is_class_root(p):
            return p
        nested = p / name
        if is_class_root(nested):
            return nested
    for child in root.iterdir():
        if not child.is_dir():
            continue
        for name in names:
            p = child / name
            if is_class_root(p):
                return p
            nested = p / name
            if is_class_root(nested):
                return nested
    raise FileNotFoundError(f"Could not find split from {names} under {root}")


def find_kaggle_dataset_root() -> Path:
    base = Path("/kaggle/input")
    if not base.exists():
        raise FileNotFoundError("/kaggle/input does not exist")

    candidates = [p for p in base.iterdir() if p.is_dir()]
    for root in candidates:
        try:
            _ = resolve_split(root, ["seg_train", "train", "training"])
            _ = resolve_split(root, ["seg_test", "test"])
            return root
        except FileNotFoundError:
            continue

    # Also check one nested level deep.
    for parent in candidates:
        for root in parent.iterdir():
            if not root.is_dir():
                continue
            try:
                _ = resolve_split(root, ["seg_train", "train", "training"])
                _ = resolve_split(root, ["seg_test", "test"])
                return root
            except FileNotFoundError:
                continue

    raise FileNotFoundError("Could not discover Intel dataset root under /kaggle/input")


def prepare_dataset(raw_root: Path, out_root: Path, val_ratio: float = 0.1, seed: int = 42) -> dict:
    if out_root.exists():
        shutil.rmtree(out_root)
    train_out = out_root / "train"
    val_out = out_root / "val"
    test_out = out_root / "test"
    train_out.mkdir(parents=True, exist_ok=True)
    val_out.mkdir(parents=True, exist_ok=True)
    test_out.mkdir(parents=True, exist_ok=True)

    train_src = resolve_split(raw_root, ["seg_train", "train", "training"])
    test_src = resolve_split(raw_root, ["seg_test", "test"])

    report = {"train_raw": {}, "train": {}, "val": {}, "test": {}}
    rng = random.Random(seed)

    for class_dir in sorted(train_src.iterdir()):
        if not class_dir.is_dir():
            continue
        files = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        report["train_raw"][class_dir.name] = len(files)
        rng.shuffle(files)
        n_val = int(len(files) * val_ratio)
        val_files = set(files[:n_val])

        dst_train_c = train_out / class_dir.name
        dst_val_c = val_out / class_dir.name
        dst_train_c.mkdir(parents=True, exist_ok=True)
        dst_val_c.mkdir(parents=True, exist_ok=True)

        t_count = 0
        v_count = 0
        for fp in files:
            if fp in val_files:
                shutil.copy2(fp, dst_val_c / fp.name)
                v_count += 1
            else:
                shutil.copy2(fp, dst_train_c / fp.name)
                t_count += 1
        report["train"][class_dir.name] = t_count
        report["val"][class_dir.name] = v_count

    for class_dir in sorted(test_src.iterdir()):
        if not class_dir.is_dir():
            continue
        files = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        dst_test_c = test_out / class_dir.name
        dst_test_c.mkdir(parents=True, exist_ok=True)
        for fp in files:
            shutil.copy2(fp, dst_test_c / fp.name)
        report["test"][class_dir.name] = len(files)

    return report


def fgsm_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float, mean, std) -> torch.Tensor:
    eps_t = torch.tensor([eps / s for s in std], device=x.device).view(1, 3, 1, 1)
    x_adv = x.detach().clone().requires_grad_(True)
    logits = model(x_adv)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    adv = x_adv + eps_t * x_adv.grad.sign()

    low = torch.tensor([(0.0 - m) / s for m, s in zip(mean, std)], device=x.device).view(1, 3, 1, 1)
    high = torch.tensor([(1.0 - m) / s for m, s in zip(mean, std)], device=x.device).view(1, 3, 1, 1)
    adv = torch.max(torch.min(adv, high), low).detach()
    model.zero_grad(set_to_none=True)
    return adv


def make_transforms(img_size: int, profile: str):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if profile == "baseline":
        tfm_train = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        tfm_train = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.10), ratio=(0.3, 3.3)),
        ])

    tfm_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return tfm_train, tfm_eval, mean, std


def evaluate(model, loader, device):
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


def evaluate_fgsm(model, loader, device, eps, mean, std):
    model.eval()
    total = 0
    correct = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_adv = fgsm_attack(model, x, y, eps, mean, std)
        with torch.no_grad():
            pred = model(x_adv).argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.size(0))
    return correct / max(total, 1)


def train_once(name, data_root, save_dir, epochs, batch_size, img_size, lr, adv_eps, profile, device):
    tfm_train, tfm_eval, mean, std = make_transforms(img_size, profile)

    train_ds = datasets.ImageFolder(data_root / "train", transform=tfm_train)
    val_ds = datasets.ImageFolder(data_root / "val", transform=tfm_eval)
    test_ds = datasets.ImageFolder(data_root / "test", transform=tfm_eval)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / "train_metrics.jsonl"

    print(f"\\n=== Training {name} ===")
    print(f"classes={train_ds.classes}, adv_eps={adv_eps}, profile={profile}")

    for epoch in range(1, epochs + 1):
        model.train()
        run_loss = 0.0
        run_total = 0
        run_correct = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            if adv_eps > 0:
                opt.zero_grad(set_to_none=True)
                adv_x = fgsm_attack(model, x, y, adv_eps, mean, std)
                adv_logits = model(adv_x)
                adv_loss = loss_fn(adv_logits, y)
                adv_loss.backward()
                opt.step()

            run_loss += float(loss.item()) * x.size(0)
            run_correct += int((logits.argmax(dim=1) == y).sum().item())
            run_total += x.size(0)

        train_loss = run_loss / run_total
        train_acc = run_correct / run_total
        val_loss, val_acc = evaluate(model, val_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "adv_eps": adv_eps,
            "profile": profile,
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\\n")

        print(
            f"{name} epoch {epoch}/{epochs} | "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
        )

    ckpt = save_dir / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": train_ds.classes,
            "img_size": img_size,
            "arch": "resnet18",
            "name": name,
            "adv_eps": adv_eps,
            "profile": profile,
        },
        ckpt,
    )

    test_loss, test_acc = evaluate(model, test_loader, device)
    fgsm_acc = evaluate_fgsm(model, test_loader, device, eps=0.01, mean=mean, std=std)

    return {
        "name": name,
        "checkpoint": str(ckpt),
        "test_loss": test_loss,
        "test_acc": test_acc,
        "fgsm_eps_0_01_acc": fgsm_acc,
        "classes": train_ds.classes,
    }


def main():
    ensure_gpu_compatible_torch()
    seed_everything(42)
    device = torch.device("cpu")
    if os.getenv("SHIRONET_FORCE_CPU") != "1" and torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device="cuda")
            device = torch.device("cuda")
        except Exception:
            device = torch.device("cpu")
    print("device:", device)
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    raw_root = find_kaggle_dataset_root()
    print("dataset root:", raw_root)
    work_root = Path("/kaggle/working")
    proc_root = work_root / "data" / "processed" / "intel_scenes"
    models_root = work_root / "models"

    prep_report = prepare_dataset(raw_root, proc_root, val_ratio=0.1, seed=42)

    baseline = train_once(
        name="baseline",
        data_root=proc_root,
        save_dir=models_root / "intel_baseline",
        epochs=10,
        batch_size=64,
        img_size=160,
        lr=5e-4,
        adv_eps=0.0,
        profile="baseline",
        device=device,
    )

    shironet = train_once(
        name="shironet",
        data_root=proc_root,
        save_dir=models_root / "intel_shironet",
        epochs=10,
        batch_size=64,
        img_size=160,
        lr=5e-4,
        adv_eps=0.01,
        profile="shironet",
        device=device,
    )

    summary = {
        "prepare_report": prep_report,
        "baseline": baseline,
        "shironet": shironet,
        "delta": {
            "test_acc": shironet["test_acc"] - baseline["test_acc"],
            "fgsm_eps_0_01_acc": shironet["fgsm_eps_0_01_acc"] - baseline["fgsm_eps_0_01_acc"],
        },
    }

    out = work_root / "benchmark_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\\nBenchmark summary:")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
