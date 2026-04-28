import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_split(root: Path, names):
    def is_class_root(p: Path):
        if not (p.exists() and p.is_dir()):
            return False
        class_dirs = [d for d in p.iterdir() if d.is_dir()]
        if not class_dirs:
            return False
        return any(any(f.is_file() for f in d.iterdir()) for d in class_dirs)

    for n in names:
        p = root / n
        if is_class_root(p):
            return p
        nested = p / n
        if is_class_root(nested):
            return nested
    for child in root.iterdir():
        if not child.is_dir():
            continue
        for n in names:
            p = child / n
            if is_class_root(p):
                return p
            nested = p / n
            if is_class_root(nested):
                return nested
    raise FileNotFoundError(f"Could not resolve split {names} under {root}")


def find_dataset_root():
    base = Path('/kaggle/input')
    for p in base.rglob('*'):
        if p.is_dir():
            try:
                _ = resolve_split(p, ['seg_train', 'train'])
                _ = resolve_split(p, ['seg_test', 'test'])
                return p
            except Exception:
                continue
    raise FileNotFoundError('Intel dataset not found under /kaggle/input')


def prepare_dataset(raw_root: Path, out_root: Path, val_ratio=0.1, seed=42):
    if out_root.exists():
        shutil.rmtree(out_root)
    (out_root / 'train').mkdir(parents=True, exist_ok=True)
    (out_root / 'val').mkdir(parents=True, exist_ok=True)
    (out_root / 'test').mkdir(parents=True, exist_ok=True)

    train_src = resolve_split(raw_root, ['seg_train', 'train'])
    test_src = resolve_split(raw_root, ['seg_test', 'test'])

    rng = random.Random(seed)
    report = {'train': {}, 'val': {}, 'test': {}}

    for c in sorted(train_src.iterdir()):
        if not c.is_dir():
            continue
        files = [f for f in c.iterdir() if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        rng.shuffle(files)
        n_val = int(len(files) * val_ratio)
        val_files = set(files[:n_val])

        tdir = out_root / 'train' / c.name
        vdir = out_root / 'val' / c.name
        tdir.mkdir(parents=True, exist_ok=True)
        vdir.mkdir(parents=True, exist_ok=True)

        t_cnt = 0
        v_cnt = 0
        for f in files:
            if f in val_files:
                shutil.copy2(f, vdir / f.name)
                v_cnt += 1
            else:
                shutil.copy2(f, tdir / f.name)
                t_cnt += 1
        report['train'][c.name] = t_cnt
        report['val'][c.name] = v_cnt

    for c in sorted(test_src.iterdir()):
        if not c.is_dir():
            continue
        files = [f for f in c.iterdir() if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        d = out_root / 'test' / c.name
        d.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, d / f.name)
        report['test'][c.name] = len(files)

    return report


def fgsm_attack(model, x, y, eps):
    adv = x.detach().clone().requires_grad_(True)
    logits = model(adv)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    out = torch.clamp(adv + eps * adv.grad.sign(), -3.0, 3.0).detach()
    model.zero_grad(set_to_none=True)
    return out


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss_sum += float(loss.item()) * x.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += x.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def main():
    seed_everything(42)
    device = torch.device('cpu')
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        if major >= 7:
            device = torch.device('cuda')
    print('device:', device)
    if torch.cuda.is_available():
        print('gpu:', torch.cuda.get_device_name(0), 'capability:', torch.cuda.get_device_capability(0))

    raw_root = find_dataset_root()
    print('dataset root:', raw_root)

    work = Path('/kaggle/working')
    proc = work / 'data' / 'processed' / 'intel_scenes'
    model_dir = work / 'models' / 'intel_run_kaggle'
    model_dir.mkdir(parents=True, exist_ok=True)

    prep = prepare_dataset(raw_root, proc, val_ratio=0.1, seed=42)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_size = 160
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.10), ratio=(0.3, 3.3)),
    ])
    tf_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.ImageFolder(proc / 'train', transform=tf_train)
    val_ds = datasets.ImageFolder(proc / 'val', transform=tf_eval)
    test_ds = datasets.ImageFolder(proc / 'test', transform=tf_eval)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    model = models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    loss_fn = nn.CrossEntropyLoss()

    metrics_file = model_dir / 'train_metrics.jsonl'
    epochs = 12
    for epoch in range(1, epochs + 1):
        eps = 0.002 + (0.012 - 0.002) * ((epoch - 1) / (epochs - 1))
        model.train()
        total = 0
        correct = 0
        loss_sum = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            opt.zero_grad(set_to_none=True)
            x_adv = fgsm_attack(model, x, y, eps)
            adv_logits = model(x_adv)
            adv_loss = loss_fn(adv_logits, y)
            adv_loss.backward()
            opt.step()

            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += x.size(0)
            loss_sum += float(loss.item()) * x.size(0)

        train_loss = loss_sum / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)
        row = {
            'epoch': epoch,
            'adv_eps': eps,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'arch': 'shufflenet_v2_x0_5',
        }
        with metrics_file.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row) + '\n')
        print(f"Epoch {epoch}/{epochs} | eps={eps:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    test_loss, test_acc = evaluate(model, test_loader, device)
    ckpt = model_dir / 'shironet_shufflenet_v2_x0_5.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': train_ds.classes,
        'img_size': img_size,
        'arch': 'shufflenet_v2_x0_5'
    }, ckpt)

    summary = {
        'dataset_root': str(raw_root),
        'prepare_report': prep,
        'arch': 'shufflenet_v2_x0_5',
        'epochs': epochs,
        'adv_schedule': {'start': 0.002, 'end': 0.012},
        'test_loss': test_loss,
        'test_acc': test_acc,
        'checkpoint': str(ckpt),
        'metrics': str(metrics_file),
    }
    (model_dir / 'run_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
