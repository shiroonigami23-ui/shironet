# ShiroNet

![Version](https://img.shields.io/badge/version-v0.1.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.11%2B-3776AB)
![Build Status](https://img.shields.io/badge/build-pending-lightgrey)

ShiroNet is a lightweight, robust, and adversarially hardened vision framework designed for distilled-dataset training and edge inference.

## Project Overview

ShiroNet focuses on practical vision training under resource constraints by combining dataset distillation, adversarial hardening, and deployment-oriented model design. The project is structured for rapid iteration and production readiness.

## Architecture (Distillation Focus)

The architecture separates concerns into:
- Data pipeline and distillation logs (`src/data`, `data/`)
- Model design and edge-friendly backbones (`src/models`)
- Adversarial training hooks (`src/training`)
- Integration layer for Neon Postgres and Supabase (`src/config.py`)

See [docs/architecture/system-overview.md](docs/architecture/system-overview.md) for diagrams and milestone tables.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Quick Start

```bash
python -m src
```

Notebook workflow:
1. Open `notebooks/01_baseline_test.ipynb` for baseline sanity checks.
2. Open `notebooks/02_distillation_trial.ipynb` for distillation + adversarial iteration.
3. Use Kaggle API for dataset pulls and Hugging Face Hub for large checkpoint storage.

CLI workflow:
```bash
python scripts/kaggle_fetch.py --dataset puneet6060/intel-image-classification --out data/raw
python src/data/prepare_dataset.py --input-root data/raw --output-root data/processed/intel_scenes --val-ratio 0.1
python src/train.py --data-root data/processed/intel_scenes --epochs 10 --batch-size 32 --img-size 160 --lr 5e-4 --adv-eps 0.005 --save-dir models/intel_run --pretrained
python scripts/hf_upload.py --repo-name shironet-edge --local-dir models/intel_run --path-in-repo checkpoints/intel-run --private
```

## Tech Stack

- Neon (serverless Postgres)
- Supabase (auth, storage, edge integrations)
- MyTorch (custom training framework bridge)
- PyTorch + TensorRT (training and inference)

## Benchmark Snapshot

Kaggle apples-to-apples benchmark on Intel Scenes (same split and backbone):
- Baseline test accuracy: `82.13%`
- ShiroNet test accuracy: `85.07%`
- Delta: `+2.93` points
- Baseline FGSM (`eps=0.01`): `26.93%`
- ShiroNet FGSM (`eps=0.01`): `73.03%`
- Robustness delta: `+46.10` points

Detailed artifact: `docs/assets/benchmark_kaggle_v1/report.md`

## License

MIT License. See [LICENSE](LICENSE).
