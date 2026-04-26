# Kaggle + HF Workflow

## 1) Kaggle dataset fetch (real command)
1. Authenticate once:
   - Env vars: `KAGGLE_USERNAME`, `KAGGLE_KEY`
   - Or file: `~/.kaggle/kaggle.json`
2. Run:
   - `python scripts/kaggle_fetch.py --dataset owner/dataset-name --out data/raw`

Example:
- `python scripts/kaggle_fetch.py --dataset puneet6060/intel-image-classification --out data/raw`

## 2) Clean and preprocess dataset
- `python src/data/prepare_dataset.py --input-root data/raw --output-root data/processed/intel_scenes --val-ratio 0.1`
- This removes corrupt images and creates clean `train/val/test` splits plus a report file.

## 3) Train locally and save checkpoints
- Run real training with augmentation:
  - `python src/train.py --data-root data/processed/intel_scenes --epochs 10 --batch-size 32 --img-size 160 --lr 5e-4 --adv-eps 0.005 --save-dir models/intel_run --pretrained`
- Checkpoints and metrics are written under `models/intel_run`.

## 4) Upload large checkpoints to Hugging Face Hub (real command)
1. Export token:
   - `HF_TOKEN=...`
   - `HF_USERNAME=ShiroOnigami23`
2. Run:
   - `python scripts/hf_upload.py --repo-name shironet-edge --local-dir models/intel_run --path-in-repo checkpoints/intel-run --private`

Example:
- `python scripts/hf_upload.py --repo-name shironet-edge --local-dir models/intel_run --path-in-repo checkpoints/intel-run --private`

## Suggested storage split
- Supabase Storage: distilled datasets, manifests, logs
- Hugging Face Hub: model checkpoints and inference exports
