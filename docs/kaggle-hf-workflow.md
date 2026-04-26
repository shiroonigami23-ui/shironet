# Kaggle + HF Workflow

## 1) Kaggle dataset fetch (real command)
1. Authenticate once:
   - Env vars: `KAGGLE_USERNAME`, `KAGGLE_KEY`
   - Or file: `~/.kaggle/kaggle.json`
2. Run:
   - `python scripts/kaggle_fetch.py --dataset owner/dataset-name --out data/raw`

Example:
- `python scripts/kaggle_fetch.py --dataset paultimothymooney/chest-xray-pneumonia --out data/raw`

## 2) Train locally and save checkpoints
- Save checkpoints under `models/` (already ignored by git).

## 3) Upload large checkpoints to Hugging Face Hub (real command)
1. Export token:
   - `HF_TOKEN=...`
2. Run:
   - `python scripts/hf_upload.py --repo-id <username>/<model-repo> --local-dir models --path-in-repo checkpoints --private`

Example:
- `python scripts/hf_upload.py --repo-id shiroonigami23-ui/shironet-edge --local-dir models --path-in-repo checkpoints --private`

## Suggested storage split
- Supabase Storage: distilled datasets, manifests, logs
- Hugging Face Hub: model checkpoints and inference exports