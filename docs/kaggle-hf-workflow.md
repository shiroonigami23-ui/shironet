# Kaggle + HF workflow helpers

## Kaggle dataset fetch
1. Set `KAGGLE_USERNAME` and `KAGGLE_KEY`.
2. Use notebook cells to run:
   - `!kaggle datasets download -d <owner/dataset> -p ../data/raw --unzip`

## Hugging Face model storage
1. Set `HF_TOKEN`.
2. Use `huggingface_hub` in training scripts to push large checkpoints.

## Suggested bucket split
- Supabase Storage: distilled datasets, metadata snapshots
- Hugging Face Hub: model checkpoints and inference-ready artifacts
