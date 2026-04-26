# System Overview

## High-Level Diagram

```mermaid
flowchart TD
    A[Raw Dataset Sources] --> B[Distillation Pipeline]
    B --> C[MyTorch Training Core]
    C --> D[Adversarial Hooks]
    D --> E[Edge Inference Export]
    C --> F[Neon Postgres Metadata]
    C --> G[Supabase Storage]
    G --> H[Hugging Face Model Sync]
```

## Progress Table

| Milestone | Status | Notes |
|---|---|---|
| Repo Bootstrap | Complete | Core structure and config done |
| Baseline Training Notebook | In Progress | `01_baseline_test.ipynb` |
| Distillation Trial | In Progress | `02_distillation_trial.ipynb` |
| HF Checkpoint Sync | Planned | Script + token setup |
| Adversarial Eval Harness | Planned | Add robust test suite |

## Assets

- Dashboard Chart: `docs/assets/training_curve.png`
- Model Animation: `docs/assets/model_evolution.gif`
