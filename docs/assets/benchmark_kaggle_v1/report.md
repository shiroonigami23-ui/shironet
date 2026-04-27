# Kaggle Benchmark V1 (Intel Scenes)

Baseline and ShiroNet were trained and evaluated on the same processed Intel Scenes split (6 classes) in Kaggle.

## Results

- Baseline test accuracy: **82.13%**
- ShiroNet test accuracy: **85.07%**
- Accuracy gain: **+2.93 points**

- Baseline FGSM accuracy (eps=0.01): **26.93%**
- ShiroNet FGSM accuracy (eps=0.01): **73.03%**
- Robustness gain: **+46.10 points**

## Interpretation

ShiroNet is measurably better than the baseline in this benchmark:
- Better clean-data generalization
- Much stronger adversarial resilience under FGSM attack

## Artifact

- `benchmark_summary.json`