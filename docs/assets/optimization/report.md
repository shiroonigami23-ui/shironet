# ShiroNet Optimization Snapshot

## Goal Alignment
We are optimizing for:
- Lighter model size
- Faster inference
- Maintained or improved task performance

## Architecture Profiling (CPU, 160x160, batch=1)

| Architecture | Params | Latency (ms/image) |
|---|---:|---:|
| resnet18 | 11,179,590 | 19.50 |
| mobilenet_v3_small | 1,524,006 | 16.41 |
| shufflenet_v2_x0_5 | 347,942 | 14.69 |

## Real Edge Run (ShuffleNet)

Training config:
- Dataset: Intel Scenes (processed)
- Epochs: 2
- Adv epsilon: 0.005
- Arch: shufflenet_v2_x0_5

Evaluation:
- Test accuracy: 90.80%
- FGSM accuracy (eps=0.01): 27.33%

## Interpretation
- `shufflenet_v2_x0_5` is currently best for "lighter + faster".
- Clean accuracy remains strong even with very small parameter budget.
- Robustness for the edge variant needs more adversarial-focused training epochs.

## Next Optimization Steps
1. Train `shufflenet_v2_x0_5` for 10-20 epochs with curriculum adversarial schedule.
2. Run AMP + compile on CUDA for speed.
3. Add quantization/export benchmark (TorchScript/ONNX/TensorRT).