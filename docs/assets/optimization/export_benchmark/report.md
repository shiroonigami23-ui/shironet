# Export and Latency Benchmark

- Architecture: `shufflenet_v2_x0_5`
- Checkpoint: `models\intel_shironet_edge\shironet_shufflenet_v2_x0_5.pt`
- Image size: `160`
- Batch size: `1`

| Variant | Latency (ms/image) |
|---|---:|
| FP32 CPU | 5.9384 |
| FP32 GPU | N/A |
| TorchScript CPU | 4.1909 |
| Dynamic INT8 CPU | 5.9944 |

- ONNX export: `success`
