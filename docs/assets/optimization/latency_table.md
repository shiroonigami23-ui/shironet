# ShiroNet Latency Table

Checkpoint: `models/intel_shironet_edge/shironet_shufflenet_v2_x0_5.pt`  
Architecture: `shufflenet_v2_x0_5`  
Input size: `160x160`, batch size `1`

| Variant | Latency (ms/image) | Notes |
|---|---:|---|
| FP32 CPU | 5.9384 | Baseline eager PyTorch |
| TorchScript CPU | 4.1909 | Faster than eager CPU |
| Dynamic INT8 CPU | 5.9944 | No gain for this conv-heavy model |
| FP32 GPU | N/A (local) | GPU benchmark should be run on Kaggle T4 environment |

## Export Artifacts
- `docs/assets/optimization/export_benchmark/shufflenet_v2_x0_5_ts.pt`
- `docs/assets/optimization/export_benchmark/shufflenet_v2_x0_5.onnx`
- `docs/assets/optimization/export_benchmark/shufflenet_v2_x0_5_dynamic_q.pt`
- `docs/assets/optimization/export_benchmark/export_benchmark.json`
- `docs/assets/optimization/export_benchmark/report.md`