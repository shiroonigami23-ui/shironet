# Kaggle GPU Benchmark

Kaggle kernel assets for reproducible baseline-vs-ShiroNet benchmarking.

## Files
- `benchmark.py`: end-to-end benchmark pipeline
- `kernel-metadata.json`: Kaggle kernel configuration

## Run
```bash
kaggle kernels push -p kaggle/benchmark_gpu
kaggle kernels status aryanchande23l/shironet-benchmark-gpu
```