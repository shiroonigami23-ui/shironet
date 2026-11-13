# 🗺 Roadmap — ShiroNet

> Back to the [README](../README.md).

## Where the project stands

Status: **v0.1.0**.

## Milestones

### v0.1 — Foundations (done)

- [x] Modular data / model / training / evaluation packages
- [x] Adversarial hardening with +46 pt FGSM gain on Intel Scenes
- [x] Three edge backbones and export/latency notebook
- [x] Kaggle and Hugging Face tooling

### v0.2 — Robustness & rigour

*Goal:* Make the robustness claims stronger and more general.

- [ ] PGD and AutoAttack evaluation in addition to FGSM
- [ ] Multi-seed runs with confidence intervals
- [ ] CI pipeline (GitHub Actions) running pytest + a smoke-train
- [ ] Real dataset-distillation method (gradient / trajectory matching) with ablation

### v0.3 — Edge deployment

- [ ] INT8 quantisation-aware training
- [ ] ONNX / TensorRT / TFLite export with on-device latency on Jetson and Android
- [ ] Tiny inference API + demo Space

### v1.0 — Scale-up

- [ ] CIFAR-100 and ImageNet-1k subset benchmarks
- [ ] Packaged `pip install shironet` release with model zoo

## 🔭 Long-term Vision

A go-to toolkit for "robust models on a budget" — train on a free GPU, harden against attacks, and ship a model that runs
on a phone, with every number reproducible from a notebook.

## How to help

Pick any unchecked item, open an issue to discuss the approach, then send a PR. Items in the earliest unfinished milestone are the highest priority.


---
*Last updated: 2026-11-14*
