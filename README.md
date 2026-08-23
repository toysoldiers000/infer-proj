# AI Compiler Lab

这里按 `Learning-Guide.md` 螺旋推进。当前实验是正式 Transformer Project 1 之前的 **Phase 0 / Core ML Foundation**：用同一个 ResNet-50 建立 PyTorch、`torch.compile`、ONNX Runtime、Core ML 的 correctness 与 benchmark 闭环。

## 启动第一个实验

```bash
uv sync
uv run jupyter lab notebooks/phase0_coreml_foundation/01_resnet50_backend_comparison.ipynb
```

模型权重、ONNX、`.mlpackage`、`.mlmodelc` 和机器相关 benchmark 结果均由 notebook 生成或校验，并已排除在 Git 之外。
