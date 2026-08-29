"""Run N Core ML predicts on a .mlpackage, then exit.

本脚本的存在理由：Instruments/xctrace 录制需要一个"生命周期可控"的目标进程。
用法（在仓库根目录）：
    uv run python tools/run_coreml_predict.py <model.mlpackage> [iterations]
配合 Instruments 录制（trace 文件由 xctrace 写到你指定的 --output 路径）：
    xctrace record --template 'Core ML' \
        --output artifacts/resnet50/coreml_fp16.trace \
        --launch .venv/bin/python tools/run_coreml_predict.py \
        artifacts/resnet50/resnet50_fp16.mlpackage 30
"""
import sys
import time

import numpy as np
import coremltools as ct

model_path = sys.argv[1]
n_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 30
unit_name = sys.argv[3].upper() if len(sys.argv) > 3 else "ALL"
compute_units = {
    "ALL": ct.ComputeUnit.ALL,
    "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
    "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
}[unit_name]

# compute_units 限定 runtime 的调度空间（notebook 的受控 lane 用的同一组参数）
model = ct.models.MLModel(model_path, compute_units=compute_units)
spec_input = model.input_description["input"]
x = np.random.default_rng(0).random((1, 3, 224, 224)).astype(np.float32)

model.predict({"input": x})  # warmup：触发 .mlmodelc 编译与首次加载（冷启动路径）
t0 = time.perf_counter()
for _ in range(n_iter):
    model.predict({"input": x})
ms = (time.perf_counter() - t0) / n_iter * 1e3
print(f"input spec: {spec_input}; {n_iter} predicts, avg {ms:.2f} ms/predict")
