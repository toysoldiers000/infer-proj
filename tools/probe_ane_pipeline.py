"""
PyTorch -> Core ML(mlprogram) -> .mlmodelc -> ANE 的三阶段探针。

阶段划分（本脚本要实证的东西）：
  A. convert  : PyTorch / TorchScript -> MIL -> mlprogram(.mlpackage)      [离线, 与设备无关]
  B. compile  : .mlpackage -> .mlmodelc                                    [离线或端上, 与具体 shape 无关]
  C. specialize: .mlmodelc + compute_units + hints + 具体 shape -> 可执行程序  [端上, 隐式发生在 load]
  D. predict  : 稳态推理

用法:
  python probe_ane_pipeline.py            # 全流程跑一遍并打印报告
  python probe_ane_pipeline.py load-only  # 只测 load(specialize)+predict, 用于观察 specialization 缓存
"""
import os
import sys
import time
import shutil
import subprocess

os.environ.setdefault("TQDM_DISABLE", "1")  # 关掉 coremltools 的进度条，保持输出干净

import numpy as np
import torch
import torch.nn as nn
import coremltools as ct

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "artifacts", "_ane_pipeline_probe")
os.makedirs(OUT, exist_ok=True)

TARGET = ct.target.iOS18  # multifunction / specializationStrategy 需要 iOS18 / macOS15+


# ---------------------------------------------------------------- 1. 模型
class Net(nn.Module):
    """base(共享) + head(每个 adapter 不同)。LoRA 场景下 head 即 adapter 增量。"""
    def __init__(self, d=1024):
        super().__init__()
        self.base = nn.Linear(d, d)
        self.head = nn.Linear(d, d)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.head(self.act(self.base(x)))


def make_adapters(seed=0):
    torch.manual_seed(seed)
    base = Net()
    a1, a2 = Net(), Net()
    a1.base.load_state_dict(base.base.state_dict())   # 共享 base
    a2.base.load_state_dict(base.base.state_dict())
    torch.manual_seed(seed + 1)
    a1.head.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)
    torch.manual_seed(seed + 2)
    a2.head.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)
    return a1.eval(), a2.eval()


def convert(model, name, dynamic=False):
    ex = torch.rand(1, 1024)
    if dynamic:
        inputs = [ct.TensorType(name="x", shape=ct.Shape(shape=(ct.RangeDim(1, 8), 1024)))]
    else:
        inputs = [ct.TensorType(name="x", shape=(1, 1024))]
    t0 = time.perf_counter()
    mlmodel = ct.convert(
        torch.jit.trace(model, ex),
        inputs=inputs,
        outputs=[ct.TensorType(name="y")],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=TARGET,
    )
    t_convert = time.perf_counter() - t0
    path = os.path.join(OUT, f"{name}.mlpackage")
    mlmodel.save(path)
    return path, t_convert, mlmodel


# ---------------------------------------------------------------- 2. compile
def compile_pkg(pkg_path):
    """compile 阶段：mlpackage -> mlmodelc。产物与 shape 无关。"""
    dst = pkg_path.replace(".mlpackage", ".mlmodelc")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    t0 = time.perf_counter()
    out = ct.models.utils.compile_model(pkg_path, destination_path=dst)
    t_compile = time.perf_counter() - t0
    return out, t_compile


def compile_cli(pkg_path):
    """等价的官方 CLI：xcrun coremlcompiler compile（Xcode build 阶段调的就是它）。"""
    t0 = time.perf_counter()
    r = subprocess.run(["xcrun", "coremlcompiler", "compile", pkg_path, OUT],
                       capture_output=True, text=True)
    return r.returncode, time.perf_counter() - t0, (r.stderr or r.stdout).strip()[:300]


# ---------------------------------------------------------------- 3. specialize + predict
def specialize_and_predict(mlmodelc, compute_units, hints=None, shapes=((1, 1024),), n_warm=20, n_run=100):
    """load = specialization；首次 predict 往往还会触发 shape 绑定。分别计时。"""
    t0 = time.perf_counter()
    m = ct.models.CompiledMLModel(mlmodelc, compute_units=compute_units, optimization_hints=hints)
    t_specialize = time.perf_counter() - t0

    res = {}
    for i, shp in enumerate(shapes):
        x = np.random.rand(*shp).astype(np.float32)
        t0 = time.perf_counter()
        m.predict({"x": x})
        t_first = time.perf_counter() - t0

        for _ in range(n_warm):
            m.predict({"x": x})
        ts = []
        for _ in range(n_run):
            t0 = time.perf_counter()
            m.predict({"x": x})
            ts.append((time.perf_counter() - t0) * 1e3)
        res[(i, shp)] = {
            "first_ms": t_first * 1e3,
            "p50_ms": float(np.percentile(ts, 50)),
            "p95_ms": float(np.percentile(ts, 95)),
            "mean_ms": float(np.mean(ts)),
        }
    return t_specialize * 1e3, res


def dir_size(p):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs)


def tree(p, max_depth=2, prefix=""):
    lines = []
    p = p.rstrip("/")
    def walk(cur, depth):
        if depth > max_depth:
            return
        for name in sorted(os.listdir(cur)):
            full = os.path.join(cur, name)
            lines.append("  " * depth + name + ("/" if os.path.isdir(full) else ""))
            if os.path.isdir(full):
                walk(full, depth + 1)
    walk(p, 0)
    return "\n".join(lines[:60])


def human(n):
    return f"{n/1024:.0f}KB" if n < 1024 * 1024 else f"{n/1024/1024:.2f}MB"


def main():
    only_load = "load-only" in sys.argv
    print("=" * 72)
    print(f"coremltools {ct.__version__} | macOS {platform_version()} | {os.uname().machine}")
    print("=" * 72)

    # ---- A. convert
    a1, a2 = make_adapters()
    p_static, t_conv, mlm = convert(a1, "adapter_1")
    print(f"\n[A] convert  PyTorch -> mlprogram            {t_conv*1e3:8.1f} ms")
    print(f"    产物 {p_static}  {human(dir_size(p_static))}")

    # ---- B. compile
    mlmodelc, t_comp = compile_pkg(p_static)
    rc, t_cli, msg = compile_cli(p_static)
    print(f"\n[B] compile  mlpackage -> mlmodelc")
    print(f"    ct.models.utils.compile_model   {t_comp*1e3:8.1f} ms")
    print(f"    xcrun coremlcompiler compile    {t_cli*1e3:8.1f} ms  (rc={rc})")
    print(f"    产物 {mlmodelc}")
    print("    .mlmodelc 内部结构:")
    for line in tree(mlmodelc).splitlines():
        print("      " + line)

    # 编译产物里可以直接读到文本 MIL —— 证明 compile 之后图还在、且未被 shape 绑定
    mil = [os.path.join(r, f) for r, _, fs in os.walk(mlmodelc) for f in fs if f.endswith(".mil")]
    if mil:
        with open(mil[0]) as fh:
            txt = fh.read()
        print(f"\n    compile 产物内含文本 MIL: {mil[0]}  ({len(txt)} chars)")
        print(f"    MIL 程序头: {txt.strip().splitlines()[0][:80]}")

    if only_load:
        report_load(mlmodelc)
        return

    # ---- C/D. specialize + predict，跨 compute units
    print(f"\n[C] specialize + [D] predict   (load = 触发 specialization)")
    print(f"    {'compute_units':<14}{'specialize':>12}{'first_ms':>11}{'p50_ms':>10}{'p95_ms':>10}")
    for cu_name in ["CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"]:
        cu = getattr(ct.ComputeUnit, cu_name)
        try:
            ts, res = specialize_and_predict(mlmodelc, cu)
            r = res[(1, 1024)]
            print(f"    {cu_name:<14}{ts:>10.1f}ms{r['first_ms']:>10.1f} "
                  f"{r['p50_ms']:>9.2f} {r['p95_ms']:>9.2f}")
        except Exception as e:
            print(f"    {cu_name:<14}FAILED: {str(e)[:70]}")

    # ---- specializationStrategy 对比
    print(f"\n[C] specializationStrategy 对比  (CPU_AND_NE)")
    for strat in [ct.SpecializationStrategy.Default, ct.SpecializationStrategy.FastPrediction]:
        ts, res = specialize_and_predict(mlmodelc, ct.ComputeUnit.CPU_AND_NE,
                                         {"specializationStrategy": strat})
        r = res[(1, 1024)]
        print(f"    {strat.name:<16} specialize={ts:>8.1f}ms  first={r['first_ms']:>7.1f}ms  "
              f"p50={r['p50_ms']:>6.2f}ms")

    # ---- 动态 shape：首次遇到新 shape 要重新 specialize
    p_dyn, _, _ = convert(a1, "adapter_1_dyn", dynamic=True)
    mlmodelc_dyn, _ = compile_pkg(p_dyn)
    print(f"\n[C] 动态 shape(1..8 x 1024)：同一份 mlmodelc，换 shape 的代价  (CPU_AND_NE)")
    ts, res = specialize_and_predict(mlmodelc_dyn, ct.ComputeUnit.CPU_AND_NE,
                                     shapes=((1, 1024), (4, 1024), (1, 1024)))
    print(f"    specialize(load) {ts:.1f}ms")
    for (i, shp), r in res.items():
        print(f"    第{i+1}个shape{shp}  first={r['first_ms']:>7.1f}ms  p50={r['p50_ms']:>6.2f}ms")
    dmil = [os.path.join(r_, f) for r_, _, fs in os.walk(mlmodelc_dyn) for f in fs if f.endswith(".mil")]
    if dmil:
        with open(dmil[0]) as fh:
            for line in fh.read().splitlines():
                if "func main" in line or "x:" in line or "tensor<" in line:
                    print(f"    MIL: {line.strip()[:90]}")

    # ---- LoRA / 多 function
    report_multifunction(a2)

    print("\n再跑一次:  python probe_ane_pipeline.py load-only   （观察 specialization 磁盘缓存命中）")


def report_load(mlmodelc):
    print(f"\n[C] 仅测 load(specialize)+predict   {mlmodelc}")
    for i in range(3):
        ts, res = specialize_and_predict(mlmodelc, ct.ComputeUnit.CPU_AND_NE, n_warm=5, n_run=30)
        r = next(iter(res.values()))
        print(f"    第{i+1}次 load: specialize={ts:>8.1f}ms   p50={r['p50_ms']:.2f}ms")


def report_multifunction(a2):
    """LoRA 阶段：多 adapter 合并成一个 mlpackage，权重共享，运行时按 function_name 选。"""
    print(f"\n[LoRA] 多 adapter -> 单 mlpackage（权重共享），运行时按 function_name 选择")
    p1 = os.path.join(OUT, "adapter_1.mlpackage")
    p2, _, _ = convert(a2, "adapter_2")

    desc = ct.utils.MultiFunctionDescriptor()
    desc.add_function(p1, src_function_name="main", target_function_name="lora_A")
    desc.add_function(p2, src_function_name="main", target_function_name="lora_B")
    desc.default_function_name = "lora_A"

    multi = os.path.join(OUT, "multifunction.mlpackage")
    if os.path.exists(multi):
        shutil.rmtree(multi)
    t0 = time.perf_counter()
    ct.utils.save_multifunction(desc, multi)
    t_save = time.perf_counter() - t0

    s1, s2, sm = dir_size(p1), dir_size(p2), dir_size(multi)
    print(f"    adapter_1 {human(s1)} + adapter_2 {human(s2)} = {human(s1+s2)}")
    print(f"    multifunction {human(sm)}   -> 共享权重省下 {human(max(0, s1+s2-sm))}"
          f"  (save {t_save*1e3:.0f}ms)")

    mlmodelc, t_comp = compile_pkg(multi)
    print(f"    compile multifunction {t_comp*1e3:.1f}ms -> {os.path.basename(mlmodelc)}")

    x = np.random.rand(1, 1024).astype(np.float32)
    for fn in ["lora_A", "lora_B"]:
        t0 = time.perf_counter()
        m = ct.models.CompiledMLModel(mlmodelc, compute_units=ct.ComputeUnit.CPU_AND_NE,
                                      function_name=fn)
        t_spec = time.perf_counter() - t0
        m.predict({"x": x})
        ts = []
        for _ in range(50):
            t0 = time.perf_counter()
            m.predict({"x": x})
            ts.append((time.perf_counter() - t0) * 1e3)
        print(f"    function={fn:<8} specialize={t_spec*1e3:>7.1f}ms  p50={np.percentile(ts,50):>6.2f}ms")


def platform_version():
    return subprocess.run(["sw_vers", "-productVersion"], capture_output=True, text=True).stdout.strip()


if __name__ == "__main__":
    main()
