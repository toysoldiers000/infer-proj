"""Checks 1 配套实验：optimized graph 节点数下降到底意味着什么。

回答三个问题：
    1. raw QDQ 的 23 个节点里，compute op 有几个是 int8 的？（答：0 个）
    2. 23 → 11 少了的是 Q/DQ，还是算子被回退了？（答：少了 12 个 Q/DQ，Gemm 全部命中 QGemm）
    3. 还剩 3 个 Q + 3 个 DQ 是谁造成的？（答：ActivationSymmetric=True 导致 Relu 无法被消除）

跑法（仓库根目录）：

    uv run python notebooks/project2_quant_compiler/check1_boundary_cost_demo.py
"""

from __future__ import annotations

import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper
from torch import nn
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
TMP = Path(tempfile.mkdtemp(prefix="check1."))
QDQ_ONNX = ROOT / "artifacts" / "project2_quantization" / "tiny_token_mlp_w8a8_qdq.onnx"
OPT_ONNX = ROOT / "artifacts" / "project2_quantization" / "tiny_token_mlp_w8a8_ort_optimized.onnx"

HIDDEN, FFN, CLASS = 256, 512, 4
rng = np.random.default_rng(7)


def rule(t: str) -> None:
    print("\n" + "=" * 84 + f"\n{t}\n" + "=" * 84)


class Reader(CalibrationDataReader):
    def __init__(self, samples):
        self.samples = samples
        self.rewind()

    def get_next(self):
        return next(self.it, None)

    def rewind(self):
        self.it = iter({"input": x} for x in self.samples)


# ---------------------------------------------------------------------------
def part_a_where_did_nodes_go() -> None:
    rule("Part A  23 → 11 少了什么：不是算子回退，是 Q/DQ 被消除")

    qdq = onnx.load(QDQ_ONNX).graph
    opt = onnx.load(OPT_ONNX).graph
    COMPUTE = {"Gemm", "QGemm", "Relu", "Conv", "MatMul", "QLinearMatMul", "QLinearAdd"}

    def split(g):
        c = Counter(n.op_type for n in g.node)
        qdq_ops = c["QuantizeLinear"] + c["DequantizeLinear"]
        compute = sum(v for k, v in c.items() if k in COMPUTE)
        return c, qdq_ops, compute

    for tag, g in [("FP32 preprocessed", onnx.load(
            ROOT / "artifacts/project2_quantization/tiny_token_mlp_fp32_preprocessed.onnx").graph),
                   ("W8A8 raw QDQ", qdq), ("W8A8 ORT optimized", opt)]:
        c, n_qdq, n_comp = split(g)
        print(f"\n  [{tag}]")
        print(f"     总节点 {len(g.node):>3}  =  Q/DQ {n_qdq:>3}  +  compute {n_comp}")
        print(f"     op counts: {dict(c)}")

    print("\n  => 从 raw QDQ 到 optimized：Q/DQ 从 18 降到 6，compute 从 5 仍是 5。")
    print("     少的 12 个全是 Q/DQ。**节点数下降是融合成功的证据，不是算子丢了。**")

    qg = Counter(n.op_type for n in qdq.node)
    og = Counter(n.op_type for n in opt.node)
    print(f"\n  Gemm {qg['Gemm']} → QGemm {og['QGemm']}：命中率 "
          f"{og['QGemm']}/{qg['Gemm']} = {og['QGemm'] / max(qg['Gemm'], 1):.0%}，fallback = "
          f"{qg['Gemm'] - og['QGemm']}")
    print("  **真正的 fallback 长这样：optimized 图里仍有 float Gemm，或者出现未融合的 DQ→Gemm→Q。**")

    print("\n  raw QDQ 阶段最关键的一列：quantized compute ops = 0")
    print("  => raw QDQ 里 Gemm 还是 float Gemm，Q/DQ 只是把 int8 数据 dequant 回 float 再算。")
    print("     **Q/DQ 是 precision contract，不是加速证明。**")


# ---------------------------------------------------------------------------
def part_b_optimized_chain() -> None:
    rule("Part B  optimized 图的真实链路：int8 链被 Relu 切成 3 段")
    opt = onnx.load(OPT_ONNX).graph
    print("  " + " → ".join(n.op_type for n in opt.node))
    print("\n  逐节点输入（注意 QGemm 有 9 个输入，weight/bias 的 DQ 全被折叠进去了）：")
    for n in opt.node:
        print(f"    {n.op_type:<17} {list(n.input)}")

    print("\n  数一数 DQ→Relu→Q 出现了几次：")
    seq = [n.op_type for n in opt.node]
    trips = sum(1 for i in range(len(seq) - 2)
                if seq[i] == "DequantizeLinear" and seq[i + 1] not in
                ("QuantizeLinear",) and seq[i + 2] == "QuantizeLinear")
    print(f"    {trips} 次 int8 → float → int8 往返")
    print("    每次往返 = 一次 dequant（int32→fp32 乘加）+ 一次 requant（带 rounding + clipping）")


# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2, self.head = nn.Linear(HIDDEN, FFN), nn.Linear(FFN, FFN), nn.Linear(FFN, CLASS)

    def forward(self, x):
        return self.head(torch.relu(self.fc2(torch.relu(self.fc1(x)))))


def part_c_symmetric_is_the_cause() -> None:
    rule("Part C  根因：ActivationSymmetric=True 让 ReLU 无法被免费实现")

    torch.manual_seed(7)
    model = MLP().eval()
    src = TMP / "mlp.onnx"
    data = [rng.normal(size=(32, HIDDEN)).astype(np.float32) for _ in range(24)]
    with torch.no_grad():
        torch.onnx.export(model, (torch.from_numpy(data[0]),), src,
                          input_names=["input"], output_names=["logits"],
                          opset_version=18, dynamo=False)
    pre = TMP / "mlp_pre.onnx"
    quant_pre_process(src, pre)

    xb = rng.normal(size=(32, HIDDEN)).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(xb)).numpy()

    print(f"  {'配置':<22}{'QDQ':>5}{'opt':>5}  链路")
    print(f"  {'-' * 78}")
    results = {}
    for tag, symmetric in [("非对称(ORT 默认)", False), ("对称(你 notebook)", True)]:
        q = TMP / f"mlp_{symmetric}.onnx"
        opt = TMP / f"mlp_{symmetric}_opt.onnx"
        extra = {"ActivationSymmetric": True, "WeightSymmetric": True} if symmetric else {}
        quantize_static(pre, q, Reader(data), quant_format=QuantFormat.QDQ,
                        per_channel=True, activation_type=QuantType.QInt8,
                        weight_type=QuantType.QInt8,
                        calibrate_method=CalibrationMethod.MinMax, extra_options=extra)

        # 关键证据：post-Relu 激活的 zero_point
        arr = {i.name: numpy_helper.to_array(i) for i in onnx.load(q).graph.initializer}
        zp = {k: int(np.asarray(v).reshape(-1)[0]) for k, v in arr.items()
              if "zero_point" in k and "weight" not in k and "bias" not in k}

        o = ort.SessionOptions()
        o.optimized_model_filepath = str(opt)
        o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        o.log_severity_level = 3
        sess = ort.InferenceSession(str(q), o, providers=["CPUExecutionProvider"])
        del sess
        chain = " → ".join(n.op_type for n in onnx.load(opt).graph.node)

        s = ort.InferenceSession(str(opt), providers=["CPUExecutionProvider"])
        for _ in range(60):
            s.run(None, {"input": xb})
        t0 = time.perf_counter()
        for _ in range(600):
            s.run(None, {"input": xb})
        ms = (time.perf_counter() - t0) / 600 * 1000
        err = np.abs(ref - s.run(None, {"input": xb})[0]).max()
        results[tag] = (ms, err)

        print(f"  {tag:<20}{len(onnx.load(q).graph.node):>5}{len(onnx.load(opt).graph.node):>5}  {chain}")
        print(f"      激活 zero_point: {zp}")
        print(f"      延迟 {ms:.4f} ms     max|INT8-FP32| = {err:.6f}（参考量级 {np.abs(ref).max():.4f}）")

    fast, slow = sorted(results.items(), key=lambda kv: kv[1][0])
    ratio = slow[1][0] / fast[1][0]
    print(f"\n  => {slow[0]} 比 {fast[0]} 慢 {ratio:.2f}×")

    print("\n  机制解释（这是本实验最值钱的一条）：")
    print("""
    int8 的取值范围是 [-128, 127]。ReLU 后激活全部 >= 0，所以 calibration 得到的
    post-ReLU 范围是 [0, max]。

    非对称量化：zero_point = -128（实测确认）。于是
        q = clip(round(x / s) + (-128), -128, 127)
    任何 x < 0 都被 clip 到 -128，反量化回来是 (-128 - (-128)) * s = 0.0。
    **ReLU 被 QuantizeLinear 的饱和截断免费实现了**，ORT 因此可以把 Relu 节点整个删掉。

    对称量化：zero_point = 0。负数会落到正常的负数码上，clip 不再等价于 ReLU，
    ORT 只能老老实实保留 float Relu，于是每个 Relu 前后各多一对 DQ/Q。
    """)
    print("  权衡提示：对称量化在部分 NPU 上更快（zero_point=0 省掉一次减法），")
    print("     所以不能无脑换。要按目标 backend 的 kernel 契约决定，并用 profiler 取证。")


# ---------------------------------------------------------------------------
def part_d_checklist() -> None:
    rule("Part D  判读 optimized graph 的 4 条 checklist（可复用到任何 backend）")
    print("""
    1) 命中率：原 float compute op 数 vs 优化后 int8 kernel 数。
       你的数据：Gemm 3 → QGemm 3，命中率 100%，fallback = 0。

    2) Q/DQ 残余：理想是入口 1 对、出口 1 对。多出来的每一对都是一次转换边界。
       你的数据：3 对，多出 2 对，对应 2 个 Relu。

    3) 边界形态：找 DQ → <非量化 op> → Q 这种三连。出现一次就说明 int8 链断一次。
       你的数据：出现 2 次。

    4) 证据强度：optimized graph 属于「静态产物」级证据，强于 kernel 名字猜测，
       弱于 runtime core placement / profiler 时间线。
       下一步必须补：逐层数值比对（Checks 2）+ profiler + 统一 benchmark。
    """)
    print("  公司 N93X 上对应的是 compiler 的 --dump-partition / --dump-lowering / fallback report：")
    print("  看哪些 node 进了 NPU subgraph、哪些 fallback 到 CPU/Fermat，口径完全一致。")


if __name__ == "__main__":
    part_a_where_did_nodes_go()
    part_b_optimized_chain()
    part_c_symmetric_is_the_cause()
    part_d_checklist()
    print(f"\n\n产物目录：{TMP}")
