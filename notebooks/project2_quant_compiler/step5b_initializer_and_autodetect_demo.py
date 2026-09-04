"""Step 5 追问配套 Demo：initializer 在量化里的角色，以及 quantize_static 的"自动识别"。

    Part A  initializer 的四种角色（用真实产物清点，不只 scale / zero_point）
    Part B  quantize_static 怎么「自动」认出 Gemm：注册表 + QDQGemm.quantize()
    Part C  weight per-channel / activation per-tensor 是谁决定的（含反例）
    Part D  最硬的一条：shape 未知时 quantizer 会**静默跳过** tensor

跑法（仓库根目录）：

    uv run python notebooks/project2_quant_compiler/step5b_initializer_and_autodetect_demo.py
"""

from __future__ import annotations

import logging
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
TMP = Path(tempfile.mkdtemp(prefix="step5b."))
QDQ_ONNX = ROOT / "artifacts" / "project2_quantization" / "tiny_token_mlp_w8a8_qdq.onnx"
FP32_ONNX = ROOT / "artifacts" / "project2_quantization" / "tiny_token_mlp_fp32_preprocessed.onnx"

K, N = 4, 8
rng = np.random.default_rng(7)


def rule(title: str) -> None:
    print("\n" + "=" * 82 + f"\n{title}\n" + "=" * 82)


class Reader(CalibrationDataReader):
    def __init__(self, samples):
        self.samples = samples
        self.rewind()

    def get_next(self):
        return next(self.it, None)

    def rewind(self):
        self.it = iter({"X": s} for s in self.samples)


# ---------------------------------------------------------------------------
def part_a_initializer_roles() -> None:
    rule("Part A  initializer 在量化里的四种角色")

    fp32 = onnx.load(FP32_ONNX)
    qdq = onnx.load(QDQ_ONNX)
    print(f"FP32 预处理后 : {len(fp32.graph.initializer)} 个 initializer")
    print(f"  {[i.name for i in fp32.graph.initializer]}")
    print(f"\nW8A8 QDQ      : {len(qdq.graph.initializer)} 个 initializer")
    print("  按角色归类：")

    buckets = {"量化后的权重": [], "权重的 scale / zp": [], "激活的 scale / zp": [],
               "bias(int32) + 它的 scale / zp": [], "其他": []}
    for i in qdq.graph.initializer:
        name = i.name
        if name.endswith("_quantized") and "bias" not in name:
            buckets["量化后的权重"].append(name)
        elif "bias_quantized" in name or ("bias" in name and ("scale" in name or "zero_point" in name)):
            buckets["bias(int32) + 它的 scale / zp"].append(name)
        elif ".weight" in name and ("scale" in name or "zero_point" in name):
            buckets["权重的 scale / zp"].append(name)
        elif "scale" in name or "zero_point" in name:
            buckets["激活的 scale / zp"].append(name)
        else:
            buckets["其他"].append(name)

    for role, names in buckets.items():
        print(f"\n  [{role}]  {len(names)} 个")
        for n in names:
            init = next(i for i in qdq.graph.initializer if i.name == n)
            arr = numpy_helper.to_array(init)
            if arr.ndim == 0:
                gran = "标量 → per-tensor"
            elif role == "量化后的权重":
                gran = "整块权重"
            else:
                gran = f"向量[{arr.size}] → per-channel"
            print(f"     {n:<38} {str(list(arr.shape)):<12} {str(arr.dtype):<8} {gran}")

    fp32_names = {i.name for i in fp32.graph.initializer}
    qdq_names = {i.name for i in qdq.graph.initializer}
    print(f"\n  FP32 里有、量化后消失的 initializer : {sorted(fp32_names - qdq_names)}")
    print("  => 原始 float 权重被**替换**掉了，不是并存：fc1.weight(float32) 变成了")
    print("     fc1.weight_quantized(int8) + fc1.weight_scale + fc1.weight_zero_point 三件。")

    print("\n  为什么 scale 必须当 initializer 存？因为 QuantizeLinear 把它当**图输入**用：")
    for n in qdq.graph.node:
        if n.op_type == "QuantizeLinear":
            print(f"     {n.name:<45} inputs={list(n.input)}")
            break
    print("  => runtime 找不到 shape 的 tensor 只能靠 shape inference 补全，")
    print("     所以 scale 这种 0 维标量也必须进 value_info / initializer 体系。")


# ---------------------------------------------------------------------------
def build_gemm(trans_b: int, name: str) -> onnx.ModelProto:
    """trans_b=1 → 权重 [N,K] 且转置（PyTorch Linear 导出的形态）"""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, K])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, N])
    W = rng.normal(size=(N, K) if trans_b else (K, N)).astype(np.float32)
    B = rng.normal(size=(N,)).astype(np.float32)
    node = helper.make_node("Gemm", ["X", "W", "Bias"], ["Y"], name=name)
    if trans_b:
        node.attribute.append(helper.make_attribute("transB", 1))
    model = helper.make_model(
        helper.make_graph([node], name, [X], [Y],
                          initializer=[numpy_helper.from_array(W, "W"),
                                       numpy_helper.from_array(B, "Bias")]),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def part_b_auto_detect() -> None:
    rule("Part B  quantize_static 怎么「自动」认出 Gemm")

    from onnxruntime.quantization.registry import QDQRegistry, QLinearOpsRegistry
    print("  ORT 内部就是两张查表（onnxruntime/quantization/registry.py）：")
    print(f"    QDQRegistry 支持的 op ({len(QDQRegistry)} 个): {sorted(QDQRegistry)}")
    print(f"    QLinearOpsRegistry 支持的 op ({len(QLinearOpsRegistry)} 个): {sorted(QLinearOpsRegistry)}")

    print("\n  Gemm 命中 QDQRegistry['Gemm'] = QDQGemm，它的 quantize() 核心逻辑：")
    print("""
    def quantize(self):
        node = self.node
        self.quantizer.quantize_activation_tensor(node.input[0])      # 输入激活
        self.quantizer.quantize_activation_tensor(node.output[0])     # 输出激活

        is_weight_per_channel, weight_axis = self.quantizer.is_tensor_per_channel(
            node.input[1], default_axis=0 if is_B_transposed(node) else 1
        )
        if is_weight_per_channel:
            self.quantizer.quantize_weight_tensor_per_channel(node.input[1], weight_axis)
        else:
            self.quantizer.quantize_weight_tensor(node.input[1])

        if len(node.input) == 3:
            if self.quantizer.is_input_a_initializer(node.input[2]):
                self.quantizer.quantize_bias_tensor(...)               # bias 特殊处理
            else:
                logging.warning(f"Bias of Gemm node ... is not constant ...")
    """)
    print("  => 是「按 op_type 查表 + 按输入位置约定」的硬编码，不是从数学语义推出来的。")
    print("     input[0]=activation, input[1]=weight, input[2]=bias 是 Gemm 的 schema 约定。")

    print("\n  实测：Gemm 节点被展开成的 Q/DQ 链")
    qdq = onnx.load(QDQ_ONNX)
    for n in qdq.graph.node:
        if n.op_type == "Gemm" and "fc1" in n.name:
            print(f"     {n.op_type} {n.name}  inputs={list(n.input)}")
    print("     ...周围：")
    for n in qdq.graph.node:
        if any("fc1" in s for s in n.input) or (n.output and "fc1" in n.output[0]):
            print(f"       {n.op_type:<18} {n.name:<45} in={list(n.input)} out={list(n.output)}")

    print("\n  不在注册表里的 op 会怎样？QDQOperatorBase 回退：")
    print("""
    def quantize(self):
        for tensor_name in chain(node.input, node.output):   # 无脑给所有输入输出插 Q/DQ
            self.quantizer.quantize_activation_tensor(tensor_name)
    """)
    print("  => 所以 Tanh / Sigmoid / Add / Mul 这类不在 QDQRegistry 里的 op 也会拿到 Q/DQ，")
    print("     但那只是「标记」，backend 未必有对应 INT8 kernel —— Q/DQ 多 ≠ 快。")


# ---------------------------------------------------------------------------
def part_c_per_channel_axis() -> None:
    rule("Part C  weight per-channel / activation per-tensor 是怎么决定的")

    samples = [rng.normal(size=(3, K)).astype(np.float32) for _ in range(16)]

    cases = [
        ("transB=1, per_channel=True ", dict(trans_b=1), dict(per_channel=True)),
        ("transB=0, per_channel=True ", dict(trans_b=0), dict(per_channel=True)),
        ("transB=1, per_channel=False", dict(trans_b=1), dict(per_channel=False)),
    ]
    print(f"  权重 shape: transB=1 时 W=[{N},{K}]；transB=0 时 W=[{K},{N}]")
    print(f"  QDQGemm 的默认 axis 规则：default_axis = 0 if is_B_transposed(node) else 1\n")
    for tag, build_kw, quant_kw in cases:
        m = build_gemm(name="g", **build_kw)
        src = TMP / f"c_{abs(hash(tag))}.onnx"
        dst = TMP / f"c_{abs(hash(tag))}_q.onnx"
        onnx.save(m, src)
        quantize_static(src, dst, Reader(samples), quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                        calibrate_method=CalibrationMethod.MinMax, **quant_kw)
        q = onnx.load(dst)
        arr = {i.name: numpy_helper.to_array(i) for i in q.graph.initializer}
        w_shape = arr["W_quantized"].shape
        ws = arr["W_scale"]
        if ws.ndim == 0:
            axis_desc = "无 axis（per-tensor）"
        else:
            hits = [i for i, d in enumerate(w_shape) if d == ws.shape[0]]
            axis_desc = f"axis={hits}（W.shape[{hits[0]}]={ws.shape[0]}）"
        print(f"  [{tag}]  W_quantized{tuple(w_shape)}  "
              f"W_scale{str(list(ws.shape)):<6}  "
              f"X_scale{str(list(arr['X_scale'].shape)):<5}  "
              f"Y_scale{str(list(arr['Y_scale'].shape)):<5}  -> {axis_desc}")

    print("\n  => 三个事实：")
    print("     1) activation 永远是 per-tensor（QDQGemm 只调 quantize_activation_tensor，无 axis 参数）")
    print("     2) weight 是否 per-channel 由你传的 per_channel=True 决定")
    print("     3) 是哪个 axis，由 op 自己给 default_axis，而它依赖 transB 这个 attribute")
    print("        —— 这又回到 Step 5：attribute 和 shape 都要已知，才能定 axis。")

    print("\n  bias 的 scale 是推出来的，不是统计出来的：")
    q = onnx.load(QDQ_ONNX)
    arr = {i.name: numpy_helper.to_array(i) for i in q.graph.initializer}
    for tag, w, b in [("fc1", "fc1.weight", "fc1.bias"), ("head", "head.weight", "head.bias")]:
        in_sc = arr["input_scale"] if tag == "fc1" else arr["/relu2/Relu_output_0_scale"]
        expect = in_sc * arr[f"{w}_scale"]
        actual = arr[f"{b}_quantized_scale"]
        print(f"     {tag:<6} max|{b}_quantized_scale - input_scale*{w}_scale| = "
              f"{np.abs(actual - expect).max():.3e}")
    print("  => bias 量化成 int32，scale = input_scale * weight_scale（* beta），zero_point = 0。")
    print("     它待在 INT32 累加器里，不需要自己的统计量。")


# ---------------------------------------------------------------------------
def part_d_silent_skip() -> None:
    rule("Part D  quantize_static 会自己兜底做 shape inference —— 但兜底的是弱版本")

    print("  1) 兜底确实存在（onnxruntime/quantization/quantize.py）：")
    print("""
    pre_processed: bool = model_has_pre_process_metadata(model)
    if not pre_processed:
        logging.warning("Please consider to run pre-processing before quantization. ...")
        model = load_model_with_shape_infer(Path(model_input))    # 内部兜底
    """)
    print("  2) 但兜底跑的是 onnx.shape_inference.infer_shapes_path（静态），")
    print("     quant_pre_process 跑的是 SymbolicShapeInference（符号）。差别实测如下：")

    from onnxruntime.quantization.shape_inference import quant_pre_process
    from onnxruntime.quantization.qdq_quantizer import QDQQuantizer
    import inspect

    sys.path.insert(0, str(Path(__file__).parent))
    from step5_preprocess_contract_demo import build_irregular_model

    raw = TMP / "d_raw.onnx"
    pre = TMP / "d_pre.onnx"
    onnx.save(build_irregular_model(), raw)
    quant_pre_process(raw, pre)
    samples = [rng.normal(size=(3, K)).astype(np.float32) for _ in range(16)]

    for tag, src in [("raw · ORT 内部兜底", raw), ("pre · 符号推理", pre)]:
        dst = TMP / f"d_{tag[:3]}.onnx"

        class Capture(logging.Handler):
            def __init__(self):
                super().__init__()
                self.msgs = []

            def emit(self, record):
                self.msgs.append(record.getMessage())

        cap = Capture()
        lg = logging.getLogger()
        lg.setLevel(logging.WARNING)
        lg.addHandler(cap)
        try:
            quantize_static(src, dst, Reader(samples), quant_format=QuantFormat.QDQ,
                            per_channel=False, activation_type=QuantType.QInt8,
                            weight_type=QuantType.QInt8, calibrate_method=CalibrationMethod.MinMax)
        finally:
            lg.removeHandler(cap)
            lg.setLevel(logging.ERROR)

        counts = Counter(n.op_type for n in onnx.load(dst).graph.node)
        shapes = {v.name: [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim]
                  for v in onnx.load(dst).graph.value_info if v.name in ("A", "A2", "R", "Y")}
        print(f"\n  [{tag}] nodes={len(onnx.load(dst).graph.node)}  "
              f"Q={counts['QuantizeLinear']}  DQ={counts['DequantizeLinear']}")
        print(f"     关键 tensor 的 shape: {shapes}")
        pre_warn = [m_ for m_ in cap.msgs if "pre-processing" in m_]
        skip_warn = [m_ for m_ in cap.msgs if "failed to infer" in m_]
        print(f"     告警: pre-processing 提示 {len(pre_warn)} 条, 静默跳过 {len(skip_warn)} 条")

    print("\n  => R 这个 tensor：静态兜底推不出（unk__0/unk__1），符号推理推得出（batch/8）。")
    print("     结果就是 31 nodes / Q7 DQ9  vs  14 nodes / Q4 DQ7。")

    print("\n  3) 还有一个更危险的分支：_is_tensor_quantizable 的最后一档")
    src = inspect.getsource(QDQQuantizer._is_tensor_quantizable)
    print("\n".join("     " + l for l in src.splitlines()[-14:]))
    print("\n     tensor 既不是 initializer、又不在 value_infos 里 → 只打一行 warning 就返回 False，")
    print("     该 tensor 被静默排除在量化之外。不抛异常、不改 exit code。")
    print("     诚实边界：我这组 toy 图没触发它（ORT 的兜底把 shape 补回来了），")
    print("     但自定义 domain 算子、shape inference 没实现的 op 会命中这条路径 ——")
    print("     这正是 AGENTS.md 里那条「exit 0 不证明任何事」在量化场景的具体形态。")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.ERROR)   # 先静音；Part D 会专门抓 WARNING
    part_a_initializer_roles()
    part_b_auto_detect()
    part_c_per_channel_axis()
    part_d_silent_skip()
    print(f"\n\n产物目录：{TMP}")
