"""Step 5 配套 Demo：quant_pre_process 到底在「预处理」什么。

四件事，各自独立可跑：

    Part 1  Protobuf 里 ONNX 图长什么样（字段号 + wire format 手解）
    Part 2  什么叫「规整的 graph」：同一数学的两份图，一份规整一份不规整
    Part 3  把 shape inference / graph optimization 与 quantization 分开，逐阶段看变化
    Part 4  为什么分开之后「更容易做 graph matching」

跑法（仓库根目录）：

    uv run python notebooks/project2_quant_compiler/step5_preprocess_contract_demo.py
"""

from __future__ import annotations

import subprocess
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
from onnxruntime.quantization.shape_inference import quant_pre_process

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
TMP = Path(tempfile.mkdtemp(prefix="step5_demo."))

M_DIM = "batch"          # 动态 batch：不规整的主要来源之一
K, N = 4, 8
rng = np.random.default_rng(7)
W_T = rng.normal(size=(N, K)).astype(np.float32)     # Gemm 用 [N, K] + transB=1
B_V = rng.normal(size=(N,)).astype(np.float32)
W_MM = np.ascontiguousarray(W_T.T)                   # MatMul 用 [K, N]


def rule(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def dim_of(vi) -> list[str]:
    t = vi.type.tensor_type
    if not t.HasField("shape"):
        return ["<no shape>"]
    return [d.dim_param if d.dim_param else str(d.dim_value) for d in t.shape.dim]


def describe(model: onnx.ModelProto, tag: str) -> None:
    g = model.graph
    print(f"\n[{tag}] nodes={len(g.node)}  initializers={len(g.initializer)}  value_info={len(g.value_info)}")
    print(f"  ops: {dict(Counter(n.op_type for n in g.node))}")
    print("  graph: " + "  ".join(f"{n.op_type}({n.output[0]})" for n in g.node))
    known = {v.name: dim_of(v) for v in g.value_info}
    init = {i.name for i in g.initializer}
    out = {o.name for o in g.output}
    unknown = [o for n in g.node for o in n.output if o not in known and o not in init and o not in out]
    print(f"  已知 shape 的中间 tensor: {known}")
    if unknown:
        print(f"  !! shape 未知的 tensor: {unknown}   <- quantizer 无法为它决定 per-channel axis")


# ---------------------------------------------------------------------------
# Part 1  Protobuf 里 ONNX 图长什么样
# ---------------------------------------------------------------------------
def build_regular_model() -> onnx.ModelProto:
    """规整版本：一个 Gemm 吃下 Wx+b，激活紧跟其后。"""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [M_DIM, K])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M_DIM, N])
    nodes = [
        helper.make_node("Gemm", ["X", "W_T", "B"], ["G"], name="gemm0", transB=1),
        helper.make_node("Relu", ["G"], ["Y"], name="relu0"),
    ]
    inits = [numpy_helper.from_array(W_T, "W_T"), numpy_helper.from_array(B_V, "B")]
    model = helper.make_model(
        helper.make_graph(nodes, "regular", [X], [Y], initializer=inits),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def part1_protobuf() -> None:
    rule("Part 1  ONNX = Protobuf：ModelProto → GraphProto → NodeProto / TensorProto")

    model = build_regular_model()
    path = TMP / "regular.onnx"
    onnx.save(model, path)

    # --- 1a. 字段编号 ---
    print("\n-- 1a. schema 视角：字段名只是给人看的，字节里只有 field number --")
    for proto_name, fields in [
        ("ModelProto", ["ir_version", "producer_name", "opset_import", "graph", "metadata_props"]),
        ("GraphProto", ["node", "initializer", "input", "output", "value_info"]),
        ("NodeProto", ["input", "output", "name", "op_type", "attribute"]),
        ("TensorProto", ["dims", "data_type", "name", "raw_data"]),
    ]:
        pairs = {f.name: f.number for f in getattr(onnx, proto_name).DESCRIPTOR.fields}
        print(f"  {proto_name:<12} " + "  ".join(f"{f}={pairs[f]}" for f in fields if f in pairs))

    print(f"\n  value_info 就是 shape inference 的产物：raw 模型里它通常是空的")
    print(f"  本例 raw value_info = {[(v.name, dim_of(v)) for v in model.graph.value_info]}")

    # --- 1b. wire format 手解 ---
    print("\n-- 1b. 字节视角：手写一个 varint 解码器，看前 64 字节 --")

    def read_varint(buf: bytes, pos: int) -> tuple[int, int]:
        """varint：每字节低 7 位是数据，最高位是「还有下一字节」标志（小端在前）。"""
        value, shift = 0, 0
        while True:
            byte = buf[pos]
            pos += 1
            value |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return value, pos
            shift += 7

    WIRE = {0: "varint", 1: "fixed64", 2: "length-delimited", 5: "fixed32"}
    raw = path.read_bytes()[:64]
    pos, shown = 0, 0
    while pos < len(raw) and shown < 8:
        key, pos = read_varint(raw, pos)
        field, wire = key >> 3, key & 7
        if wire == 0:
            val, pos = read_varint(raw, pos)
            print(f"  field={field:<2} wire={WIRE[wire]:<17} value={val}")
        elif wire == 2:
            length, pos = read_varint(raw, pos)
            payload = raw[pos:pos + length]
            hint = payload.decode("utf-8", "replace")[:24] if payload.isascii() else "<binary tensor data>"
            print(f"  field={field:<2} wire={WIRE[wire]:<17} len={length:<5} {hint!r}")
            pos += length
        else:
            pos += {1: 8, 5: 4}[wire]
            print(f"  field={field:<2} wire={WIRE[wire]}")
        shown += 1

    print("\n  对应回 schema：field 1 = ModelProto.ir_version(varint)，field 7 = ModelProto.graph(嵌套消息)")
    print("  => Protobuf 非自描述：拿到 .onnx 的人必须自己有 onnx.proto 才能解释 field 7 里是什么。")

    # --- 1c. CLI ---
    print("\n-- 1c. 不依赖 schema 也能看：protoc --decode_raw 只按 wire format 打印 --")
    print("  $ protoc --decode_raw < regular.onnx | head -26")
    try:
        out = subprocess.check_output(["protoc", "--decode_raw"], stdin=path.open("rb"), text=True)
        print("\n".join("    " + line for line in out.splitlines()[:26]))
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"    (protoc 不可用：{exc})")

    print("\n  CLI 速查：")
    print("    netron <model.onnx>          图形化看 graph（只做 inspection，不代表 device/kernel）")
    print("    check-model <model.onnx>     只查 schema 合法，不查数值正确、不查 backend 支持")


# ---------------------------------------------------------------------------
# Part 2  规整 vs 不规整
# ---------------------------------------------------------------------------
def build_irregular_model() -> onnx.ModelProto:
    """不规整版本：同一份数学，写成 quantizer 最不喜欢的形式。"""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [M_DIM, K])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M_DIM, N])

    nodes = [
        # (a) 冗余算子对：转过去又转回来
        helper.make_node("Transpose", ["X"], ["T1"], name="tp0", perm=[1, 0]),
        helper.make_node("Transpose", ["T1"], ["T2"], name="tp1", perm=[1, 0]),
        # (b) 可融合但没融合：MatMul + Add，而不是 Gemm
        helper.make_node("MatMul", ["T2", "W_MM"], ["MM"], name="mm0"),
        helper.make_node("Add", ["MM", "B"], ["A"], name="add0"),
        # (c) 可折叠常量子图：Constant * Constant
        helper.make_node("Constant", [], ["C1"], name="c1",
                         value=helper.make_tensor("c1v", TensorProto.FLOAT, [], [2.0])),
        helper.make_node("Constant", [], ["C2"], name="c2",
                         value=helper.make_tensor("c2v", TensorProto.FLOAT, [], [3.0])),
        helper.make_node("Mul", ["C1", "C2"], ["SCALE"], name="mul_c"),
        helper.make_node("Mul", ["A", "SCALE"], ["A2"], name="mul_s"),
        # (d) 恒等噪声
        helper.make_node("Identity", ["A2"], ["I"], name="id0"),
        # (e) 动态 shape 链：把静态 shape 打回未知
        helper.make_node("Shape", ["I"], ["SH"], name="sh0"),
        helper.make_node("Gather", ["SH", "GIDX"], ["G"], name="gather0", axis=0),
        helper.make_node("Unsqueeze", ["G", "AXES0"], ["U"], name="unsq0"),
        helper.make_node("Concat", ["U", "NEG1"], ["CAT"], name="concat0", axis=0),
        helper.make_node("Reshape", ["I", "CAT"], ["R"], name="reshape0"),
        helper.make_node("Relu", ["R"], ["Y"], name="relu0"),
    ]
    inits = [
        numpy_helper.from_array(W_MM, "W_MM"),
        numpy_helper.from_array(B_V, "B"),
        numpy_helper.from_array(np.array(0, dtype=np.int64), "GIDX"),   # rank-0，让 Gather 出标量
        numpy_helper.from_array(np.array([-1], dtype=np.int64), "NEG1"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "AXES0"),
    ]
    model = helper.make_model(
        helper.make_graph(nodes, "irregular", [X], [Y], initializer=inits),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def regularity_report(model: onnx.ModelProto, tag: str) -> None:
    """把「规整」落成 6 条可机检的条件。"""
    g = model.graph
    init = {i.name for i in g.initializer}
    outs = {o.name for o in g.output}
    produced = Counter(o for n in g.node for o in n.output)
    known = {v.name: dim_of(v) for v in g.value_info}
    const_out = {o for n in g.node if n.op_type == "Constant" for o in n.output}

    print(f"\n[{tag}] 规整性检查（6 条）")

    missing = [o for o in produced if o not in known and o not in init and o not in outs]
    symbolic = {n for v in g.value_info for d in v.type.tensor_type.shape.dim if d.dim_param for n in [v.name]}
    if missing:
        verdict = f"FAIL 未知={missing}"
    elif symbolic:
        verdict = f"OK(维已知，但含符号维 {sorted(symbolic)})"
    else:
        verdict = "OK(全静态)"
    print(f"  (1) 中间 tensor 的 shape 已知         : {verdict}")

    dup = [o for o, c in produced.items() if c > 1]
    print(f"  (2) 每个 value 只有唯一 producer(SSA) : {'OK' if not dup else 'FAIL ' + str(dup)}")

    def consumer_of(tensor):
        return [c.op_type for c in g.node if tensor in c.input]

    unfused = [n.name for n in g.node if n.op_type == "MatMul" and "Add" in consumer_of(n.output[0])]
    print(f"  (3) MatMul+Add 已收敛成 Gemm          : {'OK' if not unfused else 'FAIL ' + str(unfused)}")

    foldable = [n.name for n in g.node
                if n.op_type not in {"Constant", "Shape"}
                and n.input and all(i in init or i in const_out for i in n.input)]
    print(f"  (4) 常量子图已折叠                    : {'OK' if not foldable else 'FAIL ' + str(foldable)}")

    noise = [n.name for n in g.node if n.op_type in {"Identity", "Dropout"}]
    tp_pair = [n.name for n in g.node if n.op_type == "Transpose" and "Transpose" in consumer_of(n.output[0])]
    print(f"  (5) 冗余算子已消除                    : {'OK' if not noise and not tp_pair else 'FAIL ' + str(noise + tp_pair)}")

    seen, topo = {i.name for i in g.input} | init, True
    for n in g.node:
        if any(i not in seen for i in n.input):
            topo = False
        seen |= set(n.output)
    print(f"  (6) node 满足拓扑序                   : {'OK' if topo else 'FAIL'}")


def part2_regularity() -> None:
    rule("Part 2  什么叫「规整的 graph」：同一份数学，两种写法")

    reg = build_regular_model()
    irr = build_irregular_model()
    describe(reg, "规整 · 导出后")
    describe(irr, "不规整 · 导出后")
    describe(onnx.shape_inference.infer_shapes(reg), "规整 · 跑一次 onnx.shape_inference 之后")
    describe(onnx.shape_inference.infer_shapes(irr), "不规整 · 跑一次 onnx.shape_inference 之后")

    print("\n-- 两者算的是同一个函数（数值对齐，规整版未乘 6）--")
    s_reg = ort.InferenceSession(reg.SerializeToString(), providers=["CPUExecutionProvider"])
    s_irr = ort.InferenceSession(irr.SerializeToString(), providers=["CPUExecutionProvider"])
    for batch in (1, 5, 32):
        x = rng.normal(size=(batch, K)).astype(np.float32)
        a = s_reg.run(None, {"X": x})[0] * 6.0
        b = s_irr.run(None, {"X": x})[0]
        print(f"  batch={batch:<3} max|reg*6 - irr| = {np.abs(a - b).max():.3e}")

    regularity_report(onnx.shape_inference.infer_shapes(reg), "规整")
    regularity_report(onnx.shape_inference.infer_shapes(irr), "不规整")

    print("\n  => 规整不是「好看」，是 6 条可机检的条件：")
    print("     静态 shape / 单一定义 / 可融合结构已收敛成 canonical form /")
    print("     常量已折叠 / 无冗余算子 / 拓扑有序")


# ---------------------------------------------------------------------------
# Part 3  三件事分开做
# ---------------------------------------------------------------------------
class Reader(CalibrationDataReader):
    def __init__(self, samples):
        self.samples = samples
        self.rewind()

    def get_next(self):
        return next(self.it, None)

    def rewind(self):
        self.it = iter({"X": s} for s in self.samples)


def ort_basic_optimize(model: onnx.ModelProto, out: Path) -> onnx.ModelProto:
    """quant_pre_process 的第 2 步：ORT_ENABLE_BASIC 原生优化器。"""
    opt = ort.SessionOptions()
    opt.optimized_model_filepath = str(out)
    opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    opt.log_severity_level = 3
    sess = ort.InferenceSession(model.SerializeToString(), opt, providers=["CPUExecutionProvider"])
    del sess                      # 关闭 session，确保 optimized model 落盘
    return onnx.load(out)


def part3_separation() -> None:
    rule("Part 3  把 shape inference / graph optimization 与 quantization 分开")

    irr = build_irregular_model()
    raw_path, pre_path = TMP / "irr_raw.onnx", TMP / "irr_pre.onnx"
    onnx.save(irr, raw_path)

    print("\n-- 阶段 A：原始导出图 --")
    describe(irr, "A raw")

    print("\n-- 阶段 B：只做 ONNX 原生 shape inference --")
    describe(onnx.shape_inference.infer_shapes(irr), "B onnx.shape_inference")
    print("  => 遇到 Shape→Gather→Unsqueeze→Concat→Reshape 就断：")
    print("     Reshape 的 shape 输入是运行时算出来的 tensor，不是 initializer，静态推理推不出。")

    print("\n-- 阶段 C：符号 shape 推理（quant_pre_process 第 1 步，靠 sympy 做符号传播）--")
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
    c = SymbolicShapeInference.infer_shapes(onnx.load(raw_path), 2**31 - 1, False, False, 0)
    describe(c, "C symbolic shape inference")
    print("  => 把 batch 当符号变量沿 Shape/Gather/Concat/Reshape 传下去，rank 与符号维得以恢复。")

    print("\n-- 阶段 D：ORT graph optimization（quant_pre_process 第 2 步）--")
    describe(ort_basic_optimize(c, TMP / "irr_opt.onnx"), "D ORT_ENABLE_BASIC")
    print("  => 常量折叠 + MatMul+Add→Gemm 融合 + 冗余消除(Transpose 对 / Identity)")
    print("  => 这就是你说的 find define-use → 沿别名传播 → SSA 等价替换，在真实编译器里的落地。")

    print("\n-- 阶段 E：quant_pre_process 一把做完 A→D --")
    quant_pre_process(raw_path, pre_path, skip_optimization=False)
    pre = onnx.load(pre_path)
    describe(pre, "E quant_pre_process")
    print(f"  metadata 打标: {[(p.key, p.value) for p in pre.metadata_props]}")
    print("  => quantizer 用这个 tag 判断模型是否已预处理，避免重复跑。")

    print("\n-- 阶段 F：现在才轮到 quantizer --")
    samples = [rng.normal(size=(3, K)).astype(np.float32) for _ in range(16)]
    for tag, src in [("未预处理(raw)", raw_path), ("已预处理(pre)", pre_path)]:
        dst = TMP / f"q_{tag[:3]}.onnx"
        try:
            quantize_static(
                src, dst, Reader(samples),
                quant_format=QuantFormat.QDQ,
                per_channel=False,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                calibrate_method=CalibrationMethod.MinMax,
            )
            counts = Counter(n.op_type for n in onnx.load(dst).graph.node)
            print(f"  {tag:<16} nodes={len(onnx.load(dst).graph.node):<3} "
                  f"Q={counts['QuantizeLinear']:<3} DQ={counts['DequantizeLinear']:<3} {dict(counts)}")
        except Exception as exc:                       # noqa: BLE001 - demo 要看到失败本身
            print(f"  {tag:<16} quantize_static 失败: {type(exc).__name__}: {exc}")

    print("\n  => 关键：Q/DQ 插在哪，完全由「图结构」决定。多一个 Identity/Mul/Transpose，")
    print("     quantizer 就多插一对 Q/DQ 或直接跳过该 op —— 这不是数值问题，是结构问题。")


# ---------------------------------------------------------------------------
# Part 4  graph matching
# ---------------------------------------------------------------------------
def part4_matching() -> None:
    rule("Part 4  graph matching：结构稳定才能把误差归因到具体 op")

    raw_path, pre_path = TMP / "irr_raw.onnx", TMP / "irr_pre.onnx"
    samples = [rng.normal(size=(3, K)).astype(np.float32) for _ in range(16)]

    def dump_optimized(model_path: Path, optimized: Path) -> onnx.GraphProto:
        opt = ort.SessionOptions()
        opt.optimized_model_filepath = str(optimized)
        opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opt.log_severity_level = 3
        sess = ort.InferenceSession(str(model_path), opt, providers=["CPUExecutionProvider"])
        del sess
        return onnx.load(optimized).graph

    # QDQ 工具会给 tensor 改名，形如 <原名>_QuantizeLinear_Output。
    # graph matching 的做法：把后缀剥掉，再回到预处理后的 FP32 图上按名字对齐。
    SUFFIXES = ("_QuantizeLinear_Output", "_DequantizeLinear_Output", "_QuantizeLinear_Input",
                "_DequantizeLinear_Input", "_q_to_dq", "_post_dq", "_QuantizeLinear", "_DequantizeLinear")

    def strip_qdq(name: str) -> str:
        for s in SUFFIXES:
            if name.endswith(s):
                return name[: -len(s)]
        return name

    data = [rng.normal(size=(6, K)).astype(np.float32) for _ in range(8)]

    for tag, src in [("未预处理(raw)", raw_path), ("已预处理(pre)", pre_path)]:
        dst = TMP / f"m_{tag}.onnx"
        quantize_static(src, dst, Reader(samples), quant_format=QuantFormat.QDQ,
                        per_channel=False, activation_type=QuantType.QInt8,
                        weight_type=QuantType.QInt8, calibrate_method=CalibrationMethod.MinMax)

        fp32_path = TMP / f"m_{tag}_fp32_optim.onnx"
        int8_path = TMP / f"m_{tag}_int8_optim.onnx"
        fp32_g = dump_optimized(src, fp32_path)
        int8_g = dump_optimized(dst, int8_path)

        fp32_names = [o for n in fp32_g.node for o in n.output]
        int8_ops = [n.op_type for n in int8_g.node]
        int8_base = {strip_qdq(o) for n in int8_g.node for o in n.output}

        # 数值：INT8 图 vs 同一份 FP32 图
        s_fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
        s_int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
        err = max(np.abs(s_fp32.run(None, {"X": x})[0] - s_int8.run(None, {"X": x})[0]).max() for x in data)
        scale = max(np.abs(s_fp32.run(None, {"X": x})[0]).max() for x in data)

        print(f"\n[{tag}]")
        print(f"  FP32 优化后 op : {[n.op_type for n in fp32_g.node]}")
        print(f"  INT8 优化后 op : {int8_ops}")
        print(f"  INT8 计算 kernel: {[o for o in int8_ops if 'MatMul' in o or 'Gemm' in o or o in {'QLinearAdd','QLinearMul'}]}")
        print(f"  剥离 QDQ 后缀后能对上 FP32 名的 tensor : {sorted(int8_base & set(fp32_names))}")
        print(f"  对不上的（FP32 图里已经没有这个中间 tensor）: {sorted(int8_base - set(fp32_names) - {'X'})}")
        print(f"  INT8 vs FP32 最大绝对误差 = {err:.5f}  (FP32 输出量级 {scale:.3f})")

    print("\n  => 结论（这才是 graph matching 真正的价值）：")
    print("     · 未预处理的图，INT8 侧拿到 QLinearMatMul + QLinearAdd 两个 kernel，")
    print("       中间多一次 dequant→requant 往返；而且 MM/SCALE 这些名字在 FP32 优化图里根本不存在，")
    print("       误差无法归因到任何一层。")
    print("     · 已预处理的图，FP32 侧与 INT8 侧共享同一套中间 tensor 名（A / A2 / R / Y），")
    print("       差异只剩 Q/DQ，误差可以逐层定位；同时 MatMul+Add 收敛成单个 QGemm，少一次量化往返。")


def part5_per_channel_axis() -> None:
    rule("Part 5  quantizer 需要 shape + 结构，才能决定 per-channel axis 与 bias scale")

    raw_path, pre_path = TMP / "irr_raw.onnx", TMP / "irr_pre.onnx"
    samples = [rng.normal(size=(3, K)).astype(np.float32) for _ in range(16)]

    for key, tag, src in [("raw", "未预处理(raw)", raw_path), ("pre", "已预处理(pre)", pre_path)]:
        dst = TMP / f"pc_{key}.onnx"
        quantize_static(src, dst, Reader(samples), quant_format=QuantFormat.QDQ,
                        per_channel=True, activation_type=QuantType.QInt8,
                        weight_type=QuantType.QInt8, calibrate_method=CalibrationMethod.MinMax)
        m = onnx.load(dst)
        print(f"\n[{tag}] nodes={len(m.graph.node)}")
        for i in m.graph.initializer:
            if "scale" in i.name or "zero_point" in i.name:
                arr = numpy_helper.to_array(i)
                gran = "标量 = per-tensor" if arr.ndim == 0 else f"向量[{arr.size}] = per-channel"
                print(f"   {i.name:<28} dims={str(list(i.dims)):<8} {gran}")
        print(f"   ops: {dict(Counter(n.op_type for n in m.graph.node))}")

    print("\n  => 未预处理时，Q/DQ 被插到 T2/C1/C2 这类无意义的 tensor 上；bias B 被当成一个")
    print("     独立的 per-tensor 张量去量化（错的：bias 应该待在 INT32 累加器里）。")
    print("  => 预处理后，bias 变成 B_quantized_scale（长度 8 的 per-channel 向量），")
    print("     被 compiler 吸收进 Gemm 的累加器。这依赖「B 是 Gemm 的第 3 个输入」这一结构事实。")

    pre_q = onnx.load(TMP / "pc_pre.onnx")
    arr = {i.name: numpy_helper.to_array(i) for i in pre_q.graph.initializer}
    if "B_quantized_scale" in arr and "X_scale" in arr and "W_MM_scale" in arr:
        expect = arr["X_scale"] * arr["W_MM_scale"]
        actual = arr["B_quantized_scale"]
        print(f"\n  验证 bias scale 传播：max|B_quantized_scale - X_scale*W_MM_scale| = "
              f"{np.abs(actual - expect).max():.3e}")
        print("  => bias 的 scale 不是统计出来的，是从 input/weight 的 scale 推出来的；")
        print("     这种推导只有图结构稳定（B 明确挂在 Gemm 上）时才成立。")

    print("\n  CLI / 代码侧证据：跳过预处理时 ORT 自己会警告")
    print("    logger WARNING: Please consider to run pre-processing before quantization.")
    print("  它就是靠 model.metadata_props 里有没有 onnx.quant.pre_process=onnxruntime.quant 判断的。")


if __name__ == "__main__":
    part1_protobuf()
    part2_regularity()
    part3_separation()
    part4_matching()
    part5_per_channel_axis()
    print(f"\n\n产物目录：{TMP}")
