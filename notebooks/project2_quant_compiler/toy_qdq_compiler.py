"""
ToyQDQCompiler — 一个最小但真实的「NPU 厂商后端编译器」骨架
============================================================
演示厂商开发者支持 Raw QDQ 时必须做的三件事：
  Pass1  QDQ 融合          (把 DQ -> op -> Q 坍缩成 int8 kernel)
  Pass2  量化参数传播+requant 决策 (追踪 scale/zp；不同 scale 相遇要 requant)
  Pass3  算子分区          (NPU int8 子图 vs CPU float 边界)

运行：uv run python notebooks/project2_quant_compiler/toy_qdq_compiler.py

图结构按 ORT 真实 raw QDQ 建模：每个 compute op 前插 DequantizeLinear(scale_in)，
后插 QuantizeLinear(scale_out)；两个 op 之间用 Q(s)->DQ(s)（身份重量化）连接。
对称量化下 Relu 不改变 scale，所以这条边是 identity，可以被融合；
非对称下 Relu 改变 scale，边变成 requant，region 在此断裂 -> Relu 成为边界
（这就是 ORT 保守地把它留在 float 域的根因）。
"""

from dataclasses import dataclass, field


# ---------- IR 节点 ----------
@dataclass
class Node:
    op: str
    name: str
    inputs: list          # 第一个激活输入来自 DQ；权重/偏置是 initializer（不在 producer 表里）
    outputs: list


# ---------- 量化参数表（真实场景在 onnx initializer 里；这里用 dict 模拟查找表）----------
# 全部对称量化 zp=0，符合 notebook 当前配置
QP = {
    "s_x": 0.0349,        # 输入激活 scale
    "s_fc1": 0.0207,      # fc1 输出 scale
    "s_fc2": 0.0040,      # fc2 输出 scale
    "s_xskip": 0.0349,    # 残差 skip 连接的 scale（与 s_fc2 不同 -> Add 需要 requant）
    "s_out": 0.0016,      # 残差相加后输出 scale
    "s_logits": 0.0016,   # 最终 logits scale
}


# ---------- 厂商硬件能力策略表（分区依据）----------
NPU_INT8_OPS = {"Gemm", "Conv", "MatMul", "Add", "Relu", "Mul"}   # 能下 NPU 跑 int8 kernel
CPU_FLOAT_OPS = {"LayerNorm", "Softmax"}                          # 数值敏感，留 CPU/float 边界


# ---------- 前端量化器交给我们的 Raw QDQ 图 ----------
# 结构：QuantizeLinear(输入打标) -> DQ -> [Gemm->Q->DQ -> Relu->Q->DQ] x2 -> Add -> ... -> LayerNorm/Softmax
RAW = [
    Node("QuantizeLinear",   "q_in",   ["X", "s_x"],          ["x_i"]),
    Node("DequantizeLinear", "dq_in",  ["x_i", "s_x"],         ["x_f"]),
    # ---- int8 region: fc1 + relu1 ----
    Node("Gemm",             "fc1",    ["x_f", "W1", "B1"],    ["h1"]),
    Node("QuantizeLinear",   "q_fc1",  ["h1", "s_fc1"],        ["h1_i"]),
    Node("DequantizeLinear", "dq_fc1", ["h1_i", "s_fc1"],      ["h1_f"]),
    Node("Relu",             "relu1",  ["h1_f"],               ["a1"]),
    Node("QuantizeLinear",   "q_r1",   ["a1", "s_fc1"],        ["a1_i"]),
    Node("DequantizeLinear", "dq_r1",  ["a1_i", "s_fc1"],      ["a1_f"]),
    # ---- int8 region: fc2 + relu2 ----
    Node("Gemm",             "fc2",    ["a1_f", "W2", "B2"],   ["h2"]),
    Node("QuantizeLinear",   "q_fc2",  ["h2", "s_fc2"],        ["h2_i"]),
    Node("DequantizeLinear", "dq_fc2", ["h2_i", "s_fc2"],      ["h2_f"]),
    Node("Relu",             "relu2",  ["h2_f"],               ["a2"]),
    Node("QuantizeLinear",   "q_r2",   ["a2", "s_fc2"],        ["a2_i"]),
    Node("DequantizeLinear", "dq_r2",  ["a2_i", "s_fc2"],      ["a2_f"]),
    # ---- 残差 Add：两个 int8 输入 scale 不同 -> 需要内部 requant ----
    Node("DequantizeLinear", "dq_sk",  ["x_skip", "s_xskip"],  ["xsk_f"]),
    Node("Add",              "res",    ["a2_f", "xsk_f"],      ["r"]),
    Node("QuantizeLinear",   "q_res",  ["r", "s_out"],          ["r_i"]),
    Node("DequantizeLinear", "dq_outp",["r_i", "s_out"],        ["r_f"]),
    # ---- float 边界：数值敏感 op 留在 CPU（对应 AllSpark 的 Fermat 向量核 / CPU）----
    Node("LayerNorm",        "ln",     ["r_f"],                 ["ln"]),
    Node("QuantizeLinear",   "q_ln",   ["ln", "s_out"],         ["ln_i"]),
    Node("DequantizeLinear", "dq_ln",  ["ln_i", "s_out"],       ["ln_f"]),
    Node("Softmax",          "sm",     ["ln_f"],                ["sm"]),
    Node("QuantizeLinear",   "q_sm",   ["sm", "s_logits"],      ["sm_i"]),
    Node("DequantizeLinear", "dq_sm",  ["sm_i", "s_logits"],    ["Y"]),
]


# ---------- 建索引：tensor -> 生产者 / 消费者 ----------
producer_of, consumer_of = {}, {}
for n in RAW:
    for o in n.outputs:
        producer_of[o] = n
    for i in n.inputs:
        consumer_of[i] = n


def dq_before(op_node):
    """返回 op 第一个激活输入前面的 DequantizeLinear（即进入 int8 的 DQ）"""
    for i in op_node.inputs:
        if i in producer_of and producer_of[i].op == "DequantizeLinear":
            return producer_of[i]
    return None


def q_after(op_node):
    """返回 op 输出后面的 QuantizeLinear（即退出 int8 的 Q）"""
    out = op_node.outputs[0]
    if out in consumer_of and consumer_of[out].op == "QuantizeLinear":
        return consumer_of[out]
    return None


def scale_of(node):
    """从 DQ / Q 节点取 scale 名 -> 数值（Toy 里 inputs[1] 就是 scale 名）"""
    return QP[node.inputs[1]]


# ============================================================
# Pass1 + Pass3：区域融合 + 分区
#   连续 NPU_INT8 ops + 它们之间的 identity 重量化边 -> 融成一个 int8 kernel
#   CPU_FLOAT ops 打断 region -> 单独的 float 边界 kernel
# ============================================================
def compile():
    # ---- 区域划分（region = 连续同类 op）----
    regions, cur = [], []
    for n in RAW:
        if n.op in NPU_INT8_OPS:
            cur.append(n)
        elif n.op in CPU_FLOAT_OPS:
            if cur:
                regions.append(cur)
            cur = [n]                      # CPU op 自成 region
        # Quantize/Dequantize 是边标记，跳过
    if cur:
        regions.append(cur)

    kernels = []
    for reg in regions:
        is_npu = all(o.op in NPU_INT8_OPS for o in reg)
        first, last = reg[0], reg[-1]
        enter_scale = scale_of(dq_before(first)) if dq_before(first) else None
        exit_scale = scale_of(q_after(last)) if q_after(last) else None
        op_names = "+".join(o.op for o in reg)

        if is_npu:
            kind = "Q" + op_names          # 例如 QGemm+Relu+QGemm+Relu+QAdd
            note = "int8 融合 kernel（对称 Relu 前后 scale 不变，身份重量化边被融合）"
            # ---- Pass2：检测 region 内部的 requant（Add 两输入 scale 不同）----
            # Add 的激活输入直接来自各自的 DequantizeLinear，取这两个 DQ 的 scale 即可
            for o in reg:
                if o.op == "Add":
                    sa = scale_of(producer_of[o.inputs[0]])   # 输入1 的 DQ scale (s_fc2)
                    sb = scale_of(producer_of[o.inputs[1]])   # 输入2 的 DQ scale (s_xskip)
                    if sa != sb or sa != exit_scale:
                        note += f" | Add 内部 requant: {sa} & {sb} -> {exit_scale}"
        else:
            kind = "FloatKernel(" + op_names + ")"
            note = "数值敏感，保留 float；前后 DQ/Q 是 int8<->float 转换边界"

        kernels.append({
            "kind": kind, "ops": op_names,
            "target": "NPU" if is_npu else "CPU",
            "enter_s": enter_scale, "exit_s": exit_scale,
            "note": note,
        })
    return kernels


if __name__ == "__main__":
    print("=" * 72)
    print("前端量化器产出的 Raw QDQ 图节点数:", len(RAW))
    print("=" * 72)
    print("\n[Raw QDQ 图（前端交给厂商编译器的东西）]")
    for n in RAW:
        print(f"  {n.op:<16} {n.name:<10} in={n.inputs} out={n.outputs}")

    kernels = compile()

    print("\n" + "=" * 72)
    print("Toy Compiler 产出（3 趟 pass 之后）")
    print("=" * 72)
    npu, cpu = [], []
    for k in kernels:
        (npu if k["target"] == "NPU" else cpu).append(k)
        print(f"\n  [{k['target']}] {k['kind']}")
        print(f"         enter_scale={k['enter_s']}  exit_scale={k['exit_s']}")
        print(f"         说明: {k['note']}")

    print("\n" + "-" * 72)
    print("分区结果（QDQ 边界 = NPU/CPU 切分点）")
    print(f"  NPU int8 子图 : {[k['kind'] for k in npu]}")
    print(f"  CPU float 子图: {[k['kind'] for k in cpu]}")
    print(f"  转换边界张量  : int8(s_out={QP['s_out']}) -> DQ -> LayerNorm(CPU) -> ... -> DQ -> 主机(float Y)")
    print("-" * 72)
    print(f"节点数 {len(RAW)} -> kernel 数 {len(kernels)}  "
          f"(融合消除了大量 Q/DQ 往返，INT8 子图连续无切开)")
    print("=" * 72)
