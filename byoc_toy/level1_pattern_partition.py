#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Level 1：最小 BYOC —— 只做"注册图案 + 分区"，不写 codegen。

这一级要看到的东西：
  · FuseOpsByPattern 把 matmul+add+relu 切成 composite function，打上 Composite 属性
  · MergeCompositeFunctions 再包一层外层函数，打上 Codegen 属性
  · 三个 case 的分区差异（结构层失败 vs 语义层失败）

不需要 C++ / 编译器，只打印 IR。
跑法：python3 level1_pattern_partition.py

注意：本文件未在本机实跑（本机无 TVM），若 API 签名与你的版本不符，
      以 tvm/python/tvm/relax/backend/pattern_registry.py 为准。
"""

import tvm
from tvm import relax
from tvm.relax import transform
from tvm.relax.dpl import is_op, wildcard
from tvm.relax.backend.pattern_registry import (
    register_patterns,
    get_patterns_with_prefix,
)

BACKEND = "demo"
PATTERN_NAME = f"{BACKEND}.matmul_bias_relu"


# ============================================================ 造 IRModule
# 就是你的那段 CASE_SPECS + make_relax_module

CASE_SPECS = {
    "supported_with_fallback": dict(M=8, K=16, N=16, break_pattern=False),
    "pattern_mismatch":        dict(M=8, K=16, N=16, break_pattern=True),
    "support_rejected":        dict(M=8, K=12, N=15, break_pattern=False),
}


def make_relax_module(spec: dict) -> tvm.IRModule:
    m, k, n = spec["M"], spec["K"], spec["N"]
    dtype = "float32"
    builder = relax.BlockBuilder()

    x = relax.Var("x", relax.TensorType([m, k], dtype))
    weight = relax.Var("weight", relax.TensorType([k, n], dtype))
    bias = relax.Var("bias", relax.TensorType([n], dtype))

    with builder.function("main", [x, weight, bias]):
        with builder.dataflow():
            matmul = builder.emit(relax.op.matmul(x, weight), name_hint="matmul")
            if spec["break_pattern"]:
                # 数值恒等的 reshape，唯一使命是打断图案
                matmul = builder.emit(
                    relax.op.reshape(matmul, (m, n)), name_hint="reshape_break"
                )
            bias_add = builder.emit(relax.op.add(matmul, bias), name_hint="bias_add")
            activation = builder.emit(relax.op.nn.relu(bias_add), name_hint="relu")
            host_sin = builder.emit(relax.op.sin(activation), name_hint="host_sin")
            output = builder.emit_output(host_sin)
        builder.emit_func_output(output)
    return builder.get()


# ============================================ ① 结构图案 (DFPattern)

def _build_pattern():
    x, w, b = wildcard(), wildcard(), wildcard()
    mm = is_op("relax.matmul")(x, w)
    add = is_op("relax.add")(mm, b)
    out = is_op("relax.nn.relu")(add)
    # 注解：把关键子部件命名，供 codegen 取用（"哪个 wildcard 是权重"）
    annotations = {"x": x, "w": w, "b": b, "mm": mm, "add": add, "out": out}
    return out, annotations, mm


PATTERN, ANNOTATIONS, _P_MM = _build_pattern()


# ============================================ ② 语义检查 (check_func)

def _as_int(v):
    """tir.IntImm -> int；动态维度 (tir.Var) -> None"""
    if isinstance(v, tvm.tir.IntImm):
        return int(v)
    try:
        return int(v)
    except Exception:
        return None


def check_demo(context) -> bool:
    """两级过滤的第二级。context: PatternCheckContext"""
    # (a) 中间结果泄漏 —— 官方 check 里最常见的一条
    if context.has_leaking_intermediate_variables():
        return False

    # (b) 拿到匹配上的 matmul Call（annotated_expr 以 DFPattern 对象为 key）
    mm_call = context.annotated_expr[_P_MM]

    # (c) dtype 约束
    if mm_call.struct_info.dtype not in ("float32", "float16"):
        return False

    # (d) 形状对齐约束：K / N 必须是 8 的倍数
    w_sinfo = mm_call.args[1].struct_info          # weight: [K, N]
    k_dim = _as_int(w_sinfo.shape.values[0])
    n_dim = _as_int(w_sinfo.shape.values[1])
    if k_dim is None or n_dim is None:             # 动态形状，玩具后端不支持
        return False
    if k_dim % 8 or n_dim % 8:
        return False

    return True


# ============================================ ③ 注册到全局 registry

register_patterns([(PATTERN_NAME, PATTERN, ANNOTATIONS, check_demo)])


def partition_for_demo(mod, annotate_codegen: bool):
    """每个后端提供的便捷分区函数（仿 partition_for_cublas）"""
    patterns = get_patterns_with_prefix(BACKEND)
    return transform.FuseOpsByPattern(
        patterns, bind_constants=True, annotate_codegen=annotate_codegen
    )(mod)


# ============================================ 主流程

def main():
    for case, spec in CASE_SPECS.items():
        print("\n" + "=" * 72)
        print(f"  {case}   spec={spec}")
        print("=" * 72)

        mod = make_relax_module(spec)
        print("\n--- 原始 IR ---")
        print(mod.script())

        # 只切 inner composite（annotate_codegen=False）
        fused = partition_for_demo(mod, annotate_codegen=False)
        print("\n--- after FuseOpsByPattern(annotate_codegen=False) ---")
        print(fused.script())

        # 再包一层外层函数，Codegen 属性在此出现
        merged = transform.MergeCompositeFunctions()(fused)
        print("\n--- after MergeCompositeFunctions ---")
        print(merged.script())

        n_composite = sum(
            1 for f in merged.functions.values() if f.attrs and "Composite" in f.attrs
        )
        n_codegen = sum(
            1 for f in merged.functions.values() if f.attrs and "Codegen" in f.attrs
        )
        print(f"\n>>> Composite 函数: {n_composite} 个 | 带 Codegen 属性: {n_codegen} 个")
        if n_composite == 0:
            print(">>> 子图全部留在 host —— 请对照 Level 0 判断是"
                  "【DFPattern 结构层失败】还是【check 语义层拒绝】")


if __name__ == "__main__":
    main()
