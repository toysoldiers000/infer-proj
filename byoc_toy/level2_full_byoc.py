#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Level 2：完整 BYOC —— 自定义图案 + 自定义 codegen + RunCodegen + 真跑数值校验。

这一级要看到的东西：
  · 自定义 codegen 注册到 FFI key "relax.ext.demo"
  · RunCodegen 把 composite 调用替换成 R.call_dps_packed(ExternFunc("<symbol>"))
  · 生成的 C 代码被编译链接，数值与 numpy 参考一致
  · 留 host 的 sin 与 device 上的 matmul+bias+relu 之间的 conversion boundary 真的工作

跑法：
    python3 level2_full_byoc.py        # 需要 TVM + 一个 C 编译器（macOS/clang 即可）

注意：本文件未在本机实跑（本机 Python 3.14 且无 TVM），是照官方架构文档写的。
     若 API 签名与你的版本不符，优先查这两处：
       tvm/python/tvm/relax/backend/pattern_registry.py
       tvm/src/relax/transform/run_codegen.cc
"""

import numpy as np
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

M, K, N = 8, 16, 16


# ============================================ 造 IRModule

def make_relax_module() -> tvm.IRModule:
    builder = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorType([M, K], "float32"))
    w = relax.Var("weight", relax.TensorType([K, N], "float32"))
    b = relax.Var("bias", relax.TensorType([N], "float32"))

    with builder.function("main", [x, w, b]):
        with builder.dataflow():
            mm = builder.emit(relax.op.matmul(x, w), name_hint="matmul")
            ba = builder.emit(relax.op.add(mm, b), name_hint="bias_add")
            rl = builder.emit(relax.op.nn.relu(ba), name_hint="relu")
            sn = builder.emit(relax.op.sin(rl), name_hint="host_sin")   # 留在 host
            out = builder.emit_output(sn)
        builder.emit_func_output(out)
    return builder.get()


# ============================================ ①+② 图案与检查

def _build_pattern():
    x, w, b = wildcard(), wildcard(), wildcard()
    mm = is_op("relax.matmul")(x, w)
    add = is_op("relax.add")(mm, b)
    out = is_op("relax.nn.relu")(add)
    return out, {"x": x, "w": w, "b": b, "mm": mm, "add": add, "out": out}, mm


PATTERN, ANNOTATIONS, _P_MM = _build_pattern()


def _as_int(v):
    if isinstance(v, tvm.tir.IntImm):
        return int(v)
    try:
        return int(v)
    except Exception:
        return None


def check_demo(context) -> bool:
    if context.has_leaking_intermediate_variables():
        return False
    mm_call = context.annotated_expr[_P_MM]
    if mm_call.struct_info.dtype != "float32":
        return False
    w_sinfo = mm_call.args[1].struct_info
    k_dim = _as_int(w_sinfo.shape.values[0])
    n_dim = _as_int(w_sinfo.shape.values[1])
    if k_dim is None or n_dim is None:
        return False
    return k_dim % 8 == 0 and n_dim % 8 == 0


register_patterns([(PATTERN_NAME, PATTERN, ANNOTATIONS, check_demo)])


def partition_for_demo(mod):
    patterns = get_patterns_with_prefix(BACKEND)
    mod = transform.FuseOpsByPattern(
        patterns, bind_constants=True, annotate_codegen=True
    )(mod)
    return transform.MergeCompositeFunctions()(mod)


# ============================================ ③ 自定义 codegen

C_TEMPLATE = r"""
// ---- 由 BYOC demo codegen 自动生成 ----
// 对应 Composite 图案: {composite}
#include <tvm/runtime/c_runtime_api.h>
#include <dlpack/dlpack.h>

int {symbol}_(DLTensor* x, DLTensor* w, DLTensor* b, DLTensor* out) {{
  // x:[M,K]  w:[K,N]  b:[N]  out:[M,N]  float32
  const int64_t m = x->shape[0];
  const int64_t k = x->shape[1];
  const int64_t n = w->shape[1];

  const float* X = (const float*)x->data;
  const float* W = (const float*)w->data;
  const float* B = (const float*)b->data;
  float*        O = (float*)out->data;

  for (int64_t i = 0; i < m; ++i) {{
    for (int64_t j = 0; j < n; ++j) {{
      float acc = 0.0f;
      for (int64_t t = 0; t < k; ++t) {{
        acc += X[i * k + t] * W[t * n + j];
      }}
      acc += B[j];
      O[i * n + j] = acc > 0.0f ? acc : 0.0f;   // relu
    }}
  }}
  return 0;
}}

// 生成 packed 调用约定的包装（call_dps_packed 需要）
TVM_DLL_EXPORT_TYPED_FUNC({symbol}, {symbol}_);
"""


@tvm.register_func("relax.ext.demo", override=True)
def demo_compiler(funcs, options, constant_names):
    """RunCodegen 通过 FFI key "relax.ext.demo" 找到这里。

    参数：
      funcs          —— 所有带 Codegen="demo" 属性的外层函数
      options        —— 后端私有编译选项
      constant_names —— Constant -> 名字 的映射（搬权重用）
    返回：
      Array<runtime.Module>，每个函数一个
    """
    compiled = []
    create_c_module = tvm.get_global_func("runtime.CSourceModuleCreate")

    for func in funcs:
        symbol = func.attrs["global_symbol"]

        # 真实后端会在外层函数体里找到内层 private function，
        # 读它的 Composite 属性来决定生成什么代码。这里只有一种图案，直接用名字。
        composite = PATTERN_NAME
        print(f"[demo codegen] {symbol}  <-  Composite={composite}")

        src = C_TEMPLATE.format(symbol=symbol, composite=composite)
        compiled.append(create_c_module(src, "c", [symbol], []))

    return compiled


# ============================================ 主流程

def main():
    mod = make_relax_module()
    print("===== 0. 原始 IR =====")
    print(mod.script())

    # --- 分区 ---
    partitioned = partition_for_demo(mod)
    print("\n===== 1. after FuseOpsByPattern + MergeCompositeFunctions =====")
    print(partitioned.script())

    # --- 调后端 codegen ---
    after_codegen = transform.RunCodegen()(partitioned)
    print("\n===== 2. after RunCodegen =====")
    print(after_codegen.script())
    print(">>> 注意看 call_dps_packed(ExternFunc(\"...demo0\"), ...) —— 这就是卸载后的调用")

    # --- 编译 + 链接 C 模块 ---
    ex = relax.build(after_codegen, target="llvm")
    so_path = "./demo_byoc_lib.so"
    ex.export_library(so_path)
    print(f"\n===== 3. 已导出 {so_path} =====")

    # --- 跑数值校验 ---
    loaded = tvm.runtime.load_module(so_path)
    vm = relax.VirtualMachine(loaded, tvm.cpu())

    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((M, K)).astype("float32")
    w_np = rng.standard_normal((K, N)).astype("float32")
    b_np = rng.standard_normal((N,)).astype("float32")

    out = vm["main"](
        tvm.nd.array(x_np), tvm.nd.array(w_np), tvm.nd.array(b_np)
    ).numpy()

    ref = np.sin(np.maximum(x_np @ w_np + b_np, 0.0))   # device 段 + host 段
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)

    print("===== 4. 数值校验通过 =====")
    print(f"  max |out - ref| = {np.abs(out - ref).max():.3e}")
    print("  matmul+bias+relu 跑在 demo 后端（生成的 C），sin 留在 host —— 边界工作正常")


if __name__ == "__main__":
    main()
