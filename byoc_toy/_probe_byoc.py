import numpy as np
import tvm
from tvm import relax
from tvm.relax import transform
from tvm.relax.dpl import is_op, wildcard
from tvm.relax.backend.pattern_registry import register_patterns, get_patterns_with_prefix, get_pattern
from tvm.script import tirx as T   # TVM 0.26 里 TVMScript 的 TIR 方言改名为 tirx

BACKEND = "demo"
PNAME = f"{BACKEND}.matmul_bias_relu"

x, w, b = wildcard(), wildcard(), wildcard()
mm = is_op("relax.matmul")(x, w)
add = is_op("relax.add")(mm, b)
out = is_op("relax.nn.relu")(add)
ANN = {"lhs": x, "rhs": w, "bias": b, "matmul": mm, "out": out}


def check(ctx: relax.transform.PatternCheckContext) -> bool:
    lhs, rhs = ctx.annotated_expr["lhs"], ctx.annotated_expr["rhs"]
    _, k = tuple(int(v) for v in lhs.ty.shape.values)
    kk, n = tuple(int(v) for v in rhs.ty.shape.values)
    return str(lhs.ty.dtype) == "float32" and k == kk and k % 8 == 0 and n % 8 == 0


if get_pattern(PNAME) is None:
    register_patterns([(PNAME, out, ANN, check)])

builder = relax.BlockBuilder()
X = relax.Var("x", relax.TensorType([8, 16], "float32"))
W = relax.Var("w", relax.TensorType([16, 16], "float32"))
B = relax.Var("b", relax.TensorType([16], "float32"))
with builder.function("main", [X, W, B]):
    with builder.dataflow():
        o = builder.emit_output(builder.emit(relax.op.sin(
            builder.emit(relax.op.nn.relu(builder.emit(relax.op.add(
                builder.emit(relax.op.matmul(X, W)), B)))))))
    builder.emit_func_output(o)
mod = builder.get()

part = transform.FuseOpsByPattern(get_patterns_with_prefix(BACKEND), annotate_codegen=True)(mod)
part = transform.MergeCompositeFunctions()(part)

# ---- 检视外层函数结构：param 的 ty 是否可读；能否拿到内层 Composite ----
outer = [f for gv, f in part.functions.items() if f.attrs and "Codegen" in f.attrs][0]
print("=== outer func ===")
print("global_symbol:", outer.attrs["global_symbol"], " Codegen:", outer.attrs["Codegen"])
print("param0 .ty =", outer.params[0].ty)
print("body type:", type(outer.body))

blk = outer.body.blocks[0]
for bd in blk.bindings:
    print("  binding var:", bd.var.name_hint, " value type:", type(bd.value).__name__)
    if isinstance(bd.value, relax.Call):
        fn = bd.value.op
        print("    call op:", type(fn).__name__, getattr(fn, "name_hint", None))
        if isinstance(fn, relax.expr.GlobalVar):
            print("    -> composite:", part[fn].attrs.get("Composite"))


def make_tir(sym, m, k, n, dtype="float32"):
    """toy backend 的"代码生成"：直接产出 TIR，再交给 LLVM 编译"""
    @T.prim_func
    def prim(X_: T.Buffer((m, k), dtype), W_: T.Buffer((k, n), dtype),
             B_: T.Buffer((n,), dtype), O: T.Buffer((m, n), dtype)):
        for i, j, t in T.grid(m, n, k):
            with T.block("mm"):
                vi, vj, vt = T.axis.remap("SSR", [i, j, t])
                with T.init():
                    O[vi, vj] = B_[vj]
                O[vi, vj] += X_[vi, vt] * W_[vt, vj]
        for i, j in T.grid(m, n):
            with T.block("relu"):
                vi, vj = T.axis.remap("SS", [i, j])
                O[vi, vj] = T.max(O[vi, vj], T.float32(0))

    return prim.with_attr("global_symbol", sym)


def demo_compiler(funcs, options, constant_names):
    mods = []
    for f in funcs:
        sym = f.attrs["global_symbol"]
        m, k = tuple(int(v) for v in f.params[0].ty.shape.values)
        kk, n = tuple(int(v) for v in f.params[1].ty.shape.values)
        print(f"  [demo codegen] {sym}  m={m} k={k} n={n}")
        mods.append(tvm.build(make_tir(sym, m, k, n), target="llvm"))
    return mods


tvm.register_global_func("relax.ext.demo", demo_compiler, override=True)

print("\n=== RunCodegen ===")
cg = transform.RunCodegen()(part)
print(cg.script())

print("\n=== build + run ===")
ex = relax.build(cg, target="llvm")
vm = relax.VirtualMachine(ex, tvm.cpu())
rng = np.random.default_rng(7)
xa = rng.standard_normal((8, 16)).astype("float32")
wa = rng.standard_normal((16, 16)).astype("float32")
ba = rng.standard_normal(16).astype("float32")
got = vm["main"](tvm.nd.array(xa), tvm.nd.array(wa), tvm.nd.array(ba)).numpy()
ref = np.sin(np.maximum(xa @ wa + ba, 0))
print("max|d| =", np.abs(got - ref).max())

print("\n=== export_library ===")
ex.export_library("/tmp/demo_byoc.so")
vm2 = relax.VirtualMachine(tvm.runtime.load_module("/tmp/demo_byoc.so"), tvm.cpu())
got2 = vm2["main"](tvm.nd.array(xa), tvm.nd.array(wa), tvm.nd.array(ba)).numpy()
print("reload max|d| =", np.abs(got2 - ref).max())
