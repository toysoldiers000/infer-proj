import numpy as np
import tvm
from tvm import relax
from tvm.relax import transform

builder = relax.BlockBuilder()
x = relax.Var("x", relax.TensorType([8, 16], "float32"))
w = relax.Var("w", relax.TensorType([16, 16], "float32"))
b = relax.Var("b", relax.TensorType([16], "float32"))
with builder.function("main", [x, w, b]):
    with builder.dataflow():
        mm = builder.emit(relax.op.matmul(x, w))
        ba = builder.emit(relax.op.add(mm, b))
        rl = builder.emit(relax.op.nn.relu(ba))
        sn = builder.emit(relax.op.sin(rl))
        out = builder.emit_output(sn)
    builder.emit_func_output(out)
mod = builder.get()
print("=== ORIGINAL ===")
print(mod.script())

print("\n=== struct_info ===")
f = mod["main"]
print("ret_struct_info:", f.ret_struct_info)
print("param0 sinfo:", f.params[0].struct_info)
print("has attrs:", [a for a in dir(f) if "struct_info" in a])
print("lv sinfo (body binding):", f.body.blocks[0].bindings[0].var.struct_info)

print("\n=== LegalizeOps ===")
lm = transform.LegalizeOps()(mod)
print([gv.name_hint for gv in lm.functions])
print(lm.script()[:2500])

print("\n=== FuseOps ===")
try:
    fo = transform.AnnotateTIROpPattern()(lm)
    fo = transform.FuseOps()(fo)
    fo = transform.FuseTIR()(fo)
    print([gv.name_hint for gv in fo.functions if isinstance(fo[gv], tvm.tir.PrimFunc)])
    print(fo.script()[:2500])
except Exception as e:
    print("FuseOps ERR:", type(e).__name__, e)

print("\n=== build + VM ===")
try:
    ex = relax.build(mod, target="llvm")
    vm = relax.VirtualMachine(ex, tvm.cpu())
    rng = np.random.default_rng(7)
    X = rng.standard_normal((8, 16)).astype("float32")
    W = rng.standard_normal((16, 16)).astype("float32")
    B = rng.standard_normal(16).astype("float32")
    got = vm["main"](tvm.nd.array(X), tvm.nd.array(W), tvm.nd.array(B)).numpy()
    ref = np.sin(np.maximum(X @ W + B, 0))
    print("type(ex):", type(ex))
    print("max|d| =", np.abs(got - ref).max())
except Exception as e:
    import traceback
    traceback.print_exc()

print("\n=== TIR schedule API ===")
try:
    sch = tvm.tir.Schedule(lm)
    print("blocks:", [sch.get(b).name_hint for b in sch.get_blocks_in_order()])
    blk = sch.get_blocks_in_order()[0]
    loops = sch.get_loops(blk)
    print("loops:", [sch.get(l) for l in loops])
    l0, l1 = sch.split(loops[0], factors=[None, 4])
    print("after split ok")
    print(sch.mod.script()[:1200])
except Exception as e:
    import traceback
    traceback.print_exc()
