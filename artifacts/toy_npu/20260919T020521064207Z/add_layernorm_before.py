# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(p0: R.Tensor((2, 16), dtype="float16"), p1: R.Tensor((2, 16), dtype="float16"), p2: R.Tensor((16,), dtype="float16"), p3: R.Tensor((16,), dtype="float16")) -> R.Tensor((2, 16), dtype="float16"):
        with R.dataflow():
            residual_add: R.Tensor((2, 16), dtype="float16") = R.add(p0, p1)
            layernorm: R.Tensor((2, 16), dtype="float16") = R.nn.layer_norm(residual_add, p2, p3, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
            host_sin: R.Tensor((2, 16), dtype="float16") = R.sin(layernorm)
            gv: R.Tensor((2, 16), dtype="float16") = host_sin
            R.output(gv)
        return gv