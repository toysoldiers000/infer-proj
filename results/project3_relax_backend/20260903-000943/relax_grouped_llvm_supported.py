# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(private=True)
    def fused_relax_matmul_relax_add_relax_nn_relu(x: R.Tensor((8, 16), dtype="float32"), weight: R.Tensor((16, 16), dtype="float32"), bias: R.Tensor((16,), dtype="float32")) -> R.Tensor((8, 16), dtype="float32"):
        R.func_attr({"Composite": "project3_toy_npu.matmul_bias_relu", "Primitive": True})
        with R.dataflow():
            matmul: R.Tensor((8, 16), dtype="float32") = R.matmul(x, weight, out_dtype=None)
            bias_add: R.Tensor((8, 16), dtype="float32") = R.add(matmul, bias)
            gv: R.Tensor((8, 16), dtype="float32") = R.nn.relu(bias_add)
            R.output(gv)
        return gv

    @R.function
    def main(x: R.Tensor((8, 16), dtype="float32"), weight: R.Tensor((16, 16), dtype="float32"), bias: R.Tensor((16,), dtype="float32")) -> R.Tensor((8, 16), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv: R.Tensor((8, 16), dtype="float32") = cls.fused_relax_matmul_relax_add_relax_nn_relu(x, weight, bias)
            host_sin: R.Tensor((8, 16), dtype="float32") = R.sin(lv)
            gv: R.Tensor((8, 16), dtype="float32") = host_sin
            R.output(gv)
        return gv