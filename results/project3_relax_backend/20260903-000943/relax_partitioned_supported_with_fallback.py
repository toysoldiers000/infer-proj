# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def fused_relax_matmul_relax_add_relax_nn_relu_project3_toy_npu_project3_toy_npu(x: R.Tensor((8, 16), dtype="float32"), weight: R.Tensor((16, 16), dtype="float32"), bias: R.Tensor((16,), dtype="float32")) -> R.Tensor((8, 16), dtype="float32"):
        R.func_attr({"Codegen": "project3_toy_npu"})
        # from tvm.script import relax as R
        
        @R.function
        def local_func(x_1: R.Tensor((8, 16), dtype="float32"), weight_1: R.Tensor((16, 16), dtype="float32"), bias_1: R.Tensor((16,), dtype="float32")) -> R.Tensor((8, 16), dtype="float32"):
            R.func_attr({"Composite": "project3_toy_npu.matmul_bias_relu"})
            matmul: R.Tensor((8, 16), dtype="float32") = R.matmul(x_1, weight_1, out_dtype=None)
            bias_add: R.Tensor((8, 16), dtype="float32") = R.add(matmul, bias_1)
            gv: R.Tensor((8, 16), dtype="float32") = R.nn.relu(bias_add)
            return gv

        output: R.Tensor((8, 16), dtype="float32") = local_func(x, weight, bias)
        return output

    @R.function
    def main(x: R.Tensor((8, 16), dtype="float32"), weight: R.Tensor((16, 16), dtype="float32"), bias: R.Tensor((16,), dtype="float32")) -> R.Tensor((8, 16), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv: R.Tensor((8, 16), dtype="float32") = cls.fused_relax_matmul_relax_add_relax_nn_relu_project3_toy_npu_project3_toy_npu(x, weight, bias)
            host_sin: R.Tensor((8, 16), dtype="float32") = R.sin(lv)
            gv: R.Tensor((8, 16), dtype="float32") = host_sin
            R.output(gv)
        return gv