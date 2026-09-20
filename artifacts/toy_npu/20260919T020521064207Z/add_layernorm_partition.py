# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def fused_relax_add_relax_nn_layer_norm_toy_npu_v1(p0: R.Tensor((2, 16), dtype="float16"), p1: R.Tensor((2, 16), dtype="float16"), p2: R.Tensor((16,), dtype="float16"), p3: R.Tensor((16,), dtype="float16")) -> R.Tensor((2, 16), dtype="float16"):
        R.func_attr({"Codegen": "toy_npu_v1"})
        # from tvm.script import relax as R
        
        @R.function
        def local_func(p0_1: R.Tensor((2, 16), dtype="float16"), p1_1: R.Tensor((2, 16), dtype="float16"), p2_1: R.Tensor((16,), dtype="float16"), p3_1: R.Tensor((16,), dtype="float16")) -> R.Tensor((2, 16), dtype="float16"):
            R.func_attr({"Composite": "toy_npu_v1.add_layernorm"})
            with R.dataflow():
                residual_add: R.Tensor((2, 16), dtype="float16") = R.add(p0_1, p1_1)
                gv: R.Tensor((2, 16), dtype="float16") = R.nn.layer_norm(residual_add, p2_1, p3_1, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True)
                R.output(gv)
            return gv

        output: R.Tensor((2, 16), dtype="float16") = local_func(p0, p1, p2, p3)
        return output

    @R.function
    def main(p0: R.Tensor((2, 16), dtype="float16"), p1: R.Tensor((2, 16), dtype="float16"), p2: R.Tensor((16,), dtype="float16"), p3: R.Tensor((16,), dtype="float16")) -> R.Tensor((2, 16), dtype="float16"):
        cls = Module
        with R.dataflow():
            lv: R.Tensor((2, 16), dtype="float16") = cls.fused_relax_add_relax_nn_layer_norm_toy_npu_v1(p0, p1, p2, p3)
            host_sin: R.Tensor((2, 16), dtype="float16") = R.sin(lv)
            gv: R.Tensor((2, 16), dtype="float16") = host_sin
            R.output(gv)
        return gv