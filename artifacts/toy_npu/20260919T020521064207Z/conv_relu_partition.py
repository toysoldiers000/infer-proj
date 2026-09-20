# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def fused_relax_nn_conv2d_relax_nn_relu_toy_npu_v1(p0: R.Tensor((1, 16, 3, 3), dtype="float16"), p1: R.Tensor((16, 16, 1, 1), dtype="float16")) -> R.Tensor((1, 16, 3, 3), dtype="float16"):
        R.func_attr({"Codegen": "toy_npu_v1"})
        # from tvm.script import relax as R
        
        @R.function
        def local_func(p0_1: R.Tensor((1, 16, 3, 3), dtype="float16"), p1_1: R.Tensor((16, 16, 1, 1), dtype="float16")) -> R.Tensor((1, 16, 3, 3), dtype="float16"):
            R.func_attr({"Composite": "toy_npu_v1.conv_relu"})
            with R.dataflow():
                conv: R.Tensor((1, 16, 3, 3), dtype="float16") = R.nn.conv2d(p0_1, p1_1, strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype=None)
                gv: R.Tensor((1, 16, 3, 3), dtype="float16") = R.nn.relu(conv)
                R.output(gv)
            return gv

        output: R.Tensor((1, 16, 3, 3), dtype="float16") = local_func(p0, p1)
        return output

    @R.function
    def main(p0: R.Tensor((1, 16, 3, 3), dtype="float16"), p1: R.Tensor((16, 16, 1, 1), dtype="float16")) -> R.Tensor((1, 16, 3, 3), dtype="float16"):
        cls = Module
        with R.dataflow():
            lv: R.Tensor((1, 16, 3, 3), dtype="float16") = cls.fused_relax_nn_conv2d_relax_nn_relu_toy_npu_v1(p0, p1)
            host_sin: R.Tensor((1, 16, 3, 3), dtype="float16") = R.sin(lv)
            gv: R.Tensor((1, 16, 3, 3), dtype="float16") = host_sin
            R.output(gv)
        return gv