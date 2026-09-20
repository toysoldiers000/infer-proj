# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(p0: R.Tensor((1, 16, 3, 3), dtype="float16"), p1: R.Tensor((16, 16, 1, 1), dtype="float16")) -> R.Tensor((1, 16, 3, 3), dtype="float16"):
        with R.dataflow():
            conv: R.Tensor((1, 16, 3, 3), dtype="float16") = R.nn.conv2d(p0, p1, strides=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], groups=1, data_layout="NCHW", kernel_layout="OIHW", out_layout="NCHW", out_dtype=None)
            relu: R.Tensor((1, 16, 3, 3), dtype="float16") = R.nn.relu(conv)
            host_sin: R.Tensor((1, 16, 3, 3), dtype="float16") = R.sin(relu)
            gv: R.Tensor((1, 16, 3, 3), dtype="float16") = host_sin
            R.output(gv)
        return gv