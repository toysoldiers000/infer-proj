# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((8, 16), dtype="float32"), weight: R.Tensor((16, 16), dtype="float32"), bias: R.Tensor((16,), dtype="float32")) -> R.Tensor((8, 16), dtype="float32"):
        with R.dataflow():
            matmul: R.Tensor((8, 16), dtype="float32") = R.matmul(x, weight, out_dtype=None)
            reshape_break: R.Tensor((8, 16), dtype="float32") = R.reshape(matmul, R.shape([8, 16]))
            bias_add: R.Tensor((8, 16), dtype="float32") = R.add(reshape_break, bias)
            relu: R.Tensor((8, 16), dtype="float32") = R.nn.relu(bias_add)
            host_sin: R.Tensor((8, 16), dtype="float32") = R.sin(relu)
            gv: R.Tensor((8, 16), dtype="float32") = host_sin
            R.output(gv)
        return gv