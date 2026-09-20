# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(p0: R.Tensor((2, 32), dtype="float16"), p1: R.Tensor((32, 16), dtype="float16"), p2: R.Tensor((16,), dtype="float16")) -> R.Tensor((2, 16), dtype="float16"):
        with R.dataflow():
            matmul: R.Tensor((2, 16), dtype="float16") = R.matmul(p0, p1, out_dtype=None)
            bias_add: R.Tensor((2, 16), dtype="float16") = R.add(matmul, p2)
            gv: R.Tensor((2, 16), dtype="float16") = R.nn.relu(bias_add)
            R.output(gv)
        return gv