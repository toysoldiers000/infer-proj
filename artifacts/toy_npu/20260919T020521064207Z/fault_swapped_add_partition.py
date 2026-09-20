# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def fused_relax_matmul_relax_add_toy_npu_v1(p0: R.Tensor((2, 32), dtype="float16"), p1: R.Tensor((32, 16), dtype="float16"), p2: R.Tensor((16,), dtype="float16")) -> R.Tensor((2, 16), dtype="float16"):
        R.func_attr({"Codegen": "toy_npu_v1"})
        # from tvm.script import relax as R
        
        @R.function
        def local_func(p0_1: R.Tensor((2, 32), dtype="float16"), p1_1: R.Tensor((32, 16), dtype="float16"), p2_1: R.Tensor((16,), dtype="float16")) -> R.Tensor((2, 16), dtype="float16"):
            R.func_attr({"Composite": "toy_npu_v1.matmul_add"})
            with R.dataflow():
                matmul: R.Tensor((2, 16), dtype="float16") = R.matmul(p0_1, p1_1, out_dtype=None)
                gv: R.Tensor((2, 16), dtype="float16") = R.add(p2_1, matmul)
                R.output(gv)
            return gv

        output: R.Tensor((2, 16), dtype="float16") = local_func(p0, p1, p2)
        return output

    @R.function
    def main(p0: R.Tensor((2, 32), dtype="float16"), p1: R.Tensor((32, 16), dtype="float16"), p2: R.Tensor((16,), dtype="float16")) -> R.Tensor((2, 16), dtype="float16"):
        cls = Module
        with R.dataflow():
            lv: R.Tensor((2, 16), dtype="float16") = cls.fused_relax_matmul_relax_add_toy_npu_v1(p0, p1, p2)
            host_relu: R.Tensor((2, 16), dtype="float16") = R.nn.relu(lv)
            gv: R.Tensor((2, 16), dtype="float16") = host_relu
            R.output(gv)
        return gv