# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    I.module_attrs({"external_mods": [metadata["ffi.Module"][0]]})
    @R.function
    def main(p0: R.Tensor((1, 16, 3, 3), dtype="float16"), p1: R.Tensor((16, 16, 1, 1), dtype="float16")) -> R.Tensor((1, 16, 3, 3), dtype="float16"):
        with R.dataflow():
            lv = R.call_dps_packed("fused_relax_nn_conv2d_relax_nn_relu_toy_npu_v1", (p0, p1), out_ty=R.Tensor((1, 16, 3, 3), dtype="float16"))
            host_sin: R.Tensor((1, 16, 3, 3), dtype="float16") = R.sin(lv)
            gv: R.Tensor((1, 16, 3, 3), dtype="float16") = host_sin
            R.output(gv)
        return gv

# Metadata omitted. Use show_meta=True in script() method to show it.