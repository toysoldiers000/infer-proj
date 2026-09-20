# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis

@T.prim_func(s_tir=True)
def fused_relax_matmul_relax_add_toy_npu_v1(arg0: T.Buffer((2, 32), "float16", offset_factor=1), arg1: T.Buffer((32, 16), "float16", offset_factor=1), arg2: T.Buffer((16,), "float16", offset_factor=1), external_dispatch: T.Buffer((2, 16), "float16", offset_factor=1)):
    T.func_attr({"tirx.noalias": True})
    with T.sblock("external_dispatch"):
        T.reads()
        T.writes()
        T.call_packed("toy_npu_v1.execute", "{\"nodes\": [{\"arg\": 0, \"dtype\": \"float16\", \"op\": \"input\", \"shape\": [2, 32]}, {\"arg\": 1, \"dtype\": \"float16\", \"op\": \"input\", \"shape\": [32, 16]}, {\"arg\": 2, \"dtype\": \"float16\", \"op\": \"input\", \"shape\": [16]}, {\"attrs\": {}, \"dtype\": \"float16\", \"inputs\": [0, 1], \"op\": \"relax.matmul\", \"shape\": [2, 16]}, {\"attrs\": {}, \"dtype\": \"float16\", \"inputs\": [3, 2], \"op\": \"relax.add\", \"shape\": [2, 16]}], \"output\": 4, \"schema\": 1, \"symbol\": \"fused_relax_matmul_relax_add_toy_npu_v1\"}", T.tvm_stack_make_array(arg0.data, T.tvm_stack_make_shape(2, 32), 0, 2, T.float16(0.0), arg0.elem_offset), T.tvm_stack_make_array(arg1.data, T.tvm_stack_make_shape(32, 16), 0, 2, T.float16(0.0), arg1.elem_offset), T.tvm_stack_make_array(arg2.data, T.tvm_stack_make_shape(16), 0, 1, T.float16(0.0), arg2.elem_offset), T.tvm_stack_make_array(external_dispatch.data, T.tvm_stack_make_shape(2, 16), 0, 2, T.float16(0.0), external_dispatch.elem_offset))

# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis

@T.prim_func(s_tir=True)
def fused_relax_matmul_relax_add_toy_npu_v1(arg0: T.Buffer((2, 32), "float16", offset_factor=1), arg1: T.Buffer((32, 16), "float16", offset_factor=1), arg2: T.Buffer((16,), "float16", offset_factor=1), external_dispatch: T.Buffer((2, 16), "float16", offset_factor=1)):
    T.func_attr({"tirx.noalias": True})
    with T.sblock("external_dispatch"):
        T.reads()
        T.writes()
        T.call_packed("toy_npu_v1.execute", "{\"nodes\": [{\"arg\": 0, \"dtype\": \"float16\", \"op\": \"input\", \"shape\": [2, 32]}, {\"arg\": 1, \"dtype\": \"float16\", \"op\": \"input\", \"shape\": [32, 16]}, {\"arg\": 2, \"dtype\": \"float16\", \"op\": \"input\", \"shape\": [16]}, {\"attrs\": {}, \"dtype\": \"float16\", \"inputs\": [0, 1], \"op\": \"relax.matmul\", \"shape\": [2, 16]}, {\"attrs\": {}, \"dtype\": \"float16\", \"inputs\": [3, 2], \"op\": \"relax.add\", \"shape\": [2, 16]}], \"output\": 4, \"schema\": 1, \"symbol\": \"fused_relax_matmul_relax_add_toy_npu_v1\"}", T.tvm_stack_make_array(arg0.data, T.tvm_stack_make_shape(2, 32), 0, 2, T.float16(0.0), arg0.elem_offset), T.tvm_stack_make_array(arg1.data, T.tvm_stack_make_shape(32, 16), 0, 2, T.float16(0.0), arg1.elem_offset), T.tvm_stack_make_array(arg2.data, T.tvm_stack_make_shape(16), 0, 1, T.float16(0.0), arg2.elem_offset), T.tvm_stack_make_array(external_dispatch.data, T.tvm_stack_make_shape(2, 16), 0, 2, T.float16(0.0), external_dispatch.elem_offset))

# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis

@T.prim_func(s_tir=True)
def fused_relax_nn_conv2d_relax_nn_relu_toy_npu_v1(arg0: T.Buffer((1, 16, 3, 3), "float16", offset_factor=1), arg1: T.Buffer((16, 16, 1, 1), "float16", offset_factor=1), external_dispatch: T.Buffer((1, 16, 3, 3), "float16", offset_factor=1)):
    T.func_attr({"tirx.noalias": True})
    with T.sblock("external_dispatch"):
        T.reads()
        T.writes()
        T.call_packed("toy_npu_v1.execute", "{\"nodes\": [{\"arg\": 0, \"dtype\": \"float16\", \"op\": \"input\", \"shape\": [1, 16, 3, 3]}, {\"arg\": 1, \"dtype\": \"float16\", \"op\": \"input\", \"shape\": [16, 16, 1, 1]}, {\"attrs\": {}, \"dtype\": \"float16\", \"inputs\": [0, 1], \"op\": \"relax.nn.conv2d\", \"shape\": [1, 16, 3, 3]}, {\"attrs\": {}, \"dtype\": \"float16\", \"inputs\": [2], \"op\": \"relax.nn.relu\", \"shape\": [1, 16, 3, 3]}], \"output\": 3, \"schema\": 1, \"symbol\": \"fused_relax_nn_conv2d_relax_nn_relu_toy_npu_v1\"}", T.tvm_stack_make_array(arg0.data, T.tvm_stack_make_shape(1, 16, 3, 3), 0, 4, T.float16(0.0), arg0.elem_offset), T.tvm_stack_make_array(arg1.data, T.tvm_stack_make_shape(16, 16, 1, 1), 0, 4, T.float16(0.0), arg1.elem_offset), T.tvm_stack_make_array(external_dispatch.data, T.tvm_stack_make_shape(1, 16, 3, 3), 0, 4, T.float16(0.0), external_dispatch.elem_offset))

# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis

@T.prim_func(s_tir=True)
def fused_relax_add_relax_nn_layer_norm_toy_npu_v1(arg0: T.Buffer((2, 16), "float16", offset_factor=1), arg1: T.Buffer((2, 16), "float16", offset_factor=1), arg2: T.Buffer((16,), "float16", offset_factor=1), arg3: T.Buffer((16,), "float16", offset_factor=1), external_dispatch: T.Buffer((2, 16), "float16", offset_factor=1)):
    T.func_attr({"tirx.noalias": True})
    with T.sblock("external_dispatch"):
        T.reads()
        T.writes()
        T.call_packed("toy_npu_v1.execute", "{\"nodes\": [{\"arg\": 0, \"dtype\": \"float16\", \"op\": \"input\", \"shape\": [2, 16]}, {\"arg\": 1, \"dtype\": \"float16\", \"op\": \"input\", \"shape\": [2, 16]}, {\"arg\": 2, \"dtype\": \"float16\", \"op\": \"input\", \"shape\": [16]}, {\"arg\": 3, \"dtype\": \"float16\", \"op\": \"input\", \"shape\": [16]}, {\"attrs\": {}, \"dtype\": \"float16\", \"inputs\": [0, 1], \"op\": \"relax.add\", \"shape\": [2, 16]}, {\"attrs\": {\"axes\": [-1], \"epsilon\": 1e-05}, \"dtype\": \"float16\", \"inputs\": [4, 2, 3], \"op\": \"relax.nn.layer_norm\", \"shape\": [2, 16]}], \"output\": 5, \"schema\": 1, \"symbol\": \"fused_relax_add_relax_nn_layer_norm_toy_npu_v1\"}", T.tvm_stack_make_array(arg0.data, T.tvm_stack_make_shape(2, 16), 0, 2, T.float16(0.0), arg0.elem_offset), T.tvm_stack_make_array(arg1.data, T.tvm_stack_make_shape(2, 16), 0, 2, T.float16(0.0), arg1.elem_offset), T.tvm_stack_make_array(arg2.data, T.tvm_stack_make_shape(16), 0, 1, T.float16(0.0), arg2.elem_offset), T.tvm_stack_make_array(arg3.data, T.tvm_stack_make_shape(16), 0, 1, T.float16(0.0), arg3.elem_offset), T.tvm_stack_make_array(external_dispatch.data, T.tvm_stack_make_shape(2, 16), 0, 2, T.float16(0.0), external_dispatch.elem_offset))

# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis

@T.prim_func(s_tir=True)
def fused_relax_matmul_relax_add_toy_npu_v1(arg0: T.Buffer((2, 32), "float16", offset_factor=1), arg1: T.Buffer((32, 16), "float16", offset_factor=1), arg2: T.Buffer((16,), "float16", offset_factor=1), external_dispatch: T.Buffer((2, 16), "float16", offset_factor=1)):
    T.func_attr({"tirx.noalias": True})
    with T.sblock("external_dispatch"):
        T.reads()
        T.writes()
        T.call_packed("toy_npu_v1.execute", "{\"nodes\": [{\"arg\": 0, \"dtype\": \"float16\", \"op\": \"input\", \"shape\": [2, 32]}, {\"arg\": 1, \"dtype\": \"float16\", \"op\": \"input\", \"shape\": [32, 16]}, {\"arg\": 2, \"dtype\": \"float16\", \"op\": \"input\", \"shape\": [16]}, {\"attrs\": {}, \"dtype\": \"float16\", \"inputs\": [0, 1], \"op\": \"relax.matmul\", \"shape\": [2, 16]}, {\"attrs\": {}, \"dtype\": \"float16\", \"inputs\": [3, 2], \"op\": \"relax.add\", \"shape\": [2, 16]}], \"output\": 4, \"schema\": 1, \"symbol\": \"fused_relax_matmul_relax_add_toy_npu_v1\"}", T.tvm_stack_make_array(arg0.data, T.tvm_stack_make_shape(2, 32), 0, 2, T.float16(0.0), arg0.elem_offset), T.tvm_stack_make_array(arg1.data, T.tvm_stack_make_shape(32, 16), 0, 2, T.float16(0.0), arg1.elem_offset), T.tvm_stack_make_array(arg2.data, T.tvm_stack_make_shape(16), 0, 1, T.float16(0.0), arg2.elem_offset), T.tvm_stack_make_array(external_dispatch.data, T.tvm_stack_make_shape(2, 16), 0, 2, T.float16(0.0), external_dispatch.elem_offset))

# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis

@T.prim_func(s_tir=True)
def fused_relax_matmul_relax_add_toy_npu_v1(arg0: T.Buffer((2, 1024), "float16", offset_factor=1), arg1: T.Buffer((1024, 16), "float16", offset_factor=1), arg2: T.Buffer((16,), "float16", offset_factor=1), external_dispatch: T.Buffer((2, 16), "float16", offset_factor=1)):
    T.func_attr({"tirx.noalias": True})
    with T.sblock("external_dispatch"):
        T.reads()
        T.writes()
        T.call_packed("toy_npu_v1.execute", "{\"nodes\": [{\"arg\": 0, \"dtype\": \"float16\", \"op\": \"input\", \"shape\": [2, 1024]}, {\"arg\": 1, \"dtype\": \"float16\", \"op\": \"input\", \"shape\": [1024, 16]}, {\"arg\": 2, \"dtype\": \"float16\", \"op\": \"input\", \"shape\": [16]}, {\"attrs\": {}, \"dtype\": \"float16\", \"inputs\": [0, 1], \"op\": \"relax.matmul\", \"shape\": [2, 16]}, {\"attrs\": {}, \"dtype\": \"float16\", \"inputs\": [3, 2], \"op\": \"relax.add\", \"shape\": [2, 16]}], \"output\": 4, \"schema\": 1, \"symbol\": \"fused_relax_matmul_relax_add_toy_npu_v1\"}", T.tvm_stack_make_array(arg0.data, T.tvm_stack_make_shape(2, 1024), 0, 2, T.float16(0.0), arg0.elem_offset), T.tvm_stack_make_array(arg1.data, T.tvm_stack_make_shape(1024, 16), 0, 2, T.float16(0.0), arg1.elem_offset), T.tvm_stack_make_array(arg2.data, T.tvm_stack_make_shape(16), 0, 1, T.float16(0.0), arg2.elem_offset), T.tvm_stack_make_array(external_dispatch.data, T.tvm_stack_make_shape(2, 16), 0, 2, T.float16(0.0), external_dispatch.elem_offset))

# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis

@T.prim_func(s_tir=True)
def fused_relax_matmul_relax_add_toy_npu_v1(arg0: T.Buffer((2, 32), "float16", offset_factor=1), arg1: T.Buffer((32, 16), "float16", offset_factor=1), arg2: T.Buffer((16,), "float16", offset_factor=1), external_dispatch: T.Buffer((2, 16), "float16", offset_factor=1)):
    T.func_attr({"tirx.noalias": True})
    with T.sblock("external_dispatch"):
        T.reads()
        T.writes()
        T.call_packed("toy_npu_v1.execute", "{\"nodes\": [{\"arg\": 0, \"dtype\": \"float16\", \"op\": \"input\", \"shape\": [2, 32]}, {\"arg\": 1, \"dtype\": \"float16\", \"op\": \"input\", \"shape\": [32, 16]}, {\"arg\": 2, \"dtype\": \"float16\", \"op\": \"input\", \"shape\": [16]}, {\"attrs\": {}, \"dtype\": \"float16\", \"inputs\": [0, 1], \"op\": \"relax.matmul\", \"shape\": [2, 16]}, {\"attrs\": {}, \"dtype\": \"float16\", \"inputs\": [2, 3], \"op\": \"relax.add\", \"shape\": [2, 16]}], \"output\": 4, \"schema\": 1, \"symbol\": \"fused_relax_matmul_relax_add_toy_npu_v1\"}", T.tvm_stack_make_array(arg0.data, T.tvm_stack_make_shape(2, 32), 0, 2, T.float16(0.0), arg0.elem_offset), T.tvm_stack_make_array(arg1.data, T.tvm_stack_make_shape(32, 16), 0, 2, T.float16(0.0), arg1.elem_offset), T.tvm_stack_make_array(arg2.data, T.tvm_stack_make_shape(16), 0, 1, T.float16(0.0), arg2.elem_offset), T.tvm_stack_make_array(external_dispatch.data, T.tvm_stack_make_shape(2, 16), 0, 2, T.float16(0.0), external_dispatch.elem_offset))

# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis

@T.prim_func(s_tir=True)
def fused_relax_matmul_relax_add_toy_npu_v1(arg0: T.Buffer((2, 32), "float16", offset_factor=1), arg1: T.Buffer((32, 16), "float16", offset_factor=1), arg2: T.Buffer((16,), "float16", offset_factor=1), external_dispatch: T.Buffer((2, 16), "float16", offset_factor=1)):
    T.func_attr({"tirx.noalias": True})
    with T.sblock("external_dispatch"):
        T.reads()
        T.writes()
        T.call_packed("toy_npu_v1.execute", "{\"nodes\": [{\"arg\": 0, \"dtype\": \"float16\", \"op\": \"input\", \"shape\": [2, 32]}, {\"arg\": 1, \"dtype\": \"float16\", \"op\": \"input\", \"shape\": [32, 16]}, {\"arg\": 2, \"dtype\": \"float16\", \"op\": \"input\", \"shape\": [16]}, {\"attrs\": {}, \"dtype\": \"float16\", \"inputs\": [0, 1], \"op\": \"relax.matmul\", \"shape\": [2, 16]}, {\"attrs\": {}, \"dtype\": \"float16\", \"inputs\": [3, 2], \"op\": \"relax.add\", \"shape\": [2, 16]}], \"output\": 4, \"schema\": 1, \"symbol\": \"fused_relax_matmul_relax_add_toy_npu_v1\"}", T.tvm_stack_make_array(arg0.data, T.tvm_stack_make_shape(2, 32), 0, 2, T.float16(0.0), arg0.elem_offset), T.tvm_stack_make_array(arg1.data, T.tvm_stack_make_shape(32, 16), 0, 2, T.float16(0.0), arg1.elem_offset), T.tvm_stack_make_array(arg2.data, T.tvm_stack_make_shape(16), 0, 1, T.float16(0.0), arg2.elem_offset), T.tvm_stack_make_array(external_dispatch.data, T.tvm_stack_make_shape(2, 16), 0, 2, T.float16(0.0), external_dispatch.elem_offset))