# 从一段 Python `forward` 到 NPU kernel：AllSpark Alchemy、Hauk 与 Attention 的发生式源码调查

> **一句话心智模型**：Alchemy 负责把“模型作者写的 `forward`”逐步记成 AllSpark 的**图**；Hauk 负责把“某一个算子怎样在 N93X/N93P/N93E 上跑得快”记成接近 TensorIR 的**kernel 程序**；二者之间的 native `ANetwork` / Builder、以及相邻的 AGE，是把图选择、布局、内存和设备 kernel 接起来的边界。

本文按“先有问题，才发明抽象”的顺序讲解，而不是先背 IR 名词。调查的主目录是：

- `work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/alchemy`
- `work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/hauk`

为回答“一个 Hauk kernel 真正长什么样”，还查看了同一发布树中紧邻它们的 `ace/kernels`、`ace/aopi/json`。它们不是问题指定的两个目录，但正是 Hauk DSL 被使用、算子能力被描述的地方。最后以 Apache TVM、MLIR、Core ML Tools、ONNX Runtime QNN EP 的公开资料做概念校准；**这些公开框架用于解释共同的编译器思想，不能反向证明 AllSpark 的私有 native 实现与它们完全相同。**

---

## 0. 先把地图摆正：不要把三种“IR”混成一种

先问一个看似简单的问题：写下

```python
y = linear(x, w)
y = add(y, b)
```

这里的 `y` 到底是什么？在 Python 层它看着像立即算出的 ndarray；在编译器层，它常常只是“未来会有一个结果”的句柄。不同层记录的东西完全不同：

```text
模型 Python / Module.forward
          │
          │  Alchemy functional API：创建 AOperation / ATensor 句柄
          ▼
AllSpark native graph（ANetwork，Python 只见到 binding）
          │                 │
          │                 └── Builder：profile、layout、fusion/memory 开关
          │
          ├── 相邻 AGE：atvm.relay 图级 pass（不是 Alchemy 本体）
          ▼
设备算子选择 / ACE 接口契约（AOPI JSON）
          │
          ▼
Hauk Script → Hauk 扩展 TIR / PrimFunc → FFI lowering → .ll/.s/.o
          │
          ▼
Fermat / Cayley 类目标 kernel 与 runtime
```

### 0.1 三个结论，后文逐一证明

1. **Alchemy 不是一个像 Relay/Relax 那样可在 Python 中遍历的独立图 IR。**它是 Python 侧的建图前端：`Tensor` 包住 native `ATensor`，每个 functional API 直接向 opaque 的 `allspark_engine.ANetwork` 加 `AOperation`。真正的图节点和许多优化实现藏在 native binding 后面。
2. **Hauk 才是本次代码中最像 TensorIR/TIR 的自定义 IR 层。**它复用/扩展 `atvm.tir.PrimExpr`、`Buffer`、`PrimFunc`、`IRModule`，加上动态 shape、layout、cache、warp/block binding 等 NPU 语义，再交给 FFI pass 和 codegen。
3. **“图级 rewrite”和“kernel 级 schedule”是两件事。**当前 Alchemy Python 目录中 `graph_rewriting.py` 的主体被注释；相邻 AGE 才能看到 Relay 风格图 pass。Hauk 则主要处理一个 kernel 内的 tile、buffer、同步、循环与指令 lowering。

这一区分非常重要。否则很容易看到 `enable_op_fusion=True`，就误写成“Alchemy 源码实现了某某 pattern fusion”；事实上，源码只证明这个开关被转交给 native Builder，不能证明 native 内部到底匹配了哪一种图模式。

### 0.2 本文的证据标记

| 标记      | 含义                                 | 例子                                                     |
| ------- | ---------------------------------- | ------------------------------------------------------ |
| **[S]** | 本地 Python 源码直接可见                   | `flash_attention()` 调用 `network.add_flash_attention()` |
| **[D]** | 可见 docstring / 接口名，C++ FFI 实现不在当前树 | `HaukStorageRewrite()` 的“复用静态分配”说明                     |
| **[M]** | AOPI JSON 是算子可行性/签名元数据             | FlashAttention 的输入 dtype/layout 组合                     |
| **[I]** | 合理工程推断，明确不当作事实                     | native planner 可能用 profile 选 tile                      |

后文把“能证明什么、不能证明什么”一并写出来，这比把内部名词堆成一张架构图更可靠。

---

## 1. 需求从哪里来：最朴素的推理器为什么不够

先假装我们完全没有 IR，只想让一个极小模型跑起来。

### 1.1 naive 方案：Python 立刻计算

下面伪代码足以得到数值正确的 attention 单头版本。`q`、`k`、`v` 都是已经在内存里的二维数组；`D` 是每个 head 的通道数。

```python
# 演示：第 i 个 query 对所有 key 打分，再用权重混合 value。
for i in range(q_len):
    score = [dot(q[i], k[j]) / sqrt(D) for j in range(k_len)]
    score = add_mask(score, mask[i])
    prob = softmax(score)
    out[i] = sum(prob[j] * v[j] for j in range(k_len))
```

它解决了基础问题。白话说：`q[i]` 是第 `i` 个 token 发出的查询向量；它跟每一个历史 `k[j]` 点乘得到相关度；`softmax` 把相关度变成和为 1 的权重；最后按权重加权 `v[j]`。

紧凑写法才是大家熟悉的：

```text
O = softmax(Q × Kᵀ / √D + M) × V
```

其中 `Q` 可看作 `[B, Hq, Sq, D]`，`K/V` 可看作 `[B, Hkv, Sk, D]`：`B` 是 batch，`H` 是 head 数，`S` 是 token 数，`D` 是一个 head 的通道数；`M` 是 mask。先记住它是“两次矩阵乘 + 一次 reduce/softmax”，而不是先被符号吓住。

### 1.2 好奇的初学者会连续追问

**问题一：为什么不能每个算子都立刻算？**

因为编译器看不到后面。它不知道 `Linear` 的输出马上会接 `Add`，也不知道 `Q/K/V` 之后会接 RoPE、KV cache、causal mask 和 attention。于是：

- 每一步都可能把中间 tensor 写回外部内存，再读回来；
- layout 只能按默认连续数组，不能为下一台硬件单元改排布；
- 动态的 batch/sequence length 只能“来了再说”，没有预留和选型依据；
- 想把几步变成一个 FlashAttention 语义算子，也已经来不及。

**问题二：就算先画一张图，为什么还要另一层 Hauk？**

图只会说“这里是 `Softmax`、那里是 `Dense`”。但 NPU 还需要回答：一个 Dense 的 `M/N/K` 怎样切块？哪块放 L1/L2？哪一个循环绑定到 warp/block？load 下一块时能不能与计算上一块重叠？最后一块不满向量宽度怎么办？这已不是图 rewrite，而是一个小型并行程序的编译。

**问题三：为什么同一个模型要有 min / typical / max shape？**

假设 prompt 长度可能是 1 到 4096，而 90% 请求是 512。只按最大长度安排，会浪费本地存储和并行度；只按当前长度编译，又会导致频繁重编译或越界。因此真正的目标不是“支持动态 shape”这么抽象，而是：**用范围保证正确性，用 typical shape 选择更像真实负载的资源与 tile。**

这三个问题自然推出三个抽象：图 IR、kernel IR、shape/profile contract。现在回到源码看它们分别落在哪里。

---

## 2. 第一层发明：Alchemy 让 `forward` 变成一张 native 图

### 2.1 如果自己手写图，最少要维护什么

naive 图实现可以只有三种对象：

```python
class Tensor:
    producer: Op | None
    shape: list[int]
    dtype: str

class Op:
    kind: str
    inputs: list[Tensor]
    outputs: list[Tensor]
    attrs: dict

class Graph:
    ops: list[Op]
```

`linear(x, w)` 不算数值，而是往 `ops` 追加一条边：`x,w -> Linear -> y`。这样 Builder 才能向前后看，决定是否融合、如何分配内存、是否改变 layout。

Alchemy 没有重新在 Python 写这一套 `Graph/Op` 容器；它把容器放在 native engine 里。Python 用薄包装承担“写起来像 PyTorch Module、实际是在构图”的职责。

### 2.2 源码中的实际数据结构：谁保存什么

| 角色 | 实物 | 源码事实 | 初学者该如何理解 |
|---|---|---|---|
| `Network` | Python 图构建会话 | `Network._init()` 保存 `_allspark_network`、输入表、名字生成器、module call stack、ndarray keep-alive、未填权重表 | 它不是 graph 本身，更像“native graph 的 Python 会话与账本” |
| `Tensor` | `ATensor` wrapper | functional API 返回的 `Tensor` 带 native `allspark_tensor` 和 producer | 图中的边在 native `ATensor`；Python wrapper 提供 shape/value/profile API |
| `Parameter` | 延迟 materialize 的常量代理 | 第一次需要时建 native constant，并按 network 缓存 | 类似 C++ 中“资源对象 + 延迟初始化”，避免没走到的权重提前塞入图 |
| `Module` | PyTorch 风格层次结构 | `__setattr__` 注册 child module / parameter；`__call__` 维护 call stack | 主要为权重命名、debug、模型层级服务，不是每次执行真的跑 CPU `forward` 算法 |
| `ANetwork` / `AOperation` | native 图 IR | `add_input`、`mark_output`、`add_linear`、`add_flash_attention` 都直达 binding | 这是本次发布 Python 源码无法完全展开的真正 graph 节点存储 |

对应位置：[`alchemy/network.py:45-74`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/alchemy/network.py)、[`alchemy/network.py:116-193`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/alchemy/network.py)、[`alchemy/module.py`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/alchemy/module.py)、[`alchemy/parameter.py`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/alchemy/parameter.py)。

一个很工程化、也很容易遗漏的细节是 `Network.register_ndarray()`：注释明确说 native 网络对这些 weight ndarray 是**弱引用**，因此 Python 必须把 ndarray 放在 `_registered_ndarrays` 里延长生命周期。它不是性能花活，而是对象所有权合同：如果局部 numpy 数组先被 GC，native constant 就会指向失效存储。

### 2.3 `forward` 到 native graph 的发生过程

把一次 `model(**inputs)` 当作“录制”而不是“立即推理”：

```text
Builder.create_network()
  ↓
with net_guard(network):             # 把当前 Network 放到 thread/context 中
    prepare inputs → ANetwork.add_input(...)
    model.forward(...)               # 每个 F.xxx 调 native add_xxx，逐步录图
    parameter materialization        # constant / name / ndarray lifetime
    process_weights_and_op_layouts() # 可选物理 layout 覆盖
  ↓
native ABuilder.build(...)
```

`net_guard()` 在 [`network.py:489-500`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/alchemy/network.py) 切换默认网络；`_add_input()` 在 [`network.py:116-129`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/alchemy/network.py) 调 `ANetwork.add_input()`；完整 build 主线在 [`builder.py:410` 起](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/alchemy/builder.py)。

这解释了为什么看起来像普通的函数调用：

```python
# 表面：返回一个 tensor。
y = F.linear(x, weight)

# 实质：native network 添加 Linear AOperation，包装其 output(0)。
```

在 `functional.py` 中，结果经 `_create_tensor(operation.get_output(0), operation)` 包回 Python `Tensor`。`Network._set_op_name()` 还把 Module 路径、调用栈和序号编码到 op/tensor 名字里，以便 dump 或报错能回到模型中的层。

### 2.4 朴素 `Linear` 之后的第一个优化机会

普通 dense 层常写成：

```text
Y = X × Wᵀ
Z = Y + bias
```

如果 `Y` 真落到外部内存，`Add` 又读它一次，就发生一次完全没必要的 round trip。Alchemy 的 `linear()` [S] 先创建 native Linear，bias 在 Python 侧是额外的 Binary Add；这件事有两个正确结论：

1. **图语义上**能清楚看到 `Linear → Add`，因此 native Builder 有机会融合；
2. **源码上**不能断言已经融合，因为模式匹配和 codegen 不在此 Python 文件。`BuildConfig.enable_op_fusion` 只是传给 native `ABuilderConfig`（[`builder.py:79-114`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/alchemy/builder.py)）。

这就是研究编译器时“机会”和“已发生”必须分开的典型例子。

---

## 3. Alchemy 已可见的自定义优化：从朴素图一步步收紧

下面不是把所有 flag 罗列一遍，而是按“它修补了哪一种朴素方案的缺陷”分类。

### 3.1 优化总表：证据强度也一起给出

| 缺陷 | 改动 | 涉及算子/对象 | 源码中的实际做法 | 证据 |
|---|---|---|---|---|
| 每次请求形状不同，内存与 kernel 无从计划 | dynamic profile | 任意动态输入；LLM 的 token/KV | 每个输入加入 min/typical/max shape；LLM profile 传 batch、seq、KV 限制 | **[S]** |
| 默认物理布局可能让下一算子搬运很重 | layout specialization | Constant weight、compute op | 用户 JSON/dict 按 op 名传 `set_physical_layout(input_layout, output_layout)` | **[S]** |
| FP32/FP16 Dense 带宽和算力成本高 | Q/DQ 注入量化子图 | 当前只接入 `dense` / Linear | `QuantizeDynamic → Linear(int weight) → DequantizeDynamic → Add(bias)` | **[S]** |
| Attention 中间 score/prob 太大 | 语义级 FlashAttention op | `flash_attention`、`batch_flash_attention` | Python 直接创建一个 native 自定义 op，不先展开为 `BMM+mask+softmax+BMM` | **[S]**（内部 kernel 算法不可见） |
| 有些视觉感知算子不是通用 ONNX 原语组合 | 专用 cross/sparse attention op | `cross_attention`、`sparse_cross_attention` | Python 直接发 native custom op；ACE 有能力 JSON/部分 Hauk kernel | **[S]/[M]** |
| 相邻逐点/后处理 op 会造成额外调度和中间 buffer | native op fusion / mem reuse | 模式未公开 | `enable_op_fusion`、`enable_mem_reuse` 传给 Builder | **[S]**有开关，**[I]**不得臆测模式 |

### 3.2 dynamic profile：先把“动态”讲具体

假设输入 `tokens` 的逻辑 shape 是 `[B, S]`。与其写“它是动态的”，不如把合同写成：

```text
min     = [1,    1]
typical = [4,  512]
max     = [8, 4096]
```

`min` 是合法下界，`max` 是安全上界，`typical` 是调度/内存策略最该贴近的常用点。`Builder._add_optimization_profile()` 对每个动态输入调用 `set_shape_range(... MinRange/MaxRange)` 与 `set_typical_shape()`，见 [`builder.py:116-148`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/alchemy/builder.py)。

对 LLM，`_add_llm_profile()` 又传入更语义化的约束：`max_batch_size`、`max_input_len`、`max_seq_len`、`max_num_tokens`、KV cache 类型/长度、`enable_flash_attention` 等，见 [`builder.py:150-188`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/alchemy/builder.py)。这比“给每个张量写几个 -1”更接近运行时真正需要知道的资源上界。

**它解决什么？**不是凭空让动态 shape 更快；而是让 native compiler 有安全范围和典型工作点。它究竟是否多版本编译、runtime dispatch 还是单一泛化 kernel，当前 Python 源没有给出，必须视为 native 黑盒。

### 3.3 physical layout：逻辑 shape 不等于内存字节顺序

初学者容易把 `[B, S, H]` 当成全部信息。但对硬件，另一个同样重要的问题是：连续地址沿着哪一维？是否按通道块对齐？下一条 vector/matrix load 是否一次能取满？

Alchemy 的 `process_weights_and_op_layouts()` 支持两张用户配置表：

```text
weight_layout[parameter_name] = [input_layouts, output_layouts]
compute_layout[op_name]       = [input_layouts, output_layouts]
```

构图后它枚举 Constant/compute op，调用 native `set_physical_layout()`，同时输出 `weight_op_names.txt` 与 `compute_op_names.txt` 供你填配置。源码位置：[`builder.py:327-407`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/alchemy/builder.py)。

这不是“改 shape”，而是保持数学上相同的 tensor，改变**物理遍历/对齐合同**。AOPI JSON 中反复出现 `ND`、`SD`、`P1ND_128B` 一类 layout 名称，说明候选 kernel 有 layout 约束；但每个名字的完整物理编码仍属于当前 native/AOPI 定义，不应靠名字猜测。

### 3.4 Q/DQ 注入：不是一句“INT8 量化”，而是一段可见子图

先看朴素思路：把权重从 float 换成 int8，直接做 `X × W_int8`。问题来了：`X` 的数值范围仍是 float，输出的整数累加值也需要恢复单位；bias 应放在整数域还是浮点域会影响精度和实现。

当前 `LinearPerTokenPerChannelWrapper.forward()` 的可见过程是：

```text
x_float
  └─ QuantizeDynamic → (x_int, act_scale)
                         │
W_int + weight_scale ────┼─ Linear → accumulator
                         └─ DequantizeDynamic(accumulator, act_scale, weight_scale)
                                                └─ Add(float bias, if present)
```

对应 [`quantize.py:55-83`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/alchemy/quantize.py)。

这里“per-token”不是玄学：每一个 token 的 activation 可以有自己的 `act_scale`；“per-channel weight”表示 weight 的每个输出通道可有自己的 scale。代码将一维 `weight_scale` reshape 成 `[1, C_out]`，把 `act_scale` 在轴 1 unsqueeze，令它们能按矩阵输出广播。另一个 wrapper 是 per-tensor activation + per-channel weight，见同文件 `:85-111`。

**当前实现边界非常具体：**

- `apply_quantization_from_proto()` 读取量化 protobuf 和 safetensors，遍历 `named_modules_with_parent()`；
- 当前只接受 `op_type == "dense"`；
- 只接入 `input per_tensor/per_token + weight per_channel` 两组 grain；
- 文件里虽定义 `QQBatMatMulDynDQWrapper`，但当前替换分支没有把 `BatchMatMul` 接进去。

证据在 [`quantize.py:386-473`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/alchemy/quantize.py)。所以正确表述是“**Python 前端已实现 Dense 的两类 Q/DQ 注入；BMM wrapper 是已写未接线的扩展点**”，不是“Alchemy 已普遍量化所有 attention BMM”。

### 3.5 “有 graph rewriting 文件”不等于“正在做 graph rewrite”

`alchemy/graph_rewriting.py` 中能看到 `Operation` wrapper、pattern rewriter 等设计痕迹，但当前主体是注释代码。它很像一个曾计划在 Python 侧做 pattern rewrite 的草图；**在这份源码状态下，它不是活动优化路径的证据。**

反过来，相邻而不在本次主目录的 [`age/compiler/build_module.py:40-67`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/age/compiler/build_module.py) 能看到 `atvm.transform.Sequential`，其中有：

```text
AddDequantizeZeroPoint
FoldExplicitPadding
AlterUnsupportedOps
FoldWeightQuantization
SwapQuantMaxPool
RelayConvertUInt8QuantToInt8
AlterBiasAdd
AlterOpLogicalLayout
PrerequisiteProcess
```

这才是本发布树中可见的 Relay 风格图级优化证据。它能帮助理解全栈，但不要把 AGE 的 pass 错算为 “Alchemy 目录实现”。

---

## 4. 第二层发明：为什么图还不够，于是有 Hauk

### 4.1 从标量循环到设备程序

假设图已经决定有一个 `Softmax`，最朴素 kernel 仍可写成：

```python
for row in range(rows):
    m = max(x[row, c] for c in range(C))
    e = [exp(x[row, c] - m) for c in range(C)]
    s = sum(e)
    for c in range(C):
        y[row, c] = e[c] / s
```

它数值正确，还用了稳定技巧 `x - max(x)`。但 NPU 设计者马上会问：

- `C` 很大时，一个 core 怎么分段？段间 maximum / sum 怎么 reduce？
- `C` 不是向量宽度整数倍时，尾部 lane 谁来屏蔽？
- 输入在外部内存，能否先搬到 L1/shared，计算下一块时预取？
- 多 warp 的 partial result 放哪里、何时同步？
- shape 是动态时，分配最大 buffer 还是在运行时切 tile？

这正是 Hauk 的工作空间。它不改变 softmax 的数学定义，而是把“资源、布局、同步、循环绑定、特殊指令”变成可表示、可验证、可 lower 的 IR 节点。

### 4.2 Hauk 的最小结构：不是另起炉灶，而是扩展 `atvm.tir`

Hauk 中的 `HaukOuterExpr`、`HaukOuterVar`、`HaukOuterBuffer`、`HaukOuterFunc` 都继承 `atvm.tir.PrimExpr`，通过 `_ffi_api` 创建 native 节点，见 [`hauk/tir/expr.py:49-119`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/hauk/tir/expr.py)。

也就是说，Hauk 的设计不是“Python 字符串拼汇编”，而是：

```text
Python-like Hauk Script AST
    ↓ parser / registry / template specialization
atvm.tir.PrimFunc + HaukOuterExpr/Stmt/Buffer
    ↓ Hauk-specific transformation passes（FFI）
target-specific intrinsic / LLVM IR / assembly / object
```

这是它和 TVM TensorIR 最接近的地方。一个 `PrimFunc` 本质上是“一个可以独立编译的并行 kernel 函数”；`Buffer` 描述一段 tensor 存储；`Expr/Stmt` 描述读写、循环、条件、intrinsic；`IRModule` 把多个函数装在一起。

### 4.3 `ParamsTensor`：为什么一个 tensor 参数比 `(shape, dtype)` 多这么多字段

普通框架里的 tensor 参数可能只需 shape/dtype。Hauk 的 [`ParamsTensor`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/hauk/script/tir/ty.py) 还可带：

```text
layout, image_type, max_mem_size,
ori_shape / ori_offset,
L2 hint, read/write access attribute,
cache_l1 / cache_l2,
min / typical / max shape,
fold maps / fold mode,
read/write overflow mode,
allocation size, write alignment,
enable_l1_cache_optimize, write_only_once, tsharp_info ...
```

源码构造器在 [`ty.py:85-205`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/hauk/script/tir/ty.py)。它随后由 `gen_buffer_from_tensor()` 映射到 Hauk buffer 声明（[`ty.py:395-435`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/hauk/script/tir/ty.py)），最终 `decl_buffer_from_params_tensor()` 把字段交给 FFI（[`buffer.py:141-193`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/hauk/tir/buffer.py)）。

从 OS/组成原理角度看，这像把“数组指针”升级成“带虚拟/物理布局、缓存策略、访问权限、边界检查约束的 memory descriptor”。它的目的不是装饰类型，而是让 scheduler/codegen 在编译阶段能做合法性检查和资源规划。

### 4.4 模板与 parser：为何不用为每种 shape 写一份 kernel

Hauk 的 `@template`、`@func_register`、`@gop_register` 及 `FunctionTable` 处理 generic/template function。一个模板会把 dtype、立即数、binding 信息、dynamic shape 信息等作为 specialization 参数；parser 利用 `inspect.getsourcelines()` 取 Python 函数源码、转 AST、生成 `PrimFunc`。关键上下文 `IRmodulePaserContext` 保存已解析函数、source map、调用栈、template 调用信息和 target，见 [`hauk/script/parser.py:106-185`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/hauk/script/parser.py)。

用一个**非可运行、只展示设计意图**的骨架理解即可：

```python
@template(dtype=T.tmpl.dtype, vec=T.tmpl.imm.vec_len)
def vector_op(x: T.tensor, y: T.tensor):
    # “vec”不是模型 shape，而是一次 vector 操作的候选宽度。
    for block in T.grid(...):
        with T.vector_scope(...):
            value = T.load(x, predicate=tail_mask)
            T.store(y, value, predicate=tail_mask)
```

同一数学算子可针对 fp16/fp32、vector 宽度、warp binding、tile 大小生成多个特化版本。这解决了“泛型代码易写、硬件代码必须具体”的矛盾。

### 4.5 从 Hauk 源到 `.o`：真正可观察的编译出口

`hauk.driver.build()` 的注释明确描述：

```text
Python kernel → new intrinsic → .o
```

`BuildMode.ALL_WITH_DUMP` 还会输出 `.s` 和 `.ll`；`DEBUG_WITH_PASS` 可打开 pass 调试，见 [`hauk/driver/build_module.py:381-470`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/hauk/driver/build_module.py)。这给了调查者一个非常好的验证原则：

> 不要只读模板并猜它会怎样变快；在有正确 target/runtime 的环境中，用 dump 前后看 IR、LLVM IR、汇编、编译日志和性能 trace。

`kernel_type` 可取 `fermat` / `cayley` / `mix`；源码注释称它们对应不同 core kernel 类型。除非有目标硬件文档或生成汇编佐证，不应仅凭名称把它们展开成某种公开 NPU 微架构。

---

## 5. Hauk 里的“优化”到底是什么：按问题分类，而非按 pass 名背诵

### 5.1 先看暴露的 pass，再诚实处理 FFI 边界

[`hauk/tir/transform/transform.py`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/hauk/tir/transform/transform.py) 是 Python wrapper：绝大多数 pass 直接 `return _ffi_api.Xxx()`。因此下面表中的“做法”有两种强度：带说明的列来自 docstring [D]；只有名字的列只说明能力入口存在，**不能虚构其精确 rewrite 顺序或匹配条件。**

| 朴素程序的痛点 | 可见 Hauk pass/配置 | 已知目的 | 面向的算子形态 |
|---|---|---|---|
| 临时 buffer 在内层循环反复申请，峰值内存高 | `HaukStorageRewrite`、`MmbMemoryAllocation` | 移动 allocation 到尽可能外层，并尝试复用空间形成静态分配计划 | 任意有 workspace 的 vector/matrix kernel |
| L1 cache 与 atomic 读改写可能不一致 | `InsertFlush`、`AUTO_FLUSH_L1` | 在涉及 L1 cached 数据的 atomic 前插 flush/invalidate | scatter、histogram、累加类 kernel |
| load 与 compute 串行等待 | `ForDoubleBuffer` | double buffering | Conv/GEMM/attention 中分块搬运 |
| 并行循环没有映射到硬件执行实体 | `ForLoopBinder`、`PerThreadLower` | multi-warp/multi-block binding；per-warp 降至 per-thread | 大多数并行 kernel |
| 循环控制与常量开销高 | `HaukConstFold`、`ConstExprMotion`、`LastUnroll`、`LoopUnrollExpand`、`DivModOptimize` | 常量折叠、外提、展开、除模优化 | 内层 tile loop / index 计算 |
| 动态 shape 是抽象操作，后端不能直接发指令 | `SetShapeExpand`、`DynamicShapeExpand`、`UpdateBufferProperty`、`ShapeKernelCheck` | 展开 set/get dynamic shape，更新 buffer 属性，校验 shape kernel | dynamic slice/reshape/LLM batch |
| 多维/特殊 layout 难以下到地址 | `ReadWriteLayoutInfer`、`ReadWriteProcess`、`HaukFlattenBuffer` | 推断读写 layout，按 axis separator/stride flatten buffer | ND/SD/P1ND 和 layout transform |
| 尾块有过多分支/predicate | `PredMerge`、`PredIfMerge` | 合并 load/store predicate / if | vector tail、dynamic tile tail |
| 多个自定义函数独立 launch，有中间 buffer | `FunctionInline`、`fused_cfg` | inline；把多个 CustomFunction 的同名/指定 tensor 绑定为 fused kernel 接口 | elementwise + postprocess；shape func + op |

再强调一次：例如 `ForDoubleBuffer()` 的 docstring 能证明“接口意图是 double buffer”，不能证明任何一个 ACE kernel 在这次模型 build 中都启用了它。想升级证据，必须看 `ALL_WITH_DUMP`/pass dump 与板端 profile。

### 5.2 kernel fusion：它和图 fusion 有什么不同

Hauk 的 `build(inputs=[func0, func1], fused_cfg=...)` 能把多个 `CustomFunction` 做 fused build。`fused_cfg` 的值形如：

```text
func_{index}_{input|output}_{arg_name}
```

源码会为每个函数建立 bind，再把拼接后的 `fused_cfg` 放到 pass config，见 [`build_module.py:365-378`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/hauk/driver/build_module.py)、[`build_module.py:737-760`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/hauk/driver/build_module.py)。

它的思维模型是：**kernel 作者已经决定了两个小程序应共享哪些 tensor；编译器将它们作为同一个设备编译单元处理。**

而图 fusion 的思维模型是：**图优化器从大量 `Op → Op` 边里自动识别可融合 pattern。**

两者都可能减少 launch 和中间外存，但入口、证据、适用范围不一样。把它们混称为“融合”会让后续 debug 很痛苦。

### 5.3 Hauk build config 是调度/调试合同，不是模型超参数

`BuildCfg` 中的 `AUTO_FLUSH_L1`、`FORCE_BYPASS_L1`、`ENABLE_L1D_SOLVER`、`ENABLE_BATCH_MODE`、`ENABLE_SYNC_WARP`、`HAUK_ENABLE_PART_UNROLL`、`MAX_HW_WARP_NUM_PER_BLOCK`、`MAX_BLOCK_NUM_PER_CORE` 等，见 [`build_module.py:104-149`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/hauk/driver/build_module.py)。

它们回答的是“同一数学 kernel 怎样使用目标资源”；不要与模型配置里的 `hidden_size`、`head_dim`、Conv `kernel_size` 混为一类。例如源码注释明确：`hauk_enable_part_unroll=False` 时由 LLVM 做 part unroll，`True` 时由 Hauk 做。这是**编译阶段责任划分**，不改变神经网络输出。

---

## 6. 走进几类真实 kernel：从常见到较少见

这里选的例子不是随意挑算子，而是覆盖四个常见性能瓶颈：矩阵乘、卷积 tile、reduce/softmax、位置编码；最后再给一个汽车视觉中较少见的 sparse cross attention。

### 6.1 常见例子 A：Dense/GEMM 的 tile 与后处理融合

#### 先看 naive

对 `A[M,K] × B[K,N] → C[M,N]`，标量实现会让每个 `C[m,n]` 独立扫完整个 `K`：

```python
for m in range(M):
    for n in range(N):
        acc = 0
        for k in range(K):
            acc += A[m, k] * B[k, n]
        C[m, n] = acc
```

它会大量重复从外部内存读 A/B，且 bias、dequant、activation、residual 如果分开写，会额外生成中间 tensor。

#### 再追问：怎么切，才不会装不下也不会吃不满

于是切出 L1 tile：一次让一组 `M_tile × K_tile` 的 A 和 `K_tile × N_tile` 的 B 进入近端存储，算一个 `M_tile × N_tile` 的 C。Hauk/ACE 的 GEMM 模板参数出现了：`tile_l1_m/n/k`、`tile_l0_m/n`、loop count、`with_bias`、`with_deq`、`with_req`、`with_elt`、`with_apb`、dynamic M/N/K、dtype、load/matmul/post-process 实现等。这意味着后处理可以在 accumulator 仍近在 hand 的时候完成，而不是写出再读回。

在 [`ace/kernels/cayley/gemm.py:142-143`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/ace/kernels/cayley/gemm.py) 可直接看到一处固定 micro-tile 基元：

```text
M0 = 32
N0 = 16
```

它是这个模板实现中可见的**矩阵 micro-tile 形状**，不是模型的 `kernel_size`，也不是所有 GEMM 的普适硬件常数。源码随后以 `(tile_l0_m + M0 - 1) / M0`、`(tile_l0_n + N0 - 1) / N0` 向上取整处理尾块。

**涉及算子：**`Linear/Dense`、quantized GEMM；可融合 bias、dequant/requant、elementwise、APB 等后处理。AOPI 的 `dense.json` / `quantized_gemm.json` 也列出允许的 dtype/layout 组合；它们是选择候选 kernel 的合同，不是 tile 数值的来源。

### 6.2 常见例子 B：Conv 的语义 `kernel_size` 与调度 tile，千万别混

Conv2D 的模型参数里，`Kh × Kw` 是卷积核大小，例如 `3×3`。它决定一个输出 tile 需要覆盖多大的输入窗口。若某个输出块高宽是 `tile_ho, tile_wo`，stride 是 `sh, sw`，dilation 是 `dh, dw`，先用白话理解：相邻输出相隔 stride；首尾输出各自还要覆盖一个膨胀后的卷积核，所以输入窗口会比输出大。

精确写法是：

```text
tile_hi = (tile_ho - 1) × sh + (Kh - 1) × dh + 1
tile_wi = (tile_wo - 1) × sw + (Kw - 1) × dw + 1
```

[`ace/kernels/cayley/conv_split_k.py`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/ace/kernels/cayley/conv_split_k.py) 的动态 tiling 模板显式带有 `kernel_size_0/1`、`strides_0/1`、`dilation_0/1`、`tile_ci/co/m`、`loop_ci`、`loop_khkw`、L1 size、elt layout 等，并按这类关系计算可容纳的 tile。它还把某个 SD 路径的宽 tile 对齐到 8（可见 `align_to(8)`）。

这带来两个层次的参数：

| 名称 | 是什么 | 从哪里来 | 是否改变模型数学 |
|---|---|---|---|
| `Kh, Kw` | 卷积核语义大小 | 模型/Conv op 属性 | 会，换 3×3 为 5×5 就是另一个模型 |
| `tile_ho, tile_wo, tile_ci, tile_co` | kernel 一次处理多少输出/通道 | target、L1 容量、shape/profile、schedule | 不会；只是同一 Conv 的执行计划 |
| 对齐 8、ND/SD/P1ND | 物理访问/向量化约束 | layout/kernel 模板 | 不会；边界由 padding/predicate 处理 |

**调查结论：**该树没有给出“所有 Conv 固定使用 K=3、tile=某个常数”的统一表。能证明的是模板接受泛化 `kernel_size/stride/dilation`，并动态重算 H/W tile；不能把某一次硬件调优结果编造成全局 kernel size。

### 6.3 常见例子 C：Softmax 从三遍外存到一个稳定的分块 reduce

Softmax 的数值稳定版本必须先减 max。朴素实现若分成三个独立 op，可能是：

```text
Max(x) → x - max → Exp → ReduceSum → Divide
```

这在逻辑图很清楚，却可能反复写中间结果。`ace/kernels/fermat/softmax.py` 有 vector/matrix、multi-read、last/small-channel 等变体；模板参数包括 `f_align`、warp 数、axis mapping、tail alignment、mask/precision dtype 等。典型结构是：分块求 max → `exp(x-max)` → 分块/跨 warp 求 sum → 除法，尾部用 predicate 保护。

更直观的融合证据在 [`ace/kernels/fermat/softmax_fused.py:62`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/ace/kernels/fermat/softmax_fused.py)：`softmax_matrix_three_pass_reuse_input(input0, input1, output)`。从函数名、shared 临时区和模板参数 `matrix_w/matrix_c/warp_nums/binary/clip_min_value/need_clip` 可知，它把“另一路输入（例如 score + mask/bias）+ 稳定 softmax”作为一个 kernel 处理。

**它解决的根问题：**减少中间 score 的外部读写，同时不丢掉 `x-max` 的数值稳定性。这个模式非常贴近 attention 的 `QKᵀ + mask → softmax` 前半段。

### 6.4 常见例子 D：RoPE 的计算与 LUT 取舍

RoPE 不是简单给 token 加一个标量位置。对于 head 内相邻两半通道 `x0/x1`，它按 position 的 sin/cos 做二维旋转：

```text
y0 = x0 × cos - x1 × sin
y1 = x1 × cos + x0 × sin
```

这样 token 的相对位置关系会进入 Q/K 的点乘。Hauk 的 [`ace/kernels/fermat/rope.py`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/ace/kernels/fermat/rope.py) 有 `rope_vector`，模板携带 `seq_len`、`shape_offset`、`vec_len`、load/compute/store、tail mask、`is_lut`。它处理动态 shape 与尾 lane，并提供两条思路：

- **on-the-fly**：在 kernel 内产生/计算 sin/cos；
- **LUT**：从预先构造的 table 读取 sin/cos。

LUT 省去 trig 计算，却多了 table 带宽、cache/locality 的压力；哪一条快取决于目标、序列长度和缓存，不应先验地说 LUT 一定更快。Alchemy 暴露了 `rope()` 和 `rope_with_table()` 两种前端入口（[`functional.py:3390-3397`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/alchemy/functional.py)），AOPI `rope.json` / `rope_lut.json` 也能看到对应的输入契约。

### 6.5 较少见但很有代表性：sparse cross attention（自动驾驶 BEV/FOV）

⚠️ **这个工程比 LLM self-attention 少见，可先把它当作“专用算子为何值得单独建 IR”的例子，不必作为主线背熟。**

普通 self-attention 是 token 对 token 的稠密关联；自动驾驶 BEV/FOV cross attention 需要根据相机标定和 reference points，从多相机、多尺度 feature map 的少量采样位置取值，再按 attention weights 汇聚。若把它硬拆成通用 `Gather + GridSample + Mul + Reduce`，会产生大量动态索引、边界判断、临时张量和 launch。

本树可见：

- Alchemy 有 `cross_attention()`、`sparse_cross_attention()` native custom op 前端；
- AOPI `cross_attention.json` 描述了 14 输入，`sparse_cross_attention.json` 描述了更丰富的输入契约；
- Hauk 的 [`ace/kernels/fermat/sparse_cross_attention.py`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/ace/kernels/fermat/sparse_cross_attention.py) 有 `sparse_bev2fov`、`weight_filter`、`sparse_xatn_core` 等专用程序。

后者模板参数包含 camera/head/level/point 数、`warp_range` 等；代码可见 32-lane vector 形式的坐标/权重处理、有效性 predicate、四邻居双线性插值与按 head 累加。这是“自定义 IR 不只是为了更快的 MatMul，也为了把复杂的 domain contract 变成一条可优化 kernel”的好例子。

---

## 7. Attention 专项调查：从 Qwen2 图到 native op，再到 kernel 边界

### 7.1 先构造一个 Qwen2 attention 的图心智模型

`Qwen2Attention` 的构造器先决定：

```text
head_dim = config.head_dim，若未给则 hidden_size / num_attention_heads
num_key_value_groups = num_attention_heads / num_key_value_heads
scale = head_dim^(-0.5)
```

这里 `num_key_value_groups` 是 GQA（Grouped Query Attention）的关键：Q head 可以比 K/V head 多，从而减小 KV cache。源码在 [`alchemy/models/qwen2/model.py:94-121`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/alchemy/models/qwen2/model.py)。

生成的逻辑顺序是：

```text
hidden_states
   ├─ q_proj Linear ─ view/split ─ RoPE ─┐
   ├─ k_proj Linear ─ view/split ─ RoPE ─┼─ KV cache / prefix 组织 ─ FlashAttention ─ merge heads ─ o_proj Linear
   └─ v_proj Linear ─ view/split ────────┘
```

请注意一个非常实在的源码事实：这份 Qwen2 实现用**三条独立的** `q_proj/k_proj/v_proj` Linear，而不是在 Python 图里直接一个 fused QKV Linear。因此可以说“后端可能有机会融合/打包”，但不能说“Alchemy 的 Qwen2 前端已经构成 fused QKV op”。

### 7.2 普通 batch 模式：为什么 causal mask 的 diagonal 是动态属性

普通 batch 分支把 Q/K/V 组织成：

```text
Q: [B, Hq, Sq, D]
K: [B, Hkv, Sctx + Sq, D]
V: [B, Hkv, Sctx + Sq, D]
```

其中 `Sctx = prefix_len + cached_len`。为了让当前 query 的第一个 token 能看见全部历史 KV、但不能看未来 token，causal mask 的对角偏移需要 `Sk - Sq`。源码不是把这个数固定进模型，而是构造：

```text
mask_diagonal = length(K) - length(Q)
min = [0]
max = [int32_max]
```

再作为 `diagonal_tensor` 接到 `flash_attention` 的第 4 号输入位置（零基计数），见 [`qwen2/model.py:212-266`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/alchemy/models/qwen2/model.py) 和 [`functional.py:3415-3452`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/alchemy/functional.py)。

这是“动态 shape 不只是 tensor 维度，连算子属性也可能动态”的好例子。`set_dyn_attr_value_range()` 给 native planner 一个属性范围，避免把 cache 长度硬编码。

### 7.3 continuous batching：不是简单把 batch 维拼起来

continuous batching 中，不同请求在同一轮的 token 数不同。Qwen2 分支把当前 token 按 `batch_offset` 切分，再把 Q 拼成 token-flat 形式；prefix K/V、每个请求的 cached K/V 和 current K/V 则保留成 list。最终调用：

```text
BatchFlashAttention(
  q, prefix_k, prefix_v,
  batch_cached_k[], batch_cached_v[], cur_k[], cur_v[],
  batch_offset, mask[], scale, causal, ...)
```

源码见 [`qwen2/model.py:160-206`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/alchemy/models/qwen2/model.py)、[`functional.py:3454-3495`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/alchemy/functional.py)。这是一种**形状/请求组织的 IR 优化**：避免为了凑齐 batch 而将不同长度的所有请求 padding 成同一个最大长度。

#### 一个应先验证再使用的源码一致性问题

`batch_flash_attention()` 的形参默认 `texture_len_tiling_factor=1`，但函数体要求它必须等于 `-1`，否则抛 `NotImplementedError`。Qwen2 continuous 分支没有显式传该参数。这个现象至少说明当前 Python snapshot 的默认值与检查条件不一致；可能是未覆盖路径、版本演进残留或上层有别处改写。**它不是“BatchFlashAttention 不可用”的结论，而是应在实际版本做最小 smoke test 的检查项。**

### 7.4 FlashAttention：能确认的 IR 事实，和不能偷推的算法细节

Alchemy 的 `flash_attention()` 是强证据：它不是先构造四个普通 op，而是调用一个 native `add_flash_attention(query, key, value, mask, scale, is_causal, texture_len_tiling_factor, diagonal)`；动态 diagonal 时再 `set_input(4, ...)` 和设 range。也就是说，**graph IR 在此处已经把整段 attention 标为一个语义原子。**

ACE 的 `aopi/json/N93X/flash_attention.json` [M] 进一步给出当前可见的接口合同：Q/K/V 与输出为 fp16，mask 可为 fp16/fp32/bool，diagonal 为 int32，逻辑布局为 ND；`batch_flash_attention.json` 列出 Q、prefix KV、cache/current KV、batch offset、mask 等更长的输入列表。

但必须停在这里：本次树下没有找到与 `flash_attention` 一一对应、可读的 Hauk DSL kernel 模板。可见的 Hauk `softmax.py` / `softmax_fused.py`、`rope.py` 是非常相关的低层 building blocks，**不能仅凭它们就断言 native FlashAttention 正是某个特定 online-softmax tile 算法。**其 tile 尺寸、是否重算、KV 分块策略、各层存储使用，需要 `.ll/.s` dump、native source 或官方 kernel 文档才可确认。

这条“证据边界”尤其重要：FlashAttention 在论文和 CUDA 世界有常见实现套路，但不能把外部实现细节自动移植到这块 NPU。

#### 把 AOPI 当作“可下发合同”，不是性能论文

下表直接来自 `ace/aopi/json/N93X/*.json` 的 `name`、`num_input`、`data_type`、`physical_layout` 字段。它回答“这个 native op 至少接受什么”，但不回答“哪个 tile 最快”。

| native op | 输入数 → 输出数 | 可见关键 dtype 合同 | 可见 layout 合同 | 对 IR 调查的意义 |
|---|---:|---|---|---|
| `flash_attention` | 5 → 1 | Q/K/V/output fp16；mask `fp16/fp32/bool`；diagonal int32 | 全部 ND | `diagonal_tensor` 是显式动态属性/输入，而非 Python 私有变量 |
| `batch_flash_attention` | 9 → 1 | q、prefix/cache/current KV、mask/output 均 fp16；`batch_offset` int32 | 全部 ND | variable-length 请求的 list/offset 组织已经进入算子接口 |
| `rope` / `rope_lut` | 4 → 1 | input/output 可见 fp16 或 fp32 组合；offset/shape_offset int32；table fp16/fp32 | 常见 ND；部分组合 input/output 为 SD | LUT、layout 和数值 dtype 是 kernel legality 的一部分 |
| `cross_attention` | 14 → 1 | sampling/weights/FOV/BEV/output fp16；内外参 fp32 | 全部 ND | 相机几何矩阵以 fp32 留在专用算子合同中 |
| `sparse_cross_attention` | 15 → 1 | points/feature/BEV/output fp16；reference/内外参 fp32；`valid_num_query` int32 | 全部 ND | 动态有效 query 数、几何和 feature 聚合不必拆成通用图节点 |

同一份元数据还显示 `dense`、`batchmatmul`、`conv2d` 有 fp16、fp32、int8×int8→int32、以及部分 int8×int4→int32 的候选区域；`quantized_gemm` 单独把 input/weight elements、scale、zero point、quantized bias 列为 7 个输入。这与 Alchemy 的 Dense Q/DQ wrapper 相呼应，却仍不能证明某个具体模型实际选到了哪一条 feasible region。

### 7.5 Attention 周边算子与 IR 优化的对应表

| Attention 环节 | 逻辑张量/作用 | 本树中的优化表达 | 可见参数/约束 | 证据 |
|---|---|---|---|---|
| Q/K/V projection | `[*, hidden] → [*, H×D]` | Dense/GEMM template、可做 Q/DQ | `head_dim`、Hq/Hkv、M/N/K tile、`M0=32,N0=16` 局部实现 | [S] |
| reshape/transpose/split | 改逻辑视图/组织 head 与 batch | 图 op；layout 会影响物理搬运 | continuous 与非-continuous 的维度不同 | [S] |
| RoPE | Q/K 的 position rotation | native `rope` op；Hauk vector/LUT kernel | `theta`、offset、`vec_len`、LUT 开关 | [S] |
| cache/prefix | 历史 K/V 重用 | runtime 输入/输出 + dynamic range；Batch Flash 的 list 接口 | max prefix/cache length、batch offset | [S] |
| QKᵀ + mask + softmax + PV | attention 主体 | native Flash/BatchFlash semantic op | scale、causal、diagonal、mask dtype、tiling factor | [S]/[M] |
| score softmax 单独实现 | row-wise reduce | Fermat vector/matrix variants；fused mask/bias softmax | `f_align`、warp、matrix width、tail predicate | [S] |
| 视觉跨域 attention | sparse sampling + bilinear + weighted reduce | native sparse/cross op + specialized Hauk kernels | camera/head/level/point、32-lane vector 等 | [S]/[M] |

### 7.6 `texture_len_tiling_factor` 到底算什么参数

它名字中有 tiling，却不是 Conv 的 `kernel_size`。在 FlashAttention API 中它是 native attention 属性，默认/合法性与具体 runtime 的实现需要以对应版本验证；它很可能影响序列方向如何分块，但源码只把它**原样传递**，未显示选择公式。

因此应把它放入“**native semantic-op 的调度提示/属性，语义未完全公开**”栏，而不是列为已知数值 tile。对所有这种参数，一个好习惯是记录四件事：

1. 谁设置它（模型配置、Builder、runtime 还是 kernel template）；
2. 进入哪一个 IR node/attribute；
3. 可见的合法范围/默认值；
4. 改它后是否有 IR、汇编、profile 的可重复差异。

---

## 8. 参数调查总账：模型参数、profile、tile、layout 的四本账不能混

用户常问“优化后 Kernel 大小是多少”。这句话至少可能指四种不同东西；下面按源码能证实的粒度分账。

| 类别 | 例子 | 作用 | 本次能直接证实的数值/形式 | 不该误说成 |
|---|---|---|---|---|
| 模型语义参数 | Conv `Kh/Kw`、stride、dilation；attention `head_dim`、scale | 决定模型函数本身 | Conv 模板泛化携带 `kernel_size_0/1`；Qwen `scale=head_dim^-0.5` | 某个编译 tile |
| 动态 profile | input min/typical/max；KV 上限 | 决定可接受 workload 范围/典型点 | `set_shape_range` + `set_typical_shape`；LLM profile 有 max batch/seq/cache | 具体硬件微 tile |
| kernel schedule 参数 | `tile_l1_m/n/k`、L0 tile、`vec_len`、`f_align`、warp 数 | 决定同一数学算子的资源分配和循环切分 | 模板参数可见；GEMM 一处 `M0=32,N0=16`；Conv 可见宽方向 8 对齐 | 模型 hidden size 或 Conv kernel size |
| layout/存储参数 | `ND/SD/P1ND_128B`、L1/L2 cache、allocation | 决定字节如何排布/流动 | `ParamsTensor`、AOPI JSON、`set_physical_layout` | 数学 transpose 本身 |

特别列出几个可以落笔的值：

- **GEMM micro-tile：**`M0=32`、`N0=16` 在 `ace/kernels/cayley/gemm.py` 的特定实现直接赋值；不要推广为全产品唯一 tile。
- **Conv 的对齐：**`conv_split_k.py` 有 `align_to(8)` 的宽 tile 处理；这是特定 layout/path 的对齐行为，不是“卷积核宽度固定 8”。
- **Vector lane：**多处 Fermat kernel 使用 `T.vector(32, ...)`；这表明部分实现按 32 个元素的 vector primitive 写，但模板 `vec_len/f_align` 仍可能更高层决定实际工作块。
- **Qwen 默认配置值：**本树 `LLMProfile` 的默认最大 batch/prompt/seq/cache 是配置默认值，不等于所有已编译模型的实参。要看 build 日志或 `BuildConfig.to_dict()` 才能知道一次 build 的真值。
- **Attention `head_dim`：**由模型 config 决定，未给时 `hidden_size / num_attention_heads`；它不是 compiler 可随意调成更好对齐的参数，除非你改变模型结构/权重。

---

## 9. 把 Alchemy/Hauk 放到 TVM、Relay/Relax、MLIR、QNN、Apple M 系列中校准

### 9.1 先找共同骨架：它们确实“大同小异”的部分

所有成熟推理编译器最终都要完成同一件事：把“张量函数”变成“某设备可执行程序”。最常见的分层是：

```text
模型图：      哪些算子相连？shape/type/quantization 是否合法？
      ↓
图优化：      rewrite、常量折叠、融合、partition、layout 选择
      ↓
kernel IR：   loop/tile/buffer/thread-or-warp mapping/synchronization
      ↓
target code：library call、vendor graph、LLVM/assembly/binary
      ↓
runtime：     内存池、KV cache、执行、profile、artifact 加载
```

这正是 [Apache TVM Relax 文档](https://tvm.apache.org/docs/deep_dive/relax/index.html) 所说的高层 graph abstraction 与 TensorIR 跨层协作的动机；也是 [MLIR Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/) 用 target legality + rewrite patterns + type conversion 逐步 legalize 的通用模式。

### 9.2 逐项对应，而不是宣称“它们相同”

| 问题 | AllSpark 本树可见对应 | TVM Relay / Relax / TIR | MLIR | Qualcomm QNN | Apple Core ML / M 系列 |
|---|---|---|---|---|---|
| 高层图由谁保存 | native `ANetwork`；Alchemy 是前端 wrapper | Relay `Function` / Relax `IRModule` | `Operation/Value/Region` + dialect | ONNX 图被 EP partition，再构 QNN graph | Core ML MIL / `mlprogram` 模型 |
| 图级 rewrite 在哪 | 相邻 AGE 的 Relay pass；Alchemy Python graph rewriter 目前未启用 | Relay transform、Relax DPL/transform | `RewritePattern` / conversion pass | EP capability、layout transform、QNN graph finalization | converter / MIL optimization，后端多为 opaque |
| kernel 级表达 | Hauk 的扩展 `atvm.tir.PrimFunc/Buffer` | TensorIR / TE / schedule | `linalg/affine/scf/gpu` 等 lower | SDK backend graph/context，公开细节较少 | Core ML runtime/ANE backend，公开细节较少 |
| target contract | AOPI JSON + target + `ParamsTensor` layout/cache | Target、tensor intrin、codegen ABI | dialect legality、conversion target、target lowering | supported ops/dtype/HTP backend、context binary | deployment target、ML Program、Compute Units |
| 可观测产物 | Hauk `.ll/.s/.o`、native artifact | TIR dump、LLVM/CUDA/library module | IR textual dump、pass pipeline | QNN context binary、profile/trace | `.mlpackage` / compiled model、Xcode Instruments |

### 9.3 具体差异，正好解释为何不能一一类比

#### TVM：Hauk 更接近 TIR，Alchemy 更像前端 binding，不是 Relax 的同义词

- Relay/Relax 是显式可操作的 graph IR；你通常能把 `IRModule` 打印、写 pass、模式匹配。
- Hauk 使用的 `atvm.tir` 形状与 TensorIR 的思想接近：buffer、loop、lowering、target code。
- Alchemy 没把 native graph node/edge 以 Python `IRModule` 公布出来；它更像一个 PyTorch 风格的 builder front end。
- 本树的 AGE 直接用 `atvm.relay`，所以若你想做“图 pass 对照学习”，AGE/Relay 才是较近的桥；若你想学 kernel schedule，Hauk/TIR 才是较近的桥。

#### MLIR：最相像的是“合法化”思想，不是 API 名字

MLIR 的 conversion framework 明确由 `ConversionTarget`、rewrite patterns 和可选 type converter 组成，并可定义“某 op 仅在某些 dtype/attribute 下合法”。AOPI JSON 列出 op 的 dtype/layout 组合时，概念上很像 target legality table：某个 FlashAttention 组合能下发，另一个就得改写、fallback 或报错。

但 Hauk 源码没有显示它使用 MLIR；它走的是 `atvm`/TVM 血统 IR 和自己的 FFI。因此“概念等价”不等于“工程依赖等价”。

#### Qualcomm QNN：公开的是 graph/partition/context 视角，内核调度仍是 SDK 黑盒

[ONNX Runtime QNN EP 文档](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html) 描述：QNN EP 从 ONNX 构造 QNN graph，使用 HTP backend 下发 NPU；支持 QNN context binary、图 finalization 优化选项、VTCM 配置与 profile。它很适合作为“图被一个 vendor backend 接管、再序列化为 context”的现实参照。

差异是 QNN SDK 的内层编译器并不开源；公开 ONNX Runtime EP 主要告诉你 partition、支持表、配置和 profile 入口。它不像 Hauk release 中这样能直接读到模板式 kernel 源。实践时应看：是否完整 offload（禁 CPU fallback）、context 是否复用、每 op trace 是否出现，而不是只看 session 创建成功。

#### Apple M 系列 / Core ML：适合在 MacBook Air M1 上练“图与部署合同”，不是用来编 Hauk

[Core ML Tools 的 ML Program 转换文档](https://apple.github.io/coremltools/docs-guides/source/convert-to-ml-program.html) 展示了 `mlprogram` 这一可部署图形表示；Core ML runtime 再在 CPU/GPU/Apple Neural Engine 之间执行。Core ML Tools 是公开的 Python 工具，而 Apple 的最终 kernel schedule/ANE codegen 大多不以可改的 TIR 形式暴露。

所以在 M1 上你可以很好地练：模型转换、shape/precision contract、算子支持/fallback、模型包检查、端到端 latency/能耗 profiling；但不能拿 macOS 的 Core ML backend 直接验证 N93X Hauk 的 L1、Fermat/Cayley tile。

### 9.4 对你的学习路线最有价值的统一语言

以后看到任何厂商工具链，先问下面五句话：

1. 它的 **graph IR** 是什么对象？我能 dump 吗？
2. 支持表 / **legality contract** 放在哪里？dtype、shape、layout 限制是什么？
3. 图 rewrite 与 kernel schedule 分别在哪一层发生？
4. 动态 shape、layout、quantization、memory ownership 由谁维护？
5. 我能用什么 artifact、IR dump、profile 把“可能优化”升级为“这次真的优化了”？

这五问比记住某个框架的 API 更可迁移。

---

## 10. 如何自己复核：从源码检视到 M1 / SoC 的可验证实验

### 10.1 在当前源码树先做只读调查

下面三个命令不会修改模型；每个参数都尽量只做一件事。

```bash
# rg：递归查找；-n：显示行号。看 Attention 前端怎样创建 native op。
rg -n "def (flash_attention|batch_flash_attention|rope)|add_flash_attention" \
  work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/alchemy

# 看 Hauk 的 pass 入口。这里能证实“暴露了什么”，不能替代 native FFI 实现。
rg -n "def (HaukStorageRewrite|ForDoubleBuffer|HaukFlattenBuffer|DynamicShapeExpand)" \
  work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/hauk/tir/transform

# 把模板参数与实际数字分开看；避免把任意 tile 参数误读成 Conv kernel size。
rg -n "kernel_size_[01]|tile_l1_[mnk]|M0 =|N0 =|align_to\\(8\\)" \
  work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/ace/kernels
```

进一步可沿这条链 Peek Definition：`F.flash_attention` → `network.add_flash_attention` → binding 类型定义/动态库符号（若环境允许）→ AOPI JSON → Hauk/ACE 模板或 native dump。每多走一层，结论的证据等级才会提升。

### 10.2 有 Hauk 正确 target 环境时：优先 dump，不要先改 tile

Hauk 的 `BuildMode.ALL_WITH_DUMP` 是最有价值的观察接口：它的文档承诺输出 `.o`、`.s`、`.ll`。一个合理的小实验是固定同一个 toy softmax 或 GEMM：

1. 固定 dtype、shape、输入随机种子，先验证输出；
2. 开 `ALL_WITH_DUMP`，保存生成物和 build config；
3. 只改一个参数，例如 `f_align` 或一个 L1 tile；
4. 比较 IR/汇编、L1 allocation、kernel latency、tail shape 的正确性；
5. 再决定这个参数是否真有因果作用。

不要在没有 target 可执行验证的 M1 上直接调 Hauk tile；它是 N93X/P/E target 相关的 DSL。M1 最适合学习统一概念和建立对照实验。

### 10.3 MacBook Air M1：低成本、可迁移的对照练习

在 macOS 上可用 `uv` 单独建实验环境，避免污染系统 Python：

```bash
# uv venv：创建隔离环境；uv pip：只向该环境安装依赖。
uv venv .venv
source .venv/bin/activate
uv pip install coremltools torch
```

建议只做一个极小的 `Linear → Add → GELU` 或 `MatMul → Add(mask) → Softmax` 模型，分别导出/检查：

- PyTorch eager：确认数学基线；
- TVM Relax/TIR：打印 graph 与 kernel IR；
- Core ML `mlprogram`：检查转换是否接受 shape/dtype，记录 deployment target；
- 如果公司台架恰好是 Qualcomm：同一 ONNX/QDQ 模型走 QNN EP，开启“不允许 CPU fallback”，看 QNN profile/context binary；
- N93X/P/E 台架：用 AllSpark/Hauk dump 和设备 profile 比较。

这样你练到的是同一个科学方法：**固定模型语义，改变一个编译层变量，观测图/IR/artifact/profile 的差异。**而不是在不同设备上盲比一个“总耗时”。

---

## 11. 最后收束：这套设计的本质是什么

回看最初那个 naive Python 循环，所有复杂度都来自一个事实：数学公式没有告诉机器“数据在哪、何时搬、谁并行、什么形状最常见、哪些操作可以作为一个原子”。

于是这套系统自然分层：

- **Alchemy** 解决“模型作者怎样方便而准确地描述图”。它用 Module/Tensor/Parameter 保留熟悉的 Python 体验，又把真正节点落到 native `ANetwork`；profile、layout、量化包装和 custom op 是它在图边界上给编译器的额外信息。
- **Hauk** 解决“一个设备算子怎样被明确地写成可调度程序”。它把 tensor 的 layout/cache/dynamic-shape 合同、vector/matrix/warp/block、buffer、predicate、template specialization 放进扩展 TIR。
- **native Builder/AGE/ACE** 解决两者的接缝：图级合法化/布局/内存/算子选择，和目标 kernel/artifact/runtime 的落地。

因此，最好的总结不是“它有几十个 pass”，而是：

> **把原本隐含在模型代码、硬件手册和运行时经验里的约束，逐层显式化为图节点、shape/profile、layout、buffer、schedule 和 target contract；每一层只解决自己看得见的问题。**

这也解释了为什么不同框架看起来大同小异：TVM 的 Relax/TIR、MLIR 的 dialect conversion、QNN 的 graph/context、Core ML 的 MIL/ML Program，都是在不同开放程度、不同目标硬件和不同 API 取舍下，反复回答同一组问题。真正的差异常在“谁能看到/修改 kernel 级细节”“动态 shape 支持边界”“layout/quantization 合同”和“如何证明优化真的发生”。

---

## 附录 A：本次源码证据索引

| 主题 | 关键位置 |
|---|---|
| Alchemy `Network`、native graph binding、weight lifetime | [`alchemy/network.py`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/alchemy/network.py) |
| Builder profile、native flags、physical layout hook | [`alchemy/builder.py`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/alchemy/builder.py) |
| Flash/BatchFlash/RoPE 前端 | [`alchemy/functional.py:3390-3495`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/alchemy/functional.py) |
| Dense Q/DQ wrapper 与当前接入范围 | [`alchemy/quantize.py`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/alchemy/quantize.py) |
| Qwen2 attention、GQA、KV cache、两个 Flash 分支 | [`alchemy/models/qwen2/model.py:94-277`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/alchemy/models/qwen2/model.py) |
| Hauk expression/buffer 扩展 | [`hauk/tir/expr.py`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/hauk/tir/expr.py)、[`hauk/tir/buffer.py`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/hauk/tir/buffer.py) |
| Hauk ParamsTensor、dynamic shape/binding | [`hauk/script/tir/ty.py`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/hauk/script/tir/ty.py) |
| Hauk parser/template/build/dump | [`hauk/script/parser.py`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/hauk/script/parser.py)、[`hauk/driver/build_module.py`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/hauk/driver/build_module.py) |
| Hauk pass API 与 docstring | [`hauk/tir/transform/transform.py`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/hauk/tir/transform/transform.py) |
| 可见 Relay 图优化（相邻 AGE，边界校正） | [`age/compiler/build_module.py`](../work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/age/compiler/build_module.py) |
| GEMM、Conv、Softmax、RoPE、sparse attention 模板 | `ace/kernels/cayley/gemm.py`、`ace/kernels/cayley/conv_split_k.py`、`ace/kernels/fermat/{softmax.py,softmax_fused.py,rope.py,sparse_cross_attention.py}` |
| 算子 dtype/layout/输入签名契约 | `ace/aopi/json/N93X/{flash_attention,batch_flash_attention,rope,rope_lut,cross_attention,sparse_cross_attention,dense,conv2d}.json` |

## 附录 B：公开资料（用于概念对照，不替代本地源码证据）

- [Apache TVM: Relax 高层图 IR 与 TensorIR 跨层优化](https://tvm.apache.org/docs/deep_dive/relax/index.html)
- [Apache TVM: Relax VM / lowering 结构](https://tvm.apache.org/docs/arch/relax_vm.html)
- [MLIR: Dialect Conversion（legality、pattern、type converter）](https://mlir.llvm.org/docs/DialectConversion/)
- [MLIR: Pass Management](https://mlir.llvm.org/docs/PassManagement/)
- [ONNX Runtime: QNN Execution Provider / HTP / context / profiling](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
- [Apple Core ML Tools: 转换为 ML Program](https://apple.github.io/coremltools/docs-guides/source/convert-to-ml-program.html)
- [Core ML Tools 开源仓库](https://github.com/apple/coremltools)