# Swin Transformer 阶段二：实测记录与经验卡

一句话心智模型：**最终分类结果一致，也需要检查中间表示、误差分布和验证器本身，才能知道转换链证明了什么。**

本记录来自 2026-09-20 在本机执行的阶段二教程。代码由 AI 生成并自动验证；不把这次执行写成用户已独立完成的排障经历。建议打开 [教学 Notebook](02_stage2_conversion_correctness.ipynb)，先预测、再运行、最后自己解释证据。

## 实验约束与证据入口

- 模型：`microsoft/swin-tiny-patch4-window7-224`，revision `d00d478bfaf1f417d34a1186673ad4b4be264c51`，继承阶段一权重文件 hash。
- 平台：本机 Apple Silicon，macOS 26.6.2 arm64；torch 2.7.0、transformers 4.51.3、coremltools 9.0、ONNX 1.22.0、ORT 1.29.0。
- 输入：阶段一已冻结的 3 个相关图片输入，加 normalized-zero、固定种子随机数两个数值探针；全部为 FP32 `[1,3,224,224]`。探针不属于分类验证集。
- 对照：CPU FP32 PyTorch；ORT CPU 开/关图优化；TorchScript；Core ML FP32/FP16 CPU；同一生产 FP16 包的 `ALL` 配置。
- 阈值：FP32 `atol=1e-4, rtol=1e-4`；FP16 `atol=0.05, rtol=0.01`。逐元素判断，阈值在运行前设定，本次未为通过门禁而调整。
- 本轮不测性能，不报告 accuracy，不宣称已证明 ANE 执行。

[完整执行记录](../../results/swin_stage2/validation-20260920/executed.ipynb) · [运行 manifest](../../results/swin_stage2/20260920-152006-750012/manifest.json) · [机器可读总结](../../results/swin_stage2/20260920-152006-750012/summary.json) · [全部边界比较](../../results/swin_stage2/20260920-152006-750012/boundary_quality.csv)

这些大体积产物只保存在本机且被 Git 忽略；新机器需要按 Notebook 重跑。图中数值来自同一轮 raw tensors；增加的逐元素分析也使用这批冻结输出，没有重新选择更有利的样本。

## 这次实际证明了什么

|路径或检查|观测|可以下的结论|
|---|---|---|
|FP32 主链|全部选定样本/边界通过；TorchScript 与 eager 的输出逐元素相同|本组固定输入上的转换数值对齐成立|
|ONNX 生产模型|logits 最大绝对误差约 `3.815e-6`|当前 ORT CPU 路径通过筛查|
|Core ML FP32 生产模型|logits 最大绝对误差约 `4.292e-6`|当前 Core ML CPU 路径通过筛查|
|Core ML FP16 生产模型 CPU|logits 最大绝对误差约 `0.038674`，5/5 Top-1 一致|最终输出通过本轮筛查；不等于任务准确率验收|
|Core ML FP16 调试模型 CPU|五个样本均有中间边界超过阈值|中间特征仍有待解释的偏差|
|ONNX 人为故障|结构检查通过，5/5 输入首次失败边界都为 `stage1_down`；回滚恢复|负控能发现并粗定位这类已知故障|
|Patch Merging 局部对照|最大绝对误差约 `6.199e-6`|冻结的这一个真实模块输入上，PyTorch/ORT 对齐|
|输入契约|应用拒绝 batch=2、高225、float64、NaN；ORT 拒绝前三种|固定 shape/type 的错误请求有明确失败路径|

## Case 01：复制中间程序失败，问题还没到推理阶段

**类型：实际发生的教程实现问题，已修复。**

- **Symptom / Baseline：** PyTorch、ONNX 和 TorchScript 已运行；准备从 FP32 MIL 复制两个精度分支时，`copy.deepcopy(program)` 抛 `RecursionError`。
- **Hypotheses：** 是模型转换不支持某个 op，还是 Python 对象复制本身失败？
- **Evidence：** [原始失败 Notebook](../../results/swin_stage2/validation-20260920/failed_mil_deepcopy.ipynb) 的栈停在 `deepcopy`，尚未进入该次 `ct.convert` 调用。不能把它归因于 Core ML runtime 或 FP16 计算。
- **Root cause：** 本机这份 Swin MIL 对象图的递归深复制超过 Python 当时的递归限制。没有证明其他 MIL 图也会如此。
- **Fix：** 各分支从同一份 TorchScript 重新生成独立 FP32 MIL，再应用目标精度；避免前一轮 pass 改写影响下一轮。
- **Before / After：** 原流程在构建包前异常退出；修改后四个调试/生产、FP32/FP16 包都成功构建并执行。
- **Side effect：** 增加转换工作量，实验未测其时间成本；没有改变输入、权重或筛查阈值。
- **Regression / Lesson：** 全链重跑通过，原失败日志保留。先按调用栈判断失败属于 Python、转换、编译还是执行，再决定改哪一层。

## Case 02：ONNX checker 通过，仍然可以是错误模型

**类型：人为故障注入训练，已定位并回滚。**

- **Symptom / Baseline：** 在原本通过的 debug ONNX 上，只对 `stage1_down` 的第0个 feature 加0.5；上下游连接保持合法。
- **Hypotheses：** 检查器能否发现数值语义被改变？比较器能否找到第一处偏差？注入是否真的影响下游？
- **Evidence：** [故障记录](../../results/swin_stage2/20260920-152006-750012/fault_injection.json)、[各边界误差](../../results/swin_stage2/20260920-152006-750012/fault_boundaries.csv)。`onnx.checker` 通过；五个输入均在 `stage1_down` 首次超过阈值；后续边界也出现差异。
- **Root cause / Fix：** 已知的 Add 节点改变了真实数据流；重新加载 pristine ONNX 后恢复。
- **Before / After：** 无注入对齐 → 注入失败 → 回滚对齐，三个状态都有比较结果。
- **Side effect：** 这是训练图，不是可交付模型，不能写成“修复了 ONNX 编译器 bug”。
- **Regression / Lesson：** 结构合法与数值正确是两道不同的门。此实验只证明稀疏语义边界上的定位能力，没有证明可定位任意算子或任意故障。

## Case 03：中间特征不通过，最终 logits 仍通过

**类型：实际数值观测，根因尚未定位到算子。**

FP16 CPU 调试路径的各边界最坏绝对误差如下。每行取全部5个输入中的最大值，不是同一张图片的误差轨迹。

|边界|最坏 max absolute error|所有输入是否通过逐元素筛查|
|---|---:|---|
|embedding|0.021196|是|
|stage0_down|0.065253|否|
|stage1_down|0.065290|否|
|stage2_down|4.114216|否|
|stage3_out|2.922871|否|
|logits|0.038674|是|

- **Symptom / Baseline：** 与 PyTorch FP32 比较，原图/镜像输入首次失败在 `stage0_down`，中心区域输入在 `stage1_down`，两个数值探针在 `stage2_down`。FP32 对照通过。
- **Hypotheses：** 精度改写带来的误差累积、某个模块的局部数值敏感性、后端具体实现差异，都需要进一步区分。当前证据不够把它称作实现 bug，也不够说“只是正常舍入”。
- **Evidence：** [首次失败边界](../../results/swin_stage2/20260920-152006-750012/first_failures.csv)、[具体元素检查](../../results/swin_stage2/20260920-152006-750012/fp16_element_inspection.csv)、[margin 与 Top-1](../../results/swin_stage2/20260920-152006-750012/logit_margins.csv)。
- **具体数字：** `input_0/stage0_down` 有1个元素越界，reference 约 -0.465508，candidate 约 -0.530762，误差约0.065253，预算约0.054655。随机数探针的 `stage2_down` 最大差约4.114216，对应 -29.426716 → -25.312500。不能仅用整体 cosine 把这些元素隐藏。
- **另一种边界：** 镜像输入 `stage3_out` 的最大绝对误差元素在相对误差预算内，但另外30个元素超预算。“最大绝对误差”与“最严重违反门禁的元素”不是同一个概念。
- **控制组：** 本轮 debug/production 在相同 Core ML 精度和 CPU 配置下的 logits 完全相同，因此本次没有观察到额外输出改变最终 logits；这不证明其他配置也不会受到插桩影响。
- **Root cause / Fix：** 待定位，尚未修改模型。不能写“已解决 FP16 精度问题”。下一步冻结第一个失败区间的输入，对 block、downsample 和其中的归一化/线性层做细化对照。
- **Side effect / Lesson：** 全部5个输入的 Top-1 仍一致，但没有分类真值。保留 FP16 中间筛查失败状态；在解释前不能把整条 FP16 链标为全面验收通过。

还有一个可手算的现象：normalized-zero 的参考 margin 约0.027947，FP16 CPU 最大 logit 误差约0.016466，margin 小于两倍误差，却依然没有 Top-1 翻转。这说明“margin > 2×max_error”是充分条件，不是必要条件。

## Case 04：Patch Merging 的 shape 相同，元素顺序不同

**类型：玩具负控 + 真实模块局部对齐。**

- **Symptom：** 把 NHWC 的四个位置直接 reshape 到 `[1,1,8]`，shape 符合预期，内容却不符合 Patch Merging 的拼接顺序。
- **Baseline / Evidence：** Notebook §7 用具体 FP32 内存表演算：正确 `[10,11,30,31,20,21,40,41]`，错误 `[10,11,20,21,30,31,40,41]`。
- **Root cause / Fix：** reshape 保留原来的逻辑元素次序，没有执行源码里的四个 slice 再 concat；按真实顺序实现后，玩具输出匹配手算。
- **局部回归：** 抓取真实 `encoder.layers[1].downsample` 输入 `[1,784,192]`，冻结后导出 ONNX；[局部结果](../../results/swin_stage2/20260920-152006-750012/local_patch_merging.json)通过，raw input/output 单独保存。
- **边界：** 该局部检查使用 ORT FP32，不能拿它宣称 Case 03 的 Core ML FP16 异常已经排除或修复。
- **Lesson：** shape/type 是必要检查；元素对应关系与模块语义还需要数值验证。

## Case 05：静态产物要有可验证的输入边界

**类型：契约负控，已验证。**

调用前分别送入 batch=2、高度225、float64、NaN，应用全部拒绝；绕过应用直接调用 ORT，前三种由 runtime 报 shape/type 错误。证据见 [负控结果](../../results/swin_stage2/20260920-152006-750012/contract_negative_controls.csv)。

这不是动态 shape 支持实验。本轮明确选择较小的静态验证范围，代价是不能服务其他输入尺寸；若需求变化，需要重新导出和覆盖 padding/window 边界。不得只修改签名就扩大交付承诺。

## 你亲自练习时的完成要求

1. 先不运行，指出六个 tensor 在源码里的具体语义边界，尤其是 downsample 前后与最终 LayerNorm 前后。
2. 修改故障注入的位置，预测第一个失败边界，再用证据验证；只在副本上操作。
3. 手算 Patch Merging 的8次读取，解释哪里发生了拼接、哪里只是 reshape。
4. 对 Case 03 至少提出两个可被实验排除的解释，完成一次 Core ML FP16 的局部冻结输入对照后再写根因。
5. 用“背景—约束—基线—假设—改动—证据—结果—复盘”口述一张自己实际重跑并理解的经验卡，明确训练故障与真实异常的区别。

当前阶段的成果是可复现的转换与定位练习，以及真实保留的数值异常。代表性数据 accuracy、动态 shape、部署性能、实际 ANE 归属和台架迁移均待后续验证。
