# Swin Transformer 实践课程索引

一句话心智模型：先固定同一个模型和输入，再逐阶段证明结果一致，最后讨论部署收益。

依据：[项目指导思想](../../docs/swin-transformer-guideline.md)、[Learning-Guide](../../Learning-Guide.md)。本次只展开阶段二，阶段一 Notebook 中的用户笔记保持原样。

|阶段|状态与入口|核心问题|边界|
|---|---|---|---|
|01 可信 baseline|已有：[阶段一](01_stage1_trustworthy_baseline.ipynb)|输入、精度、测量口径是否一致？|本次不修改|
|02 转换与误差定位|已展开：[阶段二](02_stage2_conversion_correctness.ipynb)|从哪一个转换环节、哪一个语义边界开始偏？|静态 batch=1、224×224，浮点，不做量化|

阶段二建议分三次完成，每次先预测结果再运行。

配套 [阶段二实测经验卡](02_stage2_experience.md) 保存首次执行中真正遇到的问题、故障注入结果和未解释的 FP16 误差；不要把其中的待验证项改写成已完成的个人成果。

|学习顺序|Notebook 章节|要亲自完成的动作|完成后应能回答|
|---|---|---|---|
|1 固定契约并导出|§0～4|检查继承的输入/权重 hash；看 ONNX node、MIL op 与多输出 shape；执行 FP32 对照|为什么两条导出分支都需要 PyTorch reference？|
|2 查误差并排除解释|§5～8|看逐边界曲线；定位人为注入；回滚；冻结 Patch Merging 输入；验证越界请求|第一失败边界为什么不是根因证明？|
|3 整理工程经验|§9～12|填三张经验卡，区分真实异常与训练故障；列剩余验收条件|哪些结论已有证据，哪些不能写入简历？|

源码阅读按一次输入的执行顺序进行。Notebook 和生成器里的符号是项目代码；框架源码通过 `inspect.getfile(modeling_swin)` 获取本机真实路径，再用编辑器“转到定义”。固定版本为 torch 2.7.0、transformers 4.51.3、coremltools 9.0，完整依赖以 `uv.lock` 和运行 manifest 为准。

|阅读顺序|真实文件/符号|看什么|
|---|---|---|
|1|`tools/build_swin_stage2_notebook.py` 中 `SOURCE_RUN`、`sha256`|阶段一输入如何成为阶段二不可随意改变的契约|
|2|同文件 `LogitsOnly.forward`、`DebugOutputs.forward`|框架对象如何变为导出接口中的 tensor|
|3|transformers 的 `models/swin/modeling_swin.py`：`SwinForImageClassification.forward → SwinModel.forward → SwinEncoder.forward → SwinStage.forward`|hidden_states 在哪里收集，是否经过 downsample、最终 LayerNorm|
|4|项目 `export_onnx → ort_session → run_ort → record`|Protobuf 图到可执行 Session，再到比较器|
|5|项目 `make_mil → build_coreml → run_coreml → record`|TorchScript、MIL、包、runtime 的边界|
|6|项目故障注入单元格：`producers`、`target`、`before`、`Add`|改一条 def-use 连接，为什么下游读取值跟着变|
|7|框架 `SwinPatchMerging.forward`，项目 `capture_merge`、`FrozenPatchMerging.forward`|真实模块输入如何冻结并形成局部复现|
|8|项目 `checked_input`、`summary`|非法输入如何失败，数值门禁与任务验收如何区分|

覆盖范围：公开模型的固定输入导出、多处语义边界检查、FP32/FP16 对照、故障注入和回归。尚未覆盖：代表性带标签数据的 accuracy、动态 shape 正向支持、量化、性能优化、ANE 运行证据、公司台架实测。MIL 只保存和检查表示，没有作为独立解释器执行；不能声称数值误差已经精确定位到某个 MIL pass。

若某个文件或原理看不懂，可继续追问 AI，并指出 Notebook 章节、cell 和实际输出。课程索引负责提供路径，机制解释与短代码在 Notebook 内。

最后自行复述：从 `inputs.npz` 的一个 `[1,3,224,224]` 输入出发，说清两条转换分支、六个输出边界、比较器、局部复现与恢复步骤，并解释没有标签时为什么不能给 accuracy 结论。
