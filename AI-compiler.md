# AI 编译器岗位项目库（内部候选稿）

> 一句话心智模型：同一批工程事实，按 AI 编译器岗位口径讲清「IR / pattern / target legality / lowering / codegen / artifact / runtime trace」这条链；但绝不能把工具源码、诊断性数据或规划项包装成已完成的个人产出。
>
> 本文档汇集两份素材库中所有**面向 AI 编译器岗位**的表述（即「三项目 STARS」的 A 版与 C 版，以及「项目列表」中的风格一与风格三）：
> - `resume/三项目九种风格_STARS_简历项目描述.md`（notebook 实践，Project 1/2/3）
> - `resume/项目列表_副本.md`（codex_doc / AllSpark / ModelZoo 项目库）
>
> 规则：**风格三（二者结合版）同时属于两个岗位**，因此既出现在本文档，也出现在 `AI-edge-inference.md`；A/风格一仅入本文档，B/风格二仅入端侧文档。

## 使用说明与真实性边界

- 每段采用 **STARS**：Situation、Task、Action、Result、Skills/Scope。三种风格是三套投递口径，**不要在同一份简历里把同一工作重复写成多个项目**。
- `<sup>Fxx</sup>`：有本地源码、日志、manifest 或文档证据的事实；仍需看「能否对外使用」列。
- `<sup>Pxx</sup>`：示例数据、目标设计或尚未完成的路线；仅用于展示最终效果，投递前必须替换为本人 raw log/CSV 的结果，或删除。
- 「诊断性」表示编译/部署/trace 已发生，但正确性 Gate 未通过；它可以证明定位能力，不能证明模型加速或任务精度。
- Sparkit/ModelZoo 中「代码已具备」的能力不自动等于「本人实现」。若要写成个人贡献，需补充自己的 commit、PR、测试记录或运行日志。
- `【实测】`/`【历史快照】`/`[^n]` 的用法见文末脚注与数据来源表。

---

# 第一部分：三项目 STARS（notebook 实践）

# 1 跨框架视觉模型编译链路审计与自研 NPU 证据闭环

## 1. 基础版本

- **S（Situation）：** 同一视觉模型经框架导出、图优化和设备编译后，节点数减少、模型包变小或“编译成功”都不能证明语义未变，更不能证明自研 NPU 真正执行了预期子图。
- **T（Task）：** 建立从 PyTorch source graph 到可部署 artifact 的可追溯编译证据链，定位 frontend、graph optimizer、precision lowering 与 runtime placement 分别做了什么。
- **A（Action）：** 使用 FX / `torch.export` 检视 def-use 与 shape metadata，解析 ONNX `ModelProto` 和 MIL textual IR；以同一权重、输入、预处理和 reference logits 贯通多后端，并把模型 hash、graph stats、编译产物 hash、设备配置和 profiler 输出写入 manifest。将 ONNX 图 `122 → 57` 个节点、TorchScript inline graph `858 → 507` 个 MIL op 的变化拆解为各阶段观察，不把节点数下降直接宣称为加速。【历史快照】
- **R（Result）：** 在固定 FP16、静态 `B=16, 3×224×224` 的自研 NPU 档位完成逐元素 parity gate；记录约 `84.1 MB` 静态设备内存和约 `30.6 MB` 临时 tensor 作为基线，而非优化收益。【实测】另在 M1 多后端实验中，6 条路径预测类别一致且 tolerance gate 通过，为后续 pass / artifact 回归提供 reference。【历史快照】
- **S（Skills）：** PyTorch FX / `torch.export`、ONNX Protobuf、MIL、compiler IR dump、artifact provenance、静态 shape / dtype contract、reference parity、runtime placement 证据分级。

## 2. 激进版

**项目名：统一边缘部署 Frontend 与自研 NPU Delegate Runtime**

- **S（Situation）：** 现有多后端实验已证明 artifact、shape、precision 与 placement 必须一起管理，但每个 runtime 自己维护一次转换和 fallback，容易产生模型版本漂移。
- **T（Task）：** 将 PyTorch 导出模型接入 ExecuTorch custom backend delegate，并保留 Core ML / CPU fallback；为自研 NPU 建立 shape bucket、内容寻址 artifact cache、C++ runtime adapter 与可回放 trace。
- **A（Action）：** 假设完成 `torch.export → Edge Dialect → capability partitioner → 自研 NPU artifact/runtime` 的 AOT 流程，兼容不支持子图的 host fallback；每个 bundle 固化 model / calibration / compiler / target / input hash，并用同一 reference、错误 golden 负控和 runtime event 验证路径。ExecuTorch 的 delegate / partitioner 只作为公共 frontend 适配层，不替代自研 backend 的 target legality 与 codegen。[^4]
- **R（Result）：** 目标是在固定模型、输入和设备配置下，将端到端 P50 相对当前可用基线改善 `15%–25%`，同时使所有 fallback、产物来源和设备 placement 可追溯。[^1]
- **S（Skills）：** ExecuTorch、custom delegate、C++ runtime ABI、AOT artifact、shape bucket、content-addressable cache、Core ML / CPU fallback、端侧可观测性。

---

# 2 Compiler-aware PTQ 与 W8A8 编译契约

## 1. 基础版本

**项目名：Q/DQ Precision Contract、量化图 Canonicalization 与后端可行性审计**

- **S（Situation）：** graph 中出现 `QuantizeLinear/DequantizeLinear` 只说明 frontend 表达了低精度意图；若后端没有选到 quantized kernel，或 Q/DQ boundary 与 layout transform 过多，结果仍可能是 float fallback。
- **T（Task）：** 建立从 calibration、量化参数、Q/DQ ONNX、optimized graph 到自研 NPU capability / lowering 的 precision contract，使每一处 INT8 意图、拒绝原因和 fallback 都可检查。
- **A（Action）：** 对 FP32 模型执行 `quant_pre_process` 与 static QDQ PTQ，保存 calibration config、initializer axis、scale、ONNX hash 与 graph dump；比较 FP32 `5` 个节点、raw W8A8 QDQ `23` 个节点（`6 Q + 12 DQ`）和 ORT optimized `11` 个节点（`3 QGemm + 3 Q + 3 DQ`），并以每输出通道 scale 验证 layout / scale contract。`fc1` 的 per-channel 重建 MAE 从 `1.2251e-4` 降至 `4.7770e-5`，约改善 `2.56×`。【历史快照】
- **R（Result）：** W8A8 QDQ ONNX 从 `1.586 MB` 缩至 `0.418 MB`（约 `3.80×`）；量化图经优化后确实出现 `QGemm`，因此把“Q/DQ 存在”和“CPU quantized kernel 被选中”分开记录。【历史快照】这只证明本机 ORT CPU lane，不外推为自研 NPU 的 INT8 性能。
- **S（Skills）：** static PTQ、Q/DQ、per-channel scale、ONNX initializer / shape inference、graph canonicalization、target capability matrix、quantized lowering、fallback report。

## 2. 激进版

**项目名：面向自研 NPU 的敏感层混合精度与 Q/DQ-aware PTQ 搜索**

- **S（Situation）：** 全局 W8A8 容易被 activation outlier、残差连接、LayerNorm / attention 边界和目标 kernel 的 layout 约束拖累；单一全局 scale 无法解释“为什么这一层必须 fallback”。
- **T（Task）：** 将现有 Q/DQ contract 扩展到带标签的小型 Transformer workload：用敏感层扫描决定 FP16 保留层，用 SmoothQuant 类 activation smoothing 与 per-channel / group-wise weight quantization 控制 outlier，并只对自研 NPU 已支持的子图下发低比特 kernel。[^2]
- **A（Action）：** 假设实现 layer-wise error attribution、calibration coverage report、Q/DQ-aware partition cost、FP16 fallback policy 与 Pareto search；输出“模型质量—P50—峰值内存—边界数量”四维候选表，而不是只报一个 INT8 结果。对于 W4A8 / AWQ/GPTQ 类 weight-only 路径，只在目标 backend 的 kernel / dtype / layout contract 和 reference parity 通过后启用。[^2]
- **R（Result）：** 目标是将真实带标签验证集的任务指标损失控制在 `≤1.0 pp`，在固定端侧目标上获得 `1.4–1.8×` 的端到端 P50 改善，并将模型 artifact 再压缩 `55%–70%`；以上均为待验证的目标区间，不是当前实验结果。[^2]
- **S（Skills）：** SmoothQuant、mixed precision、layer sensitivity、AWQ / GPTQ 概念、Q/DQ-aware partition、Pareto search、量化 CI、NPU kernel contract。

---

# 3 TVM Relax Backend Contract 与 Kernel Autotuning 基线

## 1. 基础版本

**项目名：TVM Relax 自定义 Backend Contract：Pattern、Legality、Partition 与 Codegen Gate**

- **S（Situation）：** 图结构匹配成功不代表 target 能执行；即使 partition 生成了 external function，也不代表 codegen 注册、runtime module、设备执行和性能结论已经成立。
- **T（Task）：** 为自研 NPU 定义可单测的 backend contract：哪些 `matmul + bias + relu` 子图能被识别，哪些 dtype / shape / alignment / local-memory 条件必须拒绝，以及 host fallback 如何在 IR 中可见。
- **A（Action）：** 用 Relax `FusionPattern` / `PatternCheckContext` 将结构匹配与 capability check 拆开；以 `Composite`、`Codegen` attribute 和 before/after IR 写出 graph ownership，保存 support report、coverage、manifest 和 runtime gate。设计两个负控：在链中插入语义等价 reshape，使 pattern match 从 `1` 降为 `0`；将 `K=12, N=15` 设为非对齐，使结构匹配仍成功但 legality 被拒绝。【实测】
- **R（Result）：** 支持 case 中 `3/4` 个 primitive op 进入 planned backend，node coverage `75%`，`sin` 留在 host fallback，backend call boundary 为 `1`。【实测】NumPy、original Relax/LLVM 与 semantic-grouped Relax/LLVM 的最大误差 `≤6e-8`，错误 bias golden 被负控成功拦截。【实测】同时主动捕获缺失 custom codegen FFI，结论明确止于“planned partition + LLVM semantic parity”，未虚报 NPU offload 或加速。
- **S（Skills）：** TVM Relax、DPL、SSA / IRModule、pattern matching、target legality、graph partition、external codegen ABI、LLVM、negative control、compiler observability。

## 2. 激进版

**项目名：Relax → TensorIR 自研 NPU Codegen 与可复用 MetaSchedule Tuning Database**

- **S（Situation）：** 手写一个固定 tile 的 kernel 只能证明机制，无法覆盖真实模型中的不同 shape、layout、batch / sequence profile；每次 compiler 升级重新人工调参也不可规模化。
- **T（Task）：** 为自研 NPU 完成最小 external codegen / runtime，并把合法 composite region lower 到 TensorIR-like kernel contract；以 rule-based baseline 覆盖长尾，以 MetaSchedule 搜索热点 MatMul / layout kernel，并持久化调优记录。[^3]
- **A（Action）：** 假设实现 target-aware schedule rule、remote runner、同步/回读 self-check、workload hash、tuning database 和 runtime trace 回填；仅在 reference parity、错误 golden 负控、artifact hash 和实际 placement 都通过后，才将 candidate 写入 database。使用 MLIR Dialect Conversion / StableHLO 仅做前端合法化与可移植性验证，不宣称自研后端当前依赖 MLIR。[^5]
- **R（Result）：** 目标是在热点 kernel 的 rule-based baseline 上获得 `20%–35%` 的 P50 改善，并让有真实 runtime placement 的加权计算覆盖率达到 `≥80%`；这些是需要以同一 shape、同步方式、warm-up、重复次数和 P50/P95 重新验证的目标，非当前实测。[^3]
- **S（Skills）：** TVM Relax / TensorIR、MetaSchedule、schedule rule、remote measurement、tuning database、LLVM codegen、C++ runtime、MLIR legality、StableHLO、kernel profiling。

---

# 第二部分：codex_doc / AllSpark / ModelZoo 项目库（源自 项目列表_副本.md）

# 4 AllSpark `accuracy_compare` 精度验证工具封装

## 1. 基础版本（风格一｜AI 编译器岗位：编译产物精度回归与图级定位）

**项目名称：NPU 编译产物精度回归与图级根因定位工具（候选封装）**

**技术栈：** Python、ONNX Graph、AllSpark AGE/SIM、AP/AOM、JSON、DPA/Cosine、CI/Jenkins。

- **S/T：** 面对 ONNX 经图优化、Lowering 后生成的私有 AP/AOM，单看进程 exit code 无法区分“编译、golden、SIM、比较”哪一层失效；以现有工具的 AGE→golden→SIM→compare 流程为基座，建立可拆分回归链路。<sup>F01</sup>
- **A：** 将源码中的七种有效运行 pattern 标准化为可复放 profile，并以源码枚举而非 CLI help 作为唯一协议；同时复用 ONNX 拓扑排序、`first-failed-node` 与 AOM 节点遍历能力，输出失败 tensor、上游 producer、比较阈值和可归档 task JSON，而非只保留一行“Compare fail”。<sup>F01</sup>
- **R/S：** 以 pattern-5、DPA、相对误差/异常点比例均为 3% 的统一口径记录数值回归；ResNet50 正控达到 `16000/16000`，TinyViT 的 `14764/16000=0.92275` 被如实保留为**原生 Fail 的软基线**，从而把“工具链可运行”和“任务精度通过”分离。<sup>F02</sup><sup>F03</sup><sup>F04</sup>

**创新亮点：** 用“图中第一个失败节点 + 产物来源 hash”替代笼统的端到端失败结论，直接为 pass/Lowering/Kernel 排查提供候选范围。

## 2. 激进版（风格三｜创新路线：编译阶段二分、可信 golden 与持续回归）

**项目名称：面向 NPU 编译链的精度可观测与自动归因平台（后续路线）**

**技术栈：** ONNX/IR diff、metamorphic testing、artifact manifest、差分测试、nxPerf、CI。

- **S/T：** 当前工具能比较 AGE/SIM 输出，却难把误差可靠映射到“哪一次图改写、哪个子图或哪个 Kernel”；同时 golden 可能与运行产物漂移。
- **A：** 计划新增编译阶段二分：对 source ONNX、优化图、AOM 子图和最终输出建立 tensor 级 provenance DAG；配合 shape/dtype 变换、重复编译和错误 golden 的 metamorphic tests，自动区分数值回归、artifact 串包与不稳定编译。<sup>P02</sup>
- **R/S：** 目标是在每次 compiler/toolchain 升级时生成可审计回归报告：`[示例：20+ 正反例]`、首个失败节点、同批 hash、比较口径与回退建议；只有完成真实 CI 数据闭环后才可填写“定位效率提升 [示例：40%]”。<sup>P03</sup>

**创新亮点：** 把“精度比较脚本”升级为 compiler observability：失败证据可反查 IR、artifact、运行时与板端 trace，而不是靠人工翻日志。

---

# 5 从 `codex_doc/` 提炼的九种简历项目表述（AI 编译器岗）

## 1. 基础版本（风格一｜投递 AI 编译器岗位）

### 1. 私有 NPU 编译链图优化与 Artifact Contract 审计

**技术栈：** C++、Python、ONNX、Protobuf wire format、AllSpark Builder/Runtime、Docker、SHA256。

- **S/T：** 私有 AP/AOM 缺少完整公开 schema，图优化又经历多个中间阶段；任务是建立不依赖名称猜测的编译证据链。
- **A：** 解析阶段 ONNX、fusion JSON、编译日志与 AP/AOM raw contract；用稳定节点指纹和交叉核验将融合、layout 折叠与常量变化分开，并把 compiler/library/config/artifact hash 写入同一 manifest。<sup>F06</sup>
- **R/S：** 将一个后端图从 `851→394` 节点的变化拆为 56 个 fusion、120 个 layout/permute 消失和常量折叠，而非把节点下降直接宣称为速度提升；形成可迁移的 IR/Artifact 审计方法。<sup>F06</sup>

### 2. Relay/Relax 双轨 BYOC 后端证据工程

**技术栈：** Apache TVM、Relay、Relax、TensorIR、LLVM、BYOC、C++ JSON codegen、Runtime/PackedFunc。

- **S/T：** 私有库存在 Relay/TIR/runtime 血统证据，但“符号被编入”不等于某个 pass 被激活；同时现代岗位需要 Relax 后端能力。
- **A：** 建立 availability→observable→causal→runtime 四级证据模型，设计同一 micrograph 的 Relay/Relax pass、partition、codegen、runtime 对照；把 pattern、constraint、fallback 和序列化 contract 作为后端质量指标。<sup>F07</sup><sup>P04</sup>
- **R/S：** 当前可严谨表述为“完成私有编译器血统/证据审计与现代 BYOC 实验设计”；`[示例：20+ pattern 正反例]`、真实 external module 或性能结果属于后续实现，不能用 stub runtime 代替。<sup>P03</sup>

### 3. N93X 编译—交叉 Runtime—nxPerf 的分层性能诊断

**技术栈：** CMake、AArch64 toolchain/sysroot、NPU Runtime、CUDA-like event、nxPerf、SQLite、AOM/Task/Kernel/Cluster/Core。

- **S/T：** x86 编译、ARM Runtime、板端 trace 若混用环境或产物，容易把 ABI/部署问题误认为编译器性能问题。
- **A：** 将 mode-0 正式 latency、mode-12 软件 kernel trace、mode-15 one-shot Grid 分离；建立 AOM/Grid/zip 同批配对和板端 SHA256 校验，并将后端 graph 的 layout 变化映射到 trace 观察。<sup>F08</sup>
- **R/S：** 在三个 ConvNeXt 变体上获得实际 N93X 软件 trace：layout-named slice 相对 baseline 分别减少 89.89% 与 92.13%；但因严格正确性 Gate 未过，该数据只用于调度诊断，不作为优化 KPI。<sup>F09</sup>

## 2. 激进版（风格三｜投递创新/调研导向岗位）

### 1. 内容寻址的 NPU 编译 Artifact 与可重放服务

**技术栈：** content-addressable storage、SHA256、manifest、cache key、CLI、replay、failure isolation。

- **S/T：** AP/AOM 名称可相同而内容不同，普通文件名或 package ID 无法阻止跨批产物混用；长编译链也缺少可重放诊断上下文。
- **A：** 已建立 AP/AOM 首帧、IO/AOM 引用、compiler/library/config/artifact hash 的 artifact contract；下一步将 model、shape、precision、toolchain、compiler build hash 纳入 content-addressed cache key。<sup>F14</sup><sup>P06</sup>
- **R/S：** 规划将每个请求输出 `request→artifact→runtime→report` 的可回放 DAG，并以 `[示例：10 个重复/变化/故障请求]` 验证 cold/warm cache、幂等性和失败隔离；该 KPI 目前仅为目标设计。<sup>P07</sup>

### 2. Graph-to-Hardware 因果诊断：从 fusion 到 Kernel placement

**技术栈：** ONNX diff、fusion JSON、AOM Grid、nxPerf SQLite、因果消融、DDR/NPU profiling。

- **S/T：** “节点少了/静态 cost 低了”不能解释端侧 latency；trace 也可能只有 slice，不具备 producer-consumer 关系。
- **A：** 把 graph stage diff、fusion control、mode-12 placement、mode-15 Grid 和 mode-0 latency 设计为同输入、同配置、单变量的证据链，并显式记录 trace 无法回答的 DDR edge 问题。<sup>F06</sup><sup>F09</sup>
- **R/S：** 下一阶段只在正确性 Gate 通过后，才能以 `[示例：fusion on/off × memory reuse on/off × cluster]` 的受控矩阵量化贡献；目标是输出“机制证据 + 反例”，而非把静态 estimate 线性换算为毫秒。<sup>P08</sup>

### 3. DyT/SmoothQuant 的模型—编译器协同优化路线

**技术栈：** DyT、SmoothQuant、ONNX rewrite、TVM fused pattern、AllSpark Plugin、PTQ/QDQ、NPU profiling。

- **S/T：** 算法论文的算子替换可能降低逻辑图复杂度，却因 Tanh、layout 或子图边界让硬件变慢；未训练替换还可能改变模型函数。
- **A：** 已完成 DyT 与 no-permute 的图级对照和语义检查，明确区分“删除 source `Transpose`”与“后端是否仍插入 layout lowering”；把训练/真实任务精度、native/Plugin/BYOC 三条后端路径列为后续 Gate。<sup>F11</sup>
- **R/S：** 计划实现带 dtype/rank/layout/quant constraint 的 DyT/QDQ fused pattern，并以 `[示例：3 seeds × 2 precision]` 与正反例测试验证系统净收益；未完成训练、正确性和真实后端前不使用任何论文式加速数字。<sup>P09</sup>

---

# 6 ModelZoo + Sparkit 独立项目（AI 编译器岗）

## 1. 基础版本（风格一｜AI 编译器岗位：Manifest/Conan 驱动的模型供应链与增量编译）

**项目名称：Sparkit NPU Model Zoo 编译供应链与变体治理（代码核验/候选个人化项目）**

**技术栈：** Python CLI、Conan、Manifest/Recipe、Docker、AllSpark/AQuant、CMake、JSON、文件锁。

- **S/T：** Model Zoo 同时管理 source、quant、AOM、数据集、芯片、精度和工具链变体，手工复制会导致依赖串包与不可复现。
- **A：** 基于 Sparkit 的声明式 `mz_manifest` 与 Conan package resolution，梳理 source→quant→AOM 的依赖传播、工具链部署、薄 Recipe 复用与失败隔离；把“声明的 variant”与“真实内容 identity”明确区分。<sup>F15</sup>
- **R/S：** 冻结设计审计快照记录 92 个 AOM 模型、316 个模型包引用；当前工作区静态 recipe inventory 则为 96 个 source、73 个 quant、168 个 AOM。两组数据都是“声明/文件库存”，不是构建、部署或性能通过数；同时识别 package ID 未包含模型/完整 config digest、`target_state` 不可续跑等边界，避免把现有缓存误称为内容寻址增量编译。<sup>F16</sup>

**创新亮点：** 以输入内容 hash、镜像 digest、compiler/runtime/firmware identity 替代仅靠 URL/tag 的版本判断，构建真正可追溯的编译供应链。<sup>P06</sup>

## 2. 激进版（风格三｜创新路线：面向设备农场的内容寻址控制平面）

**项目名称：Sparkit 可恢复工作流、板卡租约与性能回归控制平面（后续路线）**

**技术栈：** DAG scheduler、artifact store、lease/queue、content hash、workflow orchestration、observability、CI/Nightly。

- **S/T：** 现有 `target_state` 更接近一轮结束后的结果快照，不能从中断 stage 恢复；共享 cache、板卡和长编译链会放大重复工作与串包风险。<sup>F19</sup>
- **A：** 规划以 immutable RunSpec 和内容 hash 建模 source/quant/AOM/runtime stages，引入 per-device lease、幂等 worker、断点恢复和“数据完整性优先于缓存命中”的调度策略。<sup>P10</sup>
- **R/S：** 用 `[示例：同一模型矩阵的 20 次 nightly]` 验证 P50、cache hit、失败恢复率和板卡利用率；诸如“18 小时降至 6.5 小时、31% 升至 73%”属于设计文档的 TARGET-ASSUMED 示例，当前绝不可写成实绩。<sup>P11</sup>

---

# 附一：三项目 STARS 脚注（知识边界与验证门槛）

[^1]: **统一 delegate runtime 的 `15%–25%` P50 目标。** 这是常见的“减少重复转换、稳定 shape bucket、减少不必要 fallback / boundary”的项目目标，不是当前任何设备的实测。最小实现不是一次支持全模型：先让一个固定 shape 的 `MatMul + Bias + ReLU` 子图走 ExecuTorch delegate 和自研 NPU runtime。完成条件：同一 model/input/compiler hash、reference 全元素 parity、错误 golden 负控、runtime placement、raw latency、P50/P95 与 fallback report 齐全。学习顺序：`torch.export` → ExecuTorch Edge Dialect / partitioner → backend preprocess / runtime ABI → artifact manifest。官方入口可参考 [ExecuTorch Backends and Delegates](https://docs.pytorch.org/executorch/stable/compiler-delegate-and-partitioner.html)。

[^2]: **混合精度的“`≤1.0 pp` 质量损失、`1.4–1.8×` P50、`55%–70%` artifact 压缩”目标。** 这三个数都是以真实带标签 Transformer workload 为前提的候选验收范围，不能从当前 synthetic TinyTokenMLP 外推。知识边界包括：代表性 calibration、layer-wise sensitivity、residual / LayerNorm 数值稳定性、Q/DQ 语义、目标 kernel 的 dtype/layout legality，以及质量—时延 Pareto 选择。最小项目：只对 3 个 Linear 层比较全 FP16、全 W8A8、1 个敏感层 FP16 fallback；先用标签任务指标，再测目标端 P50/P95。SmoothQuant / AWQ / GPTQ 只能作为后续可选方法，不能替代这一条基础证据链。

[^3]: **MetaSchedule / external codegen 的“`20%–35%` P50、`≥80%` 加权计算覆盖”目标。** 这是 kernel tuning 项目的预期区间，不是当前 toy C 或 Relax CPU 结果。当前 notebook 只证明 planned partition 与 LLVM semantic parity，尚没有目标设备 codegen/runtime。最小项目：只 lower 一个 fixed-shape MatMul composite，先有 rule-based TIR schedule，再以 1 个可控 tile 参数形成小搜索空间；每个候选必须 build、在目标端同步运行、写已知模式回读并记录测量失败原因。TVM MetaSchedule 的核心是“生成 schedule → 在真实硬件测量 → 将 schedule trace 和测量结果写入 database”，可参考 [Apache TVM MetaSchedule](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html)。

[^4]: **ExecuTorch custom delegate 是拓展，不是现有实现。** 它的价值是把“哪个子图归 backend、哪个 op fallback”放到公开、可维护的 Edge Dialect / partitioner 接口中。它不自动产生自研 NPU codegen，也不保证所有设备共用一个 binary；每个 target artifact 仍必须独立验证。

[^5]: **MLIR / StableHLO 是概念和验证扩展，不是现有自研编译器的既有依赖。** MLIR Dialect Conversion 提供 `ConversionTarget + RewritePattern + TypeConverter` 的 legality 模型，适合拿来校准当前 Relax support check 的设计；StableHLO 可作为框架与编译器之间的可移植 op 语义边界。最小项目是实现一个只覆盖 `MatMul/Add/ReLU` 的 dialect legalization report，而不是重写整条编译栈。参考 [MLIR Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/) 与 [StableHLO Specification](https://openxla.org/stablehlo/spec)。

# 附二：数据与表述来源说明（F/P 标记表，源自 项目列表_副本.md）

| 标记 | 正文中的数据/结论 | 类型 | 来源与边界 | 可否直接用于外投简历 |
|---|---|---|---|---|
| F01 | 7 种有效 run pattern、图拓扑/首失败节点/ASIC client 能力；CLI help 与源码枚举有不一致 | 源码事实 | `accuracy_compare.py:547-554,1445-1555,1995-2157`；`accuracy_utils.py:127-204`；CLI 为 `accuracy_compare_tool.py:15-25` | 可写“基于该工具封装”，但“本人完成封装”需有自己的代码/提交 |
| F02 | pattern-5、DPA、3% relative error、3% outlier ratio | 当前实验规范 | `current_experiment.yaml:e1_2.standard_accuracy_policy`；`03...md:883` | 可写为数值比较口径，不可写成任务精度 |
| F03 | ResNet50 `16000/16000`、accuracy `1.0` | 本地控制结果 | `current_experiment.yaml:evidence.e1_2.resnet_pattern5_control` | 可写“工具链数值正控”，不可写 ImageNet Top-1 |
| F04 | TinyViT `14764/16000=0.92275`、原生 verdict fail | 本地软 Gate | `current_experiment.yaml:evidence.e1_2.tinyvit_provisional_accuracy_baseline`；`03...md:884` | 只可内部写“软基线/回归参照”；不建议外投为结果 |
| F05 | ASIC 客户端上传/执行/回传能力 | 源码事实 | `asic_client.py:44-201` | 可写工具能力；负控和 hash 封装是 P01 |
| F06 | `851→394`、56 fusion、120 permute 消失 | E2/E3 图审计事实 | `03_experiment_roadmap_ai_compiler.md:455-467` | 可写图优化分析；不可推导 latency |
| F07 | 四级 evidence model、Relay/TIR 有而 Relax 无的审计边界 | E2 文档/二进制核验 | `06_allspark_tvm_lineage_and_route_review.md:5-18,90-101` | 可写“证据审计”；不可写还原私有源码 |
| F08 | mode-0/12/15 的采集边界与配对要求 | 方法/实物核验 | `01_repository_knowledge_map.md:387-415`；`03...md:933-944` | 可写方法能力，须有本人 trace |
| F09 | mode-12 layout slice -89.89%/-92.13%，正确性未过 | 板端诊断性实测 | `eonvnext-guide-model12.md:766-839` | 不可作为模型加速 KPI；可用于面试讲诊断边界 |
| F10 | 固定 HF source→ONNX→ORT golden→AOM/AP→ARM/N93X Gate | 已落地流程/部分 Gate | `08_e12_hf_modelzoo_pipeline.md:1-50,477-510`；`current_experiment.yaml` | 可写流程与正/负控；具体模型结果须按 Gate |
| F11 | ConvNeXt/DyT/no-permute、72 `Transpose`、layout rewrite 语义 | 本地导出/图验证 | `convnext-report.md:1-91` | 可写图改造与实验设计；未训练模型不可写精度改善 |
| F12 | 5 warm-up+30 samples、3.5562 ms、-34.85%，但 strict gate fail | 板端诊断性实测 | `convnext-report.md:213-228`；`eonvnext-guide.md:548-590` | **不可直接外投为加速**；修复 golden 后重测 |
| F13 | 当前量化 ONNX 分支没有唯一 `*_proto.bin`，INT8 AOM 被阻断 | 实物 Gate | `08_e12_hf_modelzoo_pipeline.md:22-34,221-251` | 可写为质量门禁/负结论，不可写 W8A8 成功 |
| F14 | AP/AOM 同名不等于同内容；hash/raw contract 边界 | E2 审计事实 | `06...md:19-64`；`03...md:275-408` | 可写 artifact provenance 方法 |
| F15 | Sparkit Manifest→Conan→quant→AOM 编排 | 源码/设计核验 | `07_sparkit_design.md:134-179,520-618` | 可写“代码/架构核验”；个人实现需个人证据 |
| F16 | 冻结快照：92 个 AOM 模型、316 个包引用；当前静态 recipe inventory：96 source、73 quant、168 AOM | Manifest/目录库存 | 前者：`07_sparkit_design.md:28-50,1417-1425`；后者：当前 `modelzoo/mz_recipe` 静态核验 | 仅可写准确注明版本/日期的库存；都不是构建或部署通过数 |
| F17 | ONNX/LLM Runner 指标产生位置不同 | 源码核验 | `07_sparkit_design.md:709-845` | 可写指标口径治理能力 |
| F18 | 65 个测试、33 通过、1 个遗留导入失败 | 本地代码核验 | `07_sparkit_design.md:28-50` | 可写测试现状，不可说全量测试通过 |
| F19 | `target_state` 不可续跑 | 源码/设计核验 | `07_sparkit_design.md:520-532,1439-1444` | 可写发现的技术债/改造动机 |
| F20 | 历史 GPTQ：196 个 Linear、权重 -62.76%，CSV schema 有错位 | 历史 Git 工件/复盘 | `02_knowledge_gap_and_closure.md:24-33`；当前 prompt 也明确禁止把错位日志时间当正式数字 | 只能写历史分析/假设，不得当作个人端侧加速结果 |
| P01 | SHA256 bundle、负控退出码强制传播 | 待实现封装 | 由 `E0.4/E1.2` Gate 设计而来：`03...md:470-641` | 实现并保留 raw log 后才可写“已实现” |
| P02 | 编译阶段二分、metamorphic test、provenance DAG | 未来方案 | 本文提出的工业化路线 | 当前不可写为交付 |
| P03 | `20+` 正反例、定位效率 `40%` | 示例数据 | 仅用于展示简历效果 | 必须替换为真实测试数/基线 |
| P04 | Relay/Relax 真正 pass/partition/codegen/runtime 实验 | 路线设计 | `03...md:E3.0A-E3.2` | 完成 build/test 后才可写 |
| P05 | `16` 组 PTQ 单变量消融 | 路线设计 | `03...md:E2.2` | 未执行，不可写结果 |
| P06 | content-addressed cache key | 未来方案 | `03...md:E3.3-E3.4`、Sparkit 技术债 | 未实现，不可写 cache 命中 |
| P07 | `10` 个 replay job | 路线设计 | `03...md:E3.4` | 未执行，不可写服务成效 |
| P08 | fusion/memory/cluster 因果矩阵 | 路线设计 | `03...md:E1.4` | 未执行，不可填写速度/内存收益 |
| P09 | `3 seeds × 2 precision` | 路线设计 | `03...md:E2.4` | 未训练/未验证，不可写论文复现结果 |
| P10 | RunSpec/DAG/lease/恢复 | 未来方案 | Sparkit 技术债与本文设计 | 未实现，不可写平台可靠性成果 |
| P11 | 20 nightly、18h→6.5h、31%→73% | TARGET-ASSUMED 示例 | `07_sparkit_design.md:1535-1552,1583-1587` | **仅示例，必须删除或以真实数据替换** |

## 投递前的最小替换清单

1. 用自己执行的命令、image/toolchain hash、raw CSV/log 替换所有 `Pxx`。
2. 删除所有严格正确性 Gate 未通过的性能数字；保留它们用于面试时讲工程边界即可。
3. 将“代码核验/平台已有”改成可被 Git commit、PR、测试或运行日志证明的个人动作。
4. 一份简历每个主题只选一种风格；AI 编译器岗选「第一部分（1A/2A/3A）+ 第二部分（风格一/三）」，端侧岗选对应端侧文档。
