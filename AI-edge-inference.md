# AI 端侧推理岗位项目库（内部候选稿）

> 一句话心智模型：同一批工程事实，按 AI 端侧推理部署岗位口径讲清「精度、端到端 latency、内存、部署包完整性、设备 trace」这条闭环；但绝不能把工具源码、诊断性数据或规划项包装成已完成的个人产出。
>
> 本文档汇集两份素材库中所有**面向 AI 端侧模型推理优化岗位**的表述（即「三项目 STARS」的 B 版与 C 版，以及「项目列表」中的风格二与风格三）：
> - `resume/三项目九种风格_STARS_简历项目描述.md`（notebook 实践，Project 1/2/3）
> - `resume/项目列表_副本.md`（codex_doc / AllSpark / ModelZoo 项目库）
>
> 规则：**风格三（二者结合版）同时属于两个岗位**，因此既出现在本文档，也出现在 `AI-compiler.md`；B/风格二仅入本文档，A/风格一仅入编译器文档。

## 使用说明与真实性边界

- 每段采用 **STARS**：Situation、Task、Action、Result、Skills/Scope。三种风格是三套投递口径，**不要在同一份简历里把同一工作重复写成多个项目**。
- `<sup>Fxx</sup>`：有本地源码、日志、manifest 或文档证据的事实；仍需看「能否对外使用」列。
- `<sup>Pxx</sup>`：示例数据、目标设计或尚未完成的路线；仅用于展示最终效果，投递前必须替换为本人 raw log/CSV 的结果，或删除。
- 「诊断性」表示编译/部署/trace 已发生，但正确性 Gate 未通过；它可以证明定位能力，不能证明模型加速或任务精度。
- Sparkit/ModelZoo 中「代码已具备」的能力不自动等于「本人实现」。若要写成个人贡献，需补充自己的 commit、PR、测试记录或运行日志。
- `【实测】`/`【历史快照】`/`[^n]` 的用法见文末脚注与数据来源表。

---

# 第一部分：三项目 STARS（notebook 实践）

# 1 端侧视觉模型多后端部署、冷启动拆分与性能诊断闭环

## 1. 基础版本

**项目名：端侧视觉模型多后端部署、冷启动拆分与性能诊断闭环**

- **S（Situation）：** 端侧部署常把 CPU、GPU、NPU、精度和 runtime 差异混在同一张“速度排名”里，导致无法解释瓶颈，也无法制定启动与缓存策略。
- **T（Task）：** 在同一 ResNet-50、相同输入和预处理下，建立 cold path、warm steady path、模型包大小、数值误差与 profiler 的统一比较协议，并迁移到自研 NPU 的交付验证流程。
- **A（Action）：** 固定 `warm-up=5`、`30` 次测量和 `4` 个 CPU thread，分别记录 eager CPU、`torch.compile` CPU、MPS 与 Core ML 等路径的原始 latency；将 export / compile / load / first inference 拆开，使用 runtime trace 与 profiler 区分“预估 placement”和“实际执行”。对自研 NPU 的 3 个视觉变体建立 trace 解析和 artifact 校验管线；发现 parity 未过时，明确阻断性能发布，而非将诊断 trace 伪装为产品 latency。【实测】
- **R（Result）：** 在 M1 的历史快照中，CPU steady P50 从 eager 的 `28.14 ms` 降至 `torch.compile` 的 `25.39 ms`；`torch.compile` 首次编译成本对应约 `7,337` 次调用的 break-even。平台路径中，MPS FP32 P50 为 `15.48 ms`、Core ML FP16/ALL 为 `1.87 ms`，但该差异包含 runtime、device placement 和 precision，未归因成单一 compiler 收益。【历史快照】FP16 artifact `51.15 MB` 相对 FP32 的 `102.20 MB` 约减半。【历史快照】
- **S（Skills）：** 端侧 benchmark protocol、P50/P95、cold/warm 分离、Core ML / ONNX Runtime profiling、模型包审计、设备 smoke / parity gate、性能结论的因果边界。

## 2. 激进版

**项目名：统一边缘部署 Frontend 与自研 NPU Delegate Runtime**

- **S（Situation）：** 现有多后端实验已证明 artifact、shape、precision 与 placement 必须一起管理，但每个 runtime 自己维护一次转换和 fallback，容易产生模型版本漂移。
- **T（Task）：** 将 PyTorch 导出模型接入 ExecuTorch custom backend delegate，并保留 Core ML / CPU fallback；为自研 NPU 建立 shape bucket、内容寻址 artifact cache、C++ runtime adapter 与可回放 trace。
- **A（Action）：** 假设完成 `torch.export → Edge Dialect → capability partitioner → 自研 NPU artifact/runtime` 的 AOT 流程，兼容不支持子图的 host fallback；每个 bundle 固化 model / calibration / compiler / target / input hash，并用同一 reference、错误 golden 负控和 runtime event 验证路径。ExecuTorch 的 delegate / partitioner 只作为公共 frontend 适配层，不替代自研 backend 的 target legality 与 codegen。[^4]
- **R（Result）：** 目标是在固定模型、输入和设备配置下，将端到端 P50 相对当前可用基线改善 `15%–25%`，同时使所有 fallback、产物来源和设备 placement 可追溯。[^1]
- **S（Skills）：** ExecuTorch、custom delegate、C++ runtime ABI、AOT artifact、shape bucket、content-addressable cache、Core ML / CPU fallback、端侧可观测性。

---

# 2 W8A8 数值鲁棒性、边界开销归因与端侧量化 Gate

## 1. 基础版本

**项目名：W8A8 数值鲁棒性、边界开销归因与端侧量化 Gate**

- **S（Situation）：** 端侧量化目标不是“模型变成 INT8”，而是在代表性数据上保持任务质量，并让减少的存储 / MAC 成本超过 Q/DQ、layout 与 fallback 的边界代价。
- **T（Task）：** 用受控数据分布、统一 benchmark 和 profiler，区分数值一致性、分布漂移和真正的低精度 runtime 收益；为自研 NPU 的量化部署设置进入性能测试前的 Gate。
- **A（Action）：** 在 `warm-up=50`、`800` 次运行、`4` 个 CPU thread 的 ORT CPU lane 中对 FP32 与 W8A8 做 P50/P95；同时在 in-distribution synthetic 与 shifted/outlier synthetic 两组输入上统计 saturation、logit error、cosine 和 Top-1 agreement，并解析 profiler 中 Q/DQ / layout conversion 的耗时占比。
- **R（Result）：** FP32 P50 `0.1315 ms`，W8A8 P50 `0.0809 ms`，即 `1.63×` 的**本机 CPU toy-workload** 加速；in-distribution 的 mean cosine `0.999781`、Top-1 agreement `99.22%`，而 shifted/outlier 输入的 saturation `7.979%`、agreement 降至 `93.36%`。Q/DQ / layout conversion 占 profiler node 时间 `42.7%`，直接说明下一步应优先处理 boundary 而非盲目继续压低 bit-width。【历史快照】这些 synthetic 指标不是带标签任务 accuracy。
- **S（Skills）：** calibration、outlier / saturation diagnosis、accuracy-vs-agreement 边界、latency profiling、Q/DQ boundary cost、量化 regression gate、端侧部署质量门禁。

## 2. 激进版

**项目名：面向自研 NPU 的敏感层混合精度与 Q/DQ-aware PTQ 搜索**

- **S（Situation）：** 全局 W8A8 容易被 activation outlier、残差连接、LayerNorm / attention 边界和目标 kernel 的 layout 约束拖累；单一全局 scale 无法解释“为什么这一层必须 fallback”。
- **T（Task）：** 将现有 Q/DQ contract 扩展到带标签的小型 Transformer workload：用敏感层扫描决定 FP16 保留层，用 SmoothQuant 类 activation smoothing 与 per-channel / group-wise weight quantization 控制 outlier，并只对自研 NPU 已支持的子图下发低比特 kernel。[^2]
- **A（Action）：** 假设实现 layer-wise error attribution、calibration coverage report、Q/DQ-aware partition cost、FP16 fallback policy 与 Pareto search；输出“模型质量—P50—峰值内存—边界数量”四维候选表，而不是只报一个 INT8 结果。对于 W4A8 / AWQ/GPTQ 类 weight-only 路径，只在目标 backend 的 kernel / dtype / layout contract 和 reference parity 通过后启用。[^2]
- **R（Result）：** 目标是将真实带标签验证集的任务指标损失控制在 `≤1.0 pp`，在固定端侧目标上获得 `1.4–1.8×` 的端到端 P50 改善，并将模型 artifact 再压缩 `55%–70%`；以上均为待验证的目标区间，不是当前实验结果。[^2]
- **S（Skills）：** SmoothQuant、mixed precision、layer sensitivity、AWQ / GPTQ 概念、Q/DQ-aware partition、Pareto search、量化 CI、NPU kernel contract。

---

# 3 端侧 Offload / Fallback 可观测性与可信 Benchmark Gate

## 1. 基础版本

**项目名：端侧 Offload / Fallback 可观测性与可信 Benchmark Gate**

- **S（Situation）：** 很多端侧优化只展示“某段图被 delegate”，却无法回答 fallback 在哪里、是否发生跨设备 copy、是否真的在目标设备执行；直接给 speedup 会把静态计划误写成实测。
- **T（Task）：** 把部署优化前置为一套证据 gate：先验证 graph ownership、artifact、runtime placement 和输出，再允许写 P50/P95 与 profiler 结论。
- **A（Action）：** 在 Relax 实验中把 node coverage 的分母固定为 original primitive op count，明确区分 node / FLOPs / byte / latency coverage；在 toy C codegen 中先做每档 parity，再比较同一生成代码家族的 loop tiling。`512³` workload 上，naive C `149.69 ms`、tile-128 C `11.28 ms`，仅用于演示 loop order / tile 对 cache locality 的机制，且仍明显慢于 NumPy / BLAS，未作跨实现性能宣传。【实测】
- **R（Result）：** 形成 `pattern → support → partition → codegen → runtime → correctness → performance` 的发布顺序；external codegen 未注册时，runtime gate 正确报错并阻断 latency 结论。这使“未真实执行的 benchmark 不入库”成为机制，而不是靠人工提醒。【实测】
- **S（Skills）：** edge delegate / fallback diagnosis、coverage definition、buffer / boundary reasoning、tiling、benchmark self-check、reference parity、artifact manifest、P50/P95 发布纪律。

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

## 1. 基础版本（风格二｜端侧模型推理岗位：从 golden 到 NPU/ASIC 的部署正确性 Gate）

**项目名称：端侧 NPU 推理精度与部署完整性验证封装（候选封装）**

**技术栈：** Python、SSH/SFTP、AArch64、AllSpark Hyper Runtime、FP16 binary I/O、SHA256、N93X SoC。

- **S/T：** 端侧部署常出现“AP/AOM、input、golden 来自不同 run”或“比较失败却 exit 0”的假成功；目标是把精度比较从本地脚本提升为可交付的设备 Gate。
- **A：** 在工具已有 ASIC 客户端上传、远端执行、结果回传能力之上，封装同批 AP/AOM/input/golden 的 SHA256 清单、唯一远端目录、AArch64 Runtime 正控和 sign-flipped golden 负控；比较器失败必须传播为非零退出码。<sup>F05</sup><sup>P01</sup>
- **R/S：** 形成“来源完整性 → shape/dtype/finite → 数值比较 → 失败传播”的分层验收；DPA 仅描述逐元素数值一致性，真实分类/检测指标仍由带标签数据集单独评估，避免把随机输入比较误写成端侧任务精度。<sup>F02</sup>

**创新亮点：** 将部署包视为不可变 artifact bundle；任何 hash、输出 shape 或负控行为异常都先阻断性能测试，减少把环境污染误诊为算子精度问题。

## 2. 激进版（风格三｜创新路线：编译阶段二分、可信 golden 与持续回归）

**项目名称：面向 NPU 编译链的精度可观测与自动归因平台（后续路线）**

**技术栈：** ONNX/IR diff、metamorphic testing、artifact manifest、差分测试、nxPerf、CI。

- **S/T：** 当前工具能比较 AGE/SIM 输出，却难把误差可靠映射到“哪一次图改写、哪个子图或哪个 Kernel”；同时 golden 可能与运行产物漂移。
- **A：** 计划新增编译阶段二分：对 source ONNX、优化图、AOM 子图和最终输出建立 tensor 级 provenance DAG；配合 shape/dtype 变换、重复编译和错误 golden 的 metamorphic tests，自动区分数值回归、artifact 串包与不稳定编译。<sup>P02</sup>
- **R/S：** 目标是在每次 compiler/toolchain 升级时生成可审计回归报告：`[示例：20+ 正反例]`、首个失败节点、同批 hash、比较口径与回退建议；只有完成真实 CI 数据闭环后才可填写“定位效率提升 [示例：40%]”。<sup>P03</sup>

**创新亮点：** 把“精度比较脚本”升级为 compiler observability：失败证据可反查 IR、artifact、运行时与板端 trace，而不是靠人工翻日志。

---

# 5 从 `codex_doc/` 提炼的九种简历项目表述（端侧推理岗）

## 1. 基础版本（风格二｜投递端侧模型推理岗位）

### 1. 固定模型来源的 HF→ONNX→NPU 端到端部署闭环

**技术栈：** Hugging Face、PyTorch、ONNX Runtime、AQuant、AllSpark、AArch64、N93X、SHA256。

- **S/T：** vendor sample 中的 ONNX、proto、input/golden 可能跨批次，无法证明端侧结果属于同一模型来源。
- **A：** 固定 HF revision 与权重 hash，分别导出动态/静态 ONNX 并做 PyTorch/ORT parity；在编译前生成独立 ORT golden，随后串联 AOM/AP、x86/AArch64 Runtime、板端正控与错误-golden 负控。<sup>F10</sup>
- **R/S：** ResNet50 正控在标准比较下为 `16000/16000`；TinyViT 的 `0.92275` 仅作为同 cohort 的数值软基线，明确不等同于 ImageNet Top-1、厂商验收或生产正确性。<sup>F03</sup><sup>F04</sup>

### 2. ConvNeXt Layout Rewrite 与 DyT 的端侧部署诊断

**技术栈：** PyTorch、ONNX、ConvNeXt、DyT、NCHW/NHWC、AllSpark、N93X、nxPerf。

- **S/T：** 单纯用 DyT 替换 LayerNorm 并不会消掉 ConvNeXt 的显式 layout 转换，无法回答“图变小是否真正更快”。
- **A：** 将 block 中 NHWC Linear 等价改写为 NCHW `1×1 Conv`，构造 baseline/DyT/no-permute 对照，验证 ONNX 数值语义与编译后 graph/trace 的 layout 行为。<sup>F11</sup>
- **R/S：** 源 ONNX 的 72 个 `Transpose` 被 no-permute 变体消除，且完成 5 次 warm-up、30 次 N93X 采样；诊断性 E2E mean 为 3.5562 ms、相对 baseline 降 34.85%，但严格 golden Gate 未通过，外投前必须重测并只在通过后使用该性能数字。<sup>F12</sup>

### 3. PTQ 前端契约与精度回归治理

**技术栈：** PyTorch/TorchAO、ONNX Runtime QOperator/QDQ、AQuant、INT8、quant proto、calibration、mixed precision。

- **S/T：** calibration 集、Q/DQ 格式与量化 proto 若不成对，可能产生“量化成功但后端语义错误”；逐样本 cosine 又不能替代任务指标。
- **A：** 将 calibration/test 分离、resolved YAML、实际 loader 样本数、量化 ONNX/proto hash 和独立 golden 纳入 Gate；发现当前 ONNX 分支未产出唯一 proto 时停止 INT8 AOM，保持 FP16 与 W8A8 表述边界。<sup>F13</sup>
- **R/S：** 建立“先验证可消费 proto，再进入低精度编译”的质量门禁；历史 GPTQ 工件覆盖 `28×7=196` 个 Linear、权重体积下降 62.76%，但其 CSV schema 存在错位，因此只用作敏感层假设，不能作为正式量化耗时或端侧性能结论。后续以校准量、粒度、对称性、QDQ 格式的单变量矩阵和逐层敏感度/随机回退对照完成 Pareto 选择。`[示例：16 组消融]`尚未执行。<sup>F20</sup><sup>P05</sup>

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

# 6 ModelZoo + Sparkit 独立项目（端侧推理岗）

## 1. 基础版本（风格二｜端侧模型推理岗位：Model 生命周期评测与 SoC 交付编排）

**项目名称：Sparkit NPU Model Zoo 端侧部署、推理与可信 Benchmark 编排（代码核验/候选个人化项目）**

**技术栈：** N93X/N93P、AArch64、SSH/NFS、AllSpark Runtime、C++ Runner、LLM/ONNX、功耗/温度/内存采集、EvalScope。

- **S/T：** 端侧模型交付不仅要“能编译”，还要处理 Host/容器/SoC 路径、模型部署、推理指标、设备状态与运行产物回收。
- **A：** 梳理 Sparkit 的环境部署、P1 ONNX Runner、LLM Runner、精度/性能/功耗采集和结果归档链；明确 ONNX 路径由 C++ Runner 统计 H2D/Enqueue/NPU/D2H/E2E，而 LLM 指标主要解析下游日志，二者不能混用。<sup>F17</sup>
- **R/S：** 当前测试资产含 65 个 `test_*`，本地核验记录为 33 个通过、1 个遗留导入失败；这说明平台具备可测试基础但 Benchmark schema 仍缺 sample count、分位数等完整合同，不能把发布报告中的模型涨跌归为个人优化。<sup>F18</sup>

**创新亮点：** 为每次板端执行绑定 RunSpec、部署清单、原始 sample 与设备身份，使性能回归可区分模型、工具链、设备状态和采集口径。

## 2. 激进版（风格三｜创新路线：面向设备农场的内容寻址控制平面）

**项目名称：Sparkit 可恢复工作流、板卡租约与性能回归控制平面（后续路线）**

**技术栈：** DAG scheduler、artifact store、lease/queue、content hash、workflow orchestration、observability、CI/Nightly。

- **S/T：** 现有 `target_state` 更接近一轮结束后的结果快照，不能从中断 stage 恢复；共享 cache、板卡和长编译链会放大重复工作与串包风险。<sup>F19</sup>
- **A：** 规划以 immutable RunSpec 和内容 hash 建模 source/quant/AOM/runtime stages，引入 per-device lease、幂等 worker、断点恢复和“数据完整性优先于缓存命中”的调度策略。<sup>P10</sup>
- **R/S：** 用 `[示例：同一模型矩阵的 20 次 nightly]` 验证 P50、cache hit、失败恢复率和板卡利用率；诸如“18 小时降至 6.5 小时、31% 升至 73%”属于设计文档的 TARGET-ASSUMED 示例，当前绝不可写成实绩。<sup>P11</sup>

---

# 附一：三项目 STARS 脚注（知识边界与验证门槛）

[^1]: **统一 delegate runtime 的 `15%–25%` P50 目标。** 这是常见的“减少重复转换、稳定 shape bucket、减少不必要 fallback / boundary”的项目目标，不是当前任何设备的实测。最小实现不是一次支持全模型：先让一个固定 shape 的 `MatMul + Bias + ReLU` 子图走 ExecuTorch delegate 和自研 NPU runtime。完成条件：同一 model/input/compiler hash、reference 全元素 parity、错误 golden 负控、runtime placement、raw latency、P50/P95 与 fallback report 齐全。学习顺序：`torch.export` → ExecuTorch Edge Dialect / partitioner → backend preprocess / runtime ABI → artifact manifest。

[^2]: **混合精度的“`≤1.0 pp` 质量损失、`1.4–1.8×` P50、`55%–70%` artifact 压缩”目标。** 这三个数都是以真实带标签 Transformer workload 为前提的候选验收范围，不能从当前 synthetic TinyTokenMLP 外推。知识边界包括：代表性 calibration、layer-wise sensitivity、residual / LayerNorm 数值稳定性、Q/DQ 语义、目标 kernel 的 dtype/layout legality，以及质量—时延 Pareto 选择。最小项目：只对 3 个 Linear 层比较全 FP16、全 W8A8、1 个敏感层 FP16 fallback；先用标签任务指标，再测目标端 P50/P95。

[^3]: **MetaSchedule / external codegen 的“`20%–35%` P50、`≥80%` 加权计算覆盖”目标。** 这是 kernel tuning 项目的预期区间，不是当前 toy C 或 Relax CPU 结果。当前 notebook 只证明 planned partition 与 LLVM semantic parity，尚没有目标设备 codegen/runtime。最小项目：只 lower 一个 fixed-shape MatMul composite，先有 rule-based TIR schedule，再以 1 个可控 tile 参数形成小搜索空间；每个候选必须 build、在目标端同步运行、写已知模式回读并记录测量失败原因。

[^4]: **ExecuTorch custom delegate 是拓展，不是现有实现。** 它的价值是把“哪个子图归 backend、哪个 op fallback”放到公开、可维护的 Edge Dialect / partitioner 接口中。它不自动产生自研 NPU codegen，也不保证所有设备共用一个 binary；每个 target artifact 仍必须独立验证。

[^5]: **MLIR / StableHLO 是概念和验证扩展，不是现有自研编译器的既有依赖。** MLIR Dialect Conversion 提供 `ConversionTarget + RewritePattern + TypeConverter` 的 legality 模型，适合拿来校准当前 Relax support check 的设计；StableHLO 可作为框架与编译器之间的可移植 op 语义边界。

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
4. 一份简历每个主题只选一种风格；AI 端侧岗选「第一部分（1B/2B/3B）+ 第二部分（风格二/三）」，编译器岗选对应编译器文档。
