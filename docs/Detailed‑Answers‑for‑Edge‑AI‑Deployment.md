**# 端侧 AI 部署面试题（图像算法方向）详答**

\> 适用背景：超分辨率、ISP、视频链路、RKNN/QNN、NPU-DSP-CPU、算子、量化、性能调优和故障排查。

\>

\> 本文把“面试短答”和“能继续深挖的工程证据”放在一起。涉及 NX9031 的地方，明确区分官方文档、板端实测和推断；不要把通用 SoC 经验直接当成 NX9031 的硬件事实。

**## 0. 先记住一条总原则**

端侧部署不是“把模型转成一个文件”这么简单，而是一个有正确性门禁的编译与运行时链路：

\`\`\`text

训练模型/数据定义

    -> 导出与图检查

    -> 图优化、布局和量化

    -> 后端分区与硬件编译

    -> 模型文件（如 RKNN/QNN/AllSpark AP+AOM）

    -> 板端 Runtime

    -> 预处理 -> 推理 -> 后处理

    -> correctness -> 性能/功耗/稳定性

\`\`\`

任何“快了多少”的数字，都应绑定模型 hash、输入 shape/dtype、量化配置、芯片分区/频率、warm-up、重复次数、同步方式和 profiler 原始产物。节点数减少不等于性能提升，编译成功也不等于端上执行正确。

**---**

**## 一、基础流程**

**### 1. 把一个 PyTorch 图像模型部署到端侧 NPU 开发板，完整链路是什么？每一步有什么坑？**

**#### 面试短答**

1\. 固化模型和预处理：切换 \`eval()\`，固定输入 shape、颜色顺序、归一化、padding/resize、输出定义，并保存 checkpoint、代码和数据版本。

2\. PyTorch 导出 ONNX：检查 dynamic axes 是否真的需要；若硬件偏好静态 shape，优先导出静态 batch/分辨率。

3\. ONNX 图检查与化简：\`onnx.checker\`、shape inference、onnx-simplifier、ONNX Runtime reference 对齐输出。

4\. 进入端侧工具链：RKNN-Toolkit、QNN/SNPE 或厂商编译器完成算子合法性检查、图融合、布局转换、常量折叠、量化和后端分区。

5\. 生成硬件模型：例如 \`.rknn\`、QNN context binary，或 AllSpark 的 \`.ap/.aom\`；保存 resolved config 和输入产物 hash。

6\. 板端 C/C++ Runtime：初始化设备和内存，加载模型，配置 tensor dtype/layout，执行预处理、推理、后处理。

7\. 正确性门禁：与 Python/ONNX Runtime FP32 reference 比较；再用故意错误的 golden 做负控，确认比较失败会返回非零退出码。

8\. 性能实验：warm-up 后测 P50/P95、端到端 FPS、单次推理、内存、带宽、功耗；用 profiler 定位算子和设备归属。

**#### 深挖与常见坑**

\| 阶段 | 常见问题 | 建议验证 |

\|---|---|---|

\| 导出 | 训练态 BatchNorm/Dropout、动态控制流、Python 算子残留 | \`model.eval()\`；固定 seed；检查 ONNX node/opset/shape |

\| 输入 | NCHW/NHWC、RGB/BGR、\`uint8\`/\`float32\`、mean/std、量化 zero-point 不一致 | 用同一张原始图逐步 dump 预处理后的 tensor |

\| 图优化 | 算子被融合后名字改变，调试时找不到原节点；动态 shape 使优化失效 | 保存优化前后图、节点数、融合列表和 shape |

\| 后端转换 | 算子不支持、属性组合不支持、分支/控制流不能下沉 | 查看编译器 partition/fallback 日志；建立最小复现图 |

\| 量化 | 校准集不代表线上分布；激活被 clip；输出范围改变 | 逐层统计 min/max、饱和率和误差；用真实夜景/高 ISO 数据校准 |

\| 部署 | 模型文件和 Runtime/firmware 不匹配；AArch64 库误在 x86 运行 | 绑定工具链版本、固件版本、\`file\`/\`readelf\` 和 SHA256 |

\| 性能 | 把预处理/后处理排除在外，或未同步就计时；频繁 malloc/copy | 端到端和纯推理分开计时，明确同步点和 warm-up |

一句可复用的排查口径是：先锁定输入和 reference，再确认每个子图落在哪个后端，最后才优化算子和内存。

**### 2. ONNX-Simplifier 主要做哪些优化？**

**#### 面试短答**

它主要做 shape 推理、常量折叠和冗余节点消除，例如把可计算的 \`Shape/Gather/Unsqueeze/Squeeze\` 结果提前算好，移除训练遗留的 Dropout，消除等价的冗余 transpose/reshape，并检查简化前后输出一致。

**#### 需要纠正的细节**

\- \`Conv-BN\` 融合通常是后端编译器或专门 graph optimizer 的职责，不能笼统地说“onnx-simplifier 一定完成 Conv-BN 融合”。

\- 简化器不会凭空解决所有动态 shape、控制流或不支持算子；简化后仍需用目标工具链重新检查。

\- 过度简化或错误的 \`input\_shapes\` 可能把动态模型错误固定。保留原 ONNX，记录 simplifier 版本和命令。

**### 3. NPU、DSP(HVX)、ARM CPU-NEON 三者硬件定位有什么区别？分别擅长什么算子？**

**#### 面试短答**

\- **\*\*NPU/矩阵阵列\*\***：高吞吐 MAC、片上 SRAM、DMA 和专用数据流，擅长 Conv、Depthwise Conv、GEMM/MatMul、部分 attention；对小张量、复杂控制流和纯布局变换不一定高效。

\- **\*\*DSP-HVX\*\***：宽向量 SIMD，适合逐元素、定点向量、滤波、shuffle、permute、transpose、音视频/ISP 预处理。它是否存在取决于 SoC；不能把所有 SoC 的“向量核”都叫 HVX。

\- **\*\*ARM CPU-NEON\*\***：通用控制流、系统调用、复杂后处理、稀疏分支和不支持算子 fallback；开发灵活，但在大规模 MAC 和高带宽搬运上通常不如专用单元。

**#### NX9031 特例**

NX9031 的 AllSpark 文档描述的是一个 NPU 层级，cluster 内含 **\*\*Cayley\*\*** 和 **\*\*Fermat\*\*** 两类核：Cayley 面向矩阵运算，Fermat 是自研 GPGPU/SIMT 风格向量核。文档和本次板端 trace 都能看到 Cayley/Fermat，但没有证据证明存在一个可由模型编译器独立选择的“外置 HVX DSP”设备。因此面试时应说：

\> 通用 SoC 上可以是 NPU/DSP/CPU 三后端；对 NX9031，当前证据更准确的表述是“主机 CPU + NPU 内部 Cayley/Fermat 执行单元”，不能把 Fermat 直接等同于 Qualcomm HVX DSP。

**### 4. 什么叫算子 fallback？会带来什么后果？**

**#### 面试短答**

目标后端不能实现某个算子、属性组合、shape 或 dtype 时，编译器把它留在另一个后端执行，这叫 fallback 或 graph partition。另一个后端可能是 DSP、CPU、GPU 或厂商的参考实现，具体由工具链决定。

**#### 后果与确认方法**

\- 子图被切碎，设备之间产生 tensor copy、layout convert、同步和 cache flush。

\- 算子本身可能只占 1 ms，但边界 copy/等待让整段变成 5 ms；频繁小 fallback 比一个大算子更糟。

\- 带宽和功耗上升，NPU 利用率下降，端到端 FPS 可能断崖式下降。

\- “编译通过”不代表没有 fallback。应看 partition 日志、Runtime 事件、设备时间线和内存拷贝事件。

在 NX9031 的这次 ResNet50 实验中，顶层 AP 只有一个 \`AOM\_NODE\`，runtime 的 core placement 记录落到 Cayley/Fermat；这证明该次 top-level graph 交给 AllSpark NPU，但不能据此声称所有模型都不会 CPU fallback。

**### 5. FP32、FP16、INT8 的优缺点，端侧怎么选？**

\| 格式 | 优点 | 风险/代价 | 典型选择 |

\|---|---|---|---|

\| FP32 | 精度和动态范围最好，reference/debug 方便 | 内存、带宽和算力压力大 | 基线、敏感层、疑难排错 |

\| FP16 | 内存减半，现代 NPU 常有原生路径，通常精度损失小 | 不是所有算子都有同等高效路径；速度不会保证“翻倍” | CV 模型首选的高质量部署基线 |

\| INT8 | 体积/带宽约为 FP32 的 1/4，整数 MAC 便宜，吞吐和能效好 | 需要校准/QAT；激活、残差、SR 输出对误差敏感 | 分类/检测等成熟模型，经过精度门禁后使用 |

选型顺序：先用 FP32/FP16 建立正确性基线，再对候选敏感层做 INT8 消融，依据任务指标、延迟、带宽和功耗选择混合精度，而不是只看理论 TOPS。

**---**

**## 二、量化专题**

**### 6. PTQ 与 QAT 的区别、优缺点和选型？**

**#### PTQ（训练后量化）**

训练好的浮点模型直接用少量校准数据估计权重/激活范围并转换为 INT8。优点是快、无需重新训练，适合先做可行性验证和对精度不敏感的分类/检测模型；缺点是模型没有适应量化噪声，长链路、残差、超分和生成式输出可能明显掉点。

**#### QAT（量化感知训练）**

训练时插入 fake quant 或等价量化噪声，前向模拟端侧误差，反向仍使用近似梯度。优点是模型能主动适应量化，精度通常更好；缺点是要重新训练、调学习率和正则，训练数据和工具链配置必须与部署端一致。

**#### 面试决策**

先 PTQ 做基线和敏感层定位；若任务精度不达标，依次尝试校准集修正、混合精度、clip/observer 调整，再进入 QAT。超分、ISP 和颜色/纹理恢复通常优先保留首尾层或高敏感残差块为 FP16。

**### 7. PTQ 校准集如何选择？分布不一致会怎样？**

校准集不是越大越好，而是要覆盖真实激活分布：日光/夜景、不同曝光和 ISO、运动模糊、不同相机、极暗/极亮区域、线上常见分辨率和场景。它应使用与训练/推理完全相同的 decode、resize、颜色和归一化流程。

分布不一致时，observer 估出的 scale 可能过大（有效分辨率下降）或过小（大量值被 clip）；结果是暗部细节丢失、高光饱和、噪点放大、颜色偏移，且平均准确率可能掩盖极端场景灾难。应同时看逐层直方图、饱和率、分位数和按场景分桶的任务指标。

**### 8. 超分模型 INT8 后画质下降、噪点放大、颜色偏移，怎么解决？**

1\. **\*\*混合精度\*\***：输入/输出、第一层/最后一层、上采样、残差累加和颜色变换等敏感路径保持 FP16。

2\. **\*\*QAT\*\***：让网络学习抵抗量化噪声，必要时只对敏感 block 做 QAT。

3\. **\*\*重新估计 clip\*\***：比较 min-max 与 percentile/校准 observer，避免少数 outlier 让 scale 过大或范围过窄导致截断。

4\. **\*\*值域保护\*\***：明确输入输出是 \`[0,1]\`、\`[-1,1]\` 还是 YUV/线性 RGB；限制异常激活并检查量化前后零点。

5\. **\*\*按通道量化/权重重排\*\***：若后端支持，对权重使用 per-channel scale，减少不同通道动态范围差异。

6\. **\*\*逐层定位\*\***：从输入、浅层、上采样、重建头逐段替换回 FP16，找到最早出现 PSNR/色差突变的层。

7\. **\*\*接受 FP16\*\***：若画质是硬门槛，整体 FP16 可能比追求 INT8 更合理；要用实测延迟/带宽解释这项取舍。

**### 9. 什么是对称量化、非对称量化？zero-point、scale 是什么？**

常见仿射量化可写为：

\`\`\`text

q = clamp(round(x / scale) + zero\_point, qmin, qmax)

x\_hat = scale \* (q - zero\_point)

\`\`\`

\- **\*\*对称量化\*\***：\`zero\_point = 0\`（或固定在整数域中心），正负范围相同，实现简单、整数 MAC 友好；若数据明显偏向一侧，动态范围利用率可能低。

\- **\*\*非对称量化\*\***：\`zero\_point\` 可非零，使真实 0 更准确地映射到整数网格，适合非负激活或值域偏斜数据；硬件计算可能需要额外 zero-point 修正。

\- \`scale\` 决定一个量化整数对应多少真实值；过大浪费分辨率，过小会 clip。必须记录 per-tensor/per-channel、dtype、rounding 和饱和策略，不能只说“INT8”。

**### 10. PC 上 FP32 很好，量化后端侧结果完全不对，从哪些方向排查？**

按以下顺序排查，能最快区分“输入错”“量化错”“算子错”：

1\. **\*\*预处理对齐\*\***：RGB/BGR、NCHW/NHWC、resize 插值、uint8 减 128、mean/std、YUV 转换、stride/padding。

2\. **\*\*输入输出 dtype/layout\*\***：端侧 tensor 的 dtype、scale、zero-point、字节序和输出反量化是否正确。

3\. **\*\*校准数据\*\***：校准样本数、真实分布、动态范围、是否错误复用了 test 数据或空目录。

4\. **\*\*逐层误差\*\***：在 PC reference 和端侧 dump 相同中间层，找第一处误差爆炸；不要只比较最终图片。

5\. **\*\*算子/属性支持\*\***：尤其是 resize、padding、sigmoid、softmax、量化乘加、插件算子；用单算子最小图复现。

6\. **\*\*溢出/clip\*\***：检查激活饱和比例、累加器位宽、权重 scale、残差加法是否发生截断。

7\. **\*\*工具链/固件匹配\*\***：模型由哪个版本编译，板端 Runtime/firmware 是否匹配；确认没有加载旧模型文件。

8\. **\*\*负控\*\***：故意换错 golden 或打乱输入，确认 correctness harness 能返回失败；否则“exit 0”没有证据价值。

**---**

**## 三、算子与硬件**

**### 11. Transpose 一般跑 NPU 还是 DSP？为什么？什么场景可以消除？**

**#### 面试短答**

Transpose 是内存/布局重排，算术 MAC 很少，通常受访存、cache 和 DMA 限制。因此它不天然适合矩阵 MAC 阵列，可能被编译到向量单元、DSP、专用 layout engine，或在不支持时回退 CPU；具体归属必须以编译器 partition 和 runtime trace 为准。

**#### 优化方式**

\- 让相邻算子使用同一 layout，避免 NCHW↔NHWC 来回切换。

\- 权重在离线阶段预先 transpose/pack。

\- 融合 transpose + 后续 elementwise/卷积，直接改变 kernel 的索引访问顺序。

\- 使用后端支持的 tiled layout，而不是显式生成完整中间 tensor。

\- 合并连续 transpose，利用 \`permute(permute(x))\` 的等价关系消除节点。

在 NX9031 上，Fermat 是 NPU 内部的向量/SIMT 执行单元，不能把“Fermat trace”翻译成“外置 DSP”。某个 layout kernel 的具体落点仍应看实测时间线。

**### 12. Conv-BN-ReLU 融合原理和收益？**

推理阶段 BN 参数是常量。若卷积输出为 \`y = W\*x + b\`，BN 为 \`gamma\*(y-mu)/sqrt(var+eps) + beta\`，可折叠为：

\`\`\`text

W' = W \* gamma / sqrt(var + eps)

b' = (b - mu) \* gamma / sqrt(var + eps) + beta

\`\`\`

这样 Conv 和 BN 不必各读写一次完整 feature map；ReLU 还可作为同一 kernel 的 epilogue。收益是减少 kernel launch、全局内存读写、同步和中间 buffer，提升带宽利用率。注意训练态 BN、动态参数、量化边界或残差拓扑可能阻止融合；融合后要重新做数值对齐。

**### 13. 模型里大量 Concat，在 NPU 上有什么性能问题？**

Concat 往往需要分配输出并把多个输入 feature map 拷贝到不同偏移，带来读写带宽、地址计算、同步和临时内存开销。若硬件没有原生 concat，编译器会生成 copy/layout kernels，可能把本来并行的计算串起来。

优化包括：减少多分支拼接、改用加法/投影、提前规划连续 buffer、让后续算子直接消费分段布局、融合 concat 后的轻量操作，并通过 trace 验证 copy 时间是否下降。不能只看 FLOPs，因为 concat 几乎没有 FLOPs 但可能是热点。

**### 14. NCHW 和 NHWC 格式有哪些坑？**

PyTorch 通常训练 NCHW，而许多 NPU 内部或 kernel packing 偏好 NHWC/blocked layout。若编译器不能贯穿传播 layout，就会插入 transpose；网络中频繁来回切换会产生大量 layout kernel、额外 buffer 和 cache miss。

应在模型入口和后端边界明确 layout，检查每个子图的输入输出 contract；把 transpose 数量、字节数和耗时列入 profiler。常见错误还有把 \`[N,C,H,W]\` 当成 \`[N,H,W,C]\` 读取，或在后处理阶段忘记把量化/blocked 输出还原。

**### 15. 什么是 tiling？为什么需要？什么情况下触发？**

当 feature map、权重或工作集超过 NPU 片上 SRAM/缓存，编译器会把空间、通道或 batch 切成 tile，分块搬入、计算、写回 DDR，再处理下一块。这样能在有限片上存储下运行大图，但引入 DMA、边界 halo、同步和 DDR 流量。

大分辨率超分、检测 neck 的大 feature map、batch 较大时更容易触发。观察方法包括 compiler tiling report、DMA/DDR counter、tile 数和每 tile kernel 时间。tile 边界处理错误会产生接缝伪影；即使正确，tile 太小也会让调度和搬运开销超过计算收益。

**---**

**## 四、性能调优与故障排查**

**### 16. 端侧模型 FPS 很低，排查流程是什么？**

1\. **\*\*先确认正确性\*\***：模型、输入、输出和负控都通过；否则性能数字无意义。

2\. **\*\*拆分端到端阶段\*\***：预处理、H2D/DMA、模型、D2H、后处理分别计时，明确同步点。

3\. **\*\*看 profiler 时间线\*\***：按模型/AOM、task、kernel 聚合，找 P50/P95 hot spot。

4\. **\*\*确认设备归属\*\***：看是否有 CPU/DSP fallback、layout/copy、同步空洞；NX9031 使用 AllSpark/nxPerf 时看 Cayley/Fermat core placement。

5\. **\*\*区分计算与带宽\*\***：看 MAC/向量利用率、DDR 读写、cache miss、DMA 等待；检查 transpose/concat 是否占大头。

6\. **\*\*检查 tiling 和分区\*\***：tile 数、片上 SRAM 命中、cluster limit、频率和温度是否一致。

7\. **\*\*检查调度与并发\*\***：warm-up、队列深度、stream、锁、buffer reuse、线程亲和性；排除频率降档和热 throttling。

8\. **\*\*做受控消融\*\***：一次只改 layout、融合、量化、cluster 数或 batch，保留 raw trace 和结果。

**### 17. 带宽瓶颈和算力瓶颈怎么区分？**

\| 现象 | 更像算力瓶颈 | 更像带宽/访存瓶颈 |

\|---|---|---|

\| MAC/向量利用率 | 高且稳定接近饱和 | 低，常在等待数据 |

\| DDR/片外流量 | 不一定高 | 高，或 DMA/读写等待明显 |

\| 算子特征 | 大 Conv/GEMM、计算密度高 | transpose、concat、elementwise、小 batch |

\| 优化方向 | tile、并行度、kernel/FLOP 利用率 | 融合、layout、buffer reuse、压缩和减少搬运 |

更严谨的做法是用 roofline 思路比较算术强度和实测带宽/峰值算力，并结合 profiler counter。不要仅凭“CPU 很闲”判断 NPU 在算；NPU 可能在等 DDR。大多数 CV 端侧瓶颈常常是带宽和调度，而不是理论 TOPS 不够。

**### 18. 视频实时超分出现帧间闪烁，可能原因和修复？**

**#### 原因**

\- 模型是单帧空间 SR，没有时序一致性约束。

\- INT8 激活/输出量化误差随每帧内容变化，噪声和颜色在边界来回跳。

\- 自动曝光、白平衡、ISP 参数或预处理统计量每帧变化。

\- 输入帧时间戳、裁剪窗口、stride 或 chroma 对齐不稳定。

\- 异步队列、buffer reuse、读写竞态或帧序错乱；不要先把它归因于 NPU“非确定性”。

\- 频率/温度导致调度抖动通常影响延迟，不应直接造成数值闪烁，除非伴随竞态或超时丢帧。

**#### 验证和修复**

固定同一帧重复推理，检查 bitwise/PSNR 一致性；记录输入、预处理参数、输出和时间戳。再用固定 ISP 参数、FP16、单线程同步 buffer 做二分实验。修复可选时序模型/光流约束、跨帧一致性 loss、敏感层 FP16、稳定曝光白平衡、双缓冲和严格 fence；同时监控丢帧和队列深度。

**---**

**## 五、项目与开放设计题**

**### 19. 请讲一个“模型从训练到端侧上线”的项目，你如何证明贡献？**

建议按 **\*\*背景—约束—基线—假设—改动—证据—结果—复盘\*\*** 讲：

1\. 背景：例如 4K 视频超分，目标是 30 FPS、PSNR/LPIPS 不下降、功耗受限。

2\. 基线：记录 FP16/FP32、输入 shape、编译版本、端到端 P50/P95、内存和功耗。

3\. 假设：Profiler 显示 transpose/concat 和 DDR 等待是热点，而不是 Conv 算力不足。

4\. 改动：统一 layout、融合节点、对重建头保留 FP16、其余 INT8；每次只改一个变量。

5\. 证据：保存编译日志、partition、逐层误差、runtime trace、模型 hash 和控制组。

6\. 结果：同时报告任务指标、延迟分位数、FPS、带宽/功耗；说明统计方法和未解决项。

不要只说“用了 NPU 所以快了”；要能回答“哪一个 kernel、因为哪一种资源、通过什么 control 证明”。

**### 20. 设计一个实时视频超分部署方案，你会怎么拆模块？**

\`\`\`text

采集/时间戳 -> ISP/色彩转换 -> resize/归一化 -> NPU SR

       -> 色彩还原/锐化/后处理 -> 编码/显示

\`\`\`

设计重点：

\- 先确定 Y-only、YUV 或 RGB 处理，避免无谓的全分辨率三通道计算。

\- 预处理尽量复用 ISP/DMA，固定 stride、对齐和零拷贝 buffer。

\- NPU 子图固定 shape；动态分辨率用多份已编译模型或合理 padding。

\- 把首尾敏感层和颜色变换保留 FP16，量化中间计算；以视频质量和帧间一致性为门禁。

\- 采用双/三缓冲、异步流水和 fence；测量采集到显示的端到端 latency，而不是只测 NPU。

\- 设计丢帧策略：宁可丢旧帧也不要显示乱序帧；所有帧携带 sequence/timestamp。

**### 21. 什么时候采用“CPU + DSP/向量 + NPU”混合后端？**

当模型含有大块 Conv/MatMul、同时存在大量 layout/滤波/逐元素或复杂逻辑时，混合后端通常合理。切分原则是让每个子图足够大，减少跨设备拷贝，并使生产者和消费者 layout 一致。要比较三种方案：全 NPU、NPU+向量单元、NPU+CPU；记录 copy、同步和功耗，而不是只比较各自 kernel 时间。

对 NX9031，应把“混合后端”理解为主机 CPU 与 NPU 内部 Cayley/Fermat 协同，除非 SDK 明确提供独立 DSP backend。Fermat 的 vector/SIMT 特性让它在功能上像向量加速器，但 ISA、内存层级和调度归属仍不同于 HVX。

**### 22. 量化后精度掉 5%，你如何设计定位实验？**

建立阶梯式对照：

1\. FP32 reference。

2\. FP16 全模型。

3\. FP16 权重 + INT8 激活。

4\. 逐 block INT8，其他 block FP16。

5\. 首尾层/残差/上采样回退 FP16。

6\. 更换校准集或 observer/clip。

7\. 必要时 QAT。

每一步保存逐层误差、激活饱和率、场景分桶指标和端侧性能。若某一层回退就恢复质量，说明存在敏感路径；若 PC 模拟正常、板端异常，则优先查 dtype、zero-point、算子实现和 Runtime 反量化，而不是立即重训。

**### 23. 如何用 nxPerf/Profiler 证明某个 kernel 调度到了哪个设备？**

证据强度从高到低：

1\. Runtime 时间线或 core placement 直接出现 device/core 字段。

2\. 编译器 partition/调度日志明确记录 backend 和 kernel。

3\. 静态模型/AP/AOM 只证明存在某类节点或子图，不能单独证明一次运行的实际核。

4\. 仅凭 kernel 名称猜测（如包含 \`transpose\`）最弱，必须标为推断。

NX9031 的实际流程是：用 \`profiling\_mode=12\` 采集软件 timer；在需要时启用 \`npu\_fw\`、\`npu\_art\`、\`npu\_sw\`；用与 AOM stem 匹配的 \`allspark\_grid\_info.txt\` 解析；查看 \`nxperf\_core\_placement.csv\` 或 SQLite。\`mode=15\` 包含 Grid 和 timer，适合 one-shot 静态检查，不应用于正常延迟分布；要做延迟对照，使用独立 mode=0/12 重复实验。

**### 24. 面对一个“编译成功但 FPS 低”的新模型，第一天如何安排实验？**

上午：锁定模型/输入/工具链 hash，跑 Python 与端侧 correctness，保存一次 FP16 baseline。下午：开启 profiler，导出 kernel、设备归属、DDR、tile、copy 和同步数据；把时间按计算、访存、调度、fallback 分类。随后做三个最小消融：统一 layout、关闭一个大 concat、把敏感 block 回退 FP16。第二天再决定是改图、改量化、改 kernel 还是改 runtime 并发。

面试官想听到的是可证伪的实验设计，而不是“多开线程/换 INT8”。每个结论至少要有一个受控变量、一个反例或 control，以及可以指向机制的 IR/trace/逐层误差。

**---**

**## 6. NX9031 本次实验证据与面试口径**

**### 6.1 芯片设备结论**

\- **\*\*有 CPU\*\***：当前板端 guest 探针显示 \`getconf \_NPROCESSORS\_ONLN=40\`，CPU part 为 \`0xd42\` 24 个、\`0xd43\` 16 个逻辑处理器；研究报告中的历史 AD guest 记录为 34 个逻辑处理器。两者存在配置/固件/虚拟域差异，不能据此断言物理核心总数。

\- **\*\*有 NPU\*\***：可见 \`/dev/nnp0\`、\`/dev/nnp\_ctrl\`、\`/dev/nnp\_prof\`、\`/dev/nxmap\` 等节点，\`nnp\` 模块加载；报告列出 \`r52\_td\`、\`cluster\_group0..3\`、\`fermat\_cluster\`、\`profiler\` 等 NPU 子设备。

\- **\*\*没有独立 GPU 的实物证据\*\***：当前探针没有 \`/dev/dri\`、\`/dev/mali\`、GPU sysfs class 或 GPU 模块。Fermat 被官方资料称为 GPU-style/GPGPU vector core，但它属于 NPU 内部，不应对外表述为一块独立 GPU。

**### 6.2 ResNet50 编译与运行结论**

本次可复现实验使用 FP16、\`device\_version=N93X\`、\`cluster\_limit=2\`、优化级别 2、融合和内存复用开启、\`profiling\_mode=12\`：

\- 编译器内部编译时间约 **\*\*156.882 s\*\***；宿主观测到的 Docker 调用墙钟约 **\*\*245.633 s\*\***，两者口径不同。

\- 静态 NPU 总内存约 **\*\*84.0962 MB\*\***，临时 tensor 约 **\*\*30.6328 MB\*\***。

\- AP 解码为 \`GraphDef(name="\_\_graph\_\_", version="N93X")\`，顶层只有一个 \`AOM\_NODE\`（\`\_\_aom\_0\`），输入 FP16 \`[16,3,224,224]\`，输出 \`[16,1000]\`。这证明该图被封装为一次 AllSpark NPU AOM 调用，不等于每个内部 kernel 都是同一种核。

\- 编译日志含 \`fermat=302249\`、\`cayley=3557446\`，但单位和含义没有在公开 schema 中定义，不能把它们直接称为毫秒、延迟或“每 kernel cost”。同时能看到 sequence time、Cayley/Fermat 并行度、overlap、估算代价和 cluster 分配等调度权重，说明编译器内部存在 cost model，但没有导出一个可直接用于面试的最终 per-kernel cost 表。

\- 静态 Grid 文件包含 1 stream、6 tasks、1231 entries，混有 profiling marker、prefetch/cache 操作以及 compute/layout kernel 名称。

\- nxPerf 运行正确性通过：\`element\_pass\_rate=1\`，进程正常退出。加载插件时出现 \`libcublas.so\` 警告，但没有阻断本次 board route，不能把该警告误判成模型失败。

\- 解析后的 core placement 有 **\*\*4046\*\*** 条 core 行：Cayley 3566 行、446 个唯一 kernel 名称、2 个 cluster、8 个 core slot；Fermat 480 行、30 个唯一 kernel 名称、2 个 cluster、16 个 core slot。core slice duration 约为 Cayley 1719–96559 ns、Fermat 493–11662 ns。

\- SQLite 还出现 10 条异常大的 task aggregate 行和 5 条 parser relation warning；这些行不能拿来当单次 kernel 延迟。面试中应说明“使用 core placement 明细，保留 parser warning，不把聚合异常值当 latency”。

**### 6.3 能否得到 estimate cost 和调度策略？**

可以得到三层信息，但可见性不同：

\| 信息 | 本次是否拿到 | 证据/限制 |

\|---|---:|---|

\| 静态模型内存、AOM、算子/布局 kernel 列表 | 是 | AP/AOM、Grid、编译日志 |

\| 实际运行 kernel 的 Cayley/Fermat/core placement 与时间线 | 是 | nxPerf report、CSV、SQLite；需保留解析 warning |

\| 每个 kernel 的公开 estimate-cost 数值、最终选择概率/权重 | 部分 | 日志能看到 cost-model 权重，未见稳定公开 per-kernel cost schema |

因此回答面试官时应说“可以通过编译日志 + nxPerf trace 恢复调度事实和时间代价；estimate cost 只能在工具链暴露字段时读取，当前公开产物不能把 \`perf\_summary\` 两个数字冒充成 per-kernel latency”。

**### 6.4 Cayley 与 Fermat 的准确类比**

功能层面可以类比为：

\- Cayley ≈ 高吞吐矩阵/MAC 阵列，常见 Conv、MatMul、GEMM。

\- Fermat ≈ 向量/SIMT/GPGPU 风格单元，适合逐元素、向量、部分 layout/cache/控制辅助工作。

但不能简化成“Cayley=NPU、Fermat=DSP”：两者都在 NX9031 的 NPU cluster 内，由 AllSpark TaskDispatch 统一调度；Fermat 不是独立 HVX DSP，也不是独立 GPU。Transpose 是访存重排，可能被融合、落到 Fermat/其他 layout 单元、DSP 或 CPU，必须以实际工具链和 trace 为准。

**---**

**## 7. 高频答题与排查 Checklist**

**### 编译前**

\- [ ] 固定模型、权重、输入 shape、opset、预处理和输出语义。

\- [ ] 保存原始 ONNX、simplified ONNX、量化配置和校准集 manifest。

\- [ ] 用 \`onnx.checker.check\_model\`、shape inference 和 ONNX Runtime 做 reference。

**### 编译后**

\- [ ] 检查 unsupported op、partition、layout、fusion、tiling 和内存报告。

\- [ ] 核对模型文件、Runtime、firmware、SDK 版本和 SHA256。

\- [ ] 不把编译成功、节点减少或日志中的未知数字直接说成性能提升。

**### 运行时**

\- [ ] correctness 先于性能；有错误 golden 负控。

\- [ ] warm-up 后测端到端和纯推理，明确同步、线程、频率、温度和 batch。

\- [ ] profiler 看设备归属、copy、DDR、tile、利用率和 P50/P95。

**### 量化/画质**

\- [ ] 校准集覆盖真实分布和极端场景。

\- [ ] 统计逐层误差、clip 饱和率、zero-point/scale 和按场景指标。

\- [ ] 对首尾层、上采样、残差和颜色路径做 FP16/INT8 消融。

**---**

**## 8. 本文依据的仓库材料**

\- [NX9031/N93X AllSpark 生态调研报告]\(../docs/nx9031-n93x-allspark-llm-ecosystem-research-report/nx9031-n93x-allspark-llm-ecosystem-research-report.md)

\- [AllSpark 入门手册]\(../docs/spark\_guide\_140/AllSpark%20%E5%85%A5%E9%97%A8%E6%89%8B%E5%86%8C.md)

\- [AllSpark Engine 性能优化手册]\(../docs/spark\_guide\_140/AllSpark%20Engine%20%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96%E6%89%8B%E5%86%8C.md)

\- [ADK nxPerf 工具用户手册]\(<../docs/ADK nxPerf 工具用户手册.md>)

\- [当前实验 manifest]\(../codex\_work/manifests/current\_experiment.yaml)

\- [可执行 ResNet50 调度检查 Notebook]\(../codex\_work/notebooks/n93x\_resnet50\_cost\_schedule.ipynb)

\- [Notebook 最终运行 summary]\(../codex\_work/n93x\_resnet50\_cost\_schedule/runs/20260831T042610Z\_1323866/summary.json)

\- [Cayley/Fermat core placement CSV]\(../codex\_work/n93x\_resnet50\_cost\_schedule/runs/20260831T042610Z\_1323866/nxperf\_core\_placement.csv)

\> 最后一条链接中的运行目录名可能因重跑而变化；以 Notebook 输出的实际 run directory 和 hash 为准。\`perf\_summary.log\` 的单位未定义，nxPerf 解析 warning 也应随证据一并保存。