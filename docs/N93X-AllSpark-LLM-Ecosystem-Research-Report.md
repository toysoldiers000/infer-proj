\> **\*\*\<span style="color: inherit; background-color: rgb(247,105,100)">本报告由\</span> \<span style="color: inherit; background-color: rgb(247,105,100)">Claude\</span> \<span style="color: inherit; background-color: rgb(247,105,100)"> 直接生成，仅供参考，部分数据可能存在错误\</span>\*\***

\>

\> **\*\*&#x20;对标文档\*\***：《AMD 395 BOX 调研报告》（\`AMDdoc\`）、《AMD 与 N93X/NX9031 性能对比》（\`compre\`） **\*\*调研日期\*\***：2026-08 **\*\*被测平台\*\***：NX9031（设备树标识 \`n93x\`）+ AllSpark SDK v1.4.2 / v1.5.2 **\*\*竞品参照\*\***：AMD Ryzen AI Max+ 395（gfx1151，Strix Halo） **\*\*数据来源\*\***：本报告所有 NX9031 数据均为本次实机实测；AMD 数据引自 \`AMDdoc\` 与 \`compre\`，本次未复测

**\*\*\***

[npu\\\_bench.tar.gz]\(\<files/NX9031 ! N93X AllSpark LLM 生态调研报告-npu\_bench.tar.gz>)

**## 一、摘要**





**### 1.1 调研目的与范围**

本报告调研 NIO NX9031 平台的 AI 软件生态，并与竞品 AMD 395 BOX 做同口径对比。章节结构与 \`AMDdoc\` 对齐，便于并列阅读。

调研重心是**\*\*生态\*\***——从一个新模型或新算子出发，到它能在硬件上跑起来，中间要经过哪些工具、这些工具由谁维护、缺口在哪里。性能数据作为生态成熟度的佐证。

两个 10.x 地址是 Intel NUC 台架控制机（\`ic-ubu-tb-181\` / \`ic-ubu-tb-263\`），不是被测对象；SoC 挂在其 \`192.168.1.0/24\` 私网上。

**\*\*方法学声明\*\***：本次所有 NPU 侧数据由自行编写的 CUDA kernel 经 CCA 工具链（\`npcc++ --platform=ASIC\`）交叉编译后在 NPU 上实测得到，每个 benchmark 内置正确性自检（写入已知模式后回读校验），自检失败即拒绝输出性能数字。此机制在调研过程中成功拦截了一次「kernel 未实际执行却报告 291 TB/s」的伪数据。

**### 1.2 主要结论**





**#### 结论 1：NPU 带宽与算力由 \`ClusterGroupMode\` 决定，默认 AD 配置只给到 1/4 芯片**

\`/etc/config/art/cluster\_group\_config.json\` 控制 NPU 的硬件分区。实测四档：

算力随分组数**\*\*精确等比缩放\*\***（5.121 / 4 = 1.280，实测 1.279），确认这是硬件级资源隔离。AD 变体出厂配置为 \`ClusterGroupMode: 4\` + \`MajorGroups: ["Group0"]\`，即单个应用默认只能用到 1/4 的 NPU；LLM 部署脚本会将其改为 \`0\`。

**\*\*竞品含义\*\***：这是 AMD 395 完全不具备的能力——NX9031 支持类似 NVIDIA MIG 的 NPU 硬分区，可让规划、感知、LLM 等多个 AD 应用在物理隔离的分区上并行，这正是车载场景所需。代价是单应用峰值性能取决于部署配置，横向对比时必须声明 mode。

**#### 结论 2：另有 70% 的 NPU 读带宽藏在四个未文档化寄存器后面**

厂商 LLM 部署脚本 \`run.sh\` 在启动容器前会写四个寄存器。同一次启动内做 A/B：

带宽大幅提升而算力完全不变，符合内存控制器 / NoC QoS 寄存器的特征。这些写入不在任何文档中，只出现在部署脚本里，且 \`fw\_reboot\` 会将其复位。

\> 上表为同一组 grid/block 扫描范围下的对照。后续加宽扫描范围（grid 至 2176、block 至 1024）后，调优配置下的峰值进一步提高到 **\*\*读 355.83 / 写 410.59 / 混合 389.57 GB/s\*\***，见 §5.2.2。

**\*\*结论\*\***：NX9031 的 NPU 读带宽有三个档位——161.5（AD 默认分区）、204.1（全簇未调优）、**\*\*355.8 GB/s（全簇 + 调优）\*\***。任何对外性能数据必须声明处于哪一档。

**#### 结论 3：内存为 Micron LPDDR5X-8533 / 64 GB / 512-bit，理论 546 GB/s；实测读 355.8 GB/s，比 AMD 高 49%，但仅用掉 65% 理论带宽**

**\*\*这里有一个比绝对值更重要的反差\*\***：NX9031 的理论带宽是 AMD 的 **\*\*2.13 倍\*\***（546.1 vs 256 GB/s），但实测只领先 **\*\*48.7%\*\***——因为带宽利用率差了 28 个百分点。AMD 侧 93% 已接近物理极限、几乎无优化空间；NX9031 侧还有约 **\*\*190 GB/s 的理论带宽未被利用\*\***。若访存效率做到 AMD 的水平，可达 508 GB/s。内存规格与推导见 §2.3。

由于 LLM decode 是内存带宽严格受限，这一项直接决定 decode 上限。\`AMDdoc\` 的核心论断「显存带宽是本机 LLM 性能上限，且不随软件优化变化」在 NX9031 上**\*\*不成立\*\***——这里带宽本身就随软件配置变化 2.15 倍。

**#### 结论 4：Fermat 向量核算力 FP32 5.12 / FP16 10.37 TFLOPS；矩阵核算力无法经 CUDA 路径测得**

手写 FMA kernel 只能驱动 Fermat SIMT 核。实测最佳 grid 恒为 2176 = 34 核 × 64，反推 Fermat 核为 **\*\*64 lane FP32\*\***，理论峰值 32 核 × 64 lane × 2 FLOP × 1.4 GHz = 5.73 TFLOPS，实测 5.12 = **\*\*89% 效率\*\***。FP16 经 \`half2\` 打包精确 2×。

Cayley 矩阵阵列需经 \`mmb\_\*\` scope 与 fractal 路径，不暴露给 CUDA C，本次无法直接测量。由 LLM prefill 反推有效算力下界约 9 TFLOPS（见 §5.4）。

**#### 结论 5：CUDA 源码级兼容成立，但边界比预期窄**

CCA 的 \`npcc++\`（clang 13.0.1 定制）确实可直接编译标准 \`.cu\`——\`global\`、\`blockIdx/threadIdx\`、\`cudaMalloc/cudaMemcpy/cudaStreamCreate/cudaMemGetInfo\`、\`half\`/\`half2\` 内建全部可用。本次自编的三个 benchmark 全部经此路径编译并在 NPU 上跑通。但有两条实测确认的硬边界：

\* **\*\*默认流（null stream）不支持\*\***。\`kernel<<\<g,b>>>()\` 返回 \`rtErrorInvalidResourceHandle\`(33)，必须写成 \`kernel<<\<g,b,0,stream>>>\`。

\* **\*\*cuBLAS host API 无法从 CUDA 源码调用\*\***。SDK 自带的 \`allspark\_samples/CCA/cublas\_sample\` 在 v1.4.2 与 v1.5.2 下**\*\*均编译失败\*\***。

详见 §4.2.1。

**#### 结论 6：\`libcublas.so\` 是 rocBLAS 的映射层——CUDA 兼容层建在竞品的开源基础设施上**

头文件中 \`cublasHandle\_t → rocblas\_handle\`、\`cublasCreate\_v2 → rocblas\_create\_handle\`、\`cublasHgemm → rocblas\_hgemm\`；CCA include 目录直接包含 \`hip/\`、\`rocblas.h\`、\`rocblas\_module.f90\`、\`thrust/\`、\`cub/\`。编译失败的根因也在此：\`hip\_common.h\` 在 \`clang && CUDA && !HIP\` 时自动定义 \`HIP\_PLATFORM\_NVIDIA\`，而 \`npcc++\` 恰好满足该条件，与显式的 \`HIP\_PLATFORM\_AMD\` 冲突触发 \`#error\`。





**#### 结论 7：编译链路自研全栈共 9 条路径，全部单点供给**

ATVM（TVM fork）→ AGE（图编译）→ HAUK（算子编译）→ ACE（算子库）→ APEX（遗传算法调优）→ Alchemy（LLM 前端）→ AQUA（量化），外加 Graph API（TensorRT 镜像）、ONNX 前端、CCA。每层只有 NIO 一个供给方。对比 AMD 的「ROCm + llama.cpp 社区 + Lemonade 打包」，好处是版本对齐无缝，坏处是没有外部力量为新模型垫工程量。详见 §4.2。

**#### 结论 8：存在指令级仿真器，可无硬件编译并运行**

\`npcc++ --platform=SIM\` 配合 \`libjarvis\` / \`libesl2jarvis\`，\`soc.toml\` 建模精确到 LPDDR5 控制器队列深度、L2 端口号与 QoS 优先级映射（18 个段）。AMD 侧无等价物。详见 §4.2.9。

**#### 结论 9：推理服务只有一条路径；并发上限 1 是编译期固化**

镜像内是 ollama \`0.21.2-22-gf55d35e-dirty\`，新增 \`server\_alchemy.go\` 后端替换 llama.cpp，实际执行者为 \`alchemy-runner\`。日志显示 \`OLLAMA\_NUM\_PARALLEL=1\` 且引擎 \`max\_batch\_size=1\`，因此 1→8 路并发聚合吞吐恒定 34～35 token/s。放开并发需重新编译模型。AMD 侧有 llama.cpp / vLLM / Lemonade / Ollama 四条路径。详见 §4.4。

**#### 结论 10：GGUF 权重直读正在开发，是最重要的生态动向**

v1.5.2 的 \`alchemy\_compile/docs/gguf\_quantization\_design.md\` 给出完整 Q8\\\_0 导入设计，首批目标 \`bge-m3\` / \`bge-reranker-v2-m3\`，后续规划 Q4\\\_0/Q4\\\_K/Q5\\\_K/Q6\\\_K/IQ\\\*。打通后模型获取从「NIO 逐个实现（月级）」变为「复用 HuggingFace GGUF」。详见 §4.3。

**#### 结论 11：多模态前端已具备，现场部署包不完整**

Alchemy 已实现 \`qwen2\_5\_vl\`、\`qwen3\_asr\`，1.5.2 追加 \`bge\_m3\` / \`bge\_reranker\`。设备上有已编译的 Qwen3-ASR-0.6B（30 语种）与 BGE-M3 embedding 服务。但 ASR 端到端实测失败——mel 前端与 encoder 正常执行，随后在 tokenizer 加载处崩溃，\`model\_out\` 缺 \`tokenizer.json\`。未发现任何 TTS 路径。详见 §4.5。

**### 1.3 一句话竞品定位**

\> 硬件上 NX9031 的**\*\*理论内存带宽是 AMD 395 的 2.13 倍、实测领先 49%\*\***（且仍有 35% 理论余量未被利用），靠 W4A8 量化与混合注意力架构在 prefill 上大幅领先，decode 落后约 20%（受限于实现效率而非带宽）；并且具备 AMD 完全没有的 NPU 硬分区与指令级仿真器。真正的差距不在硬件而在生态——AMD 有四条推理框架路径和整个 GGUF 社区，NX9031 只有一条私有 ollama 分支和一份逐个手写的模型列表。

**\*\*\***





**## 二、产品形态**





**### 2.1 SoC 形态与变体**

NX9031 是车规 AD SoC，非独立产品形态，以域控制器内板卡形式交付。本次接触三个 SVB（Silicon Validation Board）变体：

**\*\*CPU 构成\*\***（AD 变体 guest 视角）：

\* 24 × Cortex-A78AE @ 2208 MHz（MIDR part \`0xd42\`），按 DSU 分 6 簇，每簇 4 核：\`0-3\` / \`4-7\` / … / \`20-23\`

\* 5 × Cortex-A65AE @ 1900.8 MHz（part \`0xd43\`），SMT2，共 10 逻辑核：\`24-31\`（4 物理核一簇）+ \`32-33\`（1 物理核一簇）

\* cpufreq governor = \`userspace\`，锁定最高频；\`/proc/device-tree/ddr\_dvfs/status\` = \`disabled\`

**\*\*软件基线\*\***：Ubuntu 24.04.3 LTS，内核 6.1.83-rt28（PREEMPT\\\_RT），glibc 2.39。ADK 2.5.21.15，branch \`snapshot/release/everest/rdk\_linux/2.0.6\`，commit \`2aa9629\`，build 2026-07-31。

**\*\*运行模式\*\***：系统有 Release / Debug 双模式，由 \`/usr/local/bin/setdebug\` 切换（写 \`/mnt/data/tmp/.preserve\` 标记后重启生效）。Release 模式下 rootfs overlay 为 1.0 GB 且不保留修改；Debug 模式 overlay 6.9 GB 且修改跨重启持久化。**\*\*对&#x20;\*\***\`/etc/config/art\`**\*\*&#x20;的任何配置更改必须在 Debug 模式下进行才能持久化\*\***——这是调优 NPU 分区的前置条件。

**\*\*虚拟化与安全启动\*\***：dmesg 显示 IDC 域间通信（\`VM\_AD\_SYS\` → \`NPU\_SYS\`）、多 remoteproc（isp0 三核 / codec / fsi / scp / npu\\\_rproc）与 arm-smmu-v3 多实例（\`hv\_isp\_smmu\` / \`med\_smmu\` / \`com\_smmu\` / \`npu-smmu\`）。刷机 bundle 内 images 全部 \`.signed\`（bl31 / tee / hsm / aon\\\_lk / npu\\\_fw\\\_dtb\\\_loader / kernel\\\_dtbs / codec\\\_fw\\\_dtb / mcu1\\\_fw\\\_dtb / shmoo），guest rootfs 分 \`system\` / \`adk\` / \`app\` / \`rw\` 四个镜像。

**### 2.2 NPU 硬件结构**

参数取自 SDK 仿真器器件描述（\`sim/N93X/config/device.yaml\`、\`sim/config/soc.toml\`），与驱动 probe 日志、实测算力三方印证。

**#### 计算单元**

两类核分工由编译器常量确认：HAUK 的 \`KERNEL\_TYPE\_FERMAT\` / \`KERNEL\_TYPE\_CAYLEY\` / \`KERNEL\_TYPE\_MIX\`，以及 \`AIWrapper\` 接口中 \`get\_fractal\_size()\`（Cayley 专有）与 \`get\_warp\_num()\` / \`get\_vgpr\_num\_per\_warp()\`（Fermat 专有）。

**#### Fermat 核微架构（GPU 式）**

warp 宽 32、SGPR/VGPR 分离、per-block 共享内存——与 NVIDIA 执行模型同构，这正是 CCA 能做 CUDA 源码兼容的硬件基础。设备编译时 \`npcc++\` 报告的目标架构字符串是 \`sm\_80\`。

理论峰值校验：32 Fermat（NN 簇内）× 64 lane × 2 FLOP × 1.4 GHz = **\*\*5.73 TFLOPS\*\***，实测 5.121 TFLOPS = **\*\*89% 效率\*\***，自洽。

**#### 频率与互联**

Fermat / Cayley / TaskDispatch 均 1.4 GHz；总线端口频率 2.7 GHz，位宽 256 bit，读写 outstanding 各 128。

**#### 存储层级**

HAUK scope 常量：\`global\` / \`shared\` / \`L1\` / \`lmb\` / \`rmb\` / \`psu\` / \`mmb\_sparse\` / \`mmb\_bias\` / \`mmb\_deq\` / \`apb\`。

\* \`mmb\_sparse\` → 矩阵单元原生支持稀疏权重

\* \`mmb\_bias\`、\`mmb\_deq\` → bias 加法与反量化在矩阵单元内联完成，W4A8 路径不需要独立 dequant kernel

**\*\*Cache\*\***：4 个 NN cluster L2（每簇 1）+ 1 个 Fermat cluster L2 + 4 个 Global L2 分片（端口 \`0x20\`-\`0x23\`），地址交织粒度 8。

**\*\*地址窗口别名\*\***：\`/etc/config/art/internal/addr\_config.json\` 定义四个基址——\`kBinaryBaseAddr\`、\`kGlobalL2BaseAddr\`、\`kClusterL2BaseAddr\`、\`kByPassL2BaseAddr\`。即同一片 DRAM 在不同地址窗口下有不同的 cache 行为，包含一个**\*\*显式绕过 L2\*\*** 的窗口。AD 配置与 LLM 配置使用不同的窗口布局：

LLM 配置的窗口间距为 32 GB，AD 配置为 8 GB。

**#### 驱动与设备节点**

\`nnp\` 模块 probe 出 7 个 subdevice——\`r52\_td\`（Cortex-R52 任务派发器）、\`cluster\_group0\`\\\~\\\`3\`、\`fermat\\\_cluster\`、\`profiler\`，各自独立 SMMU stream ID。NPU 专用 SMMU（\`primus npu smmu\`）报告 10 个 TCU。预留 DMA heap 两块：\`reserve\\\_npu\\\_firmware\`、\`reserve\\\_npu\\\`。

设备节点：\`/dev/nnp0\`、\`/dev/nnp\_ctrl\`、\`/dev/nnp\_prof\`（仅 Linux 变体）、\`/dev/nxmap\`、\`/dev/ttynpu0\`、\`/dev/npu\_firmware\_log\`、\`/dev/ad\_npu\_tty\_c0\`、\`/dev/ad\_npu\_procfs\_c1\`。

固件重启节点：\`/sys/devices/platform/280f2000.nnp/fw\_reboot\`。

**#### R52 调试命令通道**

\`/proc/npu/\` 暴露任务派发器命令接口，内置条目：\`r52\_ddr\_bw\_seq\_read\`、\`r52\_ddr\_bw\_seq\_write\`、\`r52\_ddr\_lat\_read\`、\`r52\_ddr\_lat\_write\`、\`r52\_dhrystone\`、\`timerstamp\`。但 release 固件未编入这些命令（返回 \`mem\_test\:Command not Found\`），仅 \`timerstamp\_offset\` 可用。**\*\*即厂内存在 NPU 侧带宽/延迟的官方测量手段，但不在交付固件中\*\***——这也是本次不得不自行编写 CCA benchmark 的原因。

**#### 档位划分**

同一套编译链路覆盖三档硬件，靠 \`cluster\_num\` 编译参数切分。

**### 2.3 内存颗粒与理论带宽**





**#### 颗粒识别**

内核 debugfs 直接暴露了 DDR 器件信息（\`/sys/kernel/debug/ddr/\`）：

设备树 \`ddr-size\` 节点为 \`0x0000001000000000\` = 64 GiB，与 \`density\` 一致。\`ddr\_dvfs\` 节点 \`status = disabled\`，即 DDR 运行在固定频率、不做动态调频。

内核启动日志显示该 AD guest 的物理内存窗口为 58 909 696 KB（56.2 GiB），其中 30 816 400 KB（29.4 GiB）被 hypervisor 与各协处理器保留，guest 实际可用 28 093 296 KB（26.8 GiB）。即 64 GB 物理内存中约 40% 分配给 AD guest 之外的用途。

**#### 总线位宽（推定）**

设备树未向 guest 暴露 DDR 控制器节点——DDR 由 hypervisor / SCP 管理，guest 侧只有 \`ddr\_dbg\` 这个通往 DDR 调试固件的 mailbox 通道，且 release 固件未编入相应命令（见 §2.2 的 R52 调试通道）。因此位宽只能由实测反证：

实测纯写峰值 410.59 GB/s 超过 384-bit 配置的理论上限，而带宽不可能超过物理上限，因此**\*\*总线位宽至少为 512 bit\*\***。512-bit 也与 64 GB 容量自洽（8 组 × 64-bit，或 32 通道 × 16-bit LPDDR5）。

\> 该结论为实测反证所得，非厂商文档或寄存器读数。若后续能通过 \`ddr\_dbg\` 通道或 SCP 固件确认实际通道数，应以其为准。





**#### 理论峰值与利用率**

\`\`\`plain&#x20;text

8533 MT/s × 512 bit ÷ 8 = 546.1 GB/s

\`\`\`





**#### 与竞品的内存子系统对比**

两条值得注意的结论：

1\. **\*\*NX9031 的理论带宽是 AMD 的 2.13 倍，但实测只领先 48.7%。\*\*** 差距全部来自利用率——AMD 侧 93.4% 已接近物理极限，NX9031 侧 65.2% 还有约 190 GB/s 未被利用。这是本平台目前**\*\*最大的、可通过软件回收的性能余量\*\***。

2\. \`AMDdoc\` 的核心论断「显存带宽是本机 LLM 性能上限，且不随软件优化变化」在 NX9031 上**\*\*不成立\*\***。这里带宽既随配置变化（161.5 / 204.1 / 355.8 GB/s 三档，见 §5.2.2），又有 35% 的理论余量尚未吃到。AMD 侧「内存频率由 8000 提到 8533 可获约 67% 提升」是其已识别的最具性价比优化方向；NX9031 侧对应方向是**\*\*提高访存效率\*\***，潜在收益远大于 67%。





**### 2.4 NPU 分区（ClusterGroupMode）**

这是 NX9031 相对 AMD 的结构性差异，单列一节。

配置文件 \`/etc/config/art/cluster\_group\_config.json\`：

\`\`\`json

{ "ClusterGroupCfg": { "ClusterGroupMode": 4, "MajorGroups": ["Group0"] } }

\`\`\`

AD 变体出厂即为 \`Mode 4\`，Linux 变体的 LLM 部署脚本改为 \`Mode 0\`。实测四档算力（详见 §5.3）：

算力等比缩放到小数点后三位，确认为硬件级隔离而非软件调度。

\`/etc/config/art/async\_copy\_buffer\_config.json\` 中的 buffer pool 配置进一步印证其用途——按应用名分配：\`arg\_app\`、\`planner\`、\`map\_loc\`、\`desens\`、\`parking\`，全部是 AD 功能模块。即 NPU 分区的设计目标就是让规划、定位、脱敏、泊车等模块在物理隔离的分区上并行。

**\*\*竞品对比\*\***：AMD 395 的 gfx1151 不提供任何分区能力；XDNA2 NPU 因用户态缺失完全不可用。NVIDIA 的对应能力是 MIG，但仅在数据中心卡（A100/H100）上提供，消费级与 Jetson 无。

**### 2.5 与 AMD 395 的形态对比**

**\*\*定位判断\*\***：两者不是同一类产品。NX9031 的直接对比对象应是车载 AD SoC（Orin / Thor / 高通 Ride）。与 AMD 并列的价值在于：用一个开放平台作参照系，量化 NX9031 在「模型上车」路径上的生态成本。

**\*\*\***





**## 三、主要应用场景分析**

平台性能画像：**\*\*带宽与算力充足、prefill 强、decode 实现效率偏低、并发被编译期配置锁死、模型上车成本高但运行时确定性好且可硬分区\*\***。

**\*\*\***





**## 四、AI 生态支持调研**





**### 4.0 生态分层总览**

**\*\*一句话概括\*\***：NX9031 在「硬件能力与编译迁移工具链」这一层比 AMD 更厚、更自洽（CUDA 兼容、仿真器、硬分区三项独有），在「模型与服务生态」这一层比 AMD 薄得多且全部单点供给。

**\*\*\***





**### 4.1 驱动与 runtime 生态**

**\*\*内核侧\*\***：\`nnp\` 驱动为 NIO 自研、非上游，与 ADK bundle 绑定发布。probe 阶段完成 7 个 subdevice 的 SMMU 页表建立、mailbox 通道创建（\`ipcm\_tx/rx\` 0～5 共 6 对）与固件加载。

**\*\*用户态\*\***：核心是 \`libalchemy\_runtime.so\`，现场存在多个副本且版本不一致：

每个应用自带一份 runtime，与宿主 rootfs 解耦——与 AMD 侧「容器镜像与 SDK 各自携带完整 ROCm 用户态」的做法一致，都是用打包换版本自由。但现场四份 runtime 跨越 4 个月，缺乏统一版本管理。

**\*\*与宿主的耦合度\*\***：ollama 容器仅 1.43 GB，通过 \`/dev/nnp0\` 与 \`/dev/nxmap\` 直通设备节点，挂载 \`/usr/local/lib\` 与 \`/etc/config/art\`（只读）。本次已验证可脱离 docker 直接在宿主运行（见 §5.5）。

**\*\*ART 运行时配置\*\***（\`/etc/config/art/\`）：

\`runtime\_cfg.json\` 中出现 \`CUDA\_LAUNCH\_BLOCKING\` 这一 CUDA 原生调试开关，进一步印证 CUDA 兼容是贯穿设计而非表面包装。

**\*\*\***





**### 4.2 编程与编译生态**

AllSpark 提供 9 条从源到设备的路径。以下按抽象层级由低到高列出，全部在 SDK 样例中有对应实例。

**#### 4.2.1 CCA —— CUDA 源码级兼容层**

\`allspark\_cca/bin\` 下四个二进制：\`npcc++\`、\`ccafe++\`、\`ccafatbinary\`、\`clang++\`。\`npcc++ --version\` 报告 clang 13.0.1。

**\*\*编译命令形态\*\***：

\`\`\`bash

\# 设备上运行（ASIC）

npcc++ --device-target=N93X --platform=ASIC -O2 kernel.cu \\

       \--art-path="$ALLSPARK\_RUNTIME\_PATH" -ccbin aarch64-linux-gnu-g++ -o kernel

\# x86 主机上仿真运行（SIM）

npcc++ --device-target=N93X --platform=SIM  -O2 kernel.cu \\

       \--art-path="$ALLSPARK\_RUNTIME\_PATH" && ./a.out

\`\`\`

**\*\*本次实测确认可用的 CUDA 特性\*\***（三个自编 benchmark 全部经此路径跑通）：

\* \`global\` kernel、\`blockIdx\` / \`blockDim\` / \`threadIdx\` / \`gridDim\`、grid-stride loop

\* \`cudaSetDevice\` / \`cudaMalloc\` / \`cudaFree\` / \`cudaMemset\` / \`cudaMemcpy\`（H2D / D2H）

\* \`cudaStreamCreate\` / \`cudaStreamSynchronize\` / \`cudaStreamDestroy\` / \`cudaDeviceReset\`

\* \`cudaGetLastError\`、\`cudaMemGetInfo\`

\* 向量类型 \`float4\`，\`#pragma unroll\`

\* \`\<cuda\_fp16.h>\`：\`half\`、\`half2\`、\`\_\_hfma2\`、\`\_\_hadd2\`、\`\_\_floats2half2\_rn\`、\`\_\_low2float\`

\* \`fmaf\` 等 device 数学函数

**\*\*本次实测确认的兼容边界\*\***：

**\*\*头文件覆盖面\*\***（\`allspark\_runtime/include/cca/cuda/include/\`）：

\`\`\`plain&#x20;text

cuda.h  cuda\_runtime.h  cuda\_runtime\_api.h  cuda\_fp16.h  cuda\_bf16.h  cuda\_fp8.h  cuda\_fp4.h

cublas.h  cublas\_v2.h  cublas\_api.h  rocblas.h  rocblas\_module.f90

cub/  thrust/  hip/  crt/  bits/  internal/

device\_atomic\_functions.h  surface\_types.h  texture\_types.h  vector\_types.h  library\_types.h

cca\_tsharp.h  \_\_clang\_npcc\_preload.h

\`\`\`

\`cuda\_runtime\_api.h\` 注明「following cuda 10.0 runtime & driver api」，但同时定义了 \`CUtensorMap\` 系列（Hopper TMA 接口），说明抽象覆盖到较新的 CUDA 接口。

**\*\*配套工具\*\***：\`ccasan-npcc++\`（sanitizer 编译驱动，配合 \`libCCASanPass.so\`）、\`cca-memcheck\`（显存越界检查，等价 \`cuda-memcheck\`）。样例另有 \`relu\`、\`cca\_tsharp\`、\`crash\_demo\`。

**\*\*生态含义\*\***：从 NVIDIA 平台迁移自定义 kernel，AMD 路线需 hipify + 手工修 API 差异；NX9031 路线是换编译器名 + 补显式 stream。这是本平台相对 AMD 的实质性生态优势，但「零改动」是不成立的。

**#### 4.2.2 cuBLAS = rocBLAS 映射层**

\`cublas\_v2.h\` → \`cublas\_v2\_wrapper.h\` → 宏映射到 rocBLAS：

\`\`\`c

\#define cublasHandle\_t     rocblas\_handle

\#define cublasCreate\_v2    rocblas\_create\_handle

\#define cublasSetStream\_v2 rocblas\_set\_stream

\#define cublasHgemm        rocblas\_hgemm

\#define cublasSgemm        rocblas\_sgemm

\#define CUBLAS\_OP\_N        rocblas\_operation\_none

\#define CUBLAS\_STATUS\_SUCCESS rocblas\_status\_success

\`\`\`

**\*\*开放与禁用范围\*\***：\`cublas\_api\_wrapper.h\` 中启用的 compute type 只有 \`CUBLAS\_COMPUTE\_32F\`；**\*\*66 项被注释禁用\*\***，包括 \`CUBLAS\_COMPUTE\_32I\`（INT8 GEMM）、\`CUBLAS\_COMPUTE\_16F\`、全部 \`CUBLAS\_GEMM\_ALGO0\~13\`。启用的 GEMM 入口为 \`Sgemm\` / \`Hgemm\` / \`Cgemm\` 及其 Batched / StridedBatched 变体，**\*\*无&#x20;\*\***\`cublasGemmEx\`。

**\*\*编译失败根因\*\***（本次定位）：\`hip/hip\_common.h\` 在满足

\`\`\`c

\#if defined(\_\_NVCC\_\_) || (defined(\_\_clang\_\_) && defined(\_\_CUDA\_\_) && !defined(\_\_HIP\_\_))

\#define \_\_HIP\_PLATFORM\_NVIDIA\_\_

\#endif

\`\`\`

时自动定义 \`HIP\_PLATFORM\_NVIDIA\`。\`npcc++\` 恰好是 clang + \`CUDA\` + 非 \`HIP\`，于是与显式的 \`HIP\_PLATFORM\_AMD\` 同时成立，\`hip\_runtime.h\` 的三分支守卫落到 \`#else\` 分支触发 \`#error\`。尝试 \`-D\_\_HIP\_\_\` 绕过后，转为 \`HIP\_vector\_type\` 与 CUDA \`uchar1\`/\`char1\` 等向量类型 typedef 冲突。

**\*\*实际可用方式\*\***：真实工程（\`resnet50\_fp16\_infer\`、\`yolo\_fp16\_infer\`、\`bert\_fp16\_run\`、\`Qcnet\`、\`llm\_infer\`）的 \`link.txt\` 显示，它们用普通 \`aarch64-linux-gnu-g++\` 编译宿主代码，把 \`libcublas.so\` 作为**\*\*引擎内部 BLAS\*\*** 链接进来，经 \`inference\_api\` / \`runtime\_api\` 头文件调用，不从 CUDA 源码 include \`cublas\_v2.h\`。

**\*\*结论\*\***：cuBLAS 在 NX9031 上是引擎内部实现细节，不是面向用户的 API。SDK 自带的 \`cublas\_sample\` 属于未经验证即交付的样例。

**\*\*竞品含义\*\***：NIO 的 CUDA 兼容层实现底座是 AMD 的 HIP/ROCm 移植设施（rocBLAS API + HIP 头文件树）。这条路线经过验证，但其上限与缺陷都继承自上游。

**#### 4.2.3 HAUK —— 算子编译（TIR / LLVM）**

基于 TVM TIR 的 NPU 后端。\`python/allspark/hauk\` 分 \`common\` / \`driver\` / \`target\` / \`tir\` / \`script\` 五部分。\`HaukTarget\` 经 \`AIWrapper\` FFI 暴露硬件参数查询：\`get\_total\_vgpr\_num\` / \`get\_total\_sgpr\_num\` / \`get\_total\_thread\_num\` / \`get\_scope\_size\` / \`get\_fractal\_size\` / \`get\_warp\_num\` / \`get\_fermat\_block\_num\` / \`get\_cayley\_block\_num\` / \`get\_fermat\_ie\_num\`。

kernel 类型三选一：\`fermat\`（向量）/ \`cayley\`（矩阵）/ \`mix\`。调度策略常量含 \`load\_balance\` 与 \`warp\_first\`。codegen 走 LLVM（\`ALLSPARK\_LLVM\_INSTALL\_PATH\`）。

**\*\*对标\*\***：等价于 AMD 侧的 Triton / TileLang。差别在 Triton 有活跃社区与跨厂商内核复用，HAUK 只服务 N93 系列。

**#### 4.2.4 ACE —— 手写算子库**

\`python/allspark/ace\` 分 \`aopi\` / \`functions\` / \`kernels\` / \`goldens\`，kernels 再分 \`cayley\` / \`fermat\` / \`gop\`。\`goldens\` 目录表明算子带参考实现用于精度回归。

**\*\*对标\*\***：等价于 cuDNN/cuBLAS 或 AMD 的 AITER。注意 \`AMDdoc\` 记录 vLLM 在 gfx1151 上「AITER 在候选列表中但未被选中，最终用 TRITON」——AMD 自有算子库在消费级 GPU 上并未真正投入使用。NX9031 侧 ACE 是主路径。

**#### 4.2.5 AGE —— 图编译**

建立在 ATVM（Apache TVM 的 NIO fork，保留 \`relay\` / \`tir\` / \`autotvm\` / \`auto\_scheduler\` / \`meta\_schedule\` / \`micro\` / \`rpc\` 全部子模块）之上。AGE 自身分 \`compiler\` / \`relay\` / \`op\` / \`task\_gen\` / \`quantization\_tools\` / \`hardware\_info\` / \`dfx\` / \`runtime\` / \`contrib\`。

\`hardware\_info\` 暴露的唯一配置入口是：

\`\`\`python

set\_hardware\_info(cluster\_num, cluster\_l2\_size, global\_l2\_size)

\`\`\`

即图编译对硬件的抽象只有三个参数：簇数与两级 L2 容量——与 §2.2 的 4 簇 + 4 GL2 分片结构一致，也与 ClusterGroupMode 的分区语义呼应。

\`compiler\` 目录下有 \`full\_affinity\_params.json\` 与 \`lazy\_mode\_params.json\` 两套参数，即存在「全亲和」与「惰性」两种编译模式。

**#### 4.2.6 Graph API —— TensorRT 风格 C++ 构图**

项目内已有实证：\`allspark-x86/bert-code\` 下同时存在 TensorRT 版 \`bert\_builder.cpp\` 与 AllSpark Graph API 版 \`main.cc\`，并附逐层对比文档 \`ALLSPARK\_VS\_TRT\_COMPARISON.md\`。两版层次结构、权重命名与 FP16 精度设置完全一致，差异仅在 QKV 融合与 attention mask 的实现选择。这直接证明 TensorRT 工程可按层平移。

样例覆盖：\`resnet50\_int8\`、\`transformer\_test\`、\`dynamic\_shape\`、\`dynamic\_attrs\`、\`if\`（控制流）、\`accuracy\_tool\`（静态/动态精度比对）。

**#### 4.2.7 ONNX 前端**

\`parser\_api.h\` / \`custom\_parser.h\` / \`weight\_creator.h\` / \`onnx-ml.pb.h\`，实现在 \`liballspark\_parsers.so\`。配套脚本：\`onnx\_name\_normalizer\`（节点名规范化）、\`onnx\_model\_output\_comparer\`（与编译产物逐输出比对）。

样例：\`resnet50\_api\_compile/runtime\`、\`resnet50\_quant\_compile/runtime\`、\`resnet50\_fp16\_plugin\_compile/runtime\`、\`resblock\_compile/runtime\`、\`bert\_large\_quant\_compile/runtime\`、\`dyn\_case\`、\`internal\_plugin\`、\`plugins\`。

**\*\*对标\*\***：AMD 侧对应 MIGraphX，其报告结论是「定位为传统 CV 与 ONNX 部署后端，LLM 场景无需考虑」。NX9031 的 ONNX 路径定位相同——qcnet / yolo / resnet / bert 这类 CV 模型走这条线。

**#### 4.2.8 Alchemy —— LLM 专用前端**

PyTorch 风格声明式建模框架，把 HuggingFace 模型编译为 AllSpark 引擎：

\`\`\`python

from alchemy.llmapi import LLMArgs, LLM

from alchemy import BuildConfig, LLMProfile

llm\_args = LLMArgs(

    model="Qwen/Qwen2.5-0.5B",                  # HF 名或本地路径

    build\_config=BuildConfig.from\_dict({

        "build\_dir": "./out", "device\_version": "N93X",

        "cluster\_num": 2,                       # 1/2/4，与 ClusterGroupMode 对应

        "network\_datatype": "float16",

        "optimization\_level": 2,

        "enable\_op\_fusion": True, "enable\_mem\_reuse": True,

        "quantize\_proto\_path": ..., "quantize\_safetensors\_path": ...,

    }),

    llm\_profile=LLMProfile.from\_dict({

        "max\_batch\_size": 1,                    # ← 并发上限在此固化

        "max\_prompt\_len": 1024,

        "max\_kv\_cache\_len": 2048,

        "kv\_cache\_type": "contiguous",          # contiguous / paged / disabled

    }),

)

LLM(llm\_args).build()

\`\`\`

**\*\*算子覆盖\*\***：\`ops.py\` 定义 68 个模块类，\`functional.py\` 提供约 140 个函数。LLM 关键算子齐备——\`flash\_attention\`、\`batch\_flash\_attention\`、\`cross\_attention\`、\`sparse\_cross\_attention\`、\`moe\`、\`rope\` / \`rope\_with\_table\`、\`rms\_norm\`、\`embedding\`、\`quantize\_dynamic\` / \`dequantize\_dynamic\`。CV 侧亦完整——\`conv1d/2d/3d\`、\`conv\_transpose\`、各类 pool/norm/pad、\`grid\_sample\`、\`nms\`、\`sparse\_conv\_relu\_quant\`。

**\*\*已实现模型\*\***：

**\*\*路线分野\*\***：AMD 侧不存在「LLM 编译」这个步骤，llama.cpp 直接加载 GGUF；NX9031 侧每个模型都必须先有 Alchemy 实现、再经一次完整编译。

**#### 4.2.9 SIM —— 指令级仿真**

\`sim/\` 下含 \`libjarvis.so\`、\`libesl2jarvis.so\`、\`libsim\_license.so\`（带 license 校验），配置分 \`device.yaml\`（器件结构）、\`simulation.yaml\`（仿真运行时）、\`soc.toml\`（SoC 级建模）。

\`soc.toml\` 建模粒度含 18 个段：\`AIDummyDDR\`、\`NPUCluster\`、\`TaskDispatch\`、\`NNL2s\`、\`NNCayleyCores\`、\`NNFermatCores\`、\`FermatL2s\`、\`FermatCores\`、\`GlobalL2s\`、\`HNA\`、\`PCA\`、\`BarrierUnit\`、\`chi2axi\`、\`TS\`、\`NPUCrossBar\`、\`SYSDMA\`、\`TDAxiCrossBar\`、\`AxiHighCrossBar\`、\`DeviceRingUT\`、\`LPDDR\`、\`dummySif\`。\`[LPDDR]\` 段可配到写地址队列深度 32、写数据队列 128、读 CAM 深度 64、page-hit limit 8、scramble 开关等。

\`simulation.yaml\` 可开 \`insttrace\_en\`、\`dcache\_dump\_en\`、\`infdumpen\` 等 dump 开关，\`thread\_num\` 默认 16。

**\*\*生态含义\*\***：算子开发与回归测试不占台架，且可获得周期级性能估计。AMD 侧不存在等价能力——gfx1151 的性能问题只能在实机上试。这是 NX9031 生态的第二个实质性优势。

**#### 4.2.10 编译产物格式**

\`\`\`plain&#x20;text

model\_out/

  llm.ap                       # 包清单：架构名、版本、各 stage 路径、dtype

  encoder\_es/  prefill\_es/  decode\_es/

    \<stage>.ap                 # stage 清单

    \<stage>\_0.safetensors      # 权重

    safe\_tensor.json

    \_\_hyper\_0/

      \_\_aom\_0.aom … \_\_aom\_N.aom   # 每个 kernel 一个 AllSpark Object Module

\`\`\`

一个模型被拆成若干 engine stage（encoder / prefill / decode），每个 stage 内含数十至数百个 \`.aom\` 目标模块。BGE-M3 的产物形如 \`\_\_b1\_s512\_bge\_m3\_xlmr\_dense.ap\` —— 文件名直接编码 batch=1、seq=512，**\*\*形状固化在产物中\*\***。

设备侧运行时最小集（\`/mnt/sys/allspark\`）：\`aexec\` + \`libalchemy\_runtime.so\` + \`liballspark\_plugins.so\` + \`libcublas.so\`，合计约 334 MB。\`aexec\` 是通用执行器，选项含 \`--loadHyperEngine=\<file>\` 与 \`--iterations=\<N>\`。

**#### 4.2.11 九条编译路径汇总**

**\*\*编译宿主要求\*\***：Alchemy 与 AGE 编译在 x86 容器内完成；ARM 二进制构建需 \`source allspark\_env.sh arm\` + \`-ccbin aarch64-linux-gnu-g++\`。runtime 提供三个 ABI：\`lib\`（x86）、\`lib\_arm\`、\`lib\_qnx\`——QNX 支持是车规场景必需项，AMD 侧无对应。

**\*\*\***





**### 4.3 量化生态**

工具链名称 **\*\*AQUA\*\***（目录 \`aqua\`，主入口 \`aquant.py\`），交付形态为独立 Docker（镜像 \`aquant\:v0.1\`，17.4 GB）。

**\*\*关键事实\*\***：\`Dockerfile\` 基于 \`pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime\`——**\*\*量化过程本身跑在 NVIDIA GPU 上，不在 NX9031 上\*\***。即工具链存在一条对 NVIDIA 的硬依赖。

**\*\*依赖\*\***：\`torchao 0.12.0+git\`（随包提供 \`torchao-0.12.0+git-cp39-abi3-linux\_x86\_64.whl\`）、\`transformers 4.53.0\`、\`datasets 3.6.0\`、\`onnx 1.16.0\`、\`onnxruntime 1.22.0\`、\`ultralytics 8.3.198\`（CV 侧）。

**\*\*算法\*\***：

**\*\*量化粒度约定\*\***（\`configs/rtn/llm.yml\`）：

\`\`\`yaml

quant:

  algorithm: MinMax

  weight: { dst\_dtype: int8, symmetric: True, granularity: per\_channel }

  act:    { dst\_dtype: int8, symmetric: True, granularity: per\_token, is\_static: False }

\`\`\`

权重 per-channel 对称、激活 per-token 动态——与设备上部署的 W4A8 / W8A8 命名一致。\`quantize.py\` 的 \`get\_quant\_axis\` 支持 \`per\_tensor\` / \`per\_channel\` / \`per\_token\` 三种 grain，并针对 \`BatchMatMul\` 的 \`transpose\_a\`/\`transpose\_b\` 自动翻转量化轴。

**\*\*覆盖范围\*\***：\`configs\` 下除 \`llm\` 外另有 \`language\`、\`vision\`、\`vision\_onnx\`、\`siamese\_network\`、\`llm-int4\`，即同一工具覆盖 LLM、VLM、CV 与度量学习模型。

**\*\*输出物\*\***：\`allspark\_quantize\_param\` proto（\`.pb\`）+ 量化后 safetensors，二者作为 \`BuildConfig\` 的 \`quantize\_proto\_path\` / \`quantize\_safetensors\_path\` 喂给 Alchemy。

**#### 正在开发的 GGUF 路径**

v1.5.2 的 \`alchemy\_compile/docs/gguf\_quantization\_design.md\` 给出完整设计：

\* 第一版仅支持 GGUF **\*\*Q8\\\_0\*\***（每 32 权重一 block：fp16 scale + int8\\[32]）

\* 首批目标模型：\`ggml-org/bge-m3-Q8\_0-GGUF\`、\`pqnet/bge-reranker-v2-m3-Q8\_0-GGUF\`

\* 后续规划 \`Q4\_0\` / \`Q4\_K\` / \`Q5\_K\` / \`Q6\_K\` / \`IQ\*\`

\* 明确**\*\*不走\*\*** Transformers 的 GGUF loader（因其会反量化回 PyTorch tensor），要保留 GGUF 量化语义

\* 明确禁止对声明为 GGUF 量化的 Linear 静默 fallback 到 fp16

\* 与现有 proto + safetensors 路径并存，同一次 build 内不可混用

同目录另有 \`gguf\_ace\_renewal\_request.md\`，即已向 ACE 算子层提出配套需求。

**\*\*评估\*\***：这是把 NX9031 接入 llama.cpp 量化生态的关键一步。若 Q4\\\_K 系列打通，模型获取成本将从「NIO 自行量化」降为「直接下载社区 GGUF」，对 day-0 能力的影响大于任何性能优化。当前状态为设计文档 + 首批 embedding 模型，尚未覆盖 LLM 主力量化格式。

**\*\*\***





**### 4.4 推理服务框架**

只有一条路径：ollama 的私有分支。

**\*\*镜像\*\***：\`adkv200\_ubuntu24\_alchemy\_ollama\:v0.1.2\`，1.43 GB，docker-compose 部署。容器需直通 \`/dev/nnp0\` 与 \`/dev/nxmap\`，挂载 \`/usr/local/lib\` → \`/opt/libs\`、\`/etc/config/art\`（只读）、模型目录 → \`/opt/models\`。

**\*\*内部构成\*\***：

**\*\*部署脚本&#x20;\*\***\`run.sh\`**\*\*&#x20;的六个步骤\*\***（本次完整还原）：

1\. \`echo 1 > /sys/devices/platform/280f2000.nnp/fw\_reboot\`

2\. 四个 devmem 寄存器写（见结论 2）

3\. 解压模型 tgz

4\. \`docker load\` 镜像

5\. \`docker compose up -d\`

6\. \`curl /api/tags\` 验证

**\*\*模型形态\*\***：\`alchemy/qwen3.5-35b-a3b\:w4a8\_0702\`，manifest 中 \`format="alchemy"\`、\`family="qwen3.5"\`、\`parameter\_size="35B\_A3B"\`、\`quantization\_level="W4A8"\`，blob 总计 23 GB / 108 个 blob。即 ollama 的模型分发机制（manifest + blob + digest）被完整复用，只是 blob 内容从 GGUF 换成 AllSpark \`.ap\`/\`.aom\`。

v1.5.2 的 \`alchemy/models/ollama\_import.py\` 把这条打包路径工程化，其 \`OllamaFeatureOverrides\` 字段揭示规划中的能力矩阵：\`enable\_vision\`、\`enable\_tools\`、\`enable\_thinking\`、\`enable\_embedding\`、\`enable\_rerank\`。

**\*\*运行时配置\*\***（\`alchemy-runner\` 加载日志）：

\`\`\`plain&#x20;text

max\_seq\_len=65536      max\_cached\_kv\_len=64512    max\_prefix\_kv\_len=1024

max\_chunk\_prefill\_len=1024                        kv\_dtype=float16

max\_batch\_size=1       prefill\_num\_for\_chunk=1    decode\_num\_for\_chunk=0

input\_tensors=85       output\_tensors=81          clusterGroupMode=0

engine\_max\_session\_num=1                          has\_vision=true

\`\`\`

**\*\*并发限制的性质\*\***：\`OLLAMA\_NUM\_PARALLEL=1\` 是服务端配置，\`max\_batch\_size=1\` 与 \`engine\_max\_session\_num=1\` 是编译进引擎的。放开并发必须重新编译模型，而非重启服务。§5.6 的并发实测是这一约束的直接结果，不应解读为硬件并发能力。

**\*\*对标 AMD\*\***：AMD 侧有 llama.cpp（ROCm/Vulkan 双后端）、vLLM（容器与 SDK 两条获取路径）、Lemonade Server（多后端编排）、Ollama 四条路径，其报告还能横向比较后端优劣。NX9031 侧无可比较对象。

**#### 模型架构副产品**

从运行时日志可完整还原所部署模型的结构，这是 GGUF 生态下不易获得的信息：

\* **\*\*混合注意力\*\***：40 层中仅第 3、7、11、…、39 层（每 4 层 1 层，共 10 层）持有 KV cache（形状 \`[seq, 2, 256]\`，即 2 个 KV head × 128 head\\\_dim 的 GQA）；其余 30 层持有 \`conv\_state [1, 8192, 4]\` 与 \`recurrent\_state [1, 32, 128, 128]\`，即门控线性注意力。\`hidden\_size=2048\`。这解释了 64512 的超长 KV 窗口为何内存代价可控。

\* **\*\*视觉塔\*\***：独立的 \`vit\_embed\_stage\`，\`vision.depth=27\`、\`hidden\_size=1152\`、\`intermediate\_size=4304\`、\`num\_heads=16\`、\`in\_chans=3\`、\`out\_hidden\_size=2048\`、\`patch\_size=16\`、\`spatial\_merge\_size=2\`；\`vit\_pixel\_values\` 形状 \`[-1, 1536]\`（16×16×3×2）。

\* **\*\*多模态 token\*\***：\`vision\_start\_token\_id=248053\`、\`vision\_end\_token\_id=248054\`、\`image\_token\_id=248056\`、\`video\_token\_id=248057\`。

\* **\*\*多模态约束\*\***：\`max\_pixels=262144\`（512×512）、\`max\_num\_of\_images\_for\_one\_req=1\`、\`max\_num\_of\_videos\_for\_one\_req=1\`、\`max\_frames\_per\_video=1\`。

即**\*\*当前部署的就是一个 VLM，图像与视频输入在引擎层已启用\*\***，但每请求限一张图 / 一帧视频。

**\*\*\***





**### 4.5 多模态**

**\*\*编译前端的模型支持\*\***：\`qwen2\_5\_vl\`（VLM）、\`qwen3\_asr\`（ASR）、\`bge\_m3\` 与 \`bge\_reranker\`（检索侧）。

**\*\*设备上的实际产物\*\***（104 AD 变体）：

**\*\*ASR 实测\*\***：模型为 \`Qwen3ASRForConditionalGeneration\` 0.0.1，config 声明支持 **\*\*30 种语言\*\***（中、英、粤、日、韩、阿、德、法、西、葡、俄、泰、越、土、印地、马来、荷、瑞典、丹麦、芬兰、波兰、捷克、菲律宾、波斯、希腊、罗马尼亚、匈牙利、马其顿等）。\`llm.ap\` 绑定三个 stage：\`encoder\_es\` / \`prefill\_es\` / \`decode\_es\`。

执行结果——mel 前端与 encoder 正常运行：

随后在 tokenizer 初始化处崩溃：

\`\`\`plain&#x20;text

thread '\<unnamed>' panicked at src/lib.rs:27:50:

called \`Result::unwrap()\` on an \`Err\` value: Error("EOF while parsing a value", line: 1, column: 0)

\`\`\`

原因是 \`model\_out\` 目录内只有 \`tokenizer\_config.json\` 与 \`vocab.json\`，**\*\*缺少 tokenizers 库所需的&#x20;\*\***\`tokenizer.json\`。这是部署包完整性问题而非能力问题，但说明该样例未经端到端验证即上台架。

上表耗时可视为「音频前端 + encoder」阶段的上界（含模型加载）。15.05 s 音频编码侧 1.71 s，约 8.8 倍实时；解码侧未测。作为参照，AMD 平台上 Qwen3-ASR-1.7B 端到端 RTF 0.081（12.4 倍实时）——但模型规模与口径均不同，不构成直接比较。

**\*\*语音合成\*\***：本次调研未在 Alchemy 模型列表、设备文件系统或 ollama 模型库中发现任何 TTS / vocoder 路径。AMD 侧至少有两条（Kokoro ONNX 与 Qwen-Omni Code2Wav），虽然全模态路线慢于实时。**\*\*这是 NX9031 相对 AMD 的明确缺口\*\***。

**\*\*多模态编排\*\***：不存在 Lemonade OmniRouter 一类的编排层。各模态是独立二进制 / 独立服务，需上层自行拼装。但 LLM 侧的 VL 能力是**\*\*单模型内置\*\***而非编排——\`AMDdoc\` 结论 1 指出「AMD 提供的多模态能力本质上属于系统编排层方案，而非单一统一模型」，NX9031 在图文这一路上恰恰是统一模型，路线相反。

**\*\*\***





**### 4.6 Agent 框架支持**

与 AMD 报告结论一致：**\*\*该层不构成约束\*\***。

OpenAI 兼容性实测：

\`\`\`plain&#x20;text

GET /v1/models

→ {"object":"list","data":[{"id":"alchemy/qwen3.5-35b-a3b\:w4a8\_0702",

                            "object":"model","owned\_by":"alchemy"}]}

\`\`\`

tool-calling 实测（\`get\_weather\` 函数定义，问 "What is the weather in Shanghai?"）：

\`\`\`json

{"choices":[{"message":{"role":"assistant","content":"",

   "reasoning":"The user is asking about the weather in Shanghai. I have a tool called

                \\"get\_weather\\" … I need to use this tool with the city parameter set to \\"Shanghai\\".",

   "tool\_calls":[{"id":"call\_yzvh7928","type":"function",

                  "function":{"name":"get\_weather","arguments":"{\\"city\\":\\"Shanghai\\"}"}}]},

   "finish\_reason":"tool\_calls"}],

 "usage":{"prompt\_tokens":273,"completion\_tokens":74,"total\_tokens":347}}

\`\`\`

\`finish\_reason\` 正确返回 \`tool\_calls\`，参数 JSON 结构正确，且额外暴露 \`reasoning\` 字段。Agent 框架接入的技术前提已具备。

**\*\*实际约束不在协议层而在并发层\*\***：\`max\_batch\_size=1\` 意味着多步 Agent 的每次工具往返都串行排队，多 Agent 并行场景线性劣化。

**\*\*\***





**### 4.7 新模型支持速度与 day-0 能力**

两个平台的模型跟进路径有本质区别：

\`\`\`plain&#x20;text

AMD:      新模型发布 → llama.cpp 社区实现（3 天～2 周）→ 社区上传 GGUF → 平台可用

NX9031:   新模型发布 → Alchemy 内实现该模型（NIO）→ AQUA 量化（NIO，需 NVIDIA GPU）

                     → 编译（NIO）→ 打包为 ollama 模型（NIO）→ 平台可用

\`\`\`

差异在第二步：AMD 侧由社区承担，NX9031 侧由 NIO 独家承担，没有外部力量分担。

**\*\*现场证据\*\***：Alchemy 1.4.2 支持 \`llama\` / \`qwen2\` / \`qwen2.5-VL\` / \`qwen3\` / \`qwen3-asr\`，1.5.2 增加 \`bge-m3\` / \`bge-reranker\`。即近期新增的是两个检索模型，LLM 侧仍以 Qwen 系列为主。\`README\` 的「支持的模型」一节只列 Qwen2 并注明「更多模型支持正在开发中」。

**\*\*迭代节奏\*\***：部署模型 tag \`w4a8\_0702\`（7 月 2 日）、ollama 镜像内二进制 7 月 3 日、\`libs\_vit\_pdd\` 7 月 20 日、ADK bundle 7 月 31 日——整条链路是**\*\*月级\*\***。

**\*\*三条改善路径的优先级\*\***：

**\*\*结论\*\***：NX9031 当前不具备 day-0 能力，短期内也不会具备。合理的产品规划假设是：模型集合在项目启动时冻结，中途更换按「月」计工期。这与 AMD 侧「拉最新 llama.cpp 重编译」的天级成本相差两个数量级。

需要注意的是，\`AMDdoc\` 也指出 AMD 的 day-0 承诺同样有水分：官方宣布 day-0 与 gfx1151 上出现可用构建产物之间存在 1～2 周时间差，且面向 vLLM 的官方优化插件 vLLM-ATOM 明确只覆盖 Instinct 系列。即 AMD 是「社区天级 + 官方无 SLA」，NX9031 是「厂商月级 + 无社区」。

**\*\*\***





**## 五、实测数据**





**### 5.1 测试方法与自检机制**

所有 NPU 侧数据由自行编写的 CUDA kernel 经 CCA 工具链交叉编译后在 NPU 上实测：

\`\`\`bash

npcc++ --device-target=N93X --platform=ASIC -O2 \<bench>.cu \\

       \--art-path="$ALLSPARK\_RUNTIME\_PATH" -ccbin aarch64-linux-gnu-g++ -o \<bench>

\`\`\`

三个 benchmark：

**\*\*自检机制拦截了一次伪数据\*\***：初版 \`npu\_membw\` 未检查 kernel launch 返回值，报告 291 208 GB/s（291 TB/s，物理不可能）。加入 \`cudaGetLastError()\` 与写回校验后暴露真因——\`rtErrorInvalidResourceHandle\`(33)，即默认流不被支持，kernel 从未执行。**\*\*任何未做正确性自检的加速器 benchmark 数字都不应采信。\*\***

CPU 侧带宽用 C + NEON 自编程序，SoC 上原生 \`gcc -O3 -march=armv8.2-a\` 编译。

**### 5.2 内存带宽**





**#### 5.2.1 CPU 侧（A78AE + NEON）**

每线程独立缓冲 192 MB（远超各级 cache），首触在设置 CPU 亲和性之后完成，\`pthread\_barrier\` 对齐起跑，每组 30 轮，每配置 3 次取最优。

**\*\*绑核策略是本项测试的关键\*\***。A78AE 按 DSU 分簇，每簇 4 核共享一个到互联的端口。首轮按 \`0,1,2,3,…\` 顺序绑核得到非单调的错误曲线（1 线程 17.98 → 2 线程 23.35 → 4 线程 23.63 → 8 线程 47.68 GB/s），因为前 4 个线程全落在同一 DSU 簇、撞上簇端口上限。改为跨簇散布（\`0,4,8,12,16,20 → 1,5,9,…\`）后曲线恢复单调：

单 DSU 簇内（仅绑核 0-3）对照：read 在 2 线程即饱和于约 23.6 GB/s，确认簇端口是第一道瓶颈。

A65AE 侧对照（绑核 24-33）：单线程 read 17.83、copy 26.56 GB/s，与 A78AE 单线程几乎相同——单线程带宽由访存路径而非核心微架构决定。

三条结论：

1\. 纯读聚合上限 65.5 GB/s，1→3 线程完全线性（每线程恒 17.9 GB/s），6 线程后进入平台期

2\. 纯写单线程即达 42.3 GB/s、上限 44.4 GB/s，几乎不随线程数变化——符合写合并 / 写缓冲特征

3\. 读写混合 86.9 GB/s **\*\*高于\*\***纯读的 65.5 GB/s，说明纯读并未打满 DRAM，是被核侧读并发度限制的





**#### 5.2.2 NPU 侧（CCA kernel）**

缓冲 1 GiB，8 轮，grid/block 各扫 18 组合取峰值。

**\*\*缓冲尺寸扫描\*\***（Mode 4 下）确认无 cache 效应——64 MiB / 256 MiB / 1 GiB / 2 GiB / 4 GiB 分别为 156.47 / 161.20 / 161.73 / 161.61 / 161.43 GB/s，从 64 MiB 起即平坦。

**\*\*扫描范围的影响\*\***：前五行为基准扫描（grid ≤408、block ≤256）。将扫描范围加宽到 grid ≤2176、block ≤1024 后，调优配置下读达 355.83 GB/s（grid=1088 block=256）、写达 410.59 GB/s（grid=408 block=512）、混合达 389.57 GB/s（grid=544 block=64）。即最优 launch 配置随访问模式变化，单一配置会低估 2\\\~3%。

**\*\*相对理论峰值\*\***（546.1 GB/s，推导见 §2.3）：读 65.2%、写 75.2%、混合 71.3%。作为对照，AMD 395 侧实测 239.2 GB/s / 理论 256 GB/s = 93.4%。**\*\*NX9031 的访存效率显著低于竞品，约 190 GB/s 理论带宽未被利用。\*\***

**\*\*主机↔设备拷贝\*\***（\`cudaMemcpy\`，256 MiB × 5）：H2D 12.2～12.4 GB/s，D2H 4.6～4.9 GB/s。统一内存架构下这一路径仍走运行时拷贝而非零拷贝，且方向不对称（D2H 仅为 H2D 的 39%）。

**#### 5.2.3 NPU vs CPU 侧对比**

NPU 经独立的 \`npu-smmu\` 与 NPU CrossBar 访存，带宽是 CPU 簇的 4～9 倍。**\*\*用 CPU 侧数字评估该平台的 LLM 能力会严重低估\*\***——这是本次调研最容易犯的方法学错误。

**### 5.3 算力（Fermat 向量核）**

寄存器驻留 FMA，16 个独立累加器，grid 扫至 2176 × 512 = 1 114 112 线程。

**\*\*缩放精确到小数点后三位\*\***，确认硬件级隔离。

**\*\*FP32 扫描曲线\*\***（Mode 4）：4352 线程 0.872 → 17 408 线程 1.024 → 69 632 线程 1.268 → 1 114 112 线程 1.276 TFLOPS，约 70 K 线程后完全平坦。

**\*\*lane 数推算\*\***：最佳 grid 恒为 2176 = 34 × 64，推断 Fermat 核为 **\*\*64 lane FP32\*\***。理论峰值 32 Fermat（NN 簇内）× 64 lane × 2 FLOP × 1.4 GHz = **\*\*5.73 TFLOPS\*\***，实测 5.121 = **\*\*89% 效率\*\***。FP16 经 \`half2\` 打包精确 2×（10.374 / 5.121 = 2.026）。

**\*\*跨机一致性\*\***：两台 SoC 的算力完全一致（5.121 / 10.374 与 5.121 / 10.379），说明这是确定性的硬件规格而非板级差异。

**\*\*重要限制\*\***：本项只驱动 Fermat SIMT 核。**\*\*16 个 Cayley 矩阵核完全未被计入\*\***——它们需经 \`mmb\_\*\` scope 与 fractal 路径，不暴露给 CUDA C。因此 5.12 TFLOPS **\*\*不是\*\*** NX9031 的 AI 算力，只是其向量算力。

**### 5.4 矩阵核算力的间接推算**

由 LLM prefill 实测反推 Cayley 路径的有效算力下界：

\`\`\`plain&#x20;text

Qwen3.5-35B-A3B 激活参数约 3B，prefill 实测 1769.8 token/s

有效算力 ≈ 2 × 3×10⁹ × 1769.8 = 1.06×10¹³ FLOP/s ≈ 10.6 TFLOPS

\`\`\`

这是**\*\*下界\*\***（未计入 attention、norm、MoE 路由等非 GEMM 开销，也未计入 W4A8 下的实际算术精度），且已超过 Fermat 向量核峰值的 2 倍——确认 prefill 主要跑在 Cayley 矩阵阵列上。

**### 5.5 LLM 推理性能**

测试对象：\`alchemy/qwen3.5-35b-a3b\:w4a8\_0702\`，运行于 **\*\*104 AD 变体\*\***，脱离 docker 直接以镜像内的 \`ollama\` + \`alchemy-runner\` 裸跑，Mode 0 + devmem 调优。

引擎参数：W4A8 权重、KV cache float16、\`max\_seq\_len 65536\`、\`max\_cached\_kv\_len 64512\`、\`max\_chunk\_prefill\_len 1024\`、\`max\_batch\_size 1\`、\`engine\_max\_session\_num 1\`、\`clusterGroupMode 0\`、\`has\_vision true\`。

三次测量内部一致性极好（decode 标准差 < 0.3%）。

**#### 两台设备的差异及排查**

同一模型、同一 mode、同一寄存器配置下，两台 SoC 存在稳定的 29% 差距：

**\*\*已逐项排除的因素\*\***：

**\*\*部分解释\*\***：79 的 NPU 读带宽低 12%（312.44 vs 355.83），但 decode 差 29%，无法完全解释。且 79 侧测量时其模型常驻内存，存在带宽争用的混杂因素。

**\*\*未能隔离的剩余因素\*\***：docker 容器化开销、板级/硅片差异。本次不再进一步排查（79 为生产环境，已停止对其操作）。

**\*\*报告采用 104 的数据\*\***，因其为专用测试机、裸跑、无其他负载。

**#### 与 AMD 的端到端对照**

AMD 数据引自 \`compre\`，本次未复测。

补充一个更同口径的 prefill 对照——\`AMDdoc\` A.7 中 AMD 走 vLLM 路径（Qwen3-Omni-30B-A3B，BF16，2011 token 提示）测得 prefill 1587.5 token/s，NX9031 在 2250 token 下为 1769.8 token/s，**\*\*快 11.5%\*\***，且 NX9031 侧是 4bit 量化模型。

**\*\*口径限制\*\***：两侧量化格式不同（UD-Q4\\\_K\\\_XL vs W4A8）、推理栈不同（llama.cpp Vulkan vs Alchemy）、prompt 长度不同、prefill 计时接口不同。这组数据回答的是「两个实际部署方案谁更快」，不能作为芯片级跑分。

**\*\*性能画像的一致解释\*\***：NX9031 实测带宽高 49%（理论高 113%）而 decode 反低 20%，说明其 decode 侧存在实现效率损失（\`max\_batch\_size=1\`、无连续批处理、混合注意力的 recurrent 状态更新可能未充分优化），而非硬件受限。prefill 侧 16 个 Cayley 矩阵核发挥了作用，大幅领先。

**### 5.6 并发扩展性**

（本项数据采集于 79，因其为当时唯一已部署 LLM 的设备；104 上的部署为本次调研后期完成，未重复该测试。）

聚合吞吐在 1～8 路之间恒定于 34～35 token/s，**\*\*完全不随并发数扩展\*\***。N=2 一行的 16.90 为异常点（该次墙钟 15.15 s 与 N=4 的 14.58 s 相当而输出仅其一半），未复现，不参与结论。

**\*\*成因是配置而非硬件\*\***：\`OLLAMA\_NUM\_PARALLEL=1\` + 引擎 \`max\_batch\_size=1\` + \`engine\_max\_session\_num=1\`。请求在队列中串行执行。这组数据只能证明「当前部署形态不支持并发」，不能证明 NPU 不具备批处理能力。

与 AMD 对照（\`AMDdoc\` A.6，Qwen3.5-35B-A3B Q4\\\_K\\\_XL、Vulkan、\`-np 8 -c 65536\`）：

AMD 侧 8 路聚合为单路的 1.25 倍（其报告归因于内存带宽受限与 MoE 专家复用率低），NX9031 侧为 1.03 倍（归因于 batch=1）。两者受限原因不同。AMD 侧另有 vLLM 路径可达 64 路并发聚合 373 token/s，NX9031 无对应物。

**### 5.7 语音识别（ASR）**

Qwen3-ASR-0.6B，\`Qwen3ASRForConditionalGeneration\` 0.0.1，支持 30 种语言，三段引擎（\`encoder\_es\` / \`prefill\_es\` / \`decode\_es\`），3.2 GB。

耗时含进程启动与模型加载，覆盖「读 wav → mel → encoder → 构造 prompt」全过程，随后崩溃：

\`\`\`plain&#x20;text

thread '\<unnamed>' panicked at src/lib.rs:27:50:

called \`Result::unwrap()\` on an \`Err\` value: Error("EOF while parsing a value", line: 1, column: 0)

\`\`\`

原因是 \`model\_out\` 目录缺少 \`tokenizer.json\`（仅有 \`tokenizer\_config.json\` 与 \`vocab.json\`）。因此**\*\*无法计算端到端 RTF\*\***；仅可给出编码侧上界：15.05 s 音频 1.71 s，约 8.8 倍实时。

AMD 侧参照：Qwen3-ASR-1.7B 端到端 RTF 0.081（12.4 倍实时），64 路并发聚合 234 倍实时，仅占 3.87 GiB 显存，且 \`/v1/audio/transcriptions\` 与 OpenAI Whisper API 兼容。**\*\*这一项 AMD 明显领先——不是硬件差距，是交付完整度差距。\*\***

**### 5.8 视觉（VLM）**

引擎日志显示所部署的 Qwen3.5-35B-A3B **\*\*内置完整视觉塔\*\***：

\`\`\`plain&#x20;text

has\_vit\_embedding\_stage=true    vision\_embed\_ctx\_num=2

vision.depth=27                 vision.hidden\_size=1152

vision.intermediate\_size=4304   vision.num\_heads=16

vision.patch\_size=16            vision.spatial\_merge\_size=2

vision.out\_hidden\_size=2048     vision.in\_chans=3

image\_token\_id=248056           video\_token\_id=248057

max\_pixels=262144               max\_frames\_per\_video=1

\`\`\`

即图像与视频输入在引擎层均已启用，\`vit\_embed\_stage\` 是独立的 engine stage。

**\*\*实测结果：失败。\*\*** 构造 224×224 与 448×448 的 PNG 经 \`/api/generate\` 的 \`images\` 字段提交，两台设备（104 裸跑、79 官方 docker 部署）返回**\*\*完全相同\*\***的错误：

\`\`\`plain&#x20;text

{"error":"alchemy completion: CreateEmbedTask failed"}

\`\`\`

引擎日志给出精确根因：

\`\`\`plain&#x20;text

[W] task\_combiner.cc:236  qwen3 pos\_embed preferred VIT safetensors directory not found

                          under /tmp/alchemy-model-XXXXXXXX, falling back to recursive model scan

[E] task\_combiner.cc:1431 CreateEmbedTask failed

\`\`\`

**\*\*模型包中缺少 ViT 位置编码的 safetensors 目录\*\***。引擎编译进了完整视觉栈，但权重未随包分发，回退到递归扫描 23 GB 模型目录后仍找不到，耗时约 60 秒才报错。

沿途还暴露了图像预处理链的依赖结构（裸跑时逐个补齐才走到这一步）：

\`\`\`plain&#x20;text

image\_processing.py → numpy → image\_utils.py → PIL

\`\`\`

即 **\*\*VLM 的图像预处理是在 runner 内跑 Python 完成的\*\***，依赖 \`/opt/allspark/python\`（\`image\_processing.py\` / \`image\_utils.py\` / \`video\_processing.py\` / \`video\_utils.py\`）加上容器 site-packages 中的 numpy 与 Pillow。这也是 1.43 GB 容器镜像的组成之一。

**\*\*与 ASR 的&#x20;\*\***\`tokenizer.json\`**\*\*&#x20;缺失是同一类问题\*\***：能力已编译进产物，但交付包不完整，端到端从未被验证过。

**### 5.9 Agent 接口**

\`/v1/models\` 与 \`/v1/chat/completions\` 均正常。tool-calling 返回 \`finish\_reason=tool\_calls\`，参数 JSON 结构正确，额外暴露 \`reasoning\` 字段。详见 §4.6。

**\*\*\***





**## 六、竞品对比总表**





**### 6.1 硬件能力**





**### 6.2 LLM 推理（Qwen3.5-35B-A3B，同模型家族）**





**### 6.3 生态**





**### 6.4 综合判断**

**\*\*NX9031 的硬件不是短板。\*\*** 内存理论带宽是 AMD 的 2.13 倍、实测超出 49%，prefill 领先 85%，并且具备 AMD 完全没有的三项能力：NPU 硬分区、指令级仿真器、CUDA 源码级兼容。decode 落后 20% 是软件实现效率问题（batch=1、无连续批处理），不是带宽或算力受限。

**\*\*NX9031 的短板全部在交付与生态\*\***：

1\. 模型可得性差两个数量级（月级 vs 天级）

2\. 推理框架只有一条私有分支，无并发服务能力

3\. 多模态三项（图像 / 语音识别 / 语音合成）在当前交付状态下**\*\*全部不可用\*\***——前两项是打包缺失，第三项是完全没有

4\. 关键性能配置未文档化（\`ClusterGroupMode\` 与四个 devmem 寄存器合计影响 2.15 倍带宽 / 4 倍算力），且默认配置只给 1/4 芯片

**\*\*对产品规划的含义\*\***：如果业务是「固定模型集合 + 车内确定性推理 + 多 AD 模块共享 NPU」，NX9031 的硬件与分区能力是明确优势；如果业务需要频繁跟进新模型、多用户并发服务或成熟多模态，当前生态无法支撑，且差距不是靠硬件迭代能补的。

**\*\*\***





**## 七、建议**

按投入产出排序：

**### P0 —— 立即可做，收益最大**

\* **\*\*补齐并强制端到端冒烟测试。\*\*** 本次发现两个交付包缺文件导致能力完全不可用：ASR 缺 \`tokenizer.json\`、VLM 缺 ViT pos\\\_embed safetensors。BGE-M3 侧已有 \`smoke\_board.py\` / \`board\_parity.py\` 的做法，应推广到所有交付模型。这两项修复后，多模态能力从「零」变为「有」，成本极低。

\* **\*\*把&#x20;\*\***\`ClusterGroupMode\`**\*\*&#x20;与 devmem 调优写入文档与默认配置。\*\*** 当前状态下，不读部署脚本源码的人拿到的是 1/4 算力 + 45% 带宽（161.5 / 355.8 GB/s）。至少应：

\- 在 SDK 文档中说明 \`cluster\_group\_config.json\` 的四档语义

\- 说明四个寄存器的作用，或将其固化进 NPU 固件初始化

\- 在 \`alchemy-runner\` 启动日志中打印当前有效带宽档位

\* **\*\*对外性能数据统一声明配置档位。\*\*** 同一颗芯片在不同配置下 NPU 读带宽相差 2.20 倍（161.5 / 204.1 / 355.8 GB/s），算力相差 4 倍。任何 benchmark 若不声明 mode 与寄存器状态则无法复现。

**### P1 —— 中期，决定竞争力**

\* **\*\*追查 190 GB/s 的访存效率缺口。\*\*** 内存为 LPDDR5X-8533 × 512-bit，理论峰值 546.1 GB/s，实测纯读仅 355.8 GB/s（65.2%）。竞品 AMD 395 的利用率为 93.4%，若 NX9031 达到同等水平可得 508 GB/s。由于 LLM decode 是带宽严格受限，这一项的潜在收益（+43%）远超其他任何优化方向，且不需要新硬件。可能的着手点：

\- 地址交织策略（\`addr\_config.json\` 的四个窗口，含 bypass-L2 窗口）

\- Global L2 分片数与交织粒度（当前 4 分片、粒度 8）

\- 访存请求的 outstanding 深度与 QoS 优先级映射（\`soc.toml\` 中 \`cfgRdCamDepth\`、\`cfgRdqosMapDramPri\` 等参数在真机上的对应寄存器）

\- \`mmb\_\*\` 路径与 Fermat 通用访存路径的带宽差异——本次只测了后者

\* **\*\*GGUF 权重直读扩展到 Q4\\\_K 系列。\*\*** 这是唯一能把模型获取从「月级自研」拉到「天级复用」的杠杆，收益高于任何性能优化项。当前设计只覆盖 Q8\\\_0 与两个 embedding 模型。

\* **\*\*验证&#x20;\*\***\`max\_batch\_size > 1\`**\*\*&#x20;的编译产物。\*\*** 当前「并发不扩展」完全由编译期配置造成，硬件是否具备批处理收益**\*\*从未被测量过\*\***。这是本次调研留下的最大空白。建议编译 \`max\_batch\_size\` = 4 / 8 的同款模型重复 §5.6。考虑到实测带宽比 AMD 高 49%、理论带宽高 113%，若批处理打通，并发聚合吞吐有望反超。

\* **\*\*修复 CCA 的 cuBLAS 头文件冲突。\*\*** 当前 SDK 自带的 \`cublas\_sample\` 在 v1.4.2 与 v1.5.2 下均编译失败，根因是 \`hip\_common.h\` 的平台自动检测与 \`npcc++\` 冲突。这是「CUDA 兼容」宣传与实际能力之间最显眼的缺口，修复成本应该很低（调整头文件守卫顺序）。

\* **\*\*支持默认流（null stream）。\*\*** 现有 CUDA 代码迁移时，每个 kernel launch 都要手工加 stream 参数，是最高频的移植摩擦点。

**### P2 —— 长期**

\* **\*\*暴露 NPU 侧性能计数器。\*\*** \`/dev/nnp\_prof\` 与 \`/proc/npu/\` 的 \`r52\_ddr\_bw\_\*\` 命令已存在但未编入 release 固件。本次不得不自行编写 CCA benchmark 才拿到带宽数据，外部评估者不具备这个条件。

\* **\*\*统一现场 runtime 版本。\*\*** 同一台设备上存在 4 个日期跨度 3 个月的 \`libalchemy\_runtime.so\`，缺乏版本管理机制。

\* **\*\*对外宣传口径。\*\*** 建议聚焦 prefill 吞吐、内存带宽、NPU 硬分区、实时性与 CUDA 迁移成本——这五项是本次实测中真实优于或独有于竞品的。避免宣传 decode 吞吐与多模态完整度。

**\*\*\***





**## 附录 A：复现步骤**





**### A.1 访问链路**

\`\`\`bash

\# 台架控制机

sshpass -p isaac.ch ssh -p 1000 isaac.cheng\@10.116.90.104

sudo su jenkins

ed2            # → sshpass -p root ssh root\@192.168.1.20

\`\`\`

\`\~jenkins/.bashrc\` 中的别名：\`ed1\`/\`s1\`/\`r1\` → \`192.168.1.10\`，\`ed2\`/\`s2\`/\`r2\` → \`192.168.1.20\`（root）；\`n1\`/\`n2\` 为 \`nio\` 用户；\`sd\*\`/\`nd\*\`（端口 222）、\`so\*\`/\`no\*\`（端口 2222）为其他通道；\`qr2\`/\`qn2\` → \`192.168.1.236\`（QNX）。

**### A.2 NPU 配置为全簇模式（持久化）**

\`\`\`bash

\# 必须先进 Debug 模式，否则 rootfs 修改不跨重启保留

setdebug info                       # Current/Target

setdebug on                         # 写 /mnt/data/tmp/.preserve

reboot

\# 重启后

cat > /etc/config/art/cluster\_group\_config.json <<'EOF'

{"ClusterGroupCfg":{"ClusterGroupMode":0}}

EOF

cat > /etc/config/art/internal/addr\_config.json <<'EOF'

{"HardwareAddr":{"kBinaryBaseAddr":"0x40000000","kGlobalL2BaseAddr":"0x840000000",

                 "kClusterL2BaseAddr":"0x1040000000","kByPassL2BaseAddr":"0x1840000000"}}

EOF

reboot

\`\`\`





**### A.3 查询内存颗粒规格**

\`\`\`bash

cat /sys/kernel/debug/ddr/vendor     # MICRON

cat /sys/kernel/debug/ddr/freq       # 8533Mbps

cat /sys/kernel/debug/ddr/density    # 64GB

cat /sys/kernel/debug/ddr/hwlpi      # off

hexdump -C /proc/device-tree/ddr-size          # 0x1000000000 = 64 GiB

cat /proc/device-tree/ddr\_dvfs/status          # disabled

\`\`\`

需 root。这些节点权限为 \`-r--------\`，普通用户不可读。

**### A.4 NPU 带宽调优寄存器（非持久，\`fw\_reboot\` 后需重做）**

\`\`\`bash

echo 1 > /sys/devices/platform/280f2000.nnp/fw\_reboot

devmem 0x280e0070 32 0xA4F8      # 启动默认 0x99E0

devmem 0x280e006c 32 0xC9E0      # 启动默认 0xA4F8

devmem 0x28540008 32 0x1e2f1     # 启动默认 0x13299

devmem 0x28550008 32 0x1e2f1     # 启动默认 0x13299

\`\`\`





**### A.5 编译 CCA benchmark**

\`\`\`bash

docker exec allsparkarm bash -c '

  source /usr/local/allspark/allspark\_env.sh arm

  cd /usr/local/allspark/npu\_membw

  npcc++ --device-target=N93X --platform=ASIC -O2 npu\_membw\.cu \\

         \--art-path="$ALLSPARK\_RUNTIME\_PATH" \\

         -ccbin aarch64-linux-gnu-g++ -o npu\_membw'

\`\`\`

要点：

\* 必须 \`-ccbin aarch64-linux-gnu-g++\`（Ubuntu 24.04 交叉工具链），**\*\*不能\*\***用 SDK 自带的 Poky 4.0.4（glibc 过旧，\`libruntime.so\` 需 \`GLIBC\_2.38\` / \`GLIBCXX\_3.4.32\`）

\* 所有 kernel launch 必须显式指定 stream

\* 设备侧运行需 \`LD\_LIBRARY\_PATH\` 含提供 \`rtNpuMemGetInfo\` 的 \`libruntime.so\`





**### A.6 不用 docker 直接跑 LLM**

适用于未安装 docker 的 SoC（如 AD 变体）：

\`\`\`bash

\# 1. 从镜像 tar 中解出 rootfs

tar xzf adkv200\_ubuntu24\_alchemy\_ollama\_docker.tar.gz

tar xf adkv200\_ubuntu24\_alchemy\_ollama\_v0.1.2.tar -C layers/

for b in layers/blobs/sha256/\*; do tar xf "$b" -C rootfs/ 2>/dev/null; done

\# 2. 解出模型

tar xzf qwen3.5-35b-a3b\_w4a8\_0702.tgz -C model/

\# 3. 启动

R=$PWD/rootfs

export LD\_LIBRARY\_PATH=$R/opt/allspark/lib:/mnt/data1/libs\_vit\_pdd:/usr/local/lib

export PYTHONPATH=$R/opt/allspark/python:$R/usr/local/lib/python3.12/dist-packages

export OLLAMA\_HOST=127.0.0.1:11435

export OLLAMA\_MODELS=$PWD/model

$R/opt/allspark/bin/ollama serve

\`\`\`

\`PYTHONPATH\` 两段缺一不可——第一段是 \`image\_processing.py\` 等预处理模块，第二段提供其依赖的 numpy 与 Pillow。纯文本推理不需要 \`PYTHONPATH\`。

**\*\*\***





**## 附录 B：术语表**

**\*\*\***





**## 附录 C：本次调研未覆盖的项目**

**\*\*\***





**## 附录 D：数据可信度说明**

**\*\*\***

*\*本报告所有 NX9031 数据可经附录 A 的步骤复现。\**