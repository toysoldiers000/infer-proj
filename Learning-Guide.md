
# AI Compiler 学习与项目路线
## Core ML Foundation + NPU Performance + Low Precision + Backend AutoTuning

## 1. 项目目标

本路线面向以下岗位方向：

- AI Compiler Engineer
- NPU Compiler Engineer
- Tensor Compiler Engineer
- Compiler Backend Engineer
- Kernel / Performance Engineer
- LLM Inference / Low-Precision Compiler Engineer

整体目标不是完成若干独立模型部署 Demo，而是建立一套完整的 AI Compiler 能力链：

```text
Model
  ↓
Frontend / IR
  ↓
Graph Optimization
  ↓
Precision Selection
  ↓
Partition / Lowering
  ↓
Implementation Selection
  ↓
Operator AutoTuning
  ↓
Codegen
  ↓
Runtime
  ↓
Hardware
  ↓
Profiling
  ↓
Performance Feedback
```

最终形成三个可独立投递、又可以互相串联的工程：

1. **Project 1 — Transformer NPU Compiler Performance Analysis**
2. **Project 2 — Compiler-aware Quantization & Mixed Precision**
3. **Project 3 — TVM Relax NPU Backend & Operator AutoTuning**

在三个正式 Project 之前，先使用手头的 Apple Silicon Mac，通过 **Core ML** 建立第一套真实的模型部署、编译、Runtime 和 Benchmark 闭环。

---

# 2. 总体架构

整个学习路线围绕“同一个模型、多个编译后端、统一评价体系”展开。

```text
                         PyTorch Model
                              │
                    Same Dataset / Input
                              │
              ┌───────────────┼────────────────┐
              │               │                │
              ▼               ▼                ▼
          Core ML          AllSpark          TVM Relax
        Foundation          NPU Path       Custom Backend
              │               │                │
             MIL             ONNX            Relax IR
              │               │                │
        Graph Pass       Graph Optimize      Partition
              │               │                │
         ML Program         AP/AOM           Lowering
              │               │                │
        CPU/GPU/ANE          NPU           AutoTuning
              │               │                │
              └───────────────┼────────────────┘
                              ▼
                    Unified Benchmark
                              │
             Accuracy / Latency / Memory
                              │
                    Compiler Analysis DB
                              │
                              ▼
                  Optimization Feedback
```

三个正式 Project 分别解决三个核心问题：

| Project | 核心问题 |
|---|---|
| Project 1 | **编译结果为什么快或为什么慢？** |
| Project 2 | **模型应该使用什么精度配置？** |
| Project 3 | **算子到底应该怎样编译才最快？** |

三者最终形成闭环：

```text
Project 3
决定“怎么编译”
      ↓
Project 2
决定“用什么精度编译”
      ↓
Project 1
验证“在真实硬件上到底快不快”
      ↓
Profiler / Performance DB
      ↓
反馈给 AutoTuner 与 Precision Search
```

---

# 3. Foundation Track — Core ML Compiler & Deployment Lab

## 3.1 Core ML 在整个路线中的定位

Core ML **不是第四个简历项目**。

它承担的是：

> 第一套真实、可直接运行、能够反复实验的 Compiler Reference Backend。

目标是在进入私有 NPU Compiler 和 TVM Backend 之前，先完整理解：

```text
PyTorch Model
      ↓
Graph Capture
      ↓
Compiler Frontend
      ↓
Intermediate Representation
      ↓
Graph Pass
      ↓
Compiled Model
      ↓
Device Specialization
      ↓
Runtime
      ↓
CPU / GPU / ANE
      ↓
Benchmark
```

根据当前 Core ML Tools 官方文档，PyTorch 模型可以先通过 TorchScript tracing 或 `torch.export` 捕获计算图，再经 Unified Conversion API 转成 Core ML；Core ML Tools 7 以后 `mlprogram` 是默认目标格式。citeturn914276search3turn914276search0

因此 Core ML 可以作为理解 AI Compiler 整体结构的第一站。

---

# 4. Core ML 实操路线

## C0 — 建立统一模型 Baseline

第一阶段先选择一个简单模型，例如：

```text
ResNet50
```

不要一开始直接上复杂 Transformer。

首先建立：

```text
PyTorch
  ↓
Reference Output
  ↓
Accuracy / Output Check
  ↓
CPU Baseline
```

需要固定：

- model version
- input shape
- preprocessing
- dataset
- PyTorch reference output
- model hash
- benchmark protocol

目的不是优化，而是建立后续所有实验共同的正确性基线。

### 对应 Compiler 概念

- Reference implementation
- Correctness Gate
- Workload
- Shape
- DType
- Reproducibility

---

# 5. C1 — PyTorch → Core ML

完成：

```text
torch.nn.Module
      ↓
Graph Capture
      ↓
coremltools.convert
      ↓
ML Program
      ↓
.mlpackage
```

重点不是记 API，而是理解：

```text
Framework Graph
      ↓
Compiler Frontend
      ↓
Compiler Representation
```

官方推荐工作流把 PyTorch 转换拆为 graph capture 和 Core ML conversion 两步。citeturn914276search3

ML Program 则是当前 Core ML 的推荐模型格式，并使用 `.mlpackage` 保存。citeturn914276search0turn914276search9

### 需要掌握的概念

- Framework graph
- Frontend
- Importer
- Static shape
- Dynamic/Flexible shape
- DType
- Deployment target
- Model artifact

### 最终产物

```text
pytorch_output.json
coreml_output.json
model.mlpackage
conversion_manifest.json
```

---

# 6. C2 — MIL 与 Graph Pass

这一阶段开始真正从“模型部署”进入“Compiler”。

研究：

```text
PyTorch Graph
      ↓
MIL
      ↓
Graph Pass
      ↓
ML Program
```

观察：

- op 类型变化
- constant folding
- cast
- reshape
- transpose
- fusion
- dead code elimination
- precision transformation

形成：

```text
MIL Before
   ↓
Pass Pipeline
   ↓
MIL After
   ↓
Graph Diff
```

### 核心问题

不是：

> 转换成功了吗？

而是：

> Compiler 到底改变了什么？

### 对应 P1/P3

这一步以后可以直接映射：

```text
Core ML MIL       → TVM Relax
MIL Pass          → Relax Pass
MIL Graph Diff    → ONNX / Relax Graph Diff
```

因此这里开发的 Graph Diff 思维可以复用到后面的所有 Project。

---

# 7. C3 — CPU / GPU / ANE Device Placement

Core ML 支持通过 compute unit 配置限制或允许模型使用不同计算设备，例如：

- CPU only
- CPU + GPU
- CPU + Neural Engine
- All available compute units

官方文档明确提供这些 execution configuration，用于执行和调试不同计算单元路径。citeturn914276search4turn914276search8

因此建立实验：

```text
Same Model
   │
   ├── CPU_ONLY
   │
   ├── CPU_AND_GPU
   │
   ├── CPU_AND_NE
   │
   └── ALL
```

比较：

```text
correctness
latency
load time
runtime variance
device usage
```

进一步使用 `MLComputePlan` 观察 ML Program operation 的预期 device usage 和 estimated cost。citeturn914276search5turn914276search7

### 需要理解的概念

- Device placement
- Backend selection
- Heterogeneous execution
- Fallback
- Cost estimation
- Runtime scheduling

### 与 P1 的映射

```text
Core ML

MIL Op
 ↓
Compute Device
 ↓
Estimated Cost
 ↓
Runtime
```

未来迁移为：

```text
NPU Compiler

ONNX Node
 ↓
AOM
 ↓
Task
 ↓
Kernel
 ↓
nxPerf
```

---

# 8. C4 — Benchmark 与 Compiler/Runtime 时间拆分

统一 Benchmark 不只记录一次 inference latency。

需要尝试区分：

```text
Conversion
   ↓
Compilation
   ↓
Model Load
   ↓
First Run
   ↓
Warm Runtime
```

Runtime 正式统计至少统一成：

```text
warm-up
   ↓
multiple iterations
   ↓
P50
P95
P99
mean
std
```

同时记录：

```text
model
shape
dtype
compute units
OS
chip
coremltools version
model hash
```

这和原 roadmap 的 benchmark 原则保持一致：正确性先于性能，同时保存 P50/P95/P99、环境、编译时间和失败结果，而不能只保存最快一次。fileciteturn11file4

---

# 9. C5 — Precision / Compression

Core ML 阶段可以先建立第一个低精度实验框架。

第一层先做：

```text
FP32
 ↓
FP16
```

ML Program 支持在转换时设置 FP16 或 FP32 compute precision，并且当前默认通常为 FP16。citeturn914276search0

之后扩展：

```text
Weight Compression
Palettization
Quantization
Pruning
```

Core ML Tools 官方 optimization 工具本身就提供模型压缩、量化、palettization 等路线，并建议同时观察 accuracy、model size 和 latency，而不是只比较模型大小。citeturn807489search0turn807489search1

### 这一阶段关注

```text
Precision
    ↓
Accuracy
    ↓
Model Size
    ↓
Latency
    ↓
Device Placement
```

### 与 Project 2 的关系

这里先建立：

> Precision × Accuracy × Performance

的分析框架。

后面 P2 再深入：

```text
Layer Sensitivity
Selective Fallback
QDQ
Mixed Precision
NPU Kernel Selection
```

---

# 10. C6 — Core ML External Configuration Search

在 Core ML 上可以提前实现一个 Search Framework。

但这里必须明确：

> 这是 Compiler Configuration Search，不是 ANE Kernel Schedule AutoTuning。

可以搜索：

```text
compute precision
compute units
shape profile
compression config
pass pipeline
```

形成：

```text
Configuration
      ↓
Compile
      ↓
Benchmark
      ↓
Score
      ↓
Rank
      ↓
Best Configuration
```

借此提前完成：

- experiment runner
- search framework
- cache
- result database
- replay
- ranking

真正的：

```text
tile_m
tile_n
tile_k
vector width
unroll
memory layout
kernel implementation
```

则留到 Project 3。

---

# 11. Core ML Foundation 最终产物

Core ML 阶段最终应形成：

```text
Core ML Conversion Pipeline
        +
MIL Graph Inspector
        +
Graph Diff
        +
CPU/GPU/ANE Benchmark
        +
ComputePlan Analyzer
        +
Precision Benchmark
        +
Experiment Runner
        +
Result Cache
```

完成之后，应该能够解释：

> 一个 PyTorch 模型如何进入 Compiler、怎样经过 IR 和 Graph Pass、最终如何变成可执行模型，以及不同设备、精度和 shape 为什么会产生不同性能。

---

# 12. Project 1 — Transformer NPU Compiler Performance Analysis

## 核心问题

> Transformer 编译以后为什么快，为什么慢？

原路线 P1 本身定义为从 ONNX、AP/AOM、SIM/ASIC 到 nxPerf，再通过 FP16/W8A8 等消融形成性能因果解释。fileciteturn11file0

## 系统链路

```text
Transformer
    ↓
ONNX
    ↓
NPU Compiler
    ↓
Graph Optimization
    ↓
Fusion
    ↓
Partition
    ↓
AP / AOM
    ↓
Runtime
    ↓
NPU
    ↓
Profiler
    ↓
AOM / Task / Kernel
    ↓
Original Graph
    ↓
Performance Attribution
```

---

## P1.1 Compiler Graph Observability

建立：

```text
Original ONNX
     ↓
Imported Graph
     ↓
Optimization
     ↓
Fusion Graph
     ↓
Backend Graph
```

分析：

- added / removed node
- added / removed edge
- fusion
- constants
- shape rewrite
- backend/custom op
- unsupported op

原 roadmap 已将编译阶段 ONNX Graph Diff 作为 P1/P3 的共享基础证据。fileciteturn12file0

---

## P1.2 Artifact Contract

研究：

```text
Graph
 ↓
Compiler
 ↓
AP / AOM
 ↓
Runtime
```

保存：

- compiler hash
- model hash
- target
- config
- artifact hash
- AOM count
- memory
- subgraph relationship

目标是理解：

> Compiler 与 Runtime 之间的 Artifact Contract。

---

## P1.3 Correctness

严格建立：

```text
Compile Success
      ≠
Runtime Success
      ≠
Correct Output
      ≠
Good Performance
```

执行逻辑：

```text
Compile
 ↓
SIM Correctness
 ↓
ARM Runtime
 ↓
ASIC Correctness
 ↓
Performance
```

原路线明确区分 SIM correctness 与 ASIC performance，并要求真正的 correctness failure 能传播成非零退出状态。fileciteturn11file1

---

## P1.4 Performance Attribution

最终建立：

```text
Total Latency
      ↓
AOM
      ↓
Task
      ↓
Kernel
      ↓
Cluster / Core
      ↓
Operator
      ↓
ONNX Node
```

回答例如：

- fusion 为什么没有带来加速？
- W8A8 为什么没有明显优于 FP16？
- 是否实际选择 INT8 Kernel？
- Q/DQ 是否消除？
- subgraph boundary 是否增加？
- memory traffic 是否成为瓶颈？

原 P1 的项目假设本身就强调：节点减少不保证 latency 降低，而 W8A8 只有在 Q/DQ 被消除/融合并实际选择 INT8 Kernel 后才有对应的硬件加速基础。fileciteturn11file0

---

## P1 最终输出

```text
Graph Diff
Artifact Inspector
Correctness Gate
Profiler Analyzer
Graph ↔ Kernel Mapping
Performance Attribution Report
```

### 对应岗位能力

- ONNX
- Graph Compiler
- NPU Compiler
- Runtime
- C/C++
- ARM
- Fusion
- Profiling
- Performance Optimization

---

# 13. Project 2 — Compiler-aware Quantization & Mixed Precision

## 核心问题

> 模型应该怎样选择低精度配置，在尽量保留精度的同时获得真实硬件收益？

原路线 P2 已覆盖 calibration、量化格式、逐层 sensitivity、DyT/SmoothQuant 和 Accuracy–Latency Pareto。fileciteturn11file3

## 系统链路

```text
FP Model
   ↓
Calibration
   ↓
INT8 / INT4
   ↓
QDQ Graph
   ↓
NPU Compiler
   ↓
Runtime
   ↓
Accuracy + Performance
   ↓
Layer Sensitivity
   ↓
Selective Fallback
   ↓
Recompile
   ↓
Mixed Precision Search
   ↓
Pareto Frontier
```

---

## P2.1 Quantization Baseline

对比：

```text
FP16
W8A8
W4A8
```

同时记录：

```text
accuracy
latency
model size
memory
QDQ count
fusion
fallback
kernel
```

---

## P2.2 Layer Sensitivity

建立：

```text
Layer
 ↓
Activation Error
 ↓
Weight Error
 ↓
SQNR / Cosine
 ↓
Saturation / Outlier
```

找到真正敏感的节点。

---

## P2.3 Selective Fallback

执行：

```text
INT8 baseline
      ↓
Sensitive Layers
      ↓
FP16 fallback
      ↓
Recompile
```

逐步测试：

```text
Top-1
Top-3
Top-5
Top-K
```

形成：

```text
Accuracy
 ▲
 │        ●
 │     ●
 │   ●
 │ ●
 └──────────────→ Latency
```

即：

> Accuracy–Latency Pareto Frontier

---

## P2.4 Compiler-aware Analysis

不能只说：

```text
FP16 fallback → accuracy improved
```

还必须继续分析：

```text
FP16 fallback
      ↓
Q/DQ boundary
      ↓
Fusion
      ↓
Subgraph
      ↓
Kernel Selection
      ↓
Latency
```

真正回答：

> 量化决策是怎样改变 Compiler Graph 和真实 Hardware Execution 的？

---

## P2.5 Algorithm Engineering

之后再把：

- DyT
- SmoothQuant

等算法接入。

目标不是单纯复现论文，而是：

```text
Paper Idea
   ↓
Model Transformation
   ↓
ONNX Graph
   ↓
Compiler Pattern
   ↓
Fusion
   ↓
NPU Runtime
   ↓
Hardware Benefit
```

---

## P2 最终输出

```text
Quantization Evaluator
Quant Config Matrix
Sensitivity Analyzer
Mixed Precision Search
QDQ / Fusion Analyzer
Accuracy-Latency Pareto
```

### 对应岗位能力

- PTQ
- INT8 / INT4
- Mixed Precision
- Quantization Compiler
- QDQ
- Transformer / LLM
- Operator Fusion
- Hardware-aware Precision Selection

---

# 14. Project 3 — TVM Relax NPU Backend & Operator AutoTuning

## 核心问题

> 给定一个算子、shape 和目标硬件，Compiler 应该怎样生成最快实现？

原始 roadmap 中 P3 已要求打通现代 Relax 的 pattern、partition、JSON codegen、runtime，并记录 partition coverage 和 fallback；最终还将编译流程封装为带 cache/replay 的自动编译服务。

在此基础上新增：

> Operator Implementation Selection + Schedule AutoTuning

作为 P3 的核心扩展。

---

# 15. P3.1 Relax Backend

首先实现：

```text
Model
 ↓
Relax IR
 ↓
Pattern Matching
 ↓
Support Check
 ↓
Partition
 ↓
Composite Subgraph
 ↓
Backend Codegen
 ↓
Runtime
```

记录：

```text
candidate ops
offloaded ops
partition coverage
fallback ratio
composite count
subgraph count
```

原路线 E3.1 本身就是通过 example NPU 打通 pattern → partition → codegen → runtime 四阶段并建立 coverage/fallback instrumentation。fileciteturn13file3

---

# 16. P3.2 Lowering

进一步进入：

```text
Relax Operator
      ↓
Backend Operator
      ↓
Kernel Primitive
```

例如：

```text
relax.matmul
      ↓
backend.matmul
      ↓
NPU GEMM
```

这里开始真正承担 Backend Ownership。

---

# 17. P3.3 Workload Extraction

将算子转换成标准 workload：

```text
op_type
shape
dtype
layout
target
```

例如：

```text
MatMul
M = 197
N = 1024
K = 1024
FP16
NPU-X
```

形成：

```text
workload_key
```

后续 AutoTuning、Cache 和性能数据库全部围绕 workload key 工作。

---

# 18. P3.4 Implementation Selection

同一个 MatMul 可能存在：

```text
GEMM A
GEMM B
small-M GEMM
large-K GEMM
FP16 Kernel
INT8 Kernel
```

Compiler 需要完成：

```text
Workload
      ↓
Candidate Implementations
      ↓
Implementation Selection
```

这是 AutoSearch 的第一层。

---

# 19. P3.5 Schedule Search

选定 implementation 后继续搜索：

```text
tile_m
tile_n
tile_k

vector_width
unroll
pipeline_stage
double_buffer
core_num
cluster
memory placement
prefetch
```

形成：

```text
Implementation
      ↓
Schedule Space
      ↓
AutoSearch
      ↓
Best Schedule
```

---

# 20. P3.6 Hardware Constraint Pruning

不能直接暴力枚举全部参数。

必须先经过：

```text
Raw Search Space
      ↓
Local Memory Constraint
      ↓
Alignment Constraint
      ↓
Parallelism Constraint
      ↓
Hardware Resource Constraint
      ↓
Legal Search Space
```

体现：

> Hardware-aware Compiler Optimization。

---

# 21. P3.7 AutoSearch Algorithm

按照三个版本升级。

### V1

```text
Grid Search
Random Search
```

### V2

```text
Evolutionary Search
```

### V3

```text
Schedule Features
      ↓
Cost Model
      ↓
Predicted Latency
      ↓
Top-K
      ↓
Hardware Measurement
      ↓
Model Update
```

即：

> Cost-model Guided AutoTuning

注意：原 roadmap 并未证明现有私有 NPU 流程已经使用 AutoScheduler/MetaSchedule。因此这一 AutoTuning 模块属于后续新增实现，不能包装成对已有厂商 Compiler 内部机制的复现。

---

# 22. P3.8 Tuning Database

搜索一次后不能每次重新搜索。

建立：

```text
Workload
    ↓
Workload Hash
    ↓
Tuning DB
   /      \
 HIT      MISS
  ↓         ↓
Schedule  AutoSearch
  ↓         ↓
Codegen ← Save Result
```

Cache key 包含：

```text
op
shape
dtype
layout
target
compiler version
kernel version
```

---

# 23. P3.9 Shape-aware AutoTuning

针对 Transformer / LLM 动态 Shape：

```text
seq 1–128
   ↓
Schedule A

seq 129–512
   ↓
Schedule B

seq 513–2048
   ↓
Schedule C
```

形成：

> Shape Bucket + Schedule Specialization

并进一步研究不同 shape 之间 schedule 能否迁移。

---

# 24. P3.10 Profile-Guided Optimization

把 P1 Profiler 与 P3 AutoTuner 接起来：

```text
Compiler
   ↓
Hardware
   ↓
Profiler
   ↓
Hot Kernel
   ↓
Workload Extraction
   ↓
AutoSearch
   ↓
New Schedule
   ↓
Recompile
   ↓
Profiler
```

最终形成：

> Profile-Guided Compiler Optimization

---

# 25. P3.11 Compiler Service

最后将 Compiler 工程化：

```text
request
 ↓
Frontend
 ↓
Partition
 ↓
Precision Policy
 ↓
AutoTuning
 ↓
Codegen
 ↓
Artifact
 ↓
Benchmark
 ↓
Report
```

支持：

```text
submit
status
replay
cache
compare
```

原 roadmap 已设计以 model/config/compiler/toolchain hash 构造缓存键，并统计 cold/warm compile latency、cache hit rate、artifact hash 和 benchmark report。fileciteturn13file0

---

# 26. P3 最终输出

```text
Relax Backend
Pattern / Partition
Lowering
Workload Extractor
Implementation Selector
Schedule Search Engine
Cost Model
Tuning Database
Compiler Cache
Runtime
Compiler Service
```

### 对应岗位能力

- TVM
- Relax
- IR
- Pass
- BYOC
- Backend
- Lowering
- Codegen
- Kernel
- AutoTuning
- Cost Model
- Compiler Infrastructure

---

# 27. 三个 Project 的最终关系

```text
                         Model
                           │
                           ▼
                      Compiler IR
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
         Project 2                  Project 3
     Precision Decision          Backend Compiler
              │                         │
      INT8 / FP16 / INT4           Partition
      Mixed Precision              Lowering
      QDQ Optimization             AutoTuning
              │                    Best Schedule
              └────────────┬────────────┘
                           ▼
                        Codegen
                           │
                           ▼
                        Hardware
                           │
                           ▼
                       Project 1
                  Runtime / Profiling
                           │
                           ▼
                  Bottleneck Analysis
                           │
               ┌───────────┴───────────┐
               ▼                       ▼
       Precision Feedback      Schedule Feedback
               │                       │
               ▼                       ▼
           Project 2               Project 3
```

---

# 28. 共享 Compiler Toolkit

不要为三个项目重复实现基础设施。

建立统一：

```text
compiler_toolkit/

graph_diff/
artifact_inspector/
evaluator/
benchmark/
profiler/
workload/
search/
manifest/
cache/
report/
```

推荐工程结构：

```text
ai_compiler_lab/

├── models/
├── datasets/
│
├── compiler_toolkit/
│   ├── graph_diff/
│   ├── artifact_inspector/
│   ├── evaluator/
│   ├── benchmark/
│   ├── profiler/
│   ├── workload/
│   ├── search/
│   ├── manifest/
│   ├── cache/
│   └── report/
│
├── backends/
│   ├── coreml/
│   ├── allspark/
│   └── tvm/
│
├── project1_npu_performance/
├── project2_quant_compiler/
└── project3_backend_autotuning/
```

统一使用：

```text
Same Model
Same Dataset
Same Input
Same Correctness Rule
Same Benchmark Protocol
Same Manifest
```

区别只在 Backend。

---

# 29. 统一实验数据结构

建议每次实验统一产生：

```text
manifest.json
command.txt
compile.log
graph_stats.json
accuracy.json
runtime.csv
profiler/
report.md
```

这延续原 roadmap 对实验可复现性的要求。fileciteturn11file4

统一数据字段：

```text
model_hash

backend
compiler_version
target

shape
dtype
precision

compile_time

artifact_hash
artifact_size

accuracy

P50
P95
P99
QPS

memory

fusion_count
partition_coverage
fallback_ratio

kernel
schedule

experiment_id
```

---

# 30. 推荐学习与开发顺序

不要按照：

```text
Project 1 全做完
        ↓
Project 2 全做完
        ↓
Project 3 全做完
```

而采用螺旋式推进。

## Phase 0 — Core ML Foundation

```text
ResNet
 ↓
PyTorch → Core ML
 ↓
MIL
 ↓
CPU/GPU/ANE
 ↓
Benchmark
 ↓
Precision
```

目标：

> 第一次理解完整 AI Compiler deployment path。

---

## Phase 1 — Core ML Transformer

换成 ViT / Transformer：

```text
Transformer
 ↓
MIL
 ↓
Graph
 ↓
Device Placement
 ↓
Benchmark
```

目标：

> 为 P1 建立 Reference Backend。

---

## Phase 2 — P3 Backend Baseline

优先开始：

```text
Relax
 ↓
Pattern
 ↓
Partition
 ↓
Codegen
 ↓
Runtime
```

目标：

> 尽早获得真正具有 Compiler Ownership 的项目。

---

## Phase 3 — P1 NPU Baseline

```text
Transformer
 ↓
ONNX
 ↓
NPU Compiler
 ↓
AP/AOM
 ↓
SIM
 ↓
Runtime
```

目标：

> 打通 NPU 正确性闭环。

---

## Phase 4 — P1 Profiler

```text
NPU
 ↓
Profiler
 ↓
Kernel
 ↓
Graph Mapping
```

目标：

> 建立硬件性能归因。

---

## Phase 5 — P2 Quantization

```text
FP16
 ↓
INT8
 ↓
Sensitivity
 ↓
Fallback
 ↓
Pareto
```

目标：

> 建立 hardware-aware precision optimization。

---

## Phase 6 — P3 AutoTuning V1

```text
Workload
 ↓
Search Space
 ↓
Constraint
 ↓
Random/Grid Search
 ↓
Hardware Benchmark
 ↓
Best Schedule
```

目标：

> 第一次完整实现 Operator AutoSearch。

---

## Phase 7 — 联合优化

```text
Profiler
 ↓
Hot Kernel
 ↓
AutoTuner
 ↓
Schedule

+

Sensitivity
 ↓
Mixed Precision
 ↓
Precision Policy
```

最终：

```text
Precision Policy
       +
Schedule Policy
       +
Hardware Feedback
       ↓
Feedback-driven Compiler
```

---

# 31. 最终项目定位

## Project 1

### Transformer NPU Compiler Performance Analysis

```text
ONNX
→ Graph Optimization
→ Fusion / Partition
→ Artifact
→ Runtime
→ NPU
→ Profiling
→ Graph-Kernel Mapping
→ Bottleneck Attribution
```

证明：

> 我能够理解编译结果最终在硬件上为什么快、为什么慢。

---

## Project 2

### Compiler-aware Quantization & Mixed Precision

```text
FP16
→ INT8 / INT4
→ Sensitivity
→ Selective Fallback
→ QDQ
→ Fusion
→ Mixed Precision Search
→ Pareto
```

证明：

> 我能够决定模型应该采用怎样的低精度策略，并解释真实硬件收益。

---

## Project 3

### TVM Relax NPU Backend & Operator AutoTuning

```text
Relax IR
→ Pattern
→ Partition
→ Lowering
→ Workload
→ Implementation Selection
→ Schedule Search
→ Cost Model
→ Codegen
→ Runtime
→ Tuning DB
```

证明：

> 我能够自己实现 Compiler Backend，并自动寻找适合目标硬件的算子实现。

---

# 32. 最终求职叙事

完整故事不是：

> 我跑过 Core ML、AllSpark、TVM 和量化。

而应该是：

> 我首先在 Apple Silicon / Core ML 上建立了从 PyTorch、Compiler IR、Graph Pass、Compiled Model 到 CPU/GPU/Neural Engine Runtime 的完整模型部署与性能测量体系。
>
> 在此基础上，我将同一套方法迁移到 NPU Compiler，进一步下钻到 Graph、Subgraph、Artifact、Task 和 Kernel，建立 Compiler Graph 到 Hardware Profiling 的性能归因链路。
>
> 随后围绕 INT8/INT4 和 Mixed Precision，研究低精度策略如何影响 QDQ、Fusion、Kernel Selection、Accuracy 与真实硬件性能。
>
> 最后基于 TVM Relax 自己实现 NPU Backend，并进一步完成 Workload Extraction、Implementation Selection、Schedule AutoSearch、Cost Model 与 Tuning Cache，使 Profiler 数据能够反馈到 Compiler 的下一轮优化。

最终形成：

```text
Graph Compiler
      +
Low Precision Compiler
      +
Backend Compiler
      +
Operator AutoTuning
      +
Runtime Profiling
      +
Compiler Infrastructure
```

也就是：

> **从模型图，到算子实现，再到真实硬件性能的完整 AI Compiler 技术栈。**

---

# 33. 路线中的角色边界

最后需要始终保持三个边界清楚。

### Core ML

负责：

> Reference Backend、Compiler 流程学习、真实设备实验和通用基础设施验证。

不声称：

> 实现了 Apple Neural Engine 内部 Kernel AutoTuning。

---

### AllSpark / NPU

负责：

> 私有 NPU Compiler、Artifact、Runtime、Profiling 和真实硬件性能闭环。

没有 ASIC 原始性能数据之前，不填写硬件加速结果。

---

### TVM

负责：

> 自己拥有 Pattern、Partition、Lowering、Codegen、Workload 与 AutoTuning 的实现。

这是整个路线里最主要的：

> **Compiler Ownership Project。**

---

## 一句话总结

整条学习路线可以最终压缩为：

> **先用 Core ML 看懂一个真实 AI Compiler 如何从模型走到硬件；再用 NPU Project 学会解释编译结果为什么快慢；用 Quantization Project 决定什么精度更优；最后通过 TVM Backend + AutoTuning 自己决定算子到底该怎样编译。**