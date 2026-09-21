"""Build stage 2. Run with uv run --locked python tools/build_swin_stage2_notebook.py."""
from pathlib import Path
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip()))


def code(source):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip()))


md('''
# Swin Transformer 阶段二：模型转换与误差定位

**一句话心智模型：把同一份 tensor 送进不同表示的同一个模型，在语义相同的边界留下输出，找到第一处超过约定误差的区间，再缩小问题。**

这是实践教程，状态为**已展开**。阶段一已经做了 Core ML 初次转换与设备配置实验；本阶段增加 ONNX、跨后端中间输出、局部复现和可验证的故障注入。依据 [Swin 路线 §三、十二](../../docs/swin-transformer-guideline.md)；学习索引见 [tutorial.md](tutorial.md)。

先从你已经保存的 `input_0: float32[1,3,224,224]` 开始，不重新 resize 图片。`N=1` 是 batch，`C=3` 是 RGB 通道，`H/W=224` 是空间。输出 `logits[1,1000]` 的最后一轴才是类别。

本机固定 coremltools 9.0 的实际流程是两条分支：

```text
阶段一 inputs.npz + 同一 checkpoint
                  ↓
             PyTorch FP32
             /           \\
  ONNX → ORT CPU       TorchScript → MIL → Core ML CPU
                                            ├─ FP32
                                            └─ FP16 → ALL（允许 CPU/GPU/ANE）
```

路线文档中的“PyTorch → ONNX → Core ML”用于表达逐阶段验证目标；这里按工具真实入口执行，**不把 ONNX 传给 ct.convert**。Core ML 的 ONNX 输入不在这份锁定版本的统一转换入口中。MIL 是 *Model Intermediate Language*，即转换器保存算子、tensor 类型和依赖关系的中间程序。

**本阶段约束：** batch=1、224×224、固定权重与输入、CPU FP32 为 reference；先不量化、不升级依赖。生产候选只输出 logits；调试候选额外输出五处特征。调试图会改变优化机会与内存生命周期，不用于性能结论。

**交付标准：** FP32 导出链可对齐；知道 FP16 从哪一处边界开始超过筛查阈值；故障注入确实被拦截并能回滚；错误 shape/dtype 在调用前被拒绝；保存 hash、输出、日志和 case。少量无标签图只证明 smoke parity，不证明 ImageNet accuracy，不证明 ANE 实际执行。
''')

md('''
## 0. 执行环境与继承阶段一

```bash
uv sync --locked
uv run jupyter lab notebooks/swin_transformer/02_stage2_conversion_correctness.ipynb
```

`--locked` 保持已有工具链版本；`uv run` 使用仓库虚拟环境。编辑器选这份 `.venv` 的 Python kernel。模型缓存和阶段一产物已存在时，整个实验不需要新数据下载。

下面默认选择最近一份有 manifest/input/reference 的阶段一记录，并打印路径。正式复现实验把 `SOURCE_RUN` 改成这个明确路径。**找不到阶段一产物就停止**，先完成阶段一，不能悄悄换成随机图片继续。参数 `MAX_SOURCE_IMAGES=None` 表示使用全部已冻结输入；大数据集可先改为 8 做转换排查，正式精度另做全量评测。
''')
code('''
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import copy, gc, hashlib, inspect, json, platform, warnings
import importlib.metadata as metadata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import onnx
import onnxruntime as ort
import coremltools as ct
from huggingface_hub import snapshot_download
from transformers import AutoModelForImageClassification
from transformers.models.swin import modeling_swin
from IPython.display import display

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'pyproject.toml').exists())
SOURCE_RUN = None  # 正式重跑填 ROOT / 'results/swin_stage1/某次运行目录'
MAX_SOURCE_IMAGES = None
CPU_THREADS = 4
torch.set_num_threads(CPU_THREADS)
torch.manual_seed(0)
RUN = ROOT / 'results/swin_stage2' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
ART = RUN / 'artifacts'
ART.mkdir(parents=True)
plt.rcParams.update({'figure.figsize': (10, 4), 'axes.grid': True, 'grid.alpha': .2})

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))

if SOURCE_RUN is None:
    candidates = sorted(p.parent for p in (ROOT / 'results/swin_stage1').glob('*/manifest.json')
                        if (p.parent / 'inputs.npz').exists() and (p.parent / 'cpu_reference.npy').exists())
    assert candidates, '请先执行阶段一并保存 inputs.npz、cpu_reference.npy、manifest.json'
    SOURCE_RUN = candidates[-1]
SOURCE_RUN = Path(SOURCE_RUN).resolve()
source = json.loads((SOURCE_RUN / 'manifest.json').read_text())
assert sha256(SOURCE_RUN / 'inputs.npz') == source['input_sha256'], '阶段一输入文件已变化'
assert source['shape'] == [1, 3, 224, 224] and source['input_dtype'] == 'float32'
assert source['model_id'] == 'microsoft/swin-tiny-patch4-window7-224'
with np.load(SOURCE_RUN / 'inputs.npz') as saved:
    keys = sorted(saved.files, key=lambda k: int(k.split('_')[-1]))
    keys = keys[:MAX_SOURCE_IMAGES]
    source_inputs = [np.ascontiguousarray(saved[k]) for k in keys]
assert source_inputs and all(x.shape == (1, 3, 224, 224) and x.dtype == np.float32
                             and np.isfinite(x).all() for x in source_inputs)
old_reference = np.load(SOURCE_RUN / 'cpu_reference.npy')[:len(keys)]
manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'source_run': str(SOURCE_RUN),
    'source_manifest_sha256': sha256(SOURCE_RUN / 'manifest.json'),
    'source_reference_sha256': sha256(SOURCE_RUN / 'cpu_reference.npy'),
    'source_input_sha256': source['input_sha256'], 'selected_input_keys': keys,
    'model_id': source['model_id'], 'revision': source['revision'],
    'platform': platform.platform(), 'machine': platform.machine(), 'cpu_threads': CPU_THREADS,
    'versions': {n: metadata.version(n) for n in ['torch', 'transformers', 'coremltools',
                 'onnx', 'onnxruntime', 'numpy']}, 'uv_lock_sha256': sha256(ROOT / 'uv.lock'),
    'input_contract': {'shape': [1, 3, 224, 224], 'dtype': 'float32'},
    'measurement': 'correctness only; no latency or accuracy claim', 'device_mapping': 'unverified',
    'dataset': source['dataset']}
write_json(RUN / 'manifest.json', manifest)
print('继承:', SOURCE_RUN, '\\n本次结果:', RUN)
print(manifest['versions'])
''')

md('''
## 1. 一个 reference，两个 wrapper，六个对齐边界

普通模型 `forward` 返回框架对象；导出 wrapper 只留下 tensor。生产 wrapper 输出 `logits`；调试 wrapper 返回 `hidden_states` 加 `logits`。

在本仓库的 transformers 4.51.3，`SwinEncoder.forward` 默认记录 **downsample 之后**的 stage 特征。最后一个 stage 没有 downsample。最终分类前还有 `SwinModel.layernorm → avgpool → classifier`，因此 `stage3_out` 不是最终归一化后的 tensor。

|名字|具体 shape|轴身份|源码位置|
|---|---|---|---|
|embedding|[1,3136,96]|batch / 56×56 token / feature|SwinEmbeddings 输出|
|stage0_down|[1,784,192]|batch / 28×28 token / feature|encoder.layers[0] 的 downsample 后|
|stage1_down|[1,196,384]|batch / 14×14 token / feature|encoder.layers[1] 的 downsample 后|
|stage2_down|[1,49,768]|batch / 7×7 token / feature|encoder.layers[2] 的 downsample 后|
|stage3_out|[1,49,768]|batch / 7×7 token / feature|encoder.layers[3] 输出、最终 LayerNorm 前|
|logits|[1,1000]|batch / class|classifier 输出|

token 与 feature 轴在这里都只是张量的身份；它们是否参与 reduction（归约、把多个元素合成一个）取决于运算。LayerNorm 沿 feature 归约，avgpool 沿 token 归约。不能把特征最大元素的位置当成分类 Top-1。

阅读顺序：`SwinForImageClassification.forward → SwinModel.forward → SwinEncoder.forward → SwinStage.forward`。下面保存本机源码路径和 hash；在 VS Code 对类名按“转到定义”即可核对，别凭不同版本的网页猜 tuple 含义。
''')
code('''
snapshot = Path(snapshot_download(source['model_id'], revision=source['revision'],
    allow_patterns=['config.json', 'preprocessor_config.json', 'model.safetensors']))
for name, expected in source['model_files'].items():
    assert sha256(snapshot / name) == expected, f'权重/配置变化: {name}'
model = AutoModelForImageClassification.from_pretrained(snapshot).eval().float().cpu()
source_file = Path(inspect.getfile(modeling_swin))
manifest['model_files'] = source['model_files']
manifest['swin_source'] = {'path': str(source_file), 'sha256': sha256(source_file)}

NAMES = ['embedding', 'stage0_down', 'stage1_down', 'stage2_down', 'stage3_out', 'logits']
class LogitsOnly(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, pixel_values):
        return self.net(pixel_values=pixel_values, return_dict=True).logits

class DebugOutputs(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, pixel_values):
        out = self.net(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
        return (*out.hidden_states, out.logits)

production, debug = LogitsOnly(model).eval(), DebugOutputs(model).eval()
# 额外输入是数值稳定性探针，不是分类验证集。normalized_zero 也不是黑色原图。
rng = np.random.default_rng(0)
xs = source_inputs + [np.zeros((1, 3, 224, 224), np.float32),
                     rng.normal(0, 1, (1, 3, 224, 224)).astype(np.float32)]
sample_ids = keys + ['probe/normalized_zero', 'probe/seeded_normal']
torch_xs = [torch.from_numpy(x) for x in xs]
np.savez(RUN / 'inputs.npz', **{f'x_{i}': x for i, x in enumerate(xs)})
manifest['input_sha256'] = sha256(RUN / 'inputs.npz')
manifest['sample_ids'] = sample_ids
with torch.inference_mode():
    ref = [dict(zip(NAMES, [v.numpy().copy() for v in debug(x)])) for x in torch_xs]
    prod_ref = [production(x).numpy().copy() for x in torch_xs]
assert [ref[0][n].shape for n in NAMES] == [(1,3136,96), (1,784,192), (1,196,384),
                                        (1,49,768), (1,49,768), (1,1000)]
assert all(np.array_equal(r['logits'], p) for r, p in zip(ref, prod_ref))
assert np.allclose(np.concatenate(prod_ref[:len(keys)]), old_reference, atol=1e-4, rtol=1e-4)
print('源码:', source_file)
display(pd.DataFrame([{'boundary': n, 'shape': ref[0][n].shape} for n in NAMES]))
''')

md('''
## 2. 比较器先接受审查

reference `[2,1]`，candidate `[2.1,0.9]`：逐元素绝对差 `[0.1,0.1]`，max/mean 都是 0.1。cosine 比方向，不控制每个元素的偏差。`[1.001,1.000] → [1.000,1.001]` 几乎同向，但 argmax 已变。

本教程逐元素检查“差值不超过固定误差 + 按 reference 大小放宽的误差”，再统计超过预算的比例。精确写法是 `abs(candidate-reference) <= atol + rtol*abs(reference)`。FP32 筛查设 `atol=1e-4, rtol=1e-4`；FP16 沿用阶段一 `atol=0.05, rtol=0.01`。这是**实验前设定的排查阈值**，不是行业统一精度标准，更不是允许准确率下降的证明。不能看到失败才悄悄放宽。

比较前拒绝 shape 不同、NaN/Inf；计算统计时转 float64，避免误差度量本身溢出。零向量 cosine 无定义，记录 NaN，不能靠加 epsilon 伪造一个“高相似度”。每个样本单独一行，保留 rare input，不用整个数据集平均掩盖失败。
''')
code('''
TOLERANCES = {'fp32': {'atol': 1e-4, 'rtol': 1e-4}, 'fp16': {'atol': .05, 'rtol': .01}}
manifest['screening_tolerances'] = TOLERANCES

def compare(a, b, precision='fp32'):
    shape_ok = a.shape == b.shape
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    if not shape_ok or not finite:
        return {'shape_ok': shape_ok, 'finite': finite, 'max_abs': np.nan,
                'mean_abs': np.nan, 'cosine': np.nan, 'exceed_fraction': np.nan, 'pass': False}
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    error = np.abs(b - a)
    tolerance = TOLERANCES[precision]
    over = error > tolerance['atol'] + tolerance['rtol'] * np.abs(a)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return {'shape_ok': True, 'finite': True, 'max_abs': float(error.max()),
            'mean_abs': float(error.mean()), 'cosine': float(np.dot(a, b) / denom) if denom else np.nan,
            'exceed_fraction': float(over.mean()), 'pass': bool(not over.any())}

def save_outputs(name, outputs):
    np.savez(RUN / f'{name}.npz', **{f's{i}__{n}': value
             for i, tensors in enumerate(outputs) for n, value in tensors.items()})

rows = []
def record(route, candidate, precision='fp32', baseline=None):
    baseline = ref if baseline is None else baseline
    assert len(candidate) == len(baseline)
    for i, (gold, actual) in enumerate(zip(baseline, candidate)):
        for name in gold:
            assert name in actual, f'缺失输出: {name}'
            rows.append({'route': route, 'sample': sample_ids[i], 'boundary': name,
                         'precision': precision, **compare(gold[name], actual[name], precision)})
    save_outputs(route, candidate)

toy_ref = np.array([[1.001, 1.000]], dtype=np.float32)
toy_swap = toy_ref[:, ::-1].copy()
assert compare(toy_ref, toy_ref)['pass']
assert not compare(toy_ref, toy_swap)['pass']
assert not compare(toy_ref, toy_ref.T)['pass']
assert not compare(toy_ref, np.full_like(toy_ref, np.nan))['pass']
assert not compare(toy_ref, np.full_like(toy_ref, np.inf))['pass']
display(pd.DataFrame([{'case': 'high cosine, flipped top1', **compare(toy_ref, toy_swap),
                      'top1_same': bool(toy_ref.argmax() == toy_swap.argmax())}]))
save_outputs('torch_reference', ref)
''')

md('''
## 3. PyTorch → ONNX：先检查图，再执行图

ONNX 全称 *Open Neural Network Exchange*，是模型交换表示，文件本身不是 runtime。它用 Protobuf 保存 `ModelProto`：`graph.node` 是算子表，`initializer` 是常量/权重，node 的 `input/output` 字符串连接 tensor；`value_info` 保存中间 tensor 的类型与 shape。`opset` 指算子定义的版本，不能只改文件中的版本号冒充转换。

这里显式选 `dynamo=False` 的 TorchScript 导出路径、opset 17、固定 shape，延续锁定的 torch 2.7.0 和 Core ML tracing 环境，减少同时变化的因素。新项目可以另开实验比较官方推荐的 `dynamo=True`，本轮不悄悄切换 exporter。`TracerWarning` 必须保存并判断：Swin 中 shape→Python bool 会固定 padding/window 分支，所以本产物只承诺 batch=1、224×224。检查其他图片只能验证同一 shape 的数据变化。

`onnx.checker` 检查图结构合法；`infer_shapes` 补静态 shape 信息；两者都不执行模型、不保证数值正确。`compiler lowering` 是把框架级运算改写成后端可表达的更基础运算，类似 C++ 经 LLVM IR 到目标指令的逐级展开。
''')
code('''
def export_onnx(net, sample, path, names, input_name='pixel_values'):
    with warnings.catch_warnings(record=True) as caught, torch.inference_mode():
        warnings.simplefilter('always')
        torch.onnx.export(net, (sample,), str(path), dynamo=False, opset_version=17,
                          input_names=[input_name], output_names=names)
    path.with_suffix('.warnings.txt').write_text('\\n'.join(
        f'{w.category.__name__}: {w.message}' for w in caught))
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    return graph

prod_path, debug_path = ART / 'swin_logits.onnx', ART / 'swin_debug.onnx'
prod_graph = export_onnx(production, torch_xs[0], prod_path, ['logits'])
debug_graph = export_onnx(debug, torch_xs[0], debug_path, NAMES)
inferred = onnx.shape_inference.infer_shapes(debug_graph, check_type=True, strict_mode=True)
onnx.save(inferred, ART / 'swin_debug_shapes.onnx')
op_counts = pd.DataFrame(Counter(n.op_type for n in debug_graph.graph.node).most_common(),
                         columns=['op', 'count'])
op_counts.to_csv(RUN / 'onnx_op_counts.csv', index=False)
manifest['onnx_export'] = {'dynamo': False, 'opset': 17, 'dynamic_axes': None}
print('opset:', [(v.domain, v.version) for v in debug_graph.opset_import])
print('node/initializer/value_info:', len(inferred.graph.node),
      len(inferred.graph.initializer), len(inferred.graph.value_info))
display(op_counts.head(16))
display(pd.DataFrame([{'name': o.name, 'shape': [d.dim_value or d.dim_param
      for d in o.type.tensor_type.shape.dim]} for o in debug_graph.graph.output]))
''')

md('''
### 检视接口：从文件追到执行器

`onnx.load(path).graph.node` 查看节点；`node.input/output` 追数据流；图太大时用 Netron 看 `swin_debug.onnx`。从仓库根目录运行 `uv run netron --help` 查看本机 CLI 参数，再运行 `uv run netron 实际路径/swin_debug.onnx`；最后一个参数就是要打开的模型文件。Netron 展示静态图，不能证明硬件执行设备。

ORT 是 ONNX Runtime。`InferenceSession` 加载图、选择 kernel 并执行。先用 `CPUExecutionProvider` 和 `ORT_DISABLE_ALL` 关掉可选图优化做语义对照，再在同一导出文件上启用 `ORT_ENABLE_ALL`。关闭优化并不意味着取消所有必要的内部准备，也不等于 PyTorch 实现。
''')
code('''
def ort_session(path, optimized=False):
    options = ort.SessionOptions()
    options.intra_op_num_threads = CPU_THREADS
    options.inter_op_num_threads = 1
    options.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_ENABLE_ALL if optimized
                                       else ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
    return ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])

def run_ort(session, arrays=xs):
    names = [o.name for o in session.get_outputs()]
    return [dict(zip(names, session.run(names, {'pixel_values': x}))) for x in arrays]

ort_debug = ort_session(debug_path)
ort_off = run_ort(ort_debug)
record('onnx_debug_off', ort_off)
ort_opt = ort_session(debug_path, optimized=True)
ort_on = run_ort(ort_opt)
record('onnx_debug_on', ort_on)
record('ort_optimization_only', ort_on, baseline=ort_off)
ort_prod = ort_session(prod_path, optimized=True)
onnx_production = run_ort(ort_prod)
record('onnx_production', onnx_production, baseline=[{'logits': p} for p in prod_ref])
record('onnx_debug_vs_production', [{'logits': p['logits']} for p in ort_on],
       baseline=onnx_production)
print('providers:', ort_debug.get_providers())
display(pd.DataFrame(rows).groupby(['route', 'boundary'], sort=False)
        .agg(max_abs=('max_abs', 'max'), all_pass=('pass', 'all')))
''')

md('''
## 4. TorchScript → MIL → Core ML：保留中间表示

先验证 `torch.jit.trace` 的输出，再检查转换器中的 MIL，最后执行 Core ML。**MIL 文本是程序表示，不是一个已经执行过的数值节点**；保存 MIL 能检查类型/算子变化，但这里的数值证据来自 Core ML 输出。若 TorchScript 正常、Core ML 异常，只能先定位到“转换 + 编译 + runtime”这段，不能仅凭结果直接指责某个 MIL pass。

`ct.convert(..., convert_to='milinternal')` 返回可检查的 program；`program.functions['main'].operations` 查看算子；随后 `source='milinternal'` 转成 ML Program。本轮保存的 MIL 是该调用默认 pass 后的表示，不冒充原始前端 IR。FP32 显式指定 `FLOAT32`，防止默认 FP16 把表示转换和精度变化混在一起；每个配置从同一份 TorchScript 重新生成 FP32 MIL，再应用目标精度策略。

**编写本课时实际遇到的问题：** 对整份 Swin MIL 做 `copy.deepcopy` 触发 `RecursionError`，失败位置在 Python 对象复制，尚未进入 Core ML 推理。MIL 内部有相互关联的 function/block/op/var 对象，不适合在教程中当作简单列表随意深复制。改为每次从固定 trace 重建 program，付出额外转换时间来隔离 pass 的可变状态；不需要修改模型权重或调高全局递归限制。这个结论只对应本图、本机版本，不能推广成“所有 MIL 都无法 deepcopy”。

Core ML 的 `.mlpackage` 是模型包；加载时还涉及本机编译与设备选择。`compute_precision` 控制图内精度转换，输入/输出接口在本实验始终 FP32；它不保证所有内部运算都用 FP16，也不保证每条硬件指令的精度。
''')
code('''
with warnings.catch_warnings(record=True) as caught, torch.inference_mode():
    warnings.simplefilter('always')
    traced_debug = torch.jit.trace(debug, torch_xs[0], check_inputs=[(torch_xs[1],)])
    traced_prod = torch.jit.trace(production, torch_xs[0], check_inputs=[(torch_xs[1],)])
(ART / 'torchscript.warnings.txt').write_text('\\n'.join(str(w.message) for w in caught))
(ART / 'torchscript_debug.txt').write_text(str(traced_debug.inlined_graph))
traced_debug.save(str(ART / 'swin_debug.pt'))
with torch.inference_mode():
    traced_values = [dict(zip(NAMES, [v.numpy() for v in traced_debug(x)])) for x in torch_xs]
record('torchscript_debug', traced_values)

def make_mil(traced, names):
    return ct.convert(traced, convert_to='milinternal',
        inputs=[ct.TensorType(name='pixel_values', shape=(1,3,224,224), dtype=np.float32)],
        outputs=[ct.TensorType(name=n, dtype=np.float32) for n in names],
        minimum_deployment_target=ct.target.macOS13, compute_precision=ct.precision.FLOAT32)

mil_debug = make_mil(traced_debug, NAMES)
mil_prod = make_mil(traced_prod, ['logits'])
(ART / 'debug_fp32.mil.txt').write_text(str(mil_debug))
(ART / 'production_fp32.mil.txt').write_text(str(mil_prod))
display(pd.DataFrame(Counter(op.op_type for op in mil_debug.functions['main'].operations)
                     .most_common(), columns=['MIL op', 'count']).head(16))
print('MIL 输出:', [(v.name, str(v.sym_type)) for v in mil_debug.functions['main'].outputs])
''')

md('''
### 每次只改变一个因素

|对照|固定什么|变化什么|能回答什么|
|---|---|---|---|
|PyTorch debug vs TorchScript|权重、输入、边界|图捕获|trace 是否改变了本组输入的输出|
|TorchScript vs Core ML FP32 CPU|FP32、输入、边界|表示/编译/runtime|这段转换执行链是否一致|
|Core ML FP32 CPU vs FP16 CPU|输入、允许设备、边界|图内精度策略|精度改写及其执行路径带来的差异|
|Core ML FP16 CPU vs ALL|同一模型包、输入|允许设备配置|设备选择策略是否影响结果|
|debug vs production|同一输入、权重、精度|额外输出|观测边界是否改变了最终输出|

`ALL` 表示允许 CPU/GPU/ANE，实际运行位置仍需 runtime trace。为了避免把调试候选当交付产物，我们对生产 FP32 CPU、生产 FP16 CPU 和生产 FP16 ALL 也独立检查 logits。中间输出只解释调试图中的边界；如果仅生产图失败，要继续在生产图上做最少量插桩，不直接套用调试图根因。
''')
code('''
def build_coreml(traced, names, name, precision):
    program = make_mil(traced, names)  # 独立 program，防止上一轮 pass 改写污染下一轮。
    converted = ct.convert(program, source='milinternal', convert_to='mlprogram',
        minimum_deployment_target=ct.target.macOS13, compute_precision=precision, skip_model_load=True)
    path = ART / f'{name}.mlpackage'
    converted.save(str(path))
    return path

def run_coreml(path, units):
    runtime = ct.models.MLModel(str(path), compute_units=units)
    values = [runtime.predict({'pixel_values': x}) for x in xs]
    del runtime
    gc.collect()
    return values

coreml = {}
for precision, policy in [('fp32', ct.precision.FLOAT32), ('fp16', ct.precision.FLOAT16)]:
    for kind, traced, names in [('debug', traced_debug, NAMES), ('production', traced_prod, ['logits'])]:
        name = f'coreml_{kind}_{precision}_cpu'
        package = build_coreml(traced, names, name, policy)
        outputs = run_coreml(package, ct.ComputeUnit.CPU_ONLY)
        coreml[name] = outputs
        baseline = ref if kind == 'debug' else [{'logits': p} for p in prod_ref]
        record(name, outputs, precision, baseline=baseline)
        if precision == 'fp16' and kind == 'production':
            all_outputs = run_coreml(package, ct.ComputeUnit.ALL)
            coreml['coreml_production_fp16_all'] = all_outputs
            record('coreml_production_fp16_all', all_outputs, 'fp16', baseline=baseline)

record('coreml_precision_only', coreml['coreml_debug_fp16_cpu'], 'fp16',
       baseline=coreml['coreml_debug_fp32_cpu'])
record('coreml_units_only', coreml['coreml_production_fp16_all'], 'fp16',
       baseline=coreml['coreml_production_fp16_cpu'])
for precision in ['fp32', 'fp16']:
    record(f'coreml_debug_vs_production_{precision}',
           [{'logits': p['logits']} for p in coreml[f'coreml_debug_{precision}_cpu']], precision,
           baseline=coreml[f'coreml_production_{precision}_cpu'])
quality = pd.DataFrame(rows)
quality.to_csv(RUN / 'boundary_quality.csv', index=False)
display(quality.groupby(['route', 'boundary'], sort=False)
        .agg(max_abs=('max_abs', 'max'), all_pass=('pass', 'all')))
''')

md('''
## 5. 看误差曲线，再决定该查哪里

下面每一点是该边界在本组输入上的**最大**绝对误差，不能把几个 stage 的误差相加。reduction、残差、归一化都会放大或压小上游误差；误差未必随层数单调增大。

“第一处失败边界”只把问题缩到上一个通过边界与当前边界之间，或表示多段微小误差累积到这里越过阈值；不等于这个 stage 必然有实现 bug。真正缩小问题要冻结该模块输入，单独比较同一模块。尚无失败也可以练习这种流程，不能编造 bug。

图中的 log 轴只用于显示不同数量级；0 在绘图时显示为 `1e-12`，CSV 仍保留原值。

还要检查具体元素：例如 reference=100、误差=0.8，在本轮预算 `0.05+0.01×100=1.05` 内；reference=0.2、误差=0.1，却超过预算0.052。下面同时记录最大绝对误差处的坐标/值，以及“误差÷该元素预算”最大的位置。比值大于1才是门禁失败。不能只盯绝对误差最大的一点，也不能把跨元素的平均 cosine 当成逐元素保证。
''')
code('''
routes_to_plot = ['onnx_debug_off', 'onnx_debug_on', 'torchscript_debug',
                  'coreml_debug_fp32_cpu', 'coreml_debug_fp16_cpu']
fig, ax = plt.subplots(figsize=(11, 4))
for route in routes_to_plot:
    # worst: 每个语义边界上，跨全部样本取最大的绝对误差。
    worst = quality[quality.route == route].groupby('boundary').max_abs.max().reindex(NAMES)
    ax.plot(NAMES, np.maximum(worst.to_numpy(), 1e-12), marker='o', label=route)
ax.set(yscale='log', ylabel='Worst max absolute error', xlabel='Semantic boundary')
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(RUN / 'boundary_errors.png', dpi=160)
plt.show()

first_failures = []
for (route, sample), group in quality.groupby(['route', 'sample'], sort=False):
    failed = set(group.loc[~group['pass'], 'boundary'])
    first_failures.append({'route': route, 'sample': sample,
                           'first_failed_boundary': next((n for n in NAMES if n in failed), None)})
failure_table = pd.DataFrame(first_failures)
failure_table.to_csv(RUN / 'first_failures.csv', index=False)
display(failure_table[failure_table.first_failed_boundary.notna()])

element_rows = []
for i, (gold, actual) in enumerate(zip(ref, coreml['coreml_debug_fp16_cpu'])):
    for name in NAMES:
        a, b = gold[name].astype(np.float64), actual[name].astype(np.float64)
        error = np.abs(b - a)
        budget = TOLERANCES['fp16']['atol'] + TOLERANCES['fp16']['rtol'] * np.abs(a)
        peak = np.unravel_index(error.argmax(), error.shape)
        breach = np.unravel_index((error / budget).argmax(), error.shape)
        element_rows.append({'sample': sample_ids[i], 'boundary': name,
            'max_abs_coordinate': str(peak), 'max_abs': float(error[peak]),
            'reference_at_max': float(a[peak]), 'candidate_at_max': float(b[peak]),
            'max_abs_within_budget': bool(error[peak] <= budget[peak]),
            'worst_budget_coordinate': str(breach), 'worst_budget_ratio': float((error / budget)[breach]),
            'reference_at_worst_budget': float(a[breach]), 'candidate_at_worst_budget': float(b[breach]),
            'exceed_elements': int((error > budget).sum()), 'total_elements': a.size})
element_table = pd.DataFrame(element_rows)
element_table.to_csv(RUN / 'fp16_element_inspection.csv', index=False)
display(element_table[element_table.exceed_elements > 0])
''')

md('''
### logits 的 margin：为什么高 cosine 也不够

假设第一名 8.000、第二名 7.995，margin（间隔）是 0.005。如果第一名误差 -0.004、第二名 +0.004，预测会翻转，其他 998 个分数完全不变也救不了 Top-1。

对每张图，若 reference 第一名与第二名之差 **大于两倍最大 logit 误差**，则任意两个类别的最坏误差也不足以翻转第一名；这是充分条件，不是必要条件。不满足只意味着不能用这个界保证，并不代表一定翻转。下面分别保存真实输入与数值探针，不计算没有真值的 accuracy。
''')
code('''
margin_rows = []
for route in ['coreml_production_fp32_cpu', 'coreml_production_fp16_cpu', 'coreml_production_fp16_all']:
    for i, (a, outputs) in enumerate(zip(prod_ref, coreml[route])):
        b = outputs['logits']
        err = compare(a, b, 'fp16' if 'fp16' in route else 'fp32')
        top2 = np.sort(a[0])[-2:]
        margin = float(top2[-1] - top2[-2])
        margin_rows.append({'route': route, 'sample': sample_ids[i],
            'sample_kind': 'frozen_image' if i < len(keys) else 'numerical_probe',
            'margin': margin, 'max_abs': err['max_abs'], 'cosine': err['cosine'],
            'top1_same': bool(a.argmax(-1)[0] == b.argmax(-1)[0]) if err['finite'] else False,
            'margin_guarantee': bool(err['finite'] and margin > 2 * err['max_abs'])})
margin_table = pd.DataFrame(margin_rows)
margin_table.to_csv(RUN / 'logit_margins.csv', index=False)
display(margin_table)
''')

md('''
## 6. 故障注入：checker 通过，数值仍然可能错误

**这是人为构造的训练故障，不是发现 ONNX/厂商存在 bug。** 在 `stage1_down` 的第一个 feature 上加 0.5，其他 feature 加 0。这是对一个真实中间 tensor 的定点扰动。只改最后导出的数组会绕过下游计算；所以我们修改 ONNX 图的生产者输出，并把 Add 接回原 tensor 名，让所有消费者真的读取被改过的值。

ONNX 的连接靠字符串名称。算法：从 `graph.output['stage1_down']` 反查 producer；若只是 Identity 别名，沿 input 找到实际生产者；把其输出重命名为 `__before_fault`；插入 `Add(before_fault, delta) → 原名`。原消费者不变，仍读原名，数值却来自 Add。类似修改 SSA def-use 链：重新定义一个值，原来的使用者随之读到新值。

验证三件事：故障图结构仍合法；第一处越过 FP32 阈值的已选边界就是 `stage1_down`；重新加载未改动的文件后全部恢复。后面某些误差可能被 LayerNorm 抑制，这不推翻注入已发生。保存的是“最早被观测到的差异”，不能凭稀疏打点声称逐算子定位。
''')
code('''
fault_graph = copy.deepcopy(debug_graph)
producers = {output: node for node in fault_graph.graph.node for output in node.output}
target = 'stage1_down'
while producers[target].op_type == 'Identity':
    target = producers[target].input[0]
producer = producers[target]
before = target + '__before_fault'
for i, output in enumerate(producer.output):
    if output == target:
        producer.output[i] = before
delta = np.zeros((384,), dtype=np.float32)  # 按最后 feature 轴 broadcast；只动第0个 feature。
delta[0] = .5
fault_graph.graph.initializer.append(onnx.numpy_helper.from_array(delta, name='__fault_delta'))
add = onnx.helper.make_node('Add', [before, '__fault_delta'], [target], name='INJECTED_stage1_feature0')
nodes = list(fault_graph.graph.node)
index = next(i for i, n in enumerate(nodes) if before in n.output)
nodes.insert(index + 1, add)
del fault_graph.graph.node[:]
fault_graph.graph.node.extend(nodes)
onnx.checker.check_model(fault_graph)
fault_path = ART / 'swin_debug_INJECTED.onnx'
onnx.save(fault_graph, fault_path)
fault_session = ort_session(fault_path)
fault_outputs = run_ort(fault_session)
record('injected_feature_fault', fault_outputs, baseline=ort_off)
fault_rows = pd.DataFrame(rows).query("route == 'injected_feature_fault'")
for sample, group in fault_rows.groupby('sample', sort=False):
    failed = set(group.loc[~group['pass'], 'boundary'])
    assert next((n for n in NAMES if n in failed), None) == 'stage1_down', (sample, failed)
restored = run_ort(ort_session(debug_path))
assert all(compare(g[n], r[n])['pass'] for g, r in zip(ort_off, restored) for n in NAMES)
write_json(RUN / 'fault_injection.json', {'kind': 'synthetic ONNX graph fault',
    'modified_tensor': target, 'first_failed_boundary': 'stage1_down', 'delta_feature0': .5,
    'checker_passed': True, 'all_samples_localized': True, 'rollback_passed': True})
fault_rows.to_csv(RUN / 'fault_boundaries.csv', index=False)
display(fault_rows.groupby('boundary', sort=False).agg(max_abs=('max_abs', 'max'), all_pass=('pass','all')))
''')

md('''
## 7. 从 stage 边界缩小到 Patch Merging

Patch Merging 的英文就是合并相邻 patch；在代码里先把 2×2 空间位置的特征拼起来，再做 LayerNorm 和 Linear。这里用 **N=1,H=2,W=2,C=2，FP32，NHWC 连续内存**：四个位置分别是 `[10,11]、[20,21]、[30,31]、[40,41]`。

|元素偏移|字节地址（基址0）|值|NHWC 下标|
|---:|---:|---:|---|
|0|0|10|[0,0,0,0]|
|1|4|11|[0,0,0,1]|
|2|8|20|[0,0,1,0]|
|3|12|21|[0,0,1,1]|
|4|16|30|[0,1,0,0]|
|5|20|31|[0,1,0,1]|
|6|24|40|[0,1,1,0]|
|7|28|41|[0,1,1,1]|

本版本源码顺序是左上、左下、右上、右下，各自保留 C 个 feature。输出 `[1,1,8]` 中最后一轴依次读取：

|输出坐标 [n,token,feature]|原字节地址展开式|原物理偏移|读到的值|
|---|---|---:|---:|
|[0,0,0]|4×((0×2+0)×2+0)|0|10|
|[0,0,1]|4×((0×2+0)×2+1)|1|11|
|[0,0,2]|4×((1×2+0)×2+0)|4|30|
|[0,0,3]|4×((1×2+0)×2+1)|5|31|
|[0,0,4]|4×((0×2+1)×2+0)|2|20|
|[0,0,5]|4×((0×2+1)×2+1)|3|21|
|[0,0,6]|4×((1×2+1)×2+0)|6|40|
|[0,0,7]|4×((1×2+1)×2+1)|7|41|

所以拼接结果 `[10,11,30,31,20,21,40,41]`，直接 reshape 则保留 `[10,11,20,21,30,31,40,41]`。shape 一模一样，语义已经错了。

一般 NHWC 元素偏移才写成 `((n*H+h)*W+w)*C+c`。**stride（步长）**是某轴坐标加1时跨过几个 storage 元素；**slice（切片）**是按起点/终点/步长选元素。`0::2` 即从0开始每隔2取一次。**layout transform** 是改变 tensor 元素的组织/访问顺序；不能仅凭图里一个 transpose 就断言设备做了整块搬运，还要看编译后的执行方式。
''')
code('''
toy = torch.tensor([[[[10.,11.], [20.,21.]], [[30.,31.], [40.,41.]]]])
parts = [toy[:,0::2,0::2,:], toy[:,1::2,0::2,:],
         toy[:,0::2,1::2,:], toy[:,1::2,1::2,:]]
merged = torch.cat(parts, -1).reshape(1, 1, 8)
wrong = toy.reshape(1, 1, 8)
assert merged.flatten().tolist() == [10,11,30,31,20,21,40,41]
assert not compare(merged.numpy(), wrong.numpy())['pass']
display(pd.DataFrame({'correct_merge': merged.flatten().tolist(),
                      'wrong_reshape': wrong.flatten().tolist()}))
''')

md('''
### 冻结模块输入，区分“上游带进来的误差”与“本模块新增的误差”

实际取 `encoder.layers[1].downsample`：输入 `[1,784,192]`（28×28 token），拼接后 `[1,196,768]`，LayerNorm 沿768个 feature 做归约，Linear 将768映射为384。这个 Linear 中768是 reduction 轴，384是输出 feature 轴。

forward hook（模块执行后的回调）同时拿到调用参数与结果，保存本次模块的真实输入；finally 中移除 hook。然后单独导出这个 Patch Merging 模块，PyTorch/ORT 都喂相同 frozen input。全网误差可能有上游累积，局部对照只衡量“这个模块在给定输入处”的差异。**局部通过不证明所有输入都通过，更不能替代全网回归。**

这是前面粗定位之后可复用的缩小方法；本轮没有事先断言 Patch Merging 存在缺陷。
''')
code('''
captured = {}
merge_module = model.swin.encoder.layers[1].downsample
def capture_merge(module, args, output):
    captured['input'] = args[0].detach().clone()
    captured['dimensions'] = tuple(int(v) for v in args[1])
    captured['output'] = output.detach().clone()
handle = merge_module.register_forward_hook(capture_merge)
try:
    with torch.inference_mode():
        production(torch_xs[0])
finally:
    handle.remove()

class FrozenPatchMerging(torch.nn.Module):
    def __init__(self, module, dimensions):
        super().__init__()
        self.module, self.dimensions = module, dimensions
    def forward(self, features):
        return self.module(features, self.dimensions)

local = FrozenPatchMerging(merge_module, captured['dimensions']).eval()
local_path = ART / 'stage1_patch_merging.onnx'
export_onnx(local, captured['input'], local_path, ['merged'], input_name='features')
local_session = ort_session(local_path)
local_out = local_session.run(['merged'], {'features': captured['input'].numpy()})[0]
local_result = compare(captured['output'].numpy(), local_out)
write_json(RUN / 'local_patch_merging.json', local_result)
np.savez(RUN / 'local_patch_merging.npz', features=captured['input'].numpy(),
         reference=captured['output'].numpy(), onnx=local_out)
assert local_result['pass'], local_result
print(captured['input'].shape, '→', local_out.shape, local_result)
''')

md('''
## 8. shape/dtype 契约：固定 shape 就应明确拒绝越界请求

本产物只支持 `[1,3,224,224]` FP32。`batch=2`、`H=225`、`float64` 不属于承诺范围。下面同时测试应用边界和 ORT 真实接口：应用先报易懂错误，绕过应用时 runtime 也应拒绝 shape/type 不匹配。

不测试“把随机 tensor 改成 225 能否侥幸跑通”然后宣称 dynamic shape。若需求改成动态 batch/分辨率，要重新导出并覆盖 batch=1/2、每个批准尺寸、window=7 的整除/非整除和 Patch Merging 奇偶尺寸等边界。Python 分支被 trace 固定时，改输入签名并不能恢复动态语义；需要选合适的 capture 路径并检查 padding、mask 和实际结果。

输入出现 NaN 则由应用明确拦截；这个检查是产品契约选择，不依赖 runtime 自动帮你检查数值。
''')
code('''
def checked_input(x):
    if x.shape != (1,3,224,224):
        raise ValueError(f'expected shape (1,3,224,224), got {x.shape}')
    if x.dtype != np.float32:
        raise TypeError(f'expected float32, got {x.dtype}')
    if not np.isfinite(x).all():
        raise ValueError('input contains NaN/Inf')
    return np.ascontiguousarray(x)

bad_inputs = {'batch2': np.repeat(xs[0], 2, axis=0),
              'height225': np.pad(xs[0], ((0,0),(0,0),(0,1),(0,0))),
              'float64': xs[0].astype(np.float64), 'nan': np.full_like(xs[0], np.nan)}
contract_rows = []
assert np.array_equal(checked_input(xs[0]), xs[0])
for case, value in bad_inputs.items():
    try:
        checked_input(value)
    except (ValueError, TypeError) as error:
        contract_rows.append({'case': case, 'layer': 'application', 'rejected': True, 'error': str(error)})
    else:
        raise AssertionError(f'应用未拦截 {case}')
    if case != 'nan':
        try:
            ort_prod.run(['logits'], {'pixel_values': value})
        except ort.capi.onnxruntime_pybind11_state.InvalidArgument as error:
            contract_rows.append({'case': case, 'layer': 'ORT', 'rejected': True, 'error': str(error)})
        else:
            raise AssertionError(f'ORT 未拒绝 {case}')
pd.DataFrame(contract_rows).to_csv(RUN / 'contract_negative_controls.csv', index=False)
display(pd.DataFrame(contract_rows))
''')

md('''
## 9. 从现象到决定：本阶段该积累什么经验

|看到的现象|先做的受控实验|可下的结论/下一步|
|---|---|---|
|checker 通过，某个 stage 开始错|相同图、输入、CPU，检查 tensor producer/consumer；回滚对照|结构合法不等于语义正确；本轮注入属于训练故障|
|TorchScript 已不对齐|回到 eager/trace 同输入，检查训练态与 shape 分支|尚未进入 Core ML，不先改 runtime|
|ORT off 通过、on 失败|固定导出文件和 EP，仅切图优化；提取最小复现|可选优化相关的嫌疑增加，仍需定位具体重写|
|FP32 Core ML 通过、FP16 不通过|CPU_ONLY 固定；看第一失败边界；冻结该模块输入|精度策略相关，仍不能凭“大误差”认定为舍入或 bug|
|debug 通过，production 失败|生产图逐步增加最少输出，每次与原图对照|观测改变优化/执行路径是待验证假设|
|cosine 高但 Top-1 变|逐图看 margin 与 max logit error|分类决策边界敏感；需真实标签判断任务影响|
|所有数值门禁都通过|再做完整标签评测、runtime trace、生产包性能|本阶段不能提前宣布部署验收完成|

**FP16 rounding vs implementation bug 怎么区分？** 先隔离输入、捕获、转换、精度和设备变量；冻结可疑块输入，对同一运算做 FP32/FP16 对照；检查 NaN/Inf、极值、reduction 长度与输入尺度；构造小输入验证公式和边界；修复后回归所有已保存样本。仅有误差小、cosine 高或“切回 FP32 恢复”都不是充分证明。

**本阶段的取舍：** 多输出方便定位，但会增加输出搬运并限制优化，所以交付候选保留 logits-only；静态 shape 缩小验证矩阵，但拒绝未批准的输入；先保留 FP32 控制组，付出包体/构建成本换取归因能力。不做无证据的 transpose/fusion 性能优化。

### 迁移到公司台架：迁移实验方法，不移植硬件结论

|问题|M1 本轮怎么做|N93X 下一步怎么做|
|---|---|---|
|模型入口|PyTorch 分别导出 ONNX、Core ML|【官方 schema】报告 §4.2.7 记录 ONNX parser 与逐输出 comparer；先验证厂商入口对本 ONNX 的支持|
|中间输出|调试 wrapper 暴露语义边界|【推断】尝试同样边界的调试模型或厂商 dump；这是迁移方案，尚未执行，不能保证任意融合内部 tensor 可取|
|设备归属|CPU_ONLY 控制组；ALL 仅表示允许设备|本阶段不推断板端设备归属；台架拿到 runtime 证据后单独记录|

来源：[N93X 调研报告 §4.2.7](../../docs/N93X-AllSpark-LLM-Ecosystem-Research-Report.md)。报告顶部声明由 AI 生成、部分数据可能有误；这里“官方 schema”只沿用报告记录的接口证据，没有重新核验 SDK，也不代表 Swin 已在台架跑通。台架扩展需要对应工具链和权限，不是本课完成条件。对外经历只写“某车规 AD SoC / 自研 NPU”，不带内部材料。
''')

md('''
## 10. 保存证据与门禁结果

`screen_pass` 只是当前输入与阈值的数值筛查；`task_accuracy` 一直是 `not_evaluated`。FP16 筛查失败会原样留在结果中，不能为了完成教程把它隐藏。进入性能阶段前，必须解释失败或选择已通过的候选。

下面保存全部 artifact 的 hash、每个 backend 的 raw tensors、误差表、图与负控。最后 FP32 主链有失败就抛出异常；证据在抛异常前已写入。预期的训练故障不计入 FP32 主链通过率。
''')
code('''
quality = pd.DataFrame(rows)
quality.to_csv(RUN / 'boundary_quality.csv', index=False)
route_status = quality.groupby('route', sort=False)['pass'].all()
fp32_routes = ['onnx_debug_off', 'onnx_debug_on', 'ort_optimization_only', 'onnx_production',
              'onnx_debug_vs_production', 'torchscript_debug', 'coreml_debug_fp32_cpu',
              'coreml_production_fp32_cpu', 'coreml_debug_vs_production_fp32']
summary = {'fp32_chain_pass': bool(route_status[fp32_routes].all()),
    'route_screens': {name: bool(value) for name, value in route_status.items()},
    'negative_controls': {'shape_dtype_nan': True, 'same_shape_wrong_layout': True,
                         'high_cosine_top1_flip': True, 'onnx_graph_fault_and_rollback': True},
    'task_accuracy': 'not_evaluated', 'device_mapping': 'unverified', 'performance': 'not_measured',
    'source_image_count': len(keys), 'numerical_probe_count': len(xs) - len(keys),
    'experience_owner': 'generated tutorial and automated execution; user hands-on completion not asserted'}
write_json(RUN / 'summary.json', summary)
manifest['artifacts'] = {str(p.relative_to(ART)): sha256(p) for p in sorted(ART.rglob('*')) if p.is_file()}
manifest['evidence_files'] = {str(p.relative_to(RUN)): sha256(p) for p in sorted(RUN.iterdir())
                              if p.is_file() and p.name != 'manifest.json'}
write_json(RUN / 'manifest.json', manifest)
display(route_status.rename('screen_pass').to_frame())
print('证据目录:', RUN)
assert summary['fp32_chain_pass'], 'FP32 主链未通过；先查看 boundary_quality.csv 和 first_failures.csv'
''')

md('''
## 11. 经验卡：跑过、调查过、能解释，是三个不同状态

每个 case 都按下面结构自己填写；可引用本轮输出，但不能把 AI 自动执行写成个人独立排障经历。尚未发生的真实故障标为“待调查”；人为注入必须明说。

```text
Case / 类型：真实问题 or 训练故障
Symptom：哪个样本、哪个边界、哪个指标异常？
Baseline：模型/输入/工具链 hash 与允许设备配置
Hypotheses：至少两个可排除的解释
Evidence：哪份 npz / CSV / 图 / 图节点支持或反驳解释？
Root cause：已证实到什么粒度？哪些仍是推断？
Fix：只改变了什么？
Before / After：同一输入的实际结果
Side effect：是否改输出、精度、内存、性能可比性？
Regression test：控制组、故障组、回滚是否都通过？
Transferable lesson：换后端还成立的方法是什么？
```

建议完成三张卡：① ONNX 故障注入与回滚；② FP32/FP16 边界误差的本机观测；③ Patch Merging 同 shape 不同元素顺序。第一张可证明你理解验证器，第二张只有实际异常与调查证据充分时才写“定位了问题”，第三张要能手算地址顺序。

面试追问练习：不看代码，复述一张图片从 `inputs.npz → wrapper → ONNX/ORT` 与 `TorchScript → MIL → Core ML` 的完整路径。再回答：为什么导出了 stage tensor 仍不能说看到生产模型的全部内部行为？为什么“第一个失败 stage”不一定是根因 stage？为什么 `ALL` 加速不等于全 ANE？为什么本轮没有准确率结论？

**下一阶段入口：** 保留通过且可解释的 logits-only FP32/FP16 包、固定验证集与本轮误差口径；有真实标签和代表性样本后再进入 quantization/calibration。若 FP16 已有未解释的误差，先调查再加 INT8，避免同时引入两个未知因素。
''')

md('''
## 12. 来源、检视与重跑

本教程按 2026-09-20 本机锁定版本核对 API；运行时版本与源码 hash 以 manifest 为准。

- [PyTorch 2.7 ONNX 导出路径](https://docs.pytorch.org/docs/2.7/onnx.html)：本轮固定 exporter 及 tracing 范围；[当前官方导出教程](https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html) 提供新 exporter 路径。
- [Hugging Face v4.51.3 Swin 源码](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/swin/modeling_swin.py)：hidden_states 与 Patch Merging 顺序；本课用本机同版本源文件核验。
- [ONNX shape inference](https://onnx.ai/onnx/api/shape_inference.html)：`value_info` 与静态 shape；[ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html)：SessionOptions、provider、run。
- [Apple MIL](https://apple.github.io/coremltools/docs-guides/source/model-intermediate-language.html)、[ML Program 精度](https://apple.github.io/coremltools/docs-guides/source/convert-to-ml-program.html)、[输入输出类型](https://apple.github.io/coremltools/docs-guides/source/model-input-and-output-types.html)：转换流程、精度与显式 FP32 接口。
- 仓库方法口径：[端侧部署详答 §一·1、§22～24](../../docs/Detailed‑Answers‑for‑Edge‑AI‑Deployment.md)；[Swin 路线 §三、十二](../../docs/swin-transformer-guideline.md)。

检查本机接口最小命令：`uv run python -c "import torch; help(torch.onnx.export)"`。`-c` 执行一小段 Python，`help` 打印已安装版本的参数。若网页与本机不同，以已锁定的版本和实际执行为准。

顺序重跑并把输出留在新目录（先用 `mkdir -p results/swin_stage2/manual-run` 建目录）：

```bash
uv run jupyter nbconvert --execute --to notebook \\
  --ExecutePreprocessor.timeout=1800 \\
  --output executed.ipynb --output-dir results/swin_stage2/manual-run \\
  notebooks/swin_transformer/02_stage2_conversion_correctness.ipynb
```

`--execute` 顺序执行；`timeout=1800` 是单格最多30分钟；`--output-dir` 只控制执行后的 Notebook，实验数据位置仍由首格 RUN 决定。返回非零退出码时保留日志，不把部分执行当全链通过。重建无输出教程使用 `uv run --locked python tools/build_swin_stage2_notebook.py`；不要在加入个人笔记后直接重建覆盖它。
''')

for index, cell in enumerate(cells):
    cell.id = f'swin-stage2-{index:02d}'
    if cell.cell_type == 'code' and ('ct.convert(' in cell.source or 'torch.jit.trace' in cell.source):
        cell.metadata['scrolled'] = True
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {'kernelspec': {'display_name': 'infer-proj (uv)', 'language': 'python', 'name': 'python3'},
               'language_info': {'name': 'python', 'version': '3.12'}}
nbf.validate(nb)
for index, cell in enumerate(cells):
    if cell.cell_type == 'code':
        compile(cell.source, f'cell-{index}', 'exec')
destination = ROOT / 'notebooks/swin_transformer/02_stage2_conversion_correctness.ipynb'
nbf.write(nb, destination)
print(destination, len(cells), 'cells')
