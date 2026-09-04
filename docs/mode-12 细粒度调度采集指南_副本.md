# ConvNeXt / DyT 在 NX9031 上的 mode-12 细粒度调度采集指南

本文针对当前工作区 `codex_work/convnext_dyt_experiment`，给出从编译到 N93X
板端 trace、解析和证据归档的完整路径。脚本都放在独立文件中自行创建，全文不
使用 bash 函数，也不依赖遗留的 `$RUN` 环境变量。

## 0. 实验边界

当前 `codex_work/manifests/current_experiment.yaml` 是 E1.2；ConvNeXt 是探索性
分支，不能把新结果写成主线 Gate。已有的
`runs/20260901T035617Z` 只有 mode-0：其 `summary/config.json` 为
`npu_hw_grid=0`，日志含 `Hardware profiling set to disable`，因此不能从该 run
推断 Cayley/Fermat 的 runtime placement。工作区的 ResNet50 mode-12 run 只能用来
验证文件格式和解析方法。

正确顺序是：

1. 保留每个 ONNX、compile config、AOM/AP、runtime、板端日志和 SHA256；
2. 先以 mode-0 做正确性和干净的多轮延迟；
3. 再以 mode-12 做软件 timer、core placement 和 kernel 时间线；
4. 需要硬件 Grid/task 事件时另开 `profiling_mode=15` 的 one-shot run；
5. mode-12/mode-15 的 instrumentation 开销不能写成产品 E2E latency。

## 1. mode-12 的能力

### 1.1 profiling mode 位

N93X nxPerf 手册的四个位为：

| 位 | 值 | 含义 |
|---|---:|---|
| bit31 | 1 | Fermat hardware Grid |
| bit30 | 2 | Cayley hardware Grid |
| bit29 | 4 | Fermat software timer |
| bit28 | 8 | Cayley software timer |

所以 `profiling_mode=12`（`0b1100`）打开两个核心族的软件 timer；`15` 打开四个
位。mode-12 能给出实际执行 slice 的 `Start`、`Duration`、cluster 和 core，不能
凭空给出编译器的逐 kernel estimate cost，也不能给出某一条 tensor edge 的精确
DDR 写回字节数。

### 1.2 证据能力矩阵

| 信息 | mode-0 | mode-12 | mode-15 + `--npu_hw_grid` |
|---|---|---|---|
| ONNX/OpFusion/tiling/cache 静态图 | 可以 | 可以 | 可以 |
| 实际 Cayley/Fermat core | 不可以 | `Monitor` + `Record` | 可以，含 Grid/task 事件 |
| 每条 slice 的 ns | 不可以 | 软件 timer，带 instrumentation | 硬件 Grid/软件 timer，侵入更大 |
| DDR 带宽 | 独立窗口测试 | `--ddr_bandwidth_period` 窗口聚合 | 同左 |
| producer→consumer 的单边写回 bytes | 不可以 | 不可以 | 通常也不可以 |
| 产品 E2E p50/p95 | 最适合 | 不适合替代 | 不适合，one-shot |

mode-12 中并行 core 的 slice 不能直接相加为一帧耗时。产品延迟仍以同一 AOM/AP
在无 nxPerf 的 mode-0 下 warm-up 后的 `BENCH_STATS` 为准。

## 2. 创建新 run

在宿主 `[H宿主]` 创建新目录，不覆盖现有 mode-0 run：

~~~bash
cd /home/nio/Downloads/nn
SOURCE_RUN="$PWD/codex_work/convnext_dyt_experiment/runs/20260901T035617Z"
RUN_ROOT="$PWD/codex_work/convnext_dyt_experiment/runs/$(date -u +%Y%m%dT%H%M%SZ)_mode12"
test -d "$SOURCE_RUN/exports"
test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT"
cp -a "$SOURCE_RUN/exports" "$RUN_ROOT/exports"
cp "$PWD/codex_work/convnext_dyt_experiment/compile_config_mode12.json" \
   "$RUN_ROOT/compile_config_mode12.json"
printf 'source_run=%s\nrun_root=%s\nprofiling_mode=12\n' \
  "$SOURCE_RUN" "$RUN_ROOT" > "$RUN_ROOT/experiment.txt"
~~~

exports 至少应包含 `baseline.onnx`、`dyt.onnx`、`dyt_nopermute.onnx`、
`input.bin`、三个对应的 `*_output_1.bin` 和 `manifest.json`。输入/输出 golden
必须和模型变体配对；当前 runtime 会从部署目录父目录读取
`../input.bin` 和 `../output_1.bin`。

### 2.1 先执行这一步，再调用后续脚本

上一节的 `RUN_ROOT` 只在当前 shell 中有效。如果另开终端，必须把实际目录重新
赋给变量。以下命令应在 `[H宿主]` 的同一个终端中执行：

~~~bash
cd ~/Downloads/nn
SOURCE_RUN="$PWD/codex_work/convnext_dyt_experiment/runs/20260901T035617Z"
RUN_ROOT="$PWD/codex_work/convnext_dyt_experiment/runs/$(date -u +%Y%m%dT%H%M%SZ)_mode12"
test -d "$SOURCE_RUN/exports"
test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT"
cp -a "$SOURCE_RUN/exports" "$RUN_ROOT/exports"
cp "$PWD/codex_work/convnext_dyt_experiment/compile_config_mode12.json" \
  "$RUN_ROOT/compile_config_mode12.json"
printf 'source_run=%s\nrun_root=%s\nprofiling_mode=12\n' "$SOURCE_RUN" "$RUN_ROOT" \
  > "$RUN_ROOT/experiment.txt"
printf 'RUN_ROOT=%s\n' "$RUN_ROOT"
~~~

如果终端重启，重新赋值为上一步打印的真实路径，例如：

~~~bash
cd ~/Downloads/nn
RUN_ROOT="$PWD/codex_work/convnext_dyt_experiment/runs/20260901T120000Z_mode12"
test -d "$RUN_ROOT/exports"
~~~

不要把示例中的时间戳原样使用，除非它确实是你刚创建的目录。

## 3. 创建 mode-12 编译脚本

创建 `codex_work/convnext_dyt_experiment/mode12/01_compile_mode12.sh`，复制后
执行 `chmod +x`。它只使用循环和普通命令，没有 bash 函数：

~~~bash
#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then echo "usage: $0 <run-root>" >&2; exit 2; fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd -P)"
RUN_ROOT="$(realpath "$1")"
case "$RUN_ROOT/" in
  "$REPO_ROOT/"*) ;;
  *) echo "run root must be below repository: $RUN_ROOT" >&2; exit 3 ;;
esac
RUN_REL="${RUN_ROOT#"$REPO_ROOT"/}"
CONFIG_SRC="$REPO_ROOT/codex_work/convnext_dyt_experiment/compile_config_mode12.json"
test -s "$CONFIG_SRC"
grep -q '"profiling_mode"[[:space:]]*:[[:space:]]*"12"' "$CONFIG_SRC"
overall=0
for variant in baseline dyt dyt_nopermute; do
  ONNX_SRC="$RUN_ROOT/exports/${variant}.onnx"
  TARGET="$RUN_ROOT/${variant}_mode12"
  test -s "$ONNX_SRC"
  if [[ -e "$TARGET" ]]; then
    echo "refusing stale compile target: $TARGET" >&2
    overall=1
    continue
  fi
  mkdir -p "$TARGET/output"
  cp "$CONFIG_SRC" "$TARGET/compile_config.json"
  printf 'variant=%s\nmode=12\nonnx=%s\n' "$variant" "$ONNX_SRC" > "$TARGET/experiment.txt"
  set +e
  docker exec -i codex_allspark_compiler bash -lc \
    "source /usr/local/allspark/allspark_env.sh; cd '/workspace/$RUN_REL/${variant}_mode12'; /usr/bin/time -v -o compile.resource.txt acompile --model_name='convnext_${variant}_mode12' --onnx='/workspace/$RUN_REL/exports/${variant}.onnx' --compile_config='/workspace/$RUN_REL/${variant}_mode12/compile_config.json' --output_dir='/workspace/$RUN_REL/${variant}_mode12/output'" \
    2>&1 | tee "$TARGET/compile.log"
  rc=${PIPESTATUS[0]}
  set -e
  printf 'variant=%s mode=12 return_code=%d\n' "$variant" "$rc" | tee "$TARGET/return_code.txt"
  if [[ $rc -ne 0 ]]; then overall=1; fi
done
exit "$overall"
~~~

编译完成后，日志不能出现
`profiling_mode is not found in json file, using default value: 0`。应同时保留
`compile.resource.txt` 和真实 `return_code.txt`；`tee` 的返回码不能代替
`PIPESTATUS[0]`。

执行方式（脚本的唯一参数就是 `RUN_ROOT`）：

~~~bash
cd ~/Downloads/nn
chmod +x codex_work/convnext_dyt_experiment/mode12/01_compile_mode12.sh
./codex_work/convnext_dyt_experiment/mode12/01_compile_mode12.sh "$RUN_ROOT"
~~~

如果只看到 `usage: ... <run-root>`，说明忘记传参数；如果看到
`refusing stale compile target`，说明该 run 已经部分或全部执行过，请保留原日志
并创建新的 run，不要直接删除旧目录。

## 4. 编译产物静态审计

创建 `codex_work/convnext_dyt_experiment/mode12/02_inventory_mode12.sh`：

~~~bash
#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then echo "usage: $0 <run-root>" >&2; exit 2; fi
RUN_ROOT="$(realpath "$1")"
OUT="$RUN_ROOT/mode12_compile_inventory.tsv"
printf 'variant\tpath\tbytes\tsha256\n' > "$OUT"
for variant in baseline dyt dyt_nopermute; do
  BUILD="$RUN_ROOT/${variant}_mode12"
  test -d "$BUILD"
  # 不能泛匹配 `using default value: 0`：例如 default_min_dim_value=0 与
  # profiling_mode 无关。
  if grep -Eq 'profiling_mode is not found in json file|\[key\]: profiling_mode .*using default value: 0' "$BUILD/compile.log"; then
    echo "ERROR: $variant fell back to mode 0" >&2
  fi
  GRID="$BUILD/build___aom_0/allspark_grid_info.txt"
  AOM_COUNT="$(find "$BUILD/output" -maxdepth 1 -type f -name '*.aom' | wc -l)"
  AP_COUNT="$(find "$BUILD/output" -maxdepth 1 -type f -name '*.ap' | wc -l)"
  printf '%s: aom=%s ap=%s grid=%s\n' "$variant" "$AOM_COUNT" "$AP_COUNT" "$GRID"
  test "$AOM_COUNT" -eq 1; test "$AP_COUNT" -eq 1; test -s "$GRID"
  find "$BUILD" -maxdepth 3 -type f \( -name '*.aom' -o -name '*.ap' -o -name 'allspark_grid_info.txt' \
    -o -name '__aom_0_model*.onnx' -o -name 'allspark_build*onnx' -o -name 'perf_summary.log' \
    -o -name 'tensor.json' -o -name 'fusion_result*' -o -name 'anchor.json' \) -print0 \
    | sort -z | while IFS= read -r -d '' item; do
      bytes="$(stat -c '%s' "$item")"
      digest="$(sha256sum "$item" | awk '{print $1}')"
      printf '%s\t%s\t%s\t%s\n' "$variant" "$item" "$bytes" "$digest"
    done >> "$OUT"
  printf 'grid_lines=%s\n' "$(wc -l < "$GRID")" > "$BUILD/grid_line_count.txt"
  awk -F'|' '{count[$2]++} END {for (task in count) print task, count[task]}' "$GRID" \
    | sort > "$BUILD/grid_task_distribution.txt"
done
cat "$OUT"
~~~

检查 `compile.log` 的 `Compile takes`、`Total mem`、`TmpTensor`、`CL2`、`GL2`，
检查 `perf_summary.log` 的 `fermat=`/`cayley=` 和 cost 权重；检查三份 ONNX 快照
（`__aom_0_model_.onnx`、`__aom_0_model_after_OpFusion.onnx`、
`allspark_build__graph___data.onnx`）的节点/layout 变化。Grid 行数含 marker、
prefetch 和 invalid，不是 runtime kernel 数；`tensor.json`/`anchor.json` 是静态
地址/cache/lifetime 线索，不是 cache hit/miss 或 DDR transaction counter。

执行方式：

~~~bash
cd ~/Downloads/nn
chmod +x codex_work/convnext_dyt_experiment/mode12/02_inventory_mode12.sh
./codex_work/convnext_dyt_experiment/mode12/02_inventory_mode12.sh "$RUN_ROOT"
cat "$RUN_ROOT/mode12_compile_inventory.tsv"
~~~

## 5. 构造板端部署目录

创建 `codex_work/convnext_dyt_experiment/mode12/03_stage_mode12_deploy.sh`。
该脚本假定 `runtime/build_arm/test_sample` 已经使用
`source /usr/local/allspark/allspark_env.sh arm N93X` 成功构建；部署目录放在
run 根目录下，避免编译容器 root 创建的目录阻止宿主写日志。

~~~bash
#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then echo "usage: $0 <run-root>" >&2; exit 2; fi
RUN_ROOT="$(realpath "$1")"
RUNTIME="$PWD/codex_work/convnext_dyt_experiment/runtime/build_arm/test_sample"
test -x "$RUNTIME"
for variant in baseline dyt dyt_nopermute; do
  BUILD="$RUN_ROOT/${variant}_mode12"
  SRC="$BUILD/output"
  DEPLOY="$RUN_ROOT/board_deploy/${variant}_mode12"
  if [[ -e "$DEPLOY" ]]; then echo "refusing stale deployment: $DEPLOY" >&2; exit 3; fi
  mkdir -p "$DEPLOY/build_arm"
  AOM_LIST="$(find "$SRC" -maxdepth 1 -type f -name '*.aom' | sort)"
  AP_LIST="$(find "$SRC" -maxdepth 1 -type f -name '*.ap' | sort)"
  [[ "$(printf '%s\n' "$AOM_LIST" | sed '/^$/d' | wc -l)" -eq 1 ]]
  [[ "$(printf '%s\n' "$AP_LIST" | sed '/^$/d' | wc -l)" -eq 1 ]]
  AOM="$(printf '%s\n' "$AOM_LIST")"
  AP="$(printf '%s\n' "$AP_LIST")"
  GRID="$BUILD/build___aom_0/allspark_grid_info.txt"
  test -s "$GRID"
  cp "$AOM" "$DEPLOY/__aom_0.aom"
  cp "$AP" "$DEPLOY/"
  cp "$GRID" "$DEPLOY/__aom_0.txt"
  cp "$RUNTIME" "$DEPLOY/build_arm/test_sample"
  cp "$RUN_ROOT/exports/input.bin" "$DEPLOY/input.bin"
  cp "$RUN_ROOT/exports/${variant}_output_1.bin" "$DEPLOY/output_1.bin"
  find "$DEPLOY" -maxdepth 2 -type f -printf '%p %s bytes\n' | sort > "$DEPLOY/files.txt"
  # `*` 会同时展开 build_arm 目录；只列出普通文件，且不让清单哈希自身。
  (cd "$DEPLOY" && find . -type f ! -name deploy.sha256 -print0 \
    | sort -z | xargs -0 sha256sum) > "$DEPLOY/deploy.sha256"
done
~~~

如果 runtime 出现 `refusing stale build directory`，不要递归删除；把旧目录安全
改名为 `build_arm.previous.<UTC timestamp>`，或在新的 runtime 副本中构建。AArch64
runtime 不能由宿主 `/usr/bin/c++` 去链接 ARM 的 `libfmt.so`。

执行方式：

~~~bash
cd ~/Downloads/nn
chmod +x codex_work/convnext_dyt_experiment/mode12/03_stage_mode12_deploy.sh
./codex_work/convnext_dyt_experiment/mode12/03_stage_mode12_deploy.sh "$RUN_ROOT"
find "$RUN_ROOT/board_deploy" -maxdepth 2 -type f -printf '%p %s bytes\n' | sort
~~~

## 6. 同步并运行 mode-12 软件 timer

创建 `codex_work/convnext_dyt_experiment/mode12/04_sync_and_profile_mode12.sh`。
每个变体使用唯一远端目录，trace 只执行一次模型（`1 1`）：

~~~bash
#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then echo "usage: $0 <run-root>" >&2; exit 2; fi
RUN_ROOT="$(realpath "$1")"
RUN_ID="$(basename "$RUN_ROOT")"
SOC_BASE="/mnt/data1/codex_convnext_mode12_${RUN_ID}"
HELPER="$PWD/codex_work/tools/n93x_soc.py"
test -f "$HELPER"
for variant in baseline dyt dyt_nopermute; do
  LOCAL="$RUN_ROOT/board_deploy/${variant}_mode12"
  REMOTE="$SOC_BASE/${variant}_mode12"
  test -d "$LOCAL"
  python3 "$HELPER" -j tb-217 -d "$REMOTE" --sync "$LOCAL" \
    2>&1 | tee "$RUN_ROOT/${variant}_mode12_board_sync.log"
  PROFILE_COMMAND="set -e; export LD_LIBRARY_PATH=/mnt/sys/allspark/lib:\${LD_LIBRARY_PATH-}; mkdir -p nxperf_mode12; cd build_arm; /usr/local/bin/nxperf profile --disable_default_enable -e npu_fw -e npu_art -e npu_sw --stats --ddr_bandwidth_period 99 --ddr_bandwidth_mode NPU -r ${variant}_mode12 -p ../nxperf_mode12 ./test_sample 1 1 ${variant}; cd ..; find nxperf_mode12 -maxdepth 1 -type f -printf '%p %s bytes\\n' | sort"
  python3 "$HELPER" -j tb-217 -d "$REMOTE" --run "$PROFILE_COMMAND" \
    2>&1 | tee "$RUN_ROOT/${variant}_mode12_board_nxperf.log"
done
~~~

日志应该包含 `Sw kernel profiling set to enable`、
`Hardware profiling set to disable`、生成的 zip 路径和 correctness 输出。
`npu_fw/npu_art/npu_sw` 三项要一起打开。mode-12 的时间线有 instrumentation，
不要用它替代干净延迟；同一部署目录应另外运行：

~~~bash
python3 codex_work/tools/n93x_soc.py -j tb-217 \
  -d "$SOC_BASE/baseline_mode12" --run \
  'set -e; export LD_LIBRARY_PATH=/mnt/sys/allspark/lib:${LD_LIBRARY_PATH-}; cd build_arm; ./test_sample 5 30 baseline' \
  2>&1 | tee "$RUN_ROOT/baseline_mode12_board_clean.log"
~~~

对 dyt 和 dyt_nopermute 重复，记录 `BENCH_STATS metric=npu/e2e` 的 p50/p95。

执行方式：

~~~bash
cd ~/Downloads/nn
chmod +x codex_work/convnext_dyt_experiment/mode12/04_sync_and_profile_mode12.sh
./codex_work/convnext_dyt_experiment/mode12/04_sync_and_profile_mode12.sh "$RUN_ROOT"
~~~

该步骤会访问 `tb-217 -> root@192.168.1.20`，并把同步/profile 日志写回
`$RUN_ROOT/*_board_sync.log`、`$RUN_ROOT/*_board_nxperf.log`。

## 7. 在板端解析 nxPerf

创建 `codex_work/convnext_dyt_experiment/mode12/05_report_mode12_on_board.sh`：

~~~bash
#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then echo "usage: $0 <run-root>" >&2; exit 2; fi
RUN_ROOT="$(realpath "$1")"
RUN_ID="$(basename "$RUN_ROOT")"
SOC_BASE="/mnt/data1/codex_convnext_mode12_${RUN_ID}"
HELPER="$PWD/codex_work/tools/n93x_soc.py"
for variant in baseline dyt dyt_nopermute; do
  REMOTE="$SOC_BASE/${variant}_mode12"
  REPORT_COMMAND="set -e; mkdir -p nxperf_report; /usr/local/bin/nxperf-report.py nxperf_mode12/${variant}_mode12.zip nxperf_report .; find nxperf_report/db -maxdepth 1 -type f -printf '%f %s bytes\\n' | sort; find nxperf_mode12 -maxdepth 1 -type f -name '*.nx_repo' -printf '%f %s bytes\\n' | sort"
  python3 "$HELPER" -j tb-217 -d "$REMOTE" --run "$REPORT_COMMAND" \
    2>&1 | tee "$RUN_ROOT/${variant}_mode12_board_report.log"
done
~~~

历史 ResNet50 report 显示 zip 会展开成 `summary/`、`traces/`、`log/`，随后生成
`nxperf_report/db/NPU-HW_swkernel_timeline.db` 和
`nxperf_mode12/<name>_Report.nx_repo`。必须用板端同版本的
`/usr/local/bin/nxperf-report.py`；不同 SDK 版本的 parser 不应混用。

执行方式：

~~~bash
cd ~/Downloads/nn
chmod +x codex_work/convnext_dyt_experiment/mode12/05_report_mode12_on_board.sh
./codex_work/convnext_dyt_experiment/mode12/05_report_mode12_on_board.sh "$RUN_ROOT"
~~~

`n93x_soc.py` 当前没有 `--pull`。建议通过 `--run` 输出 base64，再由宿主写文件。
为此创建 `codex_work/convnext_dyt_experiment/mode12/06_pull_mode12_artifacts.py`，
其职责是：对三个远端文件
`nxperf_mode12/<variant>_mode12.zip`、
`nxperf_mode12/<variant>_mode12_Report.nx_repo`、
`nxperf_report/db/NPU-HW_swkernel_timeline.db` 分别打印唯一 BEGIN/END 标记；
宿主接收标记之间的 base64，写入
`RUN_ROOT/board_trace/<variant>_mode12/`，并在
`board_trace_manifest.json` 记录 bytes、远端路径和 SHA256。实现时可直接复用
`codex_work/notebooks/build_n93x_resnet50_cost_schedule_notebook.py` 中
`pull_remote_artifact` 的实现，但不要把其 notebook cell 当成新的编译结果。

下面给出可直接复制的最小实现（这是 Python，不是 bash 函数）：

~~~python
#!/usr/bin/env python3
import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

if len(sys.argv) != 2:
    raise SystemExit("usage: 06_pull_mode12_artifacts.py <run-root>")
run_root = Path(sys.argv[1]).resolve()
repo_root = Path(__file__).resolve().parents[3]
helper = repo_root / "codex_work/tools/n93x_soc.py"
remote_base = "/mnt/data1/codex_convnext_mode12_{}".format(run_root.name)
manifest = []

for variant in ("baseline", "dyt", "dyt_nopermute"):
    remote_root = "{}/{}_mode12".format(remote_base, variant)
    local_root = run_root / "board_trace" / (variant + "_mode12")
    local_root.mkdir(parents=True, exist_ok=True)
    remote_files = (
        "nxperf_mode12/{}_mode12.zip".format(variant),
        "nxperf_mode12/{}_mode12_Report.nx_repo".format(variant),
        "nxperf_report/db/NPU-HW_swkernel_timeline.db",
    )
    for remote_relative in remote_files:
        tag = hashlib.sha256(remote_relative.encode("utf-8")).hexdigest()[:16]
        begin = "__CODEX_BEGIN_{}_{}__".format(variant, tag)
        end = "__CODEX_END_{}_{}__".format(variant, tag)
        remote_command = "set -e; test -f {0}; echo {1}; base64 {0}; echo {2}".format(
            remote_relative, begin, end
        )
        proc = subprocess.run(
            [sys.executable, str(helper), "-j", "tb-217", "-d", remote_root,
             "--run", remote_command], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, check=False
        )
        if proc.returncode != 0:
            print(proc.stdout, file=sys.stderr)
            raise SystemExit("pull failed: {} {}".format(variant, remote_relative))
        start = proc.stdout.rfind(begin)
        stop = proc.stdout.find(end, start + len(begin))
        if start < 0 or stop < 0:
            raise SystemExit("base64 markers missing")
        payload = "".join(proc.stdout[start + len(begin):stop].split())
        local_path = local_root / Path(remote_relative).name
        local_path.write_bytes(base64.b64decode(payload, validate=True))
        manifest.append({
            "variant": variant, "remote_root": remote_root,
            "remote_relative": remote_relative,
            "local_path": str(local_path.relative_to(repo_root)),
            "bytes": local_path.stat().st_size,
            "sha256": hashlib.sha256(local_path.read_bytes()).hexdigest(),
        })

(run_root / "board_trace_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print(json.dumps(manifest, indent=2, ensure_ascii=False))
~~~

运行：

~~~bash
cd ~/Downloads/nn
chmod +x codex_work/convnext_dyt_experiment/mode12/06_pull_mode12_artifacts.py
python3 codex_work/convnext_dyt_experiment/mode12/06_pull_mode12_artifacts.py \
  "$RUN_ROOT"
~~~

## 8. 解析 SQLite，得到 Cayley/Fermat placement

创建 `codex_work/convnext_dyt_experiment/mode12/07_summarize_mode12_trace.py`。
下面是一个最小、可审计的 Python 3.8 脚本；它不依赖 pandas：

~~~python
#!/usr/bin/env python3
import csv
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

if len(sys.argv) != 3:
    raise SystemExit("usage: 07_summarize_mode12_trace.py <timeline.db> <output-dir>")
db_path = Path(sys.argv[1]).resolve()
out_dir = Path(sys.argv[2]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)
con = sqlite3.connect(str(db_path))
con.row_factory = sqlite3.Row
try:
    rows = con.execute(
        "SELECT r.Id,r.MonitorID,r.Type,r.Start,r.Duration,r.Name,r.Row,"
        "r.ColorValue,r.RelationID,r.RelationLayer,r.OtherMsg,r.OtherUInt,"
        "r.RelationGroup,m.Layer1 AS cluster,m.Layer2 AS core "
        "FROM Record r LEFT JOIN Monitor m ON r.MonitorID=m.MonitorID "
        "ORDER BY r.Start,r.MonitorID"
    ).fetchall()
finally:
    con.close()

fields = ["Id","MonitorID","Type","start_ns","duration_ns","kernel","Row",
          "ColorValue","RelationID","RelationLayer","OtherMsg","OtherUInt",
          "RelationGroup","cluster","core","engine","core_location"]
core_rows = []
missing_relation = 0
for row in rows:
    core = str(row["core"] or "")
    if core.startswith("Cayley core"):
        engine = "Cayley"
    elif core.startswith("Fermat core"):
        engine = "Fermat"
    else:
        engine = "task aggregate"
    if not row["RelationID"]:
        missing_relation += 1
    if engine not in ("Cayley", "Fermat"):
        continue
    cluster = str(row["cluster"] or "")
    core_rows.append({
        "Id": row["Id"], "MonitorID": row["MonitorID"], "Type": row["Type"],
        "start_ns": row["Start"], "duration_ns": row["Duration"],
        "kernel": row["Name"], "Row": row["Row"], "ColorValue": row["ColorValue"],
        "RelationID": row["RelationID"], "RelationLayer": row["RelationLayer"],
        "OtherMsg": row["OtherMsg"], "OtherUInt": row["OtherUInt"],
        "RelationGroup": row["RelationGroup"], "cluster": cluster, "core": core,
        "engine": engine, "core_location": cluster + " / " + core,
    })

with (out_dir / "mode12_core_placement.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(core_rows)

groups = defaultdict(list)
for item in core_rows:
    groups[(item["engine"], item["kernel"])].append(item)
span_fields = ["engine","kernel","slice_records","clusters_used","core_slots_used",
               "first_start_ns","last_end_ns","observed_wall_span_ns"]
span_rows = []
for (engine, kernel), group in groups.items():
    starts = [int(item["start_ns"]) for item in group]
    ends = [int(item["start_ns"]) + int(item["duration_ns"]) for item in group]
    span_rows.append({
        "engine": engine, "kernel": kernel, "slice_records": len(group),
        "clusters_used": len(set(item["cluster"] for item in group)),
        "core_slots_used": len(set(item["core_location"] for item in group)),
        "first_start_ns": min(starts), "last_end_ns": max(ends),
        "observed_wall_span_ns": max(ends) - min(starts),
    })
span_rows.sort(key=lambda item: item["observed_wall_span_ns"], reverse=True)
with (out_dir / "mode12_kernel_span_summary.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=span_fields)
    writer.writeheader()
    writer.writerows(span_rows)

summary = {"database": str(db_path), "all_record_rows": len(rows),
           "core_slice_rows": len(core_rows),
           "relation_id_missing_rows": missing_relation,
           "caveat": "slice durations are instrumented and parallel; do not sum as E2E"}
(out_dir / "mode12_trace_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
~~~

运行：

~~~bash
cd ~/Downloads/nn
for variant in baseline dyt dyt_nopermute; do
  TRACE_DIR="$RUN_ROOT/board_trace/${variant}_mode12"
  python3 codex_work/convnext_dyt_experiment/mode12/07_summarize_mode12_trace.py \
    "$TRACE_DIR/NPU-HW_swkernel_timeline.db" "$TRACE_DIR/summary"
done
~~~

`mode12_core_placement.csv` 的一行是一个 core slice，不是整个 ONNX 节点；
`mode12_kernel_span_summary.csv` 用同一 kernel 最早开始到最晚结束的 wall span，
较适合观察并行窗口。`relation_id_missing_rows` 大于 0 时，不能仅凭 SQLite 恢复
producer→consumer 的边界关系。

## 9. 需要硬件 Grid 时，另开 mode-15

mode-12 预期日志是 `Hardware profiling set to disable`。硬件 Grid 必须重新编译
`profiling_mode=15` 的 AOM/AP，不能事后给 mode-12 命令追加参数。复制：

~~~text
codex_work/convnext_dyt_experiment/compile_config_mode12.json
 -> codex_work/convnext_dyt_experiment/mode12/compile_config_mode15.json
~~~

把 JSON 的 `"profiling_mode": "12"` 改成 `"profiling_mode": "15"`，再复制
`01_compile_mode12.sh` 为 `01_compile_mode15.sh`，同步修改目标目录、配置路径和
model name 的 mode 后缀；mode-12 和 mode-15 不共用 output/deploy 目录。

当前静态 Grid 行数指纹为 baseline=1086、dyt=768、dyt_nopermute=799；这些行含
marker/prefetch/invalid，不能当精确容量。可先用 2048，若新的 Grid info 或手册
提示不足则提高到不超过 4095。过小可能覆盖缓冲、挂起或触发 reboot，不能盲用 512。

板端 `[NPU板端]` one-shot 示例：

~~~bash
set -e
mkdir -p nxperf_mode15_grid
cd build_arm
/usr/local/bin/nxperf profile --disable_default_enable \
  -e npu_fw -e npu_art -e npu_sw --npu_hw_grid 2048 --block_mode 1 \
  --ddr_bandwidth_period 99 --ddr_bandwidth_mode NPU --stats \
  -r baseline_mode15_grid -p ../nxperf_mode15_grid ./test_sample 1 1 baseline
~~~

`--npu_hw_grid` 不能和 `--npu_hw_freq` 或 `duration` 同用；一次只下发一个 AOM。
Grid run 只做结构和硬件事件检查，不把它的时间当产品 latency。

## 10. 预期文件树和字段

每个变体完成后应有：

~~~text
<RUN_ROOT>/<variant>_mode12/
  compile.log  compile.resource.txt  compile_config.json  return_code.txt
  output/__aom_0.aom  output/<variant>.ap
  build___aom_0/__aom_0_model_.onnx
  build___aom_0/__aom_0_model_after_OpFusion.onnx
  build___aom_0/allspark_build__graph___data.onnx
  build___aom_0/allspark_grid_info.txt
  build___aom_0/perf_summary.log  tensor.json  anchor.json
  build___aom_0/fusion_result___graph__.json
<RUN_ROOT>/board_trace/<variant>_mode12/
  <variant>_mode12.zip  <variant>_mode12_Report.nx_repo
  NPU-HW_swkernel_timeline.db  summary/mode12_core_placement.csv
  summary/mode12_kernel_span_summary.csv  summary/mode12_trace_summary.json
~~~

最终汇总表至少记录：`variant`、compile return code/seconds、Grid 行数、zip/DB
SHA256、总 Record 行数、Cayley/Fermat slice 行数、clean **无 nxPerf** 的 NPU/E2E
p50/p95 和 DDR 窗口平均/峰值。只有以 mode-0 AOM 运行的 clean 数据才是产品级
latency；mode-12 AOM 的 clean 数据仍带编译期 profiling instrumentation，只可作调度
诊断对照。
## 11. estimate cost 与真实 E2E 的关系

`compile.log`/`perf_summary.log` 中的 `fermat=...`、`cayley=...`、
`sequence_time_weight`、`cayley_fermat_paral_weight`、`overlap_weight`、
`est_weight` 属于编译器 cost model 或计划搜索的输入/汇总。它们影响
OpFusion、cache release、cluster assignment 和 Cayley/Fermat 并行方案，但不是板端
某一次执行的纳秒值。`allspark_grid_info.txt` 行数也不是 latency。

SQLite `Record.Start`/`Duration` 和 `.nx_repo` 是实际运行时（带软件 timer）的
slice 时间。由于 Cayley/Fermat 重叠、同一 kernel 可能切到多个 core、以及 runtime
enqueue/synchronization、H2D/D2H 和 host 逻辑，正确的证据链是：

~~~text
compiler estimate  -> 影响调度/tiling/cache 选择
mode-12 slice ns   -> 解释实际 core、并行窗口和热点
mode-0 BENCH_STATS -> 产品级端到端时间
~~~

如果 mode-12 的 slice 看起来很短而 E2E 仍慢，应检查 host/runtime、同步、输入输出
搬运、预处理/后处理和 DDR 窗口；不要把 `est_weight` 当成“预测的毫秒值”。

## 12. 如何解释 ConvNeXt baseline、DyT 和 no-permute

当前 mode-0 静态结果已经显示：baseline 的 OpFusion 后 `ace.permute` 较多，dyt 和
dyt_nopermute 较少；no-permute 还同时改变 MatMul→Conv rewrite、Grid/task 数量和
cache 规划。因此更合理的推断是：backend 在 lowering/fusion 阶段消除了部分显式
Transpose 的物理 materialization，但这不等价于源 ONNX 节点被删除，也不能由静态
图单独推出 layout kernel 的 runtime 时间。

完成本指南后，按 kernel 名称、`engine`、`core_location`、wall span 和 DDR 窗口
交叉比对三个变体。若 dyt 的 permutation 仍存在但被融合到 Cayley kernel，trace
可能表现为 layout 独立 slice 减少、Cayley slice duration 改变，而不是 ONNX
Transpose 数量变化。若 clean E2E 与 mode-12 slice 的差距明显，则优先查找
跨 cluster 依赖、临时 tensor spill、runtime fence 和 host copy。

## 13. 常见失败与恢复

| 现象 | 首先检查 | 处理 |
|---|---|---|
| 日志出现 `profiling_mode ... default value: 0` | 变体目录的 JSON 和 `compile.log` | 开新 run，确认字符串值为 `12` |
| mode-12 没有 zip | `npu_fw/npu_art/npu_sw` 是否同时启用、是否从 `build_arm` 运行 | 保留完整 log，不把空目录当 trace |
| report parser/DB 错误 | zip 与板端 parser 是否同一 SDK | 在板端解析后再拉回，记录 parser 版本 |
| SQLite 只有 task aggregate | `Monitor.Layer2`、`Record` 行数和 parser warning | 不能填写 core placement；重跑并保留 warning |
| mode-15 hang/reboot | profiling=15、GridEntryNum、是否 one-shot | 停止重复尝试，增大容量或回到 mode-12 |
| `file in wrong format` | AArch64 runtime、linker 和 `libfmt.so` 架构 | 用 ACPU 交叉编译器，不能用 host c++ 混链 |
| `refusing stale build directory` | CMakeCache/旧 build 目录 | 安全改名或使用新 build 目录，不盲目递归删除 |

## 14. 历史 ResNet50 mode-12 sanity check

工作区已有：

~~~text
codex_work/n93x_resnet50_cost_schedule/runs/20260831T041714Z_1288616/
~~~

该 run 的 config 为 `profiling_mode=12`，板端日志明确写出：

~~~text
Sw kernel profiling set to enable.
Hardware profiling set to disable.
.../resnet50_mode12.zip (88.34KB)
~~~

解析后有 `NPU-HW_swkernel_timeline.db`、`resnet50_mode12_Report.nx_repo`、
`nxperf_core_placement.csv` 和 `nxperf_kernel_span_summary.csv`。其中 placement
CSV 有 4046 条明确 core slice（Cayley 3566、Fermat 480），`Monitor.Layer2` 形如
`Cayley core 1` / `Fermat core ...`，证明当前环境的解析路径可用。该 run 的
relation/aggregate 有告警，只作为 schema sanity check，不能替代 ConvNeXt 新测量。

完成 ConvNeXt 后在本文件末尾追加新 run id、zip/DB SHA256、三种变体的 placement
计数、热点 kernel 表和 clean mode-0 E2E；不要覆盖 mode-0 记录或把 instrumented
duration 直接写成产品延迟。

## 15. 20260901T083738Z_mode12 实际审计、板端运行与 nxPerf 解析

### 15.1 结论与证据等级

本节对应的 run 是
`codex_work/convnext_dyt_experiment/runs/20260901T083738Z_mode12`，远端目录是
`/mnt/data1/codex_convnext_mode12_20260901T083738Z_mode12`。三份 mode-12 AOM 都
成功编译、完整同步、可在 N93X 上加载；nxPerf 的软件 kernel timer 和 report parser
均实际运行成功。因此 **Cayley/Fermat placement、slice 数及命名 layout kernel 的归属
是 E2（实际板端、来源可核验）证据**。

但是三个变体的 `test_sample` 都返回 `2`，因为 FP16 golden 的
`element_pass_rate` 小于 runtime 的 `0.98` 阈值。于是下文的 p50 以及 mode-12
trace **仅是同一输入下的诊断性调度观察**，不是“正确模型的产品性能”或“DyT 等价且
加速”的结论。下一轮性能 Gate 必须先修复 golden/语义一致性问题，再用 mode-0 AOM
重测。

### 15.2 编译与本地部署 Gate

| 变体 | `return_code` | `profiling_mode` | 编译耗时 (s) | Grid 行数 | AOM 静态内存 (MB) | `TmpTensor / CL2 / GL2` (MB) |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0 | 12 | 3521.930 | 1086 | 1160.85 | 14.0215 / 11.4297 / 2.5918 |
| dyt | 0 | 12 | 423.984 | 768 | 1156.05 | 9.81055 / 9.13281 / 0.677734 |
| dyt_nopermute | 0 | 12 | 586.680 | 799 | 1156.30 | 9.86133 / 9.57031 / 0.291016 |

三个 `compile_config.json` 都有 `"profiling_mode": "12"`；没有出现
`profiling_mode is not found ... default value: 0`。日志中的
`default_min_dim_value ... default value: 0` 是另一个配置项，不能误判为 profiling
回退。每个变体恰有一个 `.aom`、一个 `.ap` 和非空的
`build___aom_0/allspark_grid_info.txt`；板端同步工具再次校验了 8 个普通文件的
SHA256。runtime 的 `file` 输出是 `ELF 64-bit ... ARM aarch64`。

首次运行 `03_stage_mode12_deploy.sh` 时发现其旧版清单命令
`sha256sum -- * build_arm/*` 把 `build_arm` 目录当作文件，因而非零退出。失败的半包
保存在 `board_deploy_stage_failed_20260901T103121Z/`，没有删除；脚本和本指南已改为
只遍历普通文件、且不哈希 `deploy.sha256` 自身。重试的完整部署包在
`board_deploy/`，其本地 `sha256sum -c deploy.sha256` 和同步器的 SoC 端复核均通过。

### 15.3 后端图变化：不是源 ONNX 节点数的直接翻版

下表由 compiler container 中的 ONNX parser 读取本 run 的三个 backend snapshot。
`layout` 是 `ace.permute` 的计数；这些快照已处在 AllSpark 前端/lowering 流程中，
不是原始导出 ONNX。

| 变体 | `__aom_0_model_.onnx`（节点 / layout） | OpFusion 后（节点 / layout） | `allspark_build__graph___data.onnx`（节点 / layout） |
|---|---:|---:|---:|
| baseline | 1557 / 152 | 772 / 77 | 371 / 89 |
| dyt | 1291 / 152 | 509 / 7 | 252 / 7 |
| dyt_nopermute | 1363 / 224 | 539 / 7 | 284 / 7 |

这证明 backend fusion/lowering 在 DyT 两个变体中将大量独立 layout 节点融合、消除或
重写了；也说明“源 ONNX 已手删 permute”不等于第一个 AOM snapshot 不会再出现内部
layout。反过来，final graph 的 7 个 `ace.permute` 也不代表只有 7 次物理搬运：一个
kernel 还可能在 tile、cluster 或 core 上展开为许多 slice。

### 15.4 静态 estimate 与实际诊断时间并非线性关系

`perf_summary.log` 给出 `fermat=...`、`cayley=...`，其单位在公开输出中未标注，
故以下称为 **compiler cost unit**，不写成 cycle 或毫秒。编译日志同时显示该计划器
启用了 `sequence_time_weight=1`、`cayley_fermat_paral_weight=10`、
`overlap_weight=1`、`est_weight=-30`，以及 cache/cluster assignment 权重；这足以
证明它们参与调度搜索，不能证明它们是逐 kernel 的实测时间。

| 变体 | Fermat cost | Cayley cost | cost 总和 | 相对 baseline | 无 nxPerf、但 mode-12 AOM 的 NPU p50 (ms) | 相对 baseline |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 313137 | 4469959 | 4783096 | — | 5.4280 | — |
| dyt | 180441 | 4462982 | 4643423 | -2.92% | 5.1290 | -5.51% |
| dyt_nopermute | 217050 | 4335608 | 4552658 | -4.82% | 3.4890 | -35.72% |

尤其是 no-permute 的静态总 cost 仅低 4.82%，诊断性 NPU p50 却低 35.72%。这不是
cost model “错误”的充分证据：三种图并不已经证明语义等价，且 scalar cost 会漏掉
critical path、Cayley/Fermat 并行、cache/spill、runtime fence 和 host H2D/D2H。
它只适合解释编译器为什么偏好某个 schedule；不能校准成 E2E 毫秒预测器。

### 15.5 板端 clean run：运行成功，正确性 Gate 未通过

运行命令使用 `5` 次 warm-up 和 `30` 次测量，并显式设置
`LD_LIBRARY_PATH=/mnt/sys/allspark/lib:${LD_LIBRARY_PATH-}`。这一组运行没有启动
nxPerf，但 AOM 仍由 `profiling_mode=12` 编译，所以不能替代 mode-0 的产品 latency。

| 变体 | NPU p50 / p95 (ms) | E2E p50 / p95 (ms) | `element_pass_rate` | runtime 返回码 | Gate |
|---|---:|---:|---:|---:|---|
| baseline | 5.4280 / 5.4420 | 5.6676 / 5.6823 | 0.004 | 2 | fail |
| dyt | 5.1290 / 5.1450 | 5.3906 / 5.4080 | 0.235 | 2 | fail |
| dyt_nopermute | 3.4890 / 3.5090 | 3.7604 / 3.7863 | 0.229 | 2 | fail |

这一步已排除“未加载 AOM / 动态库错误 / 输入文件缺失”这些部署问题，但没有定位到
golden 不一致的根因。可能是 golden 与模型/输出 tensor 不配对、变换本身改变数值，
或 runtime 与参考实现的前后处理/容差定义不一致；现有日志不足以在三者之间归因。
所以不能把该表用在模型选型决策里。

### 15.6 mode-12 placement：layout 命名 kernel 实际落在 Fermat

三次 profile 都使用：

~~~text
nxperf profile --disable_default_enable -e npu_fw -e npu_art -e npu_sw \
  --stats --ddr_bandwidth_period 99 --ddr_bandwidth_mode NPU ...
~~~

日志明确包含 `Sw kernel profiling set to enable` 与
`Hardware profiling set to disable`，并各自产生 zip、`*_Report.nx_repo` 和
`NPU-HW_swkernel_timeline.db`。以下是从 SQLite `Record LEFT JOIN Monitor` 得到的
实际 core slice；`layout-named` 以 kernel 名中含 `permute`、`transpose` 或 `layout`
的保守规则统计。

| 变体 | Record / core slice | Cayley slice（唯一 kernel） | Fermat slice（唯一 kernel） | layout-named slice | 相对 baseline layout slice |
|---|---:|---:|---:|---:|---:|
| baseline | 9238 / 9226 | 2394 (151) | 6832 (218) | 2848 | — |
| dyt | 5688 / 5680 | 2320 (145) | 3360 (105) | 288 | -89.89% |
| dyt_nopermute | 6691 / 6683 | 2331 (146) | 4352 (136) | 224 | -92.13% |

本次 trace 中的活跃 topology 是 **2 个 NPU cluster，每 cluster 观测到 4 个
Cayley core 和 8 个 Fermat core**；这是本 AOM/profile 的活跃 slot，不宣称是芯片的
物理总核数。baseline 的 2848 个 layout-named slice、DyT 的 288 个和 no-permute 的
224 个均位于 Fermat。具体例子包括：

- baseline：`__Conv_113_Conv_4_permute_input__...`、
  `__Conv_0_Conv_0_permuted_output__...`；
- dyt：`__LayoutTransform_AlterLayout_832_cut_...`、
  `__Transpose_732_Transpose_67__...`。

同一个分块 layout kernel 可在 `2 cluster × 8 Fermat core = 16` 个 core slot、并因
两次执行或多个 tile 形成 32 个 slice。因此 slice 数减少是很强的“独立 layout 工作
减少”证据，却不是该工作在一帧 critical path 上减少了相同次数或相同纳秒数的证明。
例如 dyt_nopermute 的 Fermat slice 比 dyt 多，但其诊断性 p50 更低，正好说明不能按
slice 数、也不能将 `Duration` 求和来解释 E2E。

### 15.7 trace 能回答与不能回答的问题

能回答：特定 kernel slice 实际显示在哪个 `Monitor.Layer2`（Cayley/Fermat core）、
使用哪些 cluster/core slot、以及同名 kernel 的观测 wall span。拉回后的可复查产物为：

| 变体 | zip SHA256 | software-kernel DB SHA256 |
|---|---|---|
| baseline | `98cf5a616a4060f474d9813ac8ff380e6aeb3bcd51cd9c8dbdfa70a57203cdbc` | `115b9d6629319f84d8e0f40a15fff2cb3bc1abf5a4736325ec7a5d2da7917186` |
| dyt | `8b6bab3cb3bcb31d0d13cff003517a4259b36d1e2125e7411dabba497e3d8f73` | `dae2497bd66df5a9547cb6414a1884409831f4ffc6d92b883defa98353538f53` |
| dyt_nopermute | `1eee27e56041c0bba04e4a7b3fdbe03a425bc0fb8d422a886f643bd711a366a8` | `a33df38db4e82c2b1641ec396c9810d565c8403c73f3e346c8edbd98d0090ff1` |

不能回答：本 run 的所有 `Record.RelationID` 都为空，且三个 parser log 都有
`miss fw trace, sw task id: 1` 及 `rel id 3 not found in art trace`。因此不能从这些
SQLite 表恢复 producer→consumer 的边，**更不能声称 Cayley→Fermat（或反向）某条
tensor 必然写回 DDR，或给出该边的字节数**。`--ddr_bandwidth_period 99` 只给采样窗口
配置，不是每条 edge 的 transaction counter；本次没有采集可用的窗口数值。若这个问题
是目标，应先完成正确性 Gate，再以独立 `profiling_mode=15` one-shot、经容量核对的
Grid 和可用的 DDR 窗口数据补证，仍需避免把 Grid 运行时间当产品 latency。

### 15.8 可复查文件、已知限制和唯一下一步

| 目的 | 本 run 的证据文件 |
|---|---|
| 编译配置、cost、内存、Grid | `<variant>_mode12/compile_config.json`、`compile.log`、`build___aom_0/perf_summary.log`、`allspark_grid_info.txt` |
| 后端图 | `<variant>_mode12/build___aom_0/{__aom_0_model_.onnx,__aom_0_model_after_OpFusion.onnx,allspark_build__graph___data.onnx}` |
| 同步与板端完整性 | `*_board_sync.log`、`board_deploy/<variant>_mode12/deploy.sha256` |
| clean 运行和正确性 | `*_board_clean.log` |
| nxPerf 配置与 instrumentation | `*_board_nxperf.log`、`*_board_report.log` |
| 拉回原件与 digest | `board_trace_manifest.json`、`board_trace_pull.log` |
| host SQLite 解析 | `board_trace/<variant>_mode12/summary/mode12_core_placement.csv`、`mode12_kernel_span_summary.csv`、`mode12_trace_summary.json` |

当前 phase：**mode-12 编译、部署、runtime、nxPerf 软件 trace、report、拉回及解析均完成；
模型正确性 Gate 未通过。** 缺失证据：与各变体匹配、可通过阈值的 golden，以及能关联
kernel producer/consumer 与 DDR 的硬件/窗口证据。唯一下一步是先在同一 `input.bin` 上
逐个比对 ORT FP16 golden、AOM 输出和输出 tensor 名/shape，修复或解释不一致；通过后再
重新编译/运行 mode-0 的 30+ 次产品延迟，并把 mode-12 的 placement 作为解释证据。

## 16. mode-12 深入问答：从已观测证据到架构推断

### 16.1 你的问题（原文）

~~~text
1. ace.permute是否是allspark的私有backent permute算子？能否看到算子实现？如果不能看到，我估计是运行在fermat(擅长向量计算，矩阵重排)核中的。
2. “trace 可能表现为 layout 独立 slice 减少、Cayley slice duration 改变” 这句话应该怎么理解？为我解释下slice是什么？cayley slice duration又是什么？从原理将其说明怎么表现出来的。
3. 一直说的grid行数是什么意思？和它同级和强相关的概念都有哪些也为我说明下。
4. 静态内存是否是文件大小？TmpTensor和CL2与GL2各自都是什么？特性是怎样的？一般怎么样的数据可以认为是更优秀的？这几个概念和Nvidia的内存架构又有什么关系？
5. layout和ace.permute的关系？为什么我一开始dyt_nopermute的layout还比baseline model更多？
6. 文中“final graph 的 7 个 ace.permute 也不代表只有 7 次物理搬运：一个

   kernel 还可能在 tile、cluster 或 core 上展开为许多 slice。”怎么理解？
7. 怎么理解“且 scalar cost 会漏掉

   critical path、Cayley/Fermat 并行、cache/spill、runtime fence 和 host H2D/D2H。

   它只适合解释编译器为什么偏好某个 schedule；不能校准成 E2E 毫秒预测器。”
8. 如何理解一下表格中的信息

| 变体Record       | core slice  | Cayley slice（唯一 kernel） | Fermat slice（唯一 kernel） | layout‑named slice | 相对 baseline layout slice |   |
| -------------- | ----------- | ----------------------- | ----------------------- | ------------------ | ------------------------ | - |
| baseline       | 9238 / 9226 | 2394 (151)              | 6832 (218)              | 2848               | —                        |   |
| dyt            | 5688 / 5680 | 2320 (145)              | 3360 (105)              | 288                | -89.89%                  |   |
| dyt_nopermute | 6691 / 6683 | 2331 (146)              | 4352 (136)              | 224                | -92.13%                  |   |
9. 目前观测到named layout kernel都被调度到了fermat core中，那么还有其他kernel被调度到了fermat core中吗？在什么日志或者信息文件能看到？
10. “同一个分块 layout kernel 可在 2 cluster × 8 Fermat core = 16 个 core slot”的意思是一个kernel在多个core并行工作吗？是不是类似nvidia的warp概念？如何理解“同一个分块 layout kernel 可在 2 cluster × 8 Fermat core = 16 个 core slot”
11. wall span是什么？和kernel的关系？

1. Hauk TIR DSL是什么？
2. work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/ace/kernels/fermat/permute.py这么一大堆代码是codegen的吗？是通过什么codegen的？还是开源工具改的？为我剥离冗余信息，用第一性原理分析这个代码解决什么问题的？怎么和其他工具或者代码来将其调度到指定设备中的？和什么工具调度？这个工具是tvm的什么扩展机制实现的吗？是否还有c++代码？
3. 从哪里体现出“T.matrix 在这里是 DSL 的 tile 数据容器，不能仅因

   出现 matrix 就断言改跑 Cayley。”的？
4. `work/allspark_release/AllSpark_Release/tmp/allspark/python/allspark/ace/aopi/json/N93X/permute.json` 中又有那些重要讯息？它的读取和使用的模块是谁？怎么用的？`ND / P1ND_128B / SD`又是啥意思？"physical_layout_transform.json 是相关但不同的物理 layout 变换 op。"又是啥意思？
5. 我不知道底层指令集我怎么能写kernel?假设我现在要实现一个toy kernel，根据当前信息我能知道什么？我通过这些信息怎么来写一个kernel呢？又要怎么验证呢？
6. slice是把一个kernel切成更小的kernel来调度到不同核心中运行吗？为我尽量用实际例子举例说明下，比如说明某块数据的位置，kernel会怎么处理数据？是否会runtime再选择核心再执行？ 以及"外部

   materialized transpose 可能消失，但 consumer 的 indexing/coalescing 和 tile 选择会变。"这个场景也举例说明》
7. 如何理解task/grid指令？我看fermat很多并行运算和内存概念都参考了nvidia进行设计，nvidia这些概念我也都不熟，所以遇到了也为我解释下。梳理出nvidia和当前NPU的内存模型到底是怎么样的，有什么相似和区别，以及哪些内存是我们说的显存，cpu ddr，还有哪些内存访存满，使我们需要避免的？哪些是软件开发者能控制的？怎么控制的？
8. CCA是什么也解释下。
9. physical_layout_transform和ace.permute的区别？以及底层原理是怎样的？以及"被 producer/consumer 的索引规则吸收，成为 view/metadata，或融合进

   Conv，而完全没有独立 ace.permute"应该怎么理解？举个例子说明下。
10. “Conv/Cayley consumer 要求某个 physical layout，而 DyT 替换后 tensor 的生产者边界”这里consumer是怎么提出要求的？是否是硬编码到算子的输入要求变量中？举例说明
11. "实际 trace 的 layout-named slice 则为 baseline/DyT/no-permute 的

    2,848 / 288 / 224，"中，是不是7的permute数量被slice称224个小的tile 版本的permute来计算呢？
12. 32 个 Fermat slice、使用

    16 个不同 core slot（2 cluster × 8 Fermat core）是否意味着32个slice，16个core所以需要2次内存搬运？
13. named layout才是真正执行了内存变换的layout吗？既然有named layout slice,那么有没有匿名layout slice?
14. "一个 kernel 的 grid/block 工作被“横向铺开”到 16 个 core；每个

    core 内部再由 warp/lane 纵向并行。它近似 CUDA 的“一个 grid 的多个 thread block 同时

    跑在多个 SM”，而不是“一个 kernel 只有一个 warp”。"举个微观点的例子说明。
15. fermat和cayley一般也会有同步并行的关系吗？如果两个核心都在工作，其中一个工作完了另一个没结束，是否完成早的核心需要等待呢？？
~~~

### 16.2 先建立证据边界

一句话心智模型：**把 ONNX 节点看成算法语义，把 ACE kernel 看成已选定的实现，把
mode-12 `slice` 看成该实现落到一个 core 时间线上的一次工作片段。它们不是同一个
计数单位。**

以下回答刻意分三层，防止“合理类比”被误读为芯片已公开的事实。

| 标记 | 本章含义 | 本次可复查来源 |
|---|---|---|
| **E1** | AllSpark 文档直接写明的编程/运行时语义 | `docs/spark_guide_140/AllSpark CCA 高性能编程指南.md`、`AllSpark CCA 开发手册.md`、`AllSpark 入门手册.md`、`AllSpark Engine 性能优化手册.md` |
| **E2** | 当前 SDK 文件或本次 ConvNeXt mode-12 工件直接观察到的事实 | SDK 中的 `ace/kernels/fermat/permute.py`；本 run 的 `compile.log`、`NPU-HW_swkernel_timeline.db`、`summary/*.csv` |
| **E0** | 基于 M1/Metal、CUDA 或常见 NPU 编译器的实现推断 | 明确以“推测/类比”表述，不能替代 N93X 计数器或 ISA 证据 |

尤其要记住：本 run 三个变体的 numerical pass rate 均未通过（见 15.5）。所以以下
placement、slice 与内存结论可用于**理解编译器行为**，不能用来宣称哪一个模型已经在
真实业务上更快或更正确。

### 16.3 1. `ace.permute` 是什么，能否看到实现？

结论：`ace.permute` 不是 ONNX 标准 op，而是 AllSpark SDK 的 **ACE 后端/方言算子**。
这个 SDK 已提供可读的 Hauk TIR DSL 实现；它不是完整的最终机器码实现。对于这次
N93X 配置和本模型中仍独立存在的 layout/permute kernel，E2 证据支持其由 **Fermat**
执行。

可复查的 SDK 文件在容器中：

~~~bash
docker exec -it codex_allspark_compiler bash
sed -n '1,420p' /usr/local/allspark/python/allspark/ace/kernels/fermat/permute.py
sed -n '1,300p' /usr/local/allspark/python/allspark/ace/kernels/fermat/physical_layout_transform.py
sed -n '1,220p' /usr/local/allspark/python/allspark/ace/aopi/json/N93X/permute.json
~~~

这不是仅从目录名猜测：

- `permute.py` 导入 `hauk.script.tir as T`，其主路径按输出逻辑轴算输入逻辑轴，随后
  用 `T.gather` / `T.scatter` 或连续 `T.read` / `T.write` 完成搬运；尾块用向量 predicate
  mask 保护。也就是典型的“地址计算 + 向量 load/store”，不是 MAC 阵列乘加。
- 它显式构造 `T.vector(...)`，`permute_not_last_dim_matrix` 与
  `shuffle_trans_general` 还在 tile 内做 matrix/vector shuffle。这解释了它为何适合
  Fermat 的 SIMT/向量编程模型；`T.matrix` 在这里是 DSL 的 tile 数据容器，**不能仅因
  出现 `matrix` 就断言改跑 Cayley**。
- `ace/aopi/json/N93X/permute.json` 公开了该 op 的可行 dtype 与
  `ND` / `P1ND_128B` / `SD` physical-layout 组合；同目录
  `physical_layout_transform.json` 是相关但不同的物理 layout 变换 op。
- 本 run 的三个 SQLite 中，按 kernel 名含 `permute`、`transpose`、`layout` 的
  3,360 个 slice（2,848 + 288 + 224）都关联到 `Monitor.Layer2 = Fermat core *`。更早
  的编译工件也有 `ace_fermat_kernel_builder` 处理 permute 的记录。

仍看不到的是：Hauk TIR 如何进一步 lower 成 Fermat ISA、具体 tile-search/cost model、
DMA 描述符和硬件微码。容器中 `/mnt/allspark/ace/src` 不存在，SDK 只分发了 Python
模板/元数据和二进制工具。因此严谨说法是：**可见算法级 kernel 模板，不能审计最终
指令级实现。** 也不能由此推导“所有 `ace.permute` 在所有 shape/版本下必定 Fermat”；
它可能被融合、消除，或随目标/约束选另一种实现。

#### 16.3.1 Hauk TIR DSL：不是 Python 逐元素执行，而是“用 Python 写 IR”

官方《AllSpark 入门手册》把 HAUK 称为“**AllSpark 算子编程语言**”，用于内置算子库。
`Hauk TIR DSL` 可以先按下面一句话理解：

> 你写的是看起来像 Python 的 kernel **规格/模板**；编译器读取它的 AST，把 `T.read`、
> `T.range`、`T.vector` 等翻译成 Tensor IR，再为 Fermat 或 Cayley 生成设备目标代码。它
> 不是 Python 解释器在 NPU 上逐元素跑循环。

这里的 `T` 是 `hauk.script.tir` 的 DSL 命名空间，而不是 PyTorch Tensor：

| DSL 对象 | 它在作者脑中的实物 | 它不是 |
|---|---|---|
| `T.tensor` | kernel 参数中的 device buffer，带 shape/dtype/layout 元数据 | 一块已经由 Python 装满的 `torch.Tensor` |
| `T.scalar` / `T.vector` / `T.matrix` | IR 里的标量、向量或局部 tile 值 | “自动选择 Fermat/Cayley”的标签 |
| `T.range(..., binding=...)` | 循环/并行工作域和 index 绑定 | Python `range` 在 host 上把循环真的跑完 |
| `T.read` / `T.write` / `T.gather` / `T.scatter` | 将来设备 load/store 的 IR 节点 | 当场从 CPU 内存读写数据 |
| `@template` | 编译时常量的 specialization 参数，例如 dtype、轴映射、tile size | runtime 的动态 Python `if` |
| `@func_register` | 把 Python 函数登记为可解析的 DSL function | 将 Python 函数直接作为 runtime callback |

这是 E2、可从 SDK 源码逐层核对的路径：

~~~text
permute.py 的 Python DSL
  -> Hauk AST parser (`HAUKScriptParser`, 继承 `TVMScriptParser`)
  -> Front IR（代码注释称 FIR；缩写全称未见公开定义）
  -> `driver_ffi.fir_to_hauk_tir(...)`
  -> Hauk TIR / IRModule
  -> `hauk_lower_*` 和 `hauk_tir_to_runtime(...)`
  -> target-specific intrinsic / `.o`（debug dump 时还有 `.ll`、`.s`）
  -> AllSpark task/kernel descriptor -> ART/runtime 发射
~~~

可直接在容器检视这条链，而不用猜名称：

~~~bash
docker exec -it codex_allspark_compiler bash
nl -ba /usr/local/allspark/python/allspark/hauk/script/parser.py | sed -n '220,235p;1570,1680p'
nl -ba /usr/local/allspark/python/allspark/hauk/driver/build_module.py | sed -n '206,242p;381,465p;833,862p'
~~~

第一段会看到 `HAUKScriptParser(TVMScriptParser)`；第二段会看到
`fir_to_hauk_tir`、`hauk_lower_*`、`hauk_tir_to_runtime`，以及官方源码注释的
`python kernel -> new intrinsic -> .o file`。这也给出“是否和 TVM 有关”的精确答案：
**Hauk/`atvm` 的 IR 和 parser 明确借用了/改造了 TVM Script、TIR、IRModule、Target、
PassContext 这套扩展点；Fermat/Cayley lower、intrinsic 与 runtime backend 则是私有扩展。**
仅凭 `atvm` 不能断言使用了某个公开 TVM commit，更不能把所有私有 pass 当成上游 TVM。

用最小的 SSA/define-use 心智模型看图优化很有帮助。下面是**示意 IR，不是当前 SDK
语法**：

~~~mlir
%p = "ace.permute"(%x) {dims = [0, 2, 3, 1]}
     : (tensor<1xC×H×Wxf16>) -> tensor<1xH×W×Cxf16>
%y = "ace.nn.conv2d"(%p, %w) : (...) -> tensor<...>
~~~

`%p` 只被定义一次（SSA 的 single assignment），而 `conv2d` 是它的 user。若后端证明
Conv 能以 `%x` 的 storage/layout 直接读取，就可重写为一个带 layout 属性的 Conv，并删除
`%p`。MLIR 把这种事抽象为“conversion target + rewrite pattern + 可选 type converter”；
TVM 则常在 Relay/Relax 的算子策略、layout rewrite、TIR schedule/lower 中完成。这里的
“def-use”不是玄学：就是从一个 value 的 producer 找到所有 consumer，再判断改 producer
还是改 consumer 的接口能否保持语义。

如果要让代码助手只做证据化 IR 阅读，可使用下面提示词（路径按你的 SDK 实际安装位置替换）：

~~~text
只读审计 /usr/local/allspark/python/allspark/hauk 与 ace/kernels/fermat/permute.py。
从 @func_register/@template 开始，逐步追踪到 from_source、fir_to_hauk_tir、_lower、
hauk_tir_to_runtime。对每一步输出：输入数据结构、输出数据结构、是否是公开 TVM API、
源码文件:行号、能证明的事实、不能证明的事实。用一个 SSA define-use 小例子解释
ace.permute -> consumer 的可重写条件；不要臆测 FIR、AOPI、P1 的英文全称或硬件 ISA。
~~~

#### 16.3.2 `permute.py` 是什么 codegen 输入，谁把它放到 Fermat？

它是**作为 kernel library 输入保存的模板化源码**，不是“这次编译 ConvNeXt 刚自动生成的
Python 输出”。仅凭已分发文件无法证明它最初完全由人工还是由内部生成器写出；但
`@func_register`、`@template`、可复用函数和稳定的库路径说明它在当前 pipeline 中扮演
codegen 的**输入**。真正 codegen 发生在后面的 parser/lower/FFI 过程；`build_module.py` 的 `build()` 明确允许
`kernel_type="fermat"`、`"cayley"`（以及文档字符串中的 `mix`），并由 target FFI 产出
对象文件/可选 LLVM、assembly dump。

把近 4,000 行 `permute.py` 去掉模板、tail 和 dtype 分支后，核心问题只有一句：

~~~text
对每一个输出逻辑坐标 out[n, h, w, c]，计算它来自哪个输入坐标，
读入一小块连续元素，再把值写到输出的正确物理地址；边界块不越界。
~~~

以 `dims=[0,2,3,1]` 为例，逻辑结果是 `out[n,h,w,c] = in[n,c,h,w]`。源码中的
`virtual_axis_map` 完成“输出轴 -> 输入轴”的映射；`T.range` 枚举 tile；`T.gather/scatter`
解决非连续地址；`pred_mask_*` 避免最后一块读写越界；`P1ND_128B` 分支处理对齐。它解决的
是 **address mapping + 向量化搬运 + tail correctness**，不是卷积 MAC。

“调度到指定设备”有三个不同层次，不能混为一个工具：

| 层 | 当前证据中的责任 | 对 permute 的含义 |
|---|---|---|
| kernel 作者/模板层 | 文件位于 `ace/kernels/fermat/`，Hauk build 接受 `kernel_type` | 提供 Fermat 版本的合法实现和 tile 模板 |
| AllSpark 编译器图层 | 根据 op、shape、dtype、layout、可行域、融合规则选实现并形成 task/grid | 选择“保留此 Fermat kernel、融合、改 layout 或消除” |
| runtime/硬件层 | 依赖满足后把已编译的 task 下发给允许的 cluster/core slot | 真正执行；本 mode-12 只证明最终 slice 位于 Fermat monitor |

因此并不是 `permute.py` 自己调用某个“把自己调度到 Fermat”的 Python API。它先以目录、
build target 和模板接口表达**可用实现**；图 compiler 选择后形成静态调度；runtime 负责
发射。当前分发包没有展示 AOPI JSON 的 Python loader，也没有分发 `hauk_lower_*` /
`hauk_tir_to_runtime` 的 C++ 源码；但 `liballspark.so` 导出了 Hauk 相关 C++ 符号，Python
通过 `driver_ffi` / `target_ffi` 调它们。这是“有 C++ 后端二进制、无 C++ 源码”的 E2 结论。

#### 16.3.3 为什么 `T.matrix` 不等于 Cayley 指令？

硬件归属由 **build target / kernel type / 最终 trace** 决定，不由某个局部变量的类型名决定。
当前源码直接给出三个反证：

1. `permute_not_last_dim_matrix` 和 `shuffle_trans_general` 定义在
   `ace/kernels/fermat/permute.py`；
2. 它们把 `T.matrix(...)` 当成临时 tile，配合 `T.read`、`T.write`、`T.gather`、
   `T.scatter`、`T.shuffle_trans` 完成搬运/重排；
3. 本次相同名字的最终 kernel 在 SQLite `Monitor.Layer2` 全是 `Fermat core *`。

`T.matrix((tile_h, tile_w), dtype)` 的作用可以类比 C++ 中的局部二维数组：它表达“我要在
更近的临时存储/寄存器/GSM 中一次处理一块二维数据”，而不是“发一条 Cayley MAC 指令”。
真正会让 Cayley 有意义的是 kernel 的整体 lowering 选择、矩阵乘累加语义、可映射的
MAC tile 和 `kernel_type="cayley"` target。一个 transpose tile 也可以是矩阵形状，但只做
read/shuffle/write，仍然是 Fermat 向量/访存工作。

#### 16.3.4 `aopi/json/N93X/permute.json`：它声明“可用组合”，不是 kernel 本体

该 JSON 的可审计字段是：`name=[input, output]`、一个输入一个输出，以及按 dtype 列出的
`feasible_region`。每个 region 指定 input/output 可以采用的 physical layout 组合，例如
FP16/FP32/INT8 的 `ND/P1ND_128B` 或 `SD`。它回答“**这个 op 在哪些 dtype/layout
合同下可被选择**”，不包含 for 循环、tile 参数或最终指令。

同目录的 `conv2d.json` 给了一个很重要的反例：FP16 Conv2d 的输入/输出可接受
`ND/SD/P1ND_128B`（权重是 ND）。所以后文所说“consumer 要求某个 layout”应准确理解为
“consumer kernel candidate 有一个**支持布局集合和偏好/代价**”，不是说当前所有 FP16
Conv 都硬编码只收一种 layout。

三个 layout 可从 SDK 代码获得的最低风险解释如下：

| layout | 可以确认的含义 | 不应擅自补全的部分 |
|---|---|---|
| `ND` | 普通 dense/normal-dimension tensor 表示；Hauk 把 `ND` 与 `P1ND_128B` 同归为 `is_nd_layout` | 不能仅凭名字断言必为 NCHW 或必为 NHWC；逻辑轴标签另由 op/spec 决定 |
| `P1ND_128B` | 一种被视作 ND 的 special/packed/aligned 表示；名字及 permute 源码明确出现 128B 对齐处理 | `P1` 的公开英文全称、完整 byte-addressing 规则没有在本 SDK 文档中给出 |
| `SD` | surface/blocked 表示。`get_sd_physical_shape()` 明写逻辑 NHWC 被转换为 `NHW1C1W0C0`，`W0=8`，`C0` 以 32 B 对齐并随 dtype 变化 | 不能把它误写成所有模型永远使用的“唯一 NPU layout” |

“`physical_layout_transform.json` 是相关但不同的 op”有明确语义证据：
`ace/goldens/permute.py` 的参考实现执行 `input.transpose(dims)`，改变逻辑轴映射；而
`ace/goldens/physical_layout_transform.py` 直接返回 `inputs[0]`，逻辑数学值不变。后者的
Fermat kernel 仍会 `gather/scatter`，因为它可能把同一逻辑 tensor 从 ND 重排为 SD/
P1ND 的物理字节布局。第 16.7 节将用 def-use/view 例子展开。

关于“谁读取 JSON”：在当前可读 Python SDK 中未找到对 `ace/aopi/json/N93X/*.json` 的
`json.load` 调用；读取/匹配器很可能在未分发源码的 AllSpark compiler C++/FFI 中。我们能
从文件内容和实际图融合结果证明它们是 op capability/fusion rule 资产，**不能**据此给出
一个未经源码验证的 loader 类名。`bops2bops/N93X/permute_slice_permute.json` 已直接展示一
条规则：`permute -> strided_slice -> permute` 可重写为单独的 `strided_slice`，这正是
图层读取规则后可能消掉 layout 的例子。

#### 16.3.5 不懂 ISA 时，怎样从一个 toy kernel 开始？

最合适的学习路径不是先改 `/usr/local/allspark/.../permute.py`：那会跳过 ONNX integration、
ABI、layout contract，且会污染 SDK。先走官方公开的 **CCA/Fermat 单算子**路径；它把
ISA 隐藏在 `npcc++` 后面，和先用 CUDA 写 `vector_add` 再学 PTX 是同一工程策略。

一个 toy `y[i] = x[i] + alpha` 只需要先确定六件事：

1. 语义：输入 `x[N]`，输出 `y[N]`，FP32；CPU golden 就是逐元素 `x[i]+alpha`。
2. 每线程工作：`i = blockIdx.x * blockDim.x + threadIdx.x`，一个 thread 负责一个元素。
3. launch：`grid.x = ceil(N/256)`、`block.x = 256`；最后一个 block 必须有 `if (i < N)`。
4. 内存：`x/y` 是 device Global Memory；同 warp 的连续 lane 访问连续 `x[i]`，形成合并
   访存。
5. 合同：dtype、shape、指针、alignment、输出 buffer ownership；这比算式本身更常出错。
6. 验证：先数值，再越界，再性能；顺序不能反。

最小 device 核心（示意与 CUDA/CCA 相容；host 的 allocate/copy/launch 可直接基于官方
`samples/CCA/add` 的 wrapper）：

~~~cpp
__global__ void add_alpha(const float* x, float* y, float alpha, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] + alpha;
}
~~~

验证闭环应是：在 SIM 上 `npcc++ --device-target=N93X --platform=SIM -O2` 编译；先将
`y` 与 CPU golden 做逐元素比较；再用 `cca-memcheck` 查 Global/Shared/Local 越界；最后
在 ASIC 上用 event/nxPerf 看 kernel placement、duration 与访问模式。官方手册给出这条
公开路径和 `cca-memcheck` 的限制。此 kernel 的 CCA device 代码跑 Fermat；要将它接入
AllSpark 模型还必须增加 parser/op contract、shape/dtype inference、plugin/runtime dispatch
和 golden，不能仅把 `.cu` 文件塞进 AOM。

若以后要写 ACE/Hauk kernel，才需要再补：op capability JSON、golden reference、Fermat
模板、layout/dynamic-shape 参数、图 importer/选择规则和端到端测试。当前 SDK 的 Hauk
build API 是可读的，但其一整套 authoring ABI 并未作为稳定公开教程发布；因此应先把
CCA toy kernel 做到“数值正确 + memcheck 无误 + profile 可见”，再接触私有 ACE 路径。

### 16.4 2. `slice` 与 “Cayley slice duration 改变”

一个有用的分层是：

~~~text
ONNX node（数学语义）
  -> ACE/backend kernel（一个已选实现）
     -> tile / cut（把大 tensor 分块）
        -> cluster/core 上的工作片段
           -> mode-12 SQLite Record（一个 slice：Start + Duration + MonitorID）
~~~

在本次 `NPU-HW_swkernel_timeline.db` 中每条记录的 `Type` 是 `slice`；`MonitorID` 再
关联到 `Monitor.Layer1/Layer2`，例如 `NPU cluster 0 / Cayley core 2`。因此这里的
**Cayley slice duration** 就是“某一个 Cayley core 上，这一个记录的
`Record.Duration`”。它是 core 级软件 trace 片段的观测时长，不是 ONNX node 总时长、
也不是一帧 latency。

“layout 独立 slice 减少、Cayley slice duration 改变”描述的是一种常见融合结果：

1. 原来 `Conv(NCHW) -> permute -> 下一算子(NHWC)` 中，permute 作为独立 Fermat
   kernel，会有可见的 Fermat `slice`。
2. 编译器若让 Cayley Conv 的输出直接按消费者需要的 layout 写出，或让下一个 Conv
   以等价 layout 读取，独立 Fermat slice 会消失/减少。
3. 但 Conv 的 tile 形状、地址计算、尾块处理、cache locality 和读写模式已改变；因此
   该 Conv 的 Cayley `Record.Duration` 可以变大、变小或不变。少一次独立搬运不等于
   “Cayley 本身不用付任何代价”。

这和 M1/Metal 或 CUDA 中把 transpose fuse 到 consumer kernel 的逻辑相同：外部
materialized transpose 可能消失，但 consumer 的 indexing/coalescing 和 tile 选择会变。
要验证这一现象，应比较同一语义阶段、同一输入 shape 的 slice 分布和 dependency；不能
把不同变体里名字不同的所有 `Duration` 直接求和。本 run 的 `RelationID` 全空，无法恢复
这种 producer→consumer 对应关系，所以目前只能提出该机制，不能对某一条 Conv 定量归因。

#### 16.4.1 `slice` 不是必然“新生成了一个更小 kernel”；用 2D transpose 看它

更精确地说，`slice` 是 profiler 对“某个 kernel 名在某个 monitor/core 上一次可观测
工作”的记录。它可能来自编译时 tile/cut，也可能来自 grid 中的不同 Block、同一 kernel
的第二次发射、或 profiling/repeat 产生的第二轮执行。**不能从一条 `slice` 单独断言有
一个新的 `.o` 文件，也不能从 `32 / 16` 断言只有两次搬运。**

考虑 FP16 矩阵 `A[128,128]`，要输出 `B[128,128]`，满足 `B[c,r]=A[r,c]`。一个典型
Fermat/Metal/CUDA 风格实现把它分成 `32×32` tile：

~~~text
逻辑 tensor：A 的第 r 行连续存放在 Global Memory/GLM
Grid：4 × 4 = 16 个 Block（每一个 Block 负责一个 32×32 tile）
Block：32 × 8 = 256 threads；每 thread 在循环中搬 4 个元素
每个 tile：先把 A 的连续行读入 GSM/local tile，block barrier，再按转置坐标连续写 B
~~~

例如 `Block(2,1)` 处理输入 `A[32:64, 64:96]`，输出 `B[64:96, 32:64]`。其中一个 lane
先读 `A[32,64 + lane]`；同 warp 的 32 lane 读连续地址，故能合并访问。数据进入 tile 后，
线程改用转置索引写 `B[64 + lane,32]`。若直接在 Global Memory 做 `B[c,r]=A[r,c]`，读或写
的一边很容易是大 stride；GSM tile 的价值是把“全局内存中的 stride”局限为“片上 tile 内
的重排”。CCA 指南的共享内存 transpose 例子与 Metal 的 threadgroup/SIMD group 都是
这个基本思路。

在本 run 观察到 16 个 Fermat core slot 时，可以把它理解成上述 16 个 Block 在资源允许
时可被横向铺到这些 slot。现实中还受每个 Block 的 VGPR/GSM 占用、kernel 的 grid 大小、
cluster policy 和 ready dependency 约束；如果只有 4 个 Block，就算机器有 16 个 slot 也
最多用 4 个。反之 Block 更多时，slot 会分多波（wave）复用。

“runtime 会不会再选择核心”要分清责任：编译期已经选定 kernel 类型、合法 layout、
tile/任务图和可能的 cluster 范围；runtime 在依赖 ready 后提交 task；硬件/driver 通常会
把可运行 Block 分给空闲的允许 core，这和 CUDA 将 thread block 分给可用 SM 类似。**这是
通用 GPGPU/NPU 调度机制的 E0 类比，不是本 firmware 的逐 Block 动态分派证据。** 当前
mode-12 只看得到结果落在的 core slot，`RelationID` 缺失，不能还原“哪一块由 runtime 在
何时选择到哪一个 core”的决定过程。

再看“materialized transpose 消失”的代价转移。设输入存为 NCHW：

~~~text
显式方案：
  P[n,h,w,c] = X[n,c,h,w]        # Fermat 写出一个完整 NHWC buffer P
  Y = Consumer(P)                # consumer 连续读取 channel C

融合方案：
  Y = ConsumerWithStrides(X,
      address(n,h,w,c) = base + (((n*C+c)*H+h)*W+w))
~~~

融合方案不再分配/写入 `P`，这是收益；但若 Consumer 的一个 warp 原来希望 lane 0..31
连续读取 `P[...,c:c+32]`，现在直接读取 NCHW 的 `X[...,c:c+32,...]` 时地址间距变成
`H×W`，就可能不合并。编译器可改“哪个轴映射到 lane”、改 tile 或继续使用 blocked SD
layout 来补救；这些改动会改变 Cayley consumer 的 duration。故融合是“去掉一个全 tensor
round-trip，换成更复杂的 consumer addressing/tile trade-off”，不是凭空免费。

### 16.5 3. “grid 行数”到底指什么？

本指南此前的“Grid 行数”专指各构建目录的 `allspark_grid_info.txt` **文本行数**。例如
baseline 为 1,086 行、DyT 为 768 行、DyT-no-permute 为 799 行。每一行形如
`stream_0|task_1|grid_2|<symbol>`，是编译产物中的静态 task/grid 指令条目。它是调度
图/命令复杂度的粗略代理，**不是**“GPU 一共启动了 1,086 个线程块”，也不是 runtime
trace 的 slice 数；文件还含 `InstrPre`、prefetch、`__invalid__` 等非模型算子条目。

“Grid”在当前栈里至少有三种不同含义，最容易混淆：

| 名称 | 所在层 | 它表示什么 | 不能推出什么 |
|---|---|---|---|
| `allspark_grid_info.txt` 的一行 | AllSpark 静态调度/命令层 | stream/task 下的一条 grid/命令信息 | CUDA/CCA 的 `gridDim`、实际执行次数 |
| CCA Fermat launch grid | kernel 并行编程层 | 一维/二维/三维 **Block** 的集合；Block 内再有 thread/warp | `grid_info.txt` 行数 |
| `--npu_hw_grid` | nxPerf mode-15 硬件 profiler | 硬件 Grid 事件缓冲区容量 | kernel 的计算量或产品 latency |

同级和强相关概念可按层次记：

~~~text
模型 / AOM
  -> stream -> task -> backend kernel -> allspark_grid_info 的命令行
  ->（Fermat kernel 内）grid -> block -> warp -> lane/thread
  ->（物理资源）cluster -> Cayley 或 Fermat core -> L1/L2/DDR
  ->（观测层）Monitor -> Record/slice -> Start/Duration
~~~

E1 文档的 Fermat 模型是 Grid → Block → Warp（warp 最多 32 threads），并定义
`HardwareWarp` 为 4 个 warp、128 threads 的硬件锁步单元。它和上表第一行的静态
`grid_info` 概念相关，但不是同一个对象。

#### 16.5.1 Task/grid 指令：它们是“依赖图上的命令”，不是每条一条 GPU 指令

把 AllSpark 运行时看成操作系统的 job scheduler 会比较直观：AOM 内有一张 DAG；每个
backend kernel/copy/cache 操作是 job；**Task** 是一组可下发的工作/依赖管理单位；
**stream** 是提交顺序队列；`allspark_grid_info.txt` 的 `stream|task|grid|symbol` 则是
编译期输出的静态命令/网格条目。一个 Task 完成会产生“后继可以 ready”的信号；runtime
再把 ready 工作放到 Cayley/Fermat 可执行路径。

这不是 CPU 的一条 `add` 指令，也不是 Fermat 的一个 warp。可以把它和 CUDA 的层次对照：

~~~text
AllSpark AOM DAG:    task / stream / kernel descriptor / cache op
CUDA host launch:    stream / kernel launch descriptor
Fermat device launch:grid -> block -> warp -> lane
CUDA device launch:  grid -> thread block -> warp -> thread
~~~

`allspark_grid_info.txt` 告诉我们“编译器静态安排了哪些命令符号”，mode-12 SQLite 告诉
我们“最终在哪个 core monitor 观察到 slice”。两者中间的 task dependency、event/fence、
内存地址和 per-block dispatch 并未在本 run 完整导出，所以不要把 `grid_2` 当作硬件
`blockIdx`。

#### 16.5.2 CCA 是什么？它和 Hauk 不是同一入口

当前公开文档将产品名写作 **AllSpark CCA 编程语言**，并明确说它提供“类似 CUDA 的
host + device 混合编程能力”：host 代码跑 CPU，device 代码跑 **Fermat Core**，编译器是
`npcc++`，Runtime API 也有 `cudaMalloc`/`cudaMemcpy`/`cudaLaunchKernel` 风格映射。
文档没有给出 CCA 三个字母可靠的英文全称，因此本文不编造全称。

它和 Hauk 的关系可记成：

| 入口 | 谁用它 | 代码形态 | 最适合的学习/工程用途 |
|---|---|---|---|
| CCA | 算子/插件开发者 | CUDA-like `.cu`，`__global__`、`gridDim`、`blockDim` | 公开可走的 Fermat toy/custom kernel、SIM、memcheck |
| Hauk | AllSpark 内置 kernel/后端作者 | Python embedded TIR DSL、模板、`T.read/write/gather` | 编译器内置算子库与 target-specific specialization |
| AllSpark Engine | 模型部署者 | ONNX/Engine API/config | 导入图、融合、选 kernel、生成 AOM/task |

所以“CCA 类 CUDA”并不意味着 Cayley 是 NVIDIA SM，也不意味着 Hauk 等于 CUDA Python。
它们共享一部分抽象（kernel、stream、grid、device memory），但编译器和硬件 ABI 是不同的。

#### 16.5.3 NVIDIA、Fermat/NPU、M1 的内存模型：先看可见性和生命周期，再谈芯片

下面表中的“近似”是工程心智模型；N93X 的逻辑 memory space 是否对应一颗独立 DRAM、
是否与 CPU 物理共享、cache policy 如何实现，当前公开资料不足，不能从 CUDA 名称反推。

| 层级 | CUDA/NVIDIA（离散 GPU 的典型情况） | Fermat/CCA 的 E1 语义 | M1/Metal 的常见类比 | 谁能见、生命周期 |
|---|---|---|---|---|
| thread 私有 | register；溢出后的 local memory 往往在 device DRAM | PVT，单线程可见；过度约束寄存器可能转为 PVT 访存 | thread-local/private register-like 值 | 一个 thread；随执行结束失效 |
| Block 共享 | shared memory | GSM，每 Block 独立，文档为每 core 32 KB | Metal threadgroup memory | 一个 Block/threadgroup；随 kernel/Block 结束失效 |
| core cache | SM L1/cache | 同一 Fermat core 的 Block 可见 L1 | GPU core cache | 一个 core；不能作为跨 kernel 正确性交接唯一依据 |
| device 共享 cache | chip-wide L2 | L2 对所有 Block 可见，可用于数据交换 | Apple GPU system-level cache 的功能类比 | 跨 Block；容量/一致性细节未公开 |
| device 全局 | global memory，离散卡上通常是 GDDR/HBM，即口语的“显存/VRAM” | GLM，所有 Block 可见，主要数据存取空间；通过 L2 访问 DDR | M1 GPU/ANE 可见的 unified physical DRAM 的 logical device resource | 可跨 kernel 交接、随应用 allocation 存续 |
| host | CPU DRAM | host CPU memory；Runtime 有 Host↔Device copy API | 与 GPU/ANE 共享同一物理 DRAM（UMA） | CPU 可见；是否需 copy 由 API/storage/coherency 决定 |

因此“显存”在 NVIDIA PC 上通常指 GPU 私有的 GDDR/HBM；在 N93X 文档里更严谨的说法是
**NPU 的 GLM/device-global memory space**，不要未经硬件内存拓扑证明就说它一定是一颗
独立显存芯片。N93X 是 SoC 时也可能与 CPU 共用物理 DDR 但拥有不同的映射、cache 和
同步域；即使物理 DRAM 相同，逻辑 `cudaMemcpyHostToDevice` 仍可能代表 ownership、cache
flush/invalidate 或 DMA 成本。M1 的 UMA 正好说明“共享物理内存”绝不等于“没有带宽和
同步成本”。

真正容易“满”的资源，以及开发者能控制的杠杆是：

| 饱和/错误点 | 现象 | 可控制的手段 | 不可由应用保证的部分 |
|---|---|---|---|
| DDR/GLM 带宽 | 大 feature map 的读+写、transpose/concat、strided access 使 MAC 空等数据 | 融合、减少中间 tensor、统一 layout、FP16/INT8、合理 tile、连续/coalesced mapping | 实际 DDR arbitration、其他核竞争、cache miss |
| L2 容量/transaction | tile 放不下或随机/stride miss；Fermat 文档给出 L2 miss 以 256 B line 从 DDR 取数 | 尽量连续访问、让 producer/consumer 靠近、控制 working set/tile | 具体替换策略、各 cluster 的物理容量/分配 |
| GSM | tile 太大/共享数组太多，导致不能同时 resident 足够 Block | 选 tile、复用 tile、减少无用 shared，transpose 可用 padding 避免冲突 | bank 结构和编译器最终分配 |
| VGPR/PVT | 寄存器过多降低 occupancy；强行压寄存器又可能 spill 到 PVT | 简化 live values、谨慎 unroll、调 block size/`launch_bounds` | 最终 register allocation |
| host 预后处理/拷贝 | 模型快但 H2D/D2H、颜色转换、CPU resize 占主导 | buffer reuse、异步 stream、把可融合预处理搬到 device、计时分段 | OS/driver 调度、IOMMU/coherency 开销 |

CCA 指南给出一个特别实际的 Fermat 风险：warp 连续 lane 若读连续且对齐地址，能合并；若
stride/random，L2 miss 会按 256 B line 拉 DDR，可能比真正需要的数据大很多。这里的首要
优化不是“提高 FLOPs”，而是让每次从 GLM/DDR 拿回的 cache line 被更多 lane/tile 复用。

### 16.6 4. 静态内存、`TmpTensor`、CL2、GL2 与 NVIDIA 类比

**静态内存不是 AOM 文件大小。** `.aom` 是序列化文件，包含权重、图、调度/元数据并可能
受对齐或压缩影响；编译 log 的 `Total mem size` 是编译器为运行期规划出的内存量。它们
单位相同但对象不同，不能用文件 `ls -lh` 替代运行期 memory plan。此 run 中三份
`compile.log` 的 `Total mem size` 大约为 1,160 MB，而 AOM 文件本身也在约 1.2 GB
量级，数值接近只是权重占主导时的巧合，不构成“二者相等”的证据。

`compile.log` 中的字段和最稳妥的解释如下：

| 字段 | 可确认/推断含义 | 特性与怎样才算好 |
|---|---|---|
| `TmpTensor` | E2：编译器静态内存规划报出的临时 tensor 工作集；通常是按 liveness 复用的 activation/workspace。其精确包含项未在公开手册定义。 | 在正确性不变、没有额外 spill 的前提下更小，通常更容易放进近端存储；但不是越小越好，重算或低并行度也会把它压小。 |
| `TmpTensor(cl2)` | E2 字段存在；结合 `CL2_ADDR`、`cluster_l2_size` 配置与文档的 cluster/L2 描述，E0 推断为计划放在 **cluster L2** 地址空间的临时量。 | 要和单 cluster L2 容量、实际 hit/miss、并行 cluster 数一起看；不能把它当 cache hit rate。 |
| `TmpTensor(gl2)` | E2 字段存在；结合 `GL2_ADDR`、`global_l2_size`，E0 推断为计划使用 **global L2** 地址空间的临时量。 | 减少共享 L2 的压力可能有利，但跨 cluster 共享、复用与冲突同样影响结果。 |
| `Total mem size` | E2：本次编译器报出的静态 runtime memory plan 总量。未获得公开 source 来分解固定权重、常驻 buffer、临时 buffer 的精确公式。 | 正确性、容量不溢出、低 spill 和低 latency 都成立时才有意义，不能单列排名。 |

本 run 的数值是：baseline `TmpTensor=14.02 MB (CL2=11.43, GL2=2.59)`；DyT 为
`9.81 MB (9.13, 0.68)`；DyT-no-permute 为 `9.86 MB (9.57, 0.29)`。这只能说明三种
**静态规划**不同，不能证明后两者 cache miss 更少，更不能取代 mode-15 的 DDR/cache
计数。

与 NVIDIA 的工程类比应按“作用”而不是名字硬对齐：

| AllSpark/CCA 名称 | E1 可见的作用 | CUDA/NVIDIA 的近似心智模型 | 重要差异 |
|---|---|---|---|
| PVT | 线程私有存储；CCA 文档特别提示其溢出/片外代价可能很高 | register / local memory | CUDA local memory 可能落 DRAM；PVT 不能简单当作一定在寄存器 |
| GSM | 一个 Block 共享，文档给出 32 KB | CUDA shared memory | 容量、bank 与一致性规则不应假定相同 |
| L1 | 一个 core 内多个 Block 可用 | 单个 SM 的 L1/cache | 只是粒度类比 |
| CL2（推断） | 一个 cluster 的较近层共享缓存/地址空间 | 多个 SM 共享的近端 cache 概念 | NVIDIA 公共 CUDA 模型并没有“一一对应的 cluster L2”名称 |
| GL2（推断） | NPU 更全局的 L2 地址空间/共享层 | GPU 的 chip-wide L2 概念 | 不是 CUDA 所说的 global DRAM |
| GLM/DDR | 更远端全局存储 | CUDA global memory / DRAM | 实际带宽、coherency、寻址由芯片决定 |

M1 的 UMA 也不是“没有内存成本”：CPU、GPU、ANE 共享物理 DRAM 可以避免某些显式拷贝，
但 cache 可见性、资源 storage mode、命令依赖和带宽竞争仍会产生成本。故不能把 M1 的
Unified Memory 等同于 N93X 的 CL2/GL2 分级设计。

### 16.7 5. layout 与 `ace.permute` 的关系；为何早期 no-permute 更多？

`layout` 是一个更大的概念：它描述 tensor 的**逻辑轴顺序**（如 NCHW/NHWC）以及物理
存储/分块方式（此 SDK 可见 `ND`、`SD`、`P1ND_128B`）。`ace.permute` 是为实现某一类
逻辑轴重排而生成的一个后端 op；`physical_layout_transform` 是另一类物理 layout 变换。
此外 layout 还可能被 producer/consumer 的索引规则吸收，成为 view/metadata，或融合进
Conv，而完全没有独立 `ace.permute`。

因此二者不是一一对应关系。这个实验正是反例：在较早的
`__aom_0_model_.onnx` 快照中，baseline/DyT/DyT-no-permute 的 `ace.permute` 节点数为
**152 / 152 / 224**；到 `__aom_0_model_after_OpFusion.onnx` 为 **77 / 7 / 7**；到
`allspark_build__graph___data.onnx` 为 **89 / 7 / 7**。`dyt_nopermute` 这个名称只表示
你在源模型层删了若干显式 `Permute`，不保证后端无需为以下约束补一个 layout adapter：

1. 某个 Conv/Cayley **kernel candidate** 只能接受或在某个 physical layout 下代价更低，
   而 DyT 替换后 tensor 的生产者边界、rank 或对齐约束变了；
2. 为维持 ONNX 数学语义，编译器需要在新边界重新表达 NCHW/NHWC 或 blocked layout；
3. 早期快照在 OpFusion 之前，包含随后会合并/消除的候选 adapter。

所以“224 比 152 大”只能说明**早期 lower 后候选 adapter 更多**，不是物理搬运更多。
最终图同为 7，实际 trace 的 layout-named slice 则为 baseline/DyT/no-permute 的
2,848 / 288 / 224，才是更接近运行期的证据；但它仍按名称过滤，不能等价成字节数。

#### 16.7.1 `ace.permute`、`ace.physical_layout_transform`、view/fusion 的本质差异

这四件事可用“数学逻辑值是否改变”和“是否必须另开一块 buffer”两个问题区分：

| 形式 | 数学语义 | 最低层的典型行为 | 当前 SDK 的直接证据 |
|---|---|---|---|
| `ace.permute` | 改变轴的逻辑对应；如 `P[n,h,w,c]=X[n,c,h,w]` | 通常需要根据新旧坐标做 gather/scatter；若 consumer 不支持 stride，必须 materialize | golden 是 `input.transpose(dims)`；Fermat 模板做 axis map + read/write/gather/scatter |
| `ace.physical_layout_transform` | 逻辑 tensor 值/shape 语义不变 | 改变同一逻辑元素在字节 buffer 的位置，例如 ND ↔ SD blocked | golden 直接返回输入；Fermat 模板仍存在实际 gather/scatter implementation |
| view/metadata | 逻辑坐标解释变了或折叠/stride 元数据变了 | 新 tensor descriptor 指向同一 base buffer，无数据 copy | 是否可用取决于 runtime 是否支持 stride/layout view；当前 trace 不能逐 node 证明 |
| 融入 consumer | consumer 的数学结果不变 | 不建中间 buffer；consumer 的 address formula/tile 直接访问 producer storage | 常见编译机制；本 SDK `permute_slice_permute` 规则给出一个实际消除 layout chain 的例子 |

一个最小 C++/IR 级例子：

~~~text
base X 的逻辑 shape: [N,C,H,W]，连续 NCHW storage
P = permute(X, [0,2,3,1])

若 P 只是 view：
  P.base = X
  P.shape = [N,H,W,C]
  P.stride = [C*H*W, W, 1, H*W]
  读 P[n,h,w,c] 时仍从 X[n,c,h,w] 取，不写新 buffer。

若 Consumer 只能接收连续 NHWC：
  必须 materialize，分配 P buffer，逐元素写 P[n,h,w,c]。

若 ConsumerWithStrides 支持上述 stride：
  直接把 base/stride 传给 Consumer；图中 P 可被删除。
~~~

这就是“被 producer/consumer 的索引规则吸收”的精确含义：`P` 这个 SSA value 的 user
不再需要一个独立 producer kernel，而是在 consumer 的 index 计算中把
`addr(P,n,h,w,c)` 替换成 `addr(X,n,c,h,w)`。它是 def-use rewrite，而不是把数学 transpose
错误地忽略掉。若 consumer 是 Conv，权重/输出轴的解释也必须同步保持正确；否则会出现
“图少了一个 permute 但数值错”的典型 bug。

#### 16.7.2 Consumer 的 layout “要求”怎样表达？不是 ONNX 的单一硬编码字段

ONNX Conv 的标准语义通常只表达逻辑 axis/order；backend 的实现再声明实际可接受的 dtype、
physical layout、alignment、rank、shape range 和 tile 限制。当前 N93X SDK 把至少一部分
合同放在 `ace/aopi/json/N93X/<op>.json` 的 `feasible_region`：`conv2d.json` 中 FP16 输入/
输出允许 `ND/SD/P1ND_128B`，权重为 ND；`permute.json` 也列出自己的输入/输出组合。

所以“consumer 提出要求”的实际过程更像约束求解：

~~~text
producer 的候选输出：ND 或 SD
consumer candidate A：只接受 ND，估计 cost=10
consumer candidate B：接受 ND/SD，估计 cost=12
layout transform ND->SD：估计 cost=3

若 producer=SD：A 的总 cost = 3 + 10；B 的总 cost = 12。
编译器还要同时考虑融合、内存峰值、tile 可行性、并行度，而不是只比这一行数字。
~~~

这里的 capability JSON 就像 MLIR dialect conversion 的 legality/type converter，或 TVM
算子 strategy 的 implementation predicate：它定义某个 rewrite/implementation 在什么
operand type/layout 下合法。更深的 cost、AOPI loader、融合匹配和最终选择代码在本 SDK
中没有公开源码，故不能声称每个约束都是 JSON 硬编码；它也可能来自 C++ op schema、
kernel builder 或 profile/config。重要修正是：**当前 FP16 Conv2d 并非只接受一个 layout，
但具体 fusion/candidate 仍可对某一 layout 更偏好，或在另一个 layout 下不可行。**

### 16.8 6. 为什么 final graph 的 7 个 `ace.permute` 不等于 7 次搬运？

`allspark_build__graph___data.onnx` 中的 7 是**静态 graph node 数**。一次物理工作的
数量还会被至少三件事放大：

1. **tile/cut**：一个大 feature-map 被拆为多个可装入本地存储的块；名字中的
   `_cut_0`、`_cut_1` 是可见线索。
2. **空间并行**：同一个 tile/kernel 被分配到多个 cluster/core，各 core 产生自己的
   trace Record。
3. **时间复用或多次调用**：同名 template 可为不同分块、不同 batch/循环或不同 task
   多次发射；同一个名字并不唯一标识“一次发射”。

本次 baseline 中一个实际例子为
`__Conv_0_Conv_0_permute_input__17__1`：SQLite 显示 **32 个 Fermat slice**、使用
**16 个不同 core slot**（2 cluster × 8 Fermat core）。它不是 1 个时间点的一条线。
而且每个 slice 的 duration 仅 744–978 ns、各 slice duration 之和 27,195 ns，但以
该名字所有记录的最早开始到最晚结束算的 span 是 801,143 ns。这直接证明“一个静态
op 名称 / 一个 graph node”不能换算成“一次连续物理搬运”。

这也提醒我们不要把 `ace.permute=7` 解读为“模型只读写七个 feature map”；需有
producer-consumer tensor 地址、tile 大小和 DMA/DDR transaction 才能计算字节数，而本
次缺这些 E2 证据。

#### 16.8.1 对本 run 可精确对账：224 的确是 7 个最终节点各 32 条 slice，但不能化约成“两次搬运”

这次补做了 final graph 与 SQLite `kernel` 名的逐名核对，结果比“仅按名称过滤”更强：

| 变体 | final graph 中的独立 layout node | trace 中同名的唯一 layout kernel | core slice | 每个同名 kernel 的 slice |
|---|---:|---:|---:|---:|
| baseline | 89 个 `ace.permute` | 89 | 2,848 | 32 |
| dyt | 7 个 `ace.permute` + 2 个 `ace.physical_layout_transform` | 9 | 288 | 32 |
| dyt_nopermute | 7 个 `ace.permute` | 7 | 224 | 32 |

因此对问题 11 的答案是：**对 `dyt_nopermute` 这个特定 run，224 恰好等于 7 × 32。**
可以说“7 个最终图 layout node 各被 mode-12 观测为 32 个 Fermat core slice”，不能说
“一个 permute 被编译成 32 个不同的 ONNX permute”。静态图 node 仍是 7 个，slice 是
执行观测记录。

对问题 12 的答案是：**32 slice / 16 slot 不等于两次内存搬运。** 它只给出一个抽屉原理：
32 条记录落在 16 个 slot，至少某些 slot 承担了不止一条记录。造成它的机制可能是两个
execution/repeat、两个 wave、多个 tile/cut 或 trace 的记录粒度；当前 `RelationID`/launch
ID 缺失，不能区分。若是两次完整 model invocation，可能是整个 tensor 各搬一次；若是
32 个不重叠 tile，则每个元素通常只在自己的 tile 中搬一次；若是缓存/重读，字节量又不同。
所以仅由 `32/16` 绝不能推出“恰好两次完整 DDR copy”。

对问题 13 的答案也需要分层：本 run 的 89/9/7 个最终独立 layout node 都有**同名且实际
落在 Fermat 的 trace record**，所以它们是可信的“已执行独立 layout kernel”。其中
`ace.permute` 有逻辑 transpose 语义，`ace.physical_layout_transform` 有物理重排语义；
二者的 Fermat 模板都包含真实读写路径。不过 mode-12 没有每条边的地址/transaction，
所以仍不能把它们定量为“写了多少 DDR 字节”。

“匿名 layout slice”在当前最终图与 trace 的**独立 kernel 层**没有发现：静态 node 名和
trace kernel 名一一对应。但“没有匿名独立 kernel”不代表模型中不存在未命名的 layout
工作：它可以藏在 `conv2d_postp...` 这类融合 Cayley kernel 的 indexing/tile 内，也可以
只是 descriptor/view metadata 而根本不产生 slice。故 `layout-named` 在其他模型里仍只是
筛选启发式；本 run 是因做了 final graph 逐名交叉验证才提升为强证据。

### 16.9 7. 为何 scalar cost 不能预测 E2E 毫秒？

把 compiler scalar cost 看成选 schedule 时的启发式评分，而不是秒表。它往往近似一个
candidate 的算术量、估计访存量、tile 数、资源限制的加权和；它有用，因为能在海量
schedule 中排序，但 E2E 是带依赖的并行执行图。

一个简化的时间线足以说明区别：

~~~text
Cayley Conv: 4 ms   ────────────────┐
Fermat layout: 1 ms ────┐            ├─ fence ─ 后续 0.5 ms
                         └────────────┘

若两个前驱可并行，关键路径约为 max(4, 1) + 0.5 = 4.5 ms；
把候选 kernel 的 scalar cost/时长相加却会得到 4 + 1 + 0.5 = 5.5 ms。
~~~

漏项的具体含义是：

- **critical path**：DAG 只由最长依赖链决定；并行支路不应线性相加。
- **Cayley/Fermat 并行**：E1 性能手册明确提示两类 core 可并行；实际是否重叠仍取决于
  buffer 依赖和 runtime。
- **cache/spill**：本应留在 L1/L2 的 tile 若容量或冲突不满足，会写回更远层再读回；
  静态 score 很难精确知道本次运行的 hit/miss。
- **runtime fence**：producer 未完成时 consumer 需等队列/事件；这是 OS/运行时里的
  dependency synchronization，不是算子 FLOPs。
- **H2D/D2H**：host 准备输入并交给 NPU、再取回输出的开销。M1 UMA 可能省去独立 PCIe
  copy，但不消除 cache ownership、同步和带宽成本。

本 run 的 `perf_summary.log` 中确有 Cayley/Fermat 两个静态数值，但未取得它们的公开
单位/完整公式，且正确性 Gate 失败。故它们目前只可用来解释“为何编译器倾向某 schedule”，
不能线性换算或拟合为 p50 E2E 毫秒。

### 16.10 8. 怎样读懂该表？

先把表头补全为“**全部 Record / 有实际 Cayley 或 Fermat core 的 slice**”。它不是
“一个模型有 9,238 个 kernel”。逐行读取：

| 变体 | 全部 Record / core slice | Cayley slice（去重 kernel 名） | Fermat slice（去重 kernel 名） | layout-named Fermat slice | 直接可说的结论 |
|---|---:|---:|---:|---:|---|
| baseline | 9238 / 9226 | 2394 (151) | 6832 (218) | 2848 | 有大量独立、按名称识别的 layout 工作 |
| dyt | 5688 / 5680 | 2320 (145) | 3360 (105) | 288 | layout-named slice 较 baseline 少 2,560，即 -89.89% |
| dyt_nopermute | 6691 / 6683 | 2331 (146) | 4352 (136) | 224 | layout-named slice 较 baseline 少 2,624，即 -92.13% |

括号内是同一 engine 下不同 `kernel` 名的数量，不是 core 数。`Record` 比 core slice
多出的 12/8 条是 cluster/硬件 task 等非 Cayley/Fermat-core monitor 记录。layout-named
是按名字包含 `permute`、`transpose`、`layout` 的保守筛选，可能漏掉被融合进 `Add`/`Mul`
或被改名的 layout 工作，也可能把并非 materialized copy 的 layout 名算进来。

因此它最强的结论是“本次 mode-12 可见的独立 layout **工作片段数**显著减少”；它不能
推出“减少了 89.89% 的 DDR 字节”“Fermat 总耗时减少 89.89%”或“E2E 必快 89.89%”。后两者
需要可关联的 trace、有效计时和正确性通过后的 benchmark。

### 16.11 9. Fermat 是否还执行别的 kernel，去哪里看？

是。Fermat 不只是 layout engine；当前 trace 中它还实际执行了 elementwise 和归约/融合
实现。例子包括 baseline 的 `__ReduceMean_...apex_fused_op...` 与 `__Add_...`，DyT 和
DyT-no-permute 的 `__Mul_...apex_fused_op...`、`__Add_...`。每个例子都可在 16 个
Fermat core slot 上看到多个 slice。它与“Fermat 是 GPGPU/SIMT 风格向量核”的 E1 定位一致。

最方便的已解析文件是：

~~~text
<run>/board_trace/<variant>_mode12/summary/mode12_core_placement.csv
~~~

它的 `engine`、`cluster`、`core`、`kernel`、`start_ns`、`duration_ns` 就是所需字段。
若要直接审计原始 SQLite，下面命令列出**排除 layout 关键字后**的 Fermat kernel；这只是
查看，不会改动工件：

~~~bash
sqlite3 -readonly \
  codex_work/convnext_dyt_experiment/runs/20260901T083738Z_mode12/board_trace/baseline_mode12/NPU-HW_swkernel_timeline.db \
  "SELECT r.Name, COUNT(*) AS slices, SUM(r.Duration) AS sum_slice_ns
   FROM Record r JOIN Monitor m ON r.MonitorID=m.MonitorID
   WHERE r.Type='slice' AND m.Layer2 LIKE 'Fermat core%'
     AND lower(r.Name) NOT LIKE '%layout%'
     AND lower(r.Name) NOT LIKE '%permute%'
     AND lower(r.Name) NOT LIKE '%transpose%'
   GROUP BY r.Name ORDER BY slices DESC, r.Name LIMIT 30;"
~~~

注意 `SUM(Duration)` 仅是该名字所有 core 片段的工作量累计，不能作为 wall time；这一点在
第 16.13 节会用真实数字说明。

### 16.12 10. `2 cluster × 8 Fermat core = 16 core slot` 与 warp 的关系

它表示：**一个 backend kernel 的不同工作块可以同时分派给多个物理 Fermat core**。这次
mode-12 中观察到 cluster 0 和 1 各有 `Fermat core 0..7`，所以最多有 16 个“本 AOM
实际使用并被 trace 观测到的 core 位置（slot）”。这不是声明全芯片总共有 16 个 Fermat
core，也不保证每个 kernel 每次都会占满全部 16 个。

它与 NVIDIA 的关系应分两级类比：

| 对象 | Fermat/CCA（E1） | CUDA 类比 | 是否等价 |
|---|---|---|---|
| warp | 一组最多 32 个线程；`HardwareWarp` 为 4 warp/128 threads，在**一个 Fermat core 内**锁步运行 | 一个 SM 内的 warp | 比较接近的编程模型类比 |
| Block | Grid 内的一块线程工作 | CUDA thread block | 编程层类比，不承诺资源参数相同 |
| 16 core slot | 跨 2 个 cluster 的 16 个 Fermat core 位置 | 多个 SM 同时调度多个 thread block | **不是 warp** |

所以问题中的短句应读成：一个 kernel 的 grid/block 工作被“横向铺开”到 16 个 core；每个
core 内部再由 warp/lane 纵向并行。它近似 CUDA 的“一个 grid 的多个 thread block 同时
跑在多个 SM”，而不是“一个 kernel 只有一个 warp”。

本 run 的前述 baseline permute 名称有 32 条 slice 却只用了 16 个 slot，这说明同名工作
至少需要在这些 slot 上产生多于一轮的记录（可能是不同 cut/tile/调用；没有 RelationID，
不能精确区分）。这也解释为什么“16 个 core 并行”不等于 trace 必然只有 16 行。

#### 16.12.1 一个 128×128 transpose 的微观并行例子

仍以 `A[128,128] -> B[128,128]` 为例，令 tile 为 `32×32`：

~~~text
Grid = (4,4) Block，共 16 Block。
每个 Block = (32,8) Thread，共 256 Thread = 最多 8 个 warp = 2 个 HardwareWarp。

Block(0,0): A[0:32,   0:32]   -> B[0:32,   0:32]
Block(1,0): A[0:32,  32:64]   -> B[32:64,  0:32]
...
Block(3,3): A[96:128,96:128]  -> B[96:128,96:128]
~~~

若 16 个 Fermat slot 都空闲且每个 Block 的 GSM/VGPR 资源都装得下，调度器可以把 16 个
Block 同时放到 `cluster0/core0..7` 与 `cluster1/core0..7`。这只是空间并行的最好情况。

把视角放到其中的 `Block(1,0)`：一个 `threadIdx.y=0` 的 warp 内，lane 0..31 分别读取
`A[0,32]..A[0,63]`，连续合并 load；每个线程循环处理再往下的第 8、16、24 行。它们把
读到的元素暂存进局部/GSM tile，block barrier 后，负责输出列方向的线程写入
`B[32:64,0]` 等连续地址。这里：

- **lane** 是一个 warp 内的一个 thread 索引；
- **warp** 是同一 Fermat core 内锁步执行的一组最多 32 threads；
- **Block** 是可以整体放到一个 Fermat core 的合作单元；
- **16 core slot** 是 16 个不同 core 同时承载不同 Block 的位置。

因此 core slot 对应 CUDA 的“SM 可驻留/执行 Block”，warp 对应 CUDA 的“SM 内的锁步线程
组”。它们是上下两层，不是同义词。Apple Metal 中的对应类比是：一个 compute grid 分成
threadgroup，threadgroup 又被拆成 SIMD group；Apple 也明确提示 SIMD-group width 应在
runtime 查询而非写死。当前 CCA 文档则公开 Fermat warp 上限为 32、HardwareWarp=128。

### 16.13 11. `wall span` 是什么，和 kernel 有何关系？

本次解析脚本给每个 `(engine, kernel name)` 做的 `observed_wall_span_ns` 定义是：

~~~text
min(该名字所有 slice 的 Start)
    到
max(该名字所有 slice 的 Start + Duration)
~~~

它回答的是“这个名字第一次被观测到开始，到最后一次被观测到结束，横跨了多长的时间轴”。
它**不是**以下任何一个值：单个 slice duration、所有 slice duration 的和、一次 kernel
launch 的纯执行时长、整帧 E2E latency。

原因是同名 kernel 的记录可跨多个 core、多个 tile，且中间可被别的 kernel、依赖等待或
后续同名调用隔开。上面的真实 baseline `permute_input` 例子尤其直观：32 条 slice 的
duration 总和仅 **27,195 ns**，单条为 **744–978 ns**，但 `observed_wall_span_ns` 是
**801,143 ns**。所以 span 中包含了非连续性；不能把它写成“这个 permute 跑了 0.801 ms”。

正确用法是把 wall span 当作排查线索：同一 kernel 名的 span 若异常大，去看它是否被多次
调用、是否发生 tile 波次、是否被 fence 或资源竞争隔开。要回答“单次 kernel 的真实 wall
time”则需要稳定的 launch/correlation ID；本 run 的 `RelationID` 缺失，正是不能继续归因
的原因。

### 16.14 15. Cayley 与 Fermat 的并行、同步和“谁在等谁”

结论：**它们既可以并行，也可以因 data dependency 同步；先完成的 core 通常不会傻等，
而是由 runtime/硬件继续执行其他 ready task。真正等待的是依赖链上的后继，或没有别的
可运行任务时该 core 才会空闲。** E1 性能手册明确提到 Cayley 与 Fermat kernel 可以并行，
但是否重叠取决于本模型的依赖、buffer 和 runtime。

用三种最小 DAG 看：

~~~text
独立：  Cayley Conv A          Fermat Reduce B
         ─────────────          ────────────────
         可并行；但仍会竞争 L2/DDR 带宽。

流水：  Cayley Conv A -> Fermat Permute B -> Cayley Conv C
         B 必须等 A 的输出对 B 可见；C 必须等 B。
         A 完成后，A 所在 core 可去跑别的 ready Cayley task，不必等待 B 完成。

汇合：  Cayley branch A ─┐
                         ├-> Fermat Add/Concat C
         Fermat branch B ─┘
         A 早完成只记录 event；C 必须等较慢的 B；A core 可转去做其他工作。
~~~

跨 kernel 的依赖不是寄存器传值：producer 的 PVT/GSM 随 kernel/Block 生命周期结束，
consumer 必须从跨 kernel 可见的 GLM/L2 路径读取中间结果，并由 task/event/fence 保证
“写已完成且对 consumer 可见”。是否再落到 DDR 取决于 cache hit、容量/spill、cluster
位置和缓存策略；本 run 不能为某一条 Cayley→Fermat 边给出 transaction 数。

可以把 `wait` 分成三个粒度：

| 场景 | 谁等待 | 典型机制 |
|---|---|---|
| 同一 Fermat Block 内需要共享 tile | 同一 Block 的 threads/warps 互等 | `__syncthreads` / threadgroup barrier；仅用于片上协作 |
| 两个独立 kernel 有 producer→consumer 边 | consumer task 等 producer completion/visibility | runtime event、fence、stream dependency、cache/coherency 操作 |
| 两条无依赖分支 | 无正确性等待，但可能慢下来 | 共享 DDR/L2、功耗/调度资源竞争 |

这与 M1/Metal 的 command buffer/encoder 也相似：同一 queue 的顺序编码天然建立顺序；跨
queue 或需显式可见性时要 event/fence/barrier。差异是 Core ML 不公开把一条图分别落在
GPU/ANE 的 core trace，而 AllSpark mode-12 在本实验中能看到 Cayley/Fermat monitor。

本 run 的 `RelationID` 全空，所以不能从时间线严谨证明某两个具体 slice 是否为依赖、是否
真正重叠或谁在 fence 上空等。若要回答某一条边，需有带 correlation 的 firmware trace，
或在图中人为保留唯一 kernel 名/屏障并采集硬件 event 与 DDR window；随后再用 mode-0
多次 E2E 验证等待是否进入 critical path。

### 16.14.1 本次 DyT 的 `Tanh`、baseline LN 与 layout：完整证据链和正确结论

先给结论：**DyT 的 `Tanh` 仍由 Fermat 执行；它没有被“换到 Cayley”或消失。性能调度层面
更有价值的变化是，`Tanh` 被并入一个 Fermat `apex_fused_op`，而 baseline 中大量独立
layout/permute 工作在 DyT 后被融合、消除或重写。** 因而不能把现象简称为“DyT 没有
Fermat 工作”，更不能简称为“没有 layout”；应说“独立、可观测的 layout kernel 大幅减少”。

这里必须按四个图/trace 层次读，只有最后一层能证明实际 core placement：

~~~text
源 ONNX 的语义节点
  -> ACE lowering 的 ace.* 节点
  -> OpFusion 属性中记录的 fusion group
  -> final graph 的切分名字 + 板端 Record JOIN Monitor
                                      ^ 只有这里直接写 Fermat/Cayley
~~~

#### A. DyT `Tanh` 为什么能确认在 Fermat

**源 ONNX（E2，`exports/dyt.onnx`）**的开头是：

~~~text
00 Conv_0   Conv
01 Mul_1    Mul
02 Tanh_2   Tanh
03 Mul_3    Mul
04 Add_4    Add
~~~

这说明模型语义为 `x -> Mul(alpha,x) -> Tanh -> Mul -> Add`；本导出图共计 41 个 `Tanh`。
`Tanh_2` 是 ONNX node name，`Tanh` 是 ONNX `op_type`，两者不要与最终 kernel 名混淆。

**ACE lowering（E2，`dyt_mode12/build___aom_0/__aom_0_model_.onnx`）**仍能一一看到：

~~~text
"__Mul_1_Mul_0__25"   | ace.multiply
"__Tanh_2_Tanh_0__26" | ace.tanh
~~~

名字前半段保留了源节点线索；末尾内部编号只用于编译期追踪，不是 layer 序号、时延或 core
编号。

**OpFusion 后（E2，`__aom_0_model_after_OpFusion.onnx`）**两个节点不再独立，而是一个：

~~~text
node.name    = "__Mul_1_Mul_0__25__8_apex_fused_op_0"
node.op_type = age.backend.fused_op
output_types = Type: Tensor[(1, 56, 56, 128), float16]
~~~

其 `op_attrs` 的关键原文为：

~~~text
fused_op_type="apex_fused_op",
attrs=[
  ... "fuse_root_flag": 1, "name": "__Mul_1_Mul_0__25",
      "fuse_group_id": 113 ...,
  ... "fuse_root_flag": 0, "output_anchor_index": 0,
      "name": "__Tanh_2_Tanh_0__26", "fuse_group_id": 113 ...
]
~~~

字段的准确读法如下：

| 字段 | 这里表示什么 |
|---|---|
| `age.backend.fused_op` | 后端整体编译的融合算子组，不再是一个普通 ONNX node |
| `fused_op_type="apex_fused_op"` | 该组走通用 elementwise/fused backend kernel 路径 |
| 同一个 `fuse_group_id=113` | `Mul` 与 `Tanh` 被装进同一个融合组 |
| `fuse_root_flag=1` | 融合组锚点/根；此例为 `Mul` |
| `fuse_root_flag=0` | 同组内部节点；此例为 `Tanh` |
| `tmp_anchor_index` | 融合 kernel 内部临时 SSA 值的编号 |
| `output_anchor_index=0` | 哪个内部结果成为该融合组的对外输出 |
| `(1,56,56,128), float16` | 融合组的 tensor 输出 shape/dtype；不是执行时间 |

final graph 中该组又被后端切成 `cut_0`、`cut_1` 两个生成节点。例如实际 trace 名称是：

~~~text
__Mul_1_Mul_0__25_cut_1__19_apex_fused_op_0_tmpl_0
~~~

`cut_*` / `tmpl_*` 是生成名的一部分，可用于跨文件匹配；当前公开材料不足以把它们严格
解释成“固定大小 tile”或“固定次数 DDR copy”。真正的 placement 需要连接 SQLite 的两张表：

~~~text
Record:
  Id=1219, MonitorID=0, Type=slice,
  Start=1111393406587794, Duration=1740,
  Name=__Mul_1_Mul_0__25_cut_1__19_apex_fused_op_0_tmpl_0

Monitor (用 MonitorID=0 join):
  Layer1=NPU cluster 0
  Layer2=Fermat core 0
  Layer3=-1
~~~

下一条记录 `MonitorID=1` 对应 `NPU cluster 0 / Fermat core 1`。这给出 E2 的实际执行
证据：**包含 `__Tanh_2_Tanh_0__26` 的 fusion group 被编译成 Fermat slice，而不是一个
可单独计时的 Tanh kernel。** SDK 还可见
`allspark/ace/kernels/fermat/unary.py` 中的 `T.tanh(...)`，这是“Fermat 支持 tanh”的 E1
实现旁证；调度结论依然以 `Monitor.Layer2` 为准。

#### B. baseline 的 LayerNorm 实际在哪里

baseline 源 ONNX 没有名为 `LayerNorm` 的单一节点；第一个 LN 已展开为：

~~~text
Cast -> ReduceMean -> Sub -> Pow -> ReduceMean -> Add -> Sqrt
     -> Div -> Cast -> Mul -> Add
~~~

其物理含义是：对输入 feature map `x` 求均值 `mean(x)`，得到 `x-mean(x)`，求平方均值
（方差），加 `eps`、开方、相除，再乘 `gamma`、加 `beta`。OpFusion 属性进一步保留了同一组
的来源，例如：

~~~text
__ReduceMean_2_ReduceMean_0__25_lglz_0
__Sub_7_Sub_1__26
__Pow_5_Pow_0__30_square_lglz_0
__ReduceMean_6_ReduceMean_1__31_lglz_0
__Add_9_Add_0__34
__Sqrt_10_Sqrt_0__35
__Div_11_Div_0__36
__Mul_13_Mul_0__39
__Add_14_Add_1__42
~~~

其中 `axis=[1]` 表示此处 reduce 沿 NCHW 的 channel 维；部分中间操作 `dtype="float32"`，
表明为数值稳定性升精度，最终又返回 FP16。对应 trace 的一个 fused slice 是：

~~~text
Name=__ReduceMean_2_ReduceMean_0__25_lglz_0_cut_0__13_
     apex_fused_op_0_IdentityMod_0_tmpl_0
Layer1=NPU cluster 0
Layer2=Fermat core 0
Duration=4814
~~~

按名字含 `ReduceMean` 聚合 baseline trace，结果为 **1,436 个 Fermat slice / 48 个唯一
具名 kernel，零 Cayley**。因此应说“baseline 的 LN reduction/elementwise fusion chain
在 Fermat”，不应说“存在一个独立 LayerNorm kernel 在 Fermat”。

#### C. 为什么这仍可能加速：减少的是独立 layout 边界，不是把 Tanh 删掉

本 run 的 final graph 与 trace 汇总如下：

| 变体 | final graph 中独立 layout node | layout trace slice | Fermat 总 slice |
|---|---:|---:|---:|
| baseline | 89 `ace.permute` | 2,848 | 6,832 |
| DyT | 7 `ace.permute` + 2 `ace.physical_layout_transform` | 288 | 3,360 |
| DyT-no-permute | 7 `ace.permute` | 224 | 4,352 |

所以 `DyT` **仍有 9 个独立 layout node，不能说“中间没有 layout 算子”**；但其被 mode-12
观测到的独立 layout slice 相比 baseline 从 2,848 降至 288，减少约 **89.89%**。

一个贴近此图的对比是：

~~~text
baseline（概念化，不代表每一条边都一一独立）
  Conv/Cayley -> 独立 permute/Fermat -> LN 分解/Fermat
              -> 独立 permute/Fermat -> 后续 Conv/Cayley

DyT
  Conv/Cayley -> [Mul + Tanh + scale/bias] 融合 Fermat kernel -> 后续算子
~~~

独立 `permute` 通常要对大 feature map 做 address remap/gather/scatter，并额外形成 kernel
launch、读写、同步、cache 容量压力或 spill 风险。反之，`Mul+Tanh` 融合后，Fermat 可把一个
tile 读入寄存器/局部存储后连续完成缩放和 tanh，不必为 **Mul 到 Tanh** 建立一个可观测的全
feature-map 中间边界。这是“同样用 Fermat，却可能更快”的第一性原理。

但下面三点不能越界：

1. `layout slice` 减少并不自动等于同样比例的 DDR 字节减少；本 run 没有依赖边和 transaction
   数据。
2. `Tanh` 已融合，不能从 trace 拆出它自己的耗时，也不能将该 fused kernel 的 `Duration`
   全归因给 Tanh。
3. 当前三变体 accuracy gate 未通过；诊断性 mode-0 p50
   `baseline/DyT/no-permute = 5.428/5.129/3.489 ms` 只能说明一个值得继续验证的调度趋势，
   不能作为正确模型上的正式性能结论。

`Record.Duration` 是单个并行 slice 的计时；同一 kernel 的多个 core slice 可能重叠，不能
把 32 个 slice duration 相加成 kernel latency 或 E2E latency。`Record.RelationID` 在本 run
全部为空，故也不能由这组证据声称任一 Cayley→Fermat 边必定写回 DDR。

### 16.15 可迁移但不可混同的外部参照

- [AllSpark CCA 高性能编程指南](../docs/spark_guide_140/AllSpark%20CCA%20高性能编程指南.md)
  与 [AllSpark CCA 开发手册](../docs/spark_guide_140/AllSpark%20CCA%20开发手册.md) 是本
  文对 Fermat Grid/Block/Warp、PVT/GSM/L1/L2 的 E1 来源。
- Apple Metal 的 `simdgroup`/`threadgroup` 和 CUDA 的 warp/block 是理解 Fermat 编程模型
  的有用参照，但不能替换芯片计数器。Apple Silicon 的统一内存也只说明 CPU/GPU 不必经由
  独立 PCIe 设备内存，并不等于没有 cache/fence/bandwidth 成本：
  [Apple: Porting Metal code to Apple silicon](https://developer.apple.com/documentation/apple-silicon/porting-your-metal-code-to-apple-silicon)。
- CUDA 的 Grid → Block → Warp 层次和 memory hierarchy 可用来理解“跨 core slot 不等于
  warp”，但它们不是 N93X 的公开微架构说明：
  [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)。
- MLIR 的 `conversion target`、rewrite pattern、type converter 是理解“consumer 的
  layout/type 约束如何合法化并改写 def-use 链”的开源参照，不代表 AllSpark 必然复用了
  MLIR： [MLIR Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/)。

若后续要把这些推断升级为 N93X 的 E2 结论，优先级是：先修复 golden 正确性，再采集能
关联 task/kernel 的硬件 trace 与 DDR 窗口/transaction 数据；最后以相同输入、热身后多次
mode-0 E2E 测试校验。不要从 graph node 数、Grid 行数或 software slice 数直接推带宽和
产品延迟。
