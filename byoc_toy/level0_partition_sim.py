#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BYOC 分区机制的最小可运行模拟 —— 纯 Python，零依赖，不需要安装 TVM。

目的：把 Relax BYOC 的四个机制拆开摊平给你看

  1) DFPattern              —— 结构层过滤：算子链必须在图上"直接相邻"
  2) check_func             —— 语义层过滤：dtype / 对齐 / 中间结果是否泄漏
  3) FuseOpsByPattern       —— 把匹配的链切成 composite function（打 Composite / Codegen 属性）
  4) MergeCompositeFunctions—— 把同后端的多个 composite 合并成一个，减少 conversion boundary

跑法：
    python3 level0_partition_sim.py
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

HOST = "host"
BACKEND = "demo"


# ================================================================ 图的定义

@dataclass
class Node:
    name: str
    op: str                      # "param" = 函数形参，视为 host 上的输入
    inputs: Tuple[str, ...]
    shape: Tuple[int, ...]
    dtype: str = "float32"


@dataclass
class Graph:
    nodes: Dict[str, Node]
    order: List[str]
    output: str

    def consumers(self, name: str) -> List[str]:
        return [n for n in self.order if name in self.nodes[n].inputs]


def build_graph(spec: dict) -> Graph:
    """造一张:  x -> {matmul -> [reshape] -> add -> relu} * n_blocks -> sin"""
    m, k, n = spec["M"], spec["K"], spec["N"]
    dt = spec["dtype"]
    nodes: Dict[str, Node] = {}
    order: List[str] = []

    def add(nd: Node) -> Node:
        nodes[nd.name] = nd
        order.append(nd.name)
        return nd

    add(Node("x", "param", (), (m, k), dt))
    cur, out_dim = "x", n

    for i in range(spec["n_blocks"]):
        kk, nn = (k, n) if i == 0 else (n, n)
        add(Node(f"w{i}", "param", (), (kk, nn), dt))
        add(Node(f"b{i}", "param", (), (nn,), dt))
        add(Node(f"matmul{i}", "relax.matmul", (cur, f"w{i}"), (m, nn), dt))

        mid = f"matmul{i}"
        if spec["break_pattern"]:
            # 数值恒等的 reshape，唯一使命是"插一刀"打断图案
            add(Node(f"reshape{i}", "relax.reshape", (mid,), (m, nn), dt))
            mid = f"reshape{i}"

        add(Node(f"bias_add{i}", "relax.add", (mid, f"b{i}"), (m, nn), dt))
        add(Node(f"relu{i}", "relax.nn.relu", (f"bias_add{i}",), (m, nn), dt))
        cur, out_dim = f"relu{i}", nn

    add(Node("host_sin", "relax.sin", (cur,), (m, out_dim), dt))
    return Graph(nodes, order, "host_sin")


# ============================================================ ① 结构图案
# 对应 DFPattern: relu(add(matmul(x, w), b))

PATTERN_NAME = f"{BACKEND}.matmul_bias_relu"
PATTERN_CHAIN = ("relax.matmul", "relax.add", "relax.nn.relu")


def match_chain(g: Graph, start: str, chain: Tuple[str, ...]) -> Optional[List[str]]:
    """结构匹配：沿"消费者"方向走一条 op 类型严格等于 chain 的相邻链。

    这就是 DFPattern 干的事 —— 中间被任何算子插一脚，链就断了。
    """
    if g.nodes[start].op != chain[0]:
        return None
    matched = [start]
    cur = start
    for op in chain[1:]:
        cands = [c for c in g.consumers(cur) if g.nodes[c].op == op]
        if len(cands) != 1:
            return None
        cur = cands[0]
        matched.append(cur)
    return matched


# ============================================================ ② 语义检查

@dataclass
class CheckContext:
    """对应 TVM 的 PatternCheckContext"""
    g: Graph
    matched: List[str]

    def has_leaking_intermediate_variables(self) -> bool:
        """链上除最后一个节点（= group 的输出）外，若还有链外消费者，即为泄漏"""
        inside = set(self.matched)
        for name in self.matched[:-1]:
            for c in self.g.consumers(name):
                if c not in inside:
                    return True
        return False


def check_demo(ctx: CheckContext) -> Tuple[bool, str]:
    """两级过滤的第二级。返回 (是否接受, 原因)"""
    # (a) dtype 约束
    for name in ctx.matched:
        dt = ctx.g.nodes[name].dtype
        if dt not in ("float32", "float16"):
            return False, f"dtype={dt} 不在支持列表 [float32, float16]"

    # (b) 形状对齐约束：matmul 的 K / N 必须是 8 的倍数
    mm = ctx.g.nodes[ctx.matched[0]]
    w = ctx.g.nodes[mm.inputs[1]]          # weight 形状是 [K, N]
    k_dim, n_dim = w.shape[0], w.shape[1]
    if k_dim % 8 or n_dim % 8:
        return False, f"形状不对齐: K={k_dim}, N={n_dim}（需为 8 的倍数）"

    # (c) 中间结果泄漏
    if ctx.has_leaking_intermediate_variables():
        return False, "中间结果被链外消费 (has_leaking_intermediate_variables)"

    return True, "accepted"


# ============================================================ ③ 分区

def fuse_ops_by_pattern(g, chain, check):
    """对应 FuseOpsByPattern：先结构匹配，再语义检查，通过则成组"""
    groups: List[List[str]] = []
    rejected: List[Tuple[str, str]] = []
    taken = set()

    for name in g.order:
        if name in taken:
            continue
        m = match_chain(g, name, chain)
        if m is None:
            continue                                  # 结构层就没匹配上
        ok, reason = check(CheckContext(g, m))
        if not ok:
            rejected.append((m[0], reason))           # 匹配上了，被 check 拒
            continue
        groups.append(m)
        taken.update(m)

    return groups, rejected


# ============================================================ ④ 区域与边界

def assign_regions(g, groups, merge: bool) -> Dict[str, str]:
    """merge=True 对应 MergeCompositeFunctions：同后端的 composite 合成一个区域"""
    region = {n: HOST for n in g.order}
    for i, grp in enumerate(groups):
        rid = f"{BACKEND}#0" if merge else f"{BACKEND}#{i}"
        for n in grp:
            region[n] = rid
    return region


def crossing_edges(g, region):
    """conversion boundary = 一条跨越 host/device 区域边界的数据边"""
    out = []
    for name in g.order:
        for inp in g.nodes[name].inputs:
            if region[inp] != region[name]:
                out.append((inp, name))
    return out


def report(title: str, g, groups, rejected, merge: bool):
    region = assign_regions(g, groups, merge)
    edges = crossing_edges(g, region)
    n_dev_regions = len({r for r in region.values() if r != HOST})

    print(f"\n{'=' * 68}")
    print(f"  {title}")
    print(f"{'=' * 68}")
    print(f"  Composite 组数 : {len(groups)}  ->  {[' + '.join(x) for x in groups]}")
    for node, reason in rejected:
        print(f"  [check 拒绝]   {node}: {reason}")
    if not groups and not rejected:
        print("  [DFPattern 失败] 结构层就没匹配上，check 根本没机会执行")
    print(f"  device 区域数  : {n_dev_regions}")
    print(f"  conversion boundary : {len(edges)} 条跨界边")
    for a, b in edges:
        print(f"      {a}[{region[a]}]  --cross-->  {b}[{region[b]}]")

    print("  分区后视图:")
    for name in g.order:
        nd = g.nodes[name]
        tag = f"[{region[name]}]"
        if nd.inputs:
            print(f"      {name:<14}{tag:<9} {nd.op}({', '.join(nd.inputs)})")
        else:
            print(f"      {name:<14}{tag:<9} {nd.op}   shape={nd.shape}")


# ============================================================ 主流程

CASE_SPECS = {
    "supported_with_fallback": dict(M=8, K=16, N=16, n_blocks=1, break_pattern=False, dtype="float32"),
    "pattern_mismatch":        dict(M=8, K=16, N=16, n_blocks=1, break_pattern=True,  dtype="float32"),
    "support_rejected":        dict(M=8, K=12, N=15, n_blocks=1, break_pattern=False, dtype="float32"),
    "two_blocks":              dict(M=8, K=16, N=16, n_blocks=2, break_pattern=False, dtype="float32"),
}


def main():
    print("图案:", PATTERN_NAME, "=", " -> ".join(PATTERN_CHAIN))
    for case, spec in CASE_SPECS.items():
        g = build_graph(spec)
        groups, rejected = fuse_ops_by_pattern(g, PATTERN_CHAIN, check_demo)
        report(case, g, groups, rejected, merge=False)
        if len(groups) > 1:
            report(case + "  [after MergeCompositeFunctions]", g, groups, rejected, merge=True)
    print()


if __name__ == "__main__":
    main()
