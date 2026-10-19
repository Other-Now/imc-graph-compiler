"""Graph-level passes on torch.fx GraphModules, run before import to MLIR.

fold_conv_bn  rewrites conv -> batchnorm into one conv with adjusted weights.
mark_fusion   decides which elementwise ops are fused into their consumer;
              the importer then emits each fused chain as one linalg.generic.
"""
import operator

import torch
import torch.nn as nn
import torch.nn.functional as F

# FX targets that the importer treats as elementwise (fusable) ops.
ELEMENTWISE_FUNCS = {F.relu, torch.relu, operator.add, torch.add, operator.mul, torch.mul}
ANCHOR_MODULES = (nn.Linear, nn.Conv2d)   # ops with a fusable bias epilogue


def _module(gm, node):
    return gm.get_submodule(node.target) if node.op == "call_module" else None


def fold_conv_bn(gm: torch.fx.GraphModule) -> int:
    """y = gamma * (conv(x) - mean) / sqrt(var + eps) + beta is a conv with
    per-output-channel scaled weights and a new bias. Valid in eval mode only."""
    folded = 0
    for node in list(gm.graph.nodes):
        bn = _module(gm, node)
        if not isinstance(bn, nn.BatchNorm2d):
            continue
        src = node.args[0]
        conv = _module(gm, src)
        if not isinstance(conv, nn.Conv2d) or len(src.users) != 1:
            continue                      # conv output also used elsewhere: cannot fold
        with torch.no_grad():
            s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            conv.weight.mul_(s.reshape(-1, 1, 1, 1))
            b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
            new_b = (b - bn.running_mean) * s + bn.bias
            conv.bias = nn.Parameter(new_b)
        node.replace_all_uses_with(src)
        gm.graph.erase_node(node)
        folded += 1
    gm.graph.lint()
    gm.delete_all_unused_submodules()
    gm.recompile()
    return folded


def is_elementwise(gm, node) -> bool:
    if node.op == "call_function" and node.target in ELEMENTWISE_FUNCS:
        return True
    if node.op == "call_method" and node.target in ("relu", "add", "mul"):
        return True
    return isinstance(_module(gm, node), nn.ReLU)


def has_epilogue(gm, node) -> bool:
    m = _module(gm, node)
    return isinstance(m, ANCHOR_MODULES) and m.bias is not None


def mark_fusion(gm: torch.fx.GraphModule) -> list:
    """Producer-consumer fusion of elementwise chains, including the bias add
    of a linear/conv ("epilogue fusion"). A value is fused into its consumer
    when it has exactly one user and that user is elementwise; otherwise it
    must be materialized. Sets node.meta["fuse"] and returns the groups."""
    for n in gm.graph.nodes:
        fusable_value = is_elementwise(gm, n) or has_epilogue(gm, n)
        users = list(n.users)
        n.meta["fuse"] = bool(fusable_value and len(users) == 1 and is_elementwise(gm, users[0]))
    groups, seen = [], set()
    for n in reversed(list(gm.graph.nodes)):          # walk from each group's root back
        if n in seen or n.meta.get("fuse") or not (is_elementwise(gm, n) or has_epilogue(gm, n)):
            continue
        group, stack = [], [n]
        while stack:
            m = stack.pop()
            group.append(m.name)
            seen.add(m)
            stack.extend(a for a in m.all_input_nodes if a.meta.get("fuse"))
        if len(group) > 1 or has_epilogue(gm, n):
            groups.append(list(reversed(group)))
    return list(reversed(groups))


def clear_fusion(gm):
    for n in gm.graph.nodes:
        n.meta["fuse"] = False
