"""Random FX graphs over the importer's op set, for differential testing."""
import operator
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


def random_graph(seed):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    root, g = nn.Module(), torch.fx.Graph()
    M, D = rng.randint(1, 8), rng.randint(1, 24)
    vals = [(g.placeholder("x"), (M, D))]
    for i in range(rng.randint(2, 9)):
        v, shape = vals[-1] if rng.random() < 0.6 else rng.choice(vals)
        op = rng.choice(["linear", "linear", "relu", "add", "mul", "t", "matmul", "softmax", "layernorm"])
        if op == "linear":
            out = rng.randint(1, 24)
            mod = nn.Linear(shape[1], out, bias=rng.random() < 0.8)
            root.add_module(f"l{i}", mod)
            vals.append((g.call_module(f"l{i}", (v,)), (shape[0], out)))
        elif op == "relu":
            vals.append((g.call_function(F.relu, (v,)), shape))
        elif op == "add":
            same = [n for n, s in vals if s == shape]
            vals.append((g.call_function(operator.add, (v, rng.choice(same))), shape))
        elif op == "mul":
            vals.append((g.call_function(operator.mul, (v, round(rng.uniform(-2, 2), 3))), shape))
        elif op == "t":
            vals.append((g.call_method("t", (v,)), (shape[1], shape[0])))
        elif op == "matmul":
            rhs = [(n, s) for n, s in vals if s[0] == shape[1]]
            if rhs:
                n, s = rng.choice(rhs)
                vals.append((g.call_function(operator.matmul, (v, n)), (shape[0], s[1])))
        elif op == "softmax":
            vals.append((g.call_function(F.softmax, (v,), {"dim": -1}), shape))
        elif op == "layernorm":
            mod = nn.LayerNorm(shape[1])
            with torch.no_grad():
                mod.weight.uniform_(0.5, 1.5)
                mod.bias.uniform_(-0.5, 0.5)
            root.add_module(f"ln{i}", mod)
            vals.append((g.call_module(f"ln{i}", (v,)), shape))
    g.output(vals[-1][0])
    return torch.fx.GraphModule(root, g), torch.randn(M, D)
