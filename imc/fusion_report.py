"""Kernel count and intermediate-tensor bytes before/after fusion, compared
with upstream MLIR's elementwise fusion and with TorchInductor (CPU).

    python -m imc.fusion_report      -> results/fusion.md
"""
import os
import re

import torch

from . import fxpasses, mlirrun
from .difftest import traced
from .importer import import_graph

INPUTS = {"mlp": (1, 3, 32, 32), "cnn": (1, 3, 32, 32), "transformer": (16, 64)}


def inductor_kernels(gm, x):
    """(fused C++ kernels Inductor generated, calls to extern kernels like mm/conv)."""
    try:
        from torch._inductor.utils import run_and_get_code
        torch._dynamo.reset()
        _, code = run_and_get_code(torch.compile(gm), x)
        src = "\n".join(code)
        gen = len(set(re.findall(r"^(cpp_fused_\w+) = async_compile", src, re.M)))
        ext = len(re.findall(r"extern_kernels\.\w+\(", src)) + len(re.findall(r"torch\.ops\.mkldnn\.\w+", src))
        return gen, ext
    except Exception as e:                                   # no C++ compiler, etc.
        print("inductor unavailable:", str(e).splitlines()[0])
        return None, None


def main():
    rows = []
    for name, shape in INPUTS.items():
        gm = traced(name)
        x = torch.randn(*shape)
        fxpasses.clear_fusion(gm)
        unfused, iu = import_graph(gm, [x])
        groups = fxpasses.mark_fusion(gm)
        fused, ifu = import_graph(gm, [x])
        upstream = mlirrun.mlir_opt("module {\n" + unfused + "\n}", "--linalg-fuse-elementwise-ops", "--canonicalize")
        gen, ext = inductor_kernels(gm, x)
        rows.append((name, mlirrun.count_linalg_ops(unfused), iu.stats.intermediate_bytes,
                     mlirrun.count_linalg_ops(fused), ifu.stats.intermediate_bytes,
                     mlirrun.count_linalg_ops(upstream), gen, ext, len(groups)))
    out = ["| model | kernels unfused | intermediate bytes unfused | kernels fused (mine) | intermediate bytes fused "
           "| upstream `-linalg-fuse-elementwise-ops` | Inductor fused C++ kernels + extern calls |",
           "|---|---|---|---|---|---|---|"]
    for n, ku, bu, kf, bf, kup, gen, ext, ng in rows:
        ind = f"{gen} + {ext}" if gen is not None else "n/a"
        out.append(f"| {n} | {ku} | {bu:,} | {kf} | {bf:,} ({100 * (bu - bf) / bu:.0f}% less) | {kup} | {ind} |")
    text = "\n".join(out)
    print(text)
    with open(os.path.join(mlirrun.REPO, "results", "fusion.md"), "w") as f:
        f.write("Batch 1 (CNN, MLP) and 16 tokens (transformer). Kernels = linalg ops other than fill; "
                "conv padding counts as a kernel.\n\n" + text + "\n")


if __name__ == "__main__":
    main()
