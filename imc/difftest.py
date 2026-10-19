"""Differential tests: compiled MLIR vs PyTorch.

  1. fp32 import of the three models, fused and unfused, vs eager PyTorch
     (tolerance: different summation order).
  2. int8 crossbar path (imc dialect -> tile -> lower) vs quant.Emulator:
     logits and last-layer int32 accumulators must be bit-identical.
  3. N random FX graphs, fused and unfused, vs eager.

    python -m imc.difftest --fuzz 100
"""
import argparse
import os
import time

import numpy as np
import torch
from torch.fx import symbolic_trace

from . import fxpasses, models, mlirrun, quant
from .fuzz import random_graph
from .importer import UnsupportedOpError, import_graph

CKPT = os.path.join(mlirrun.REPO, "checkpoints")
RTOL, ATOL = 1e-4, 1e-5


def load(name):
    m = models.build(name)
    p = os.path.join(CKPT, f"{name}.pt")
    if os.path.exists(p):
        m.load_state_dict(torch.load(p))
    return m.eval()


def traced(name):
    gm = symbolic_trace(load(name))
    fxpasses.fold_conv_bn(gm)
    return gm


def check_fp32(gm, x, fuse):
    fxpasses.clear_fusion(gm)
    if fuse:
        fxpasses.mark_fusion(gm)
    ref = gm(x).detach().numpy()
    fwd, imp = import_graph(gm, [x])
    out = mlirrun.compile_and_run(fwd, [x.numpy()], imp.result_types)[0]
    err = float(np.abs(out - ref).max()) if out.size else 0.0
    return np.allclose(out, ref, rtol=RTOL, atol=ATOL), err


def check_quant(gm, x, calib, cfg):
    q = quant.calibrate(gm, calib, cfg)
    emu = quant.Emulator(gm, q, cfg.rows)
    with torch.no_grad():
        ref = emu.run(x).numpy()
    ref_acc = emu.last_acc.numpy()
    fxpasses.clear_fusion(gm)
    fxpasses.mark_fusion(gm)
    fwd, imp = import_graph(gm, [x], quant=q, expose_acc=True)
    passes = ["-imc-lower-to-linalg"]          # ideal ADC: exact matmul, tiling would not change it
    if cfg.adc_bits:
        passes.insert(0, f"-imc-tile-crossbars=rows={cfg.rows} cols=256")
    out, acc = mlirrun.compile_and_run(fwd, [x.numpy()], imp.result_types, imc_passes=passes)
    same = np.array_equal(out.view(np.int32), ref.view(np.int32)) and np.array_equal(acc, ref_acc)
    return same, imp.stats.qdq_folds, fwd.count("imc.matmul")


def unsupported_cases():
    class Sig(torch.nn.Module):
        def forward(self, x):
            return torch.sigmoid(x)

    class Bcast(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.b = torch.nn.Parameter(torch.ones(3))

        def forward(self, x):
            return x + self.b

    cnn = symbolic_trace(load("cnn"))                       # BN deliberately not folded
    yield "sigmoid", symbolic_trace(Sig()), torch.randn(2, 3)
    yield "unfolded BatchNorm", cnn, torch.randn(1, 3, 32, 32)
    yield "float64 input", symbolic_trace(Sig()), torch.randn(2, 3, dtype=torch.float64)
    yield "broadcasting add", symbolic_trace(Bcast()), torch.randn(4, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuzz", type=int, default=100)
    a = ap.parse_args()
    torch.manual_seed(0)
    lines, ok = [], True

    def log(s):
        print(s, flush=True)
        lines.append(s)

    log("== fp32 import vs eager PyTorch (rtol 1e-4, atol 1e-5) ==")
    for name, x in [("mlp", torch.randn(4, 3, 32, 32)), ("cnn", torch.randn(4, 3, 32, 32)),
                    ("transformer", torch.randn(16, 64))]:
        gm = traced(name)
        for fuse in (False, True):
            good, err = check_fp32(gm, x, fuse)
            ok &= good
            log(f"{name:12s} {'fused  ' if fuse else 'unfused'} max|diff| {err:.2e}  {'OK' if good else 'FAIL'}")

    log("== int8 crossbar path vs emulator (bit-exact logits + int32 accumulators) ==")
    calib = torch.randn(256, 3, 32, 32)
    for name in ("mlp", "cnn"):
        gm = traced(name)
        for cfg in (quant.Config(), quant.Config(adc_bits=8, rows=256), quant.Config(adc_bits=6, rows=128),
                    quant.Config(adc_bits=4, rows=64, adc_calib="pct")):
            x = torch.randn(2, 3, 32, 32)
            same, folds, nmm = check_quant(gm, x, calib, cfg)
            ok &= same
            adc = f"adc {cfg.adc_bits}b/{cfg.adc_calib}" if cfg.adc_bits else "ideal ADC"
            log(f"{name:4s} rows={cfg.rows:3d} {adc:16s} imc.matmul={nmm} qdq-folds={folds}  "
                f"{'BIT-EXACT' if same else 'MISMATCH'}")

    log("== importer rejects what it cannot compile, with a clear message ==")
    for label, gm, x in unsupported_cases():
        try:
            import_graph(gm, [x])
            msg, good = "no error raised", False
        except UnsupportedOpError as e:
            msg, good = str(e), True
        ok &= good
        log(f"{label:28s} {'OK ' if good else 'FAIL'} {msg}")

    log(f"== random FX graphs: {a.fuzz} graphs x (unfused, fused) ==")
    t0, bad = time.time(), []
    for seed in range(a.fuzz):
        gm, x = random_graph(seed)
        for fuse in (False, True):
            try:
                good, err = check_fp32(gm, x, fuse)
            except Exception as e:                      # importer or toolchain failure is a failure
                good, err = False, str(e).splitlines()[0]
            if not good:
                bad.append((seed, fuse, err))
    ok &= not bad
    log(f"{2 * a.fuzz - len(bad)}/{2 * a.fuzz} passed in {time.time() - t0:.0f}s"
        + "".join(f"\n  FAIL seed={s} fused={f}: {e}" for s, f, e in bad))

    os.makedirs(os.path.join(mlirrun.REPO, "results"), exist_ok=True)
    with open(os.path.join(mlirrun.REPO, "results", "difftest.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
