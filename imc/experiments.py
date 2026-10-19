"""CIFAR-10 accuracy under int8 / ADC / crossbar choices, plus the tiling and
cost sweep. Calibration uses the first 1000 *training* images; accuracy is
on the full 10,000-image test set. Accuracy comes from quant.Emulator, which
difftest.py checks bit-exact against the compiled MLIR.

    python -m imc.experiments       -> results/accuracy.md, results/tiling.md
"""
import os
import time

import torch
from torch.fx import symbolic_trace

from . import crossbar, data, fxpasses, quant
from .difftest import load
from .mlirrun import REPO

OUT = os.path.join(REPO, "results")


def fp32_acc(model, x, y):
    with torch.no_grad():
        return sum((model(x[i:i + 500]).argmax(1) == y[i:i + 500]).sum().item() for i in range(0, len(x), 500)) / len(x)


def main():
    torch.set_num_threads(os.cpu_count())
    xtr, _ = data.train_set()
    calib = xtr[:1000]
    del xtr
    xte, yte = data.test_set()
    acc_lines = ["| model | configuration | test accuracy | change vs fp32 |", "|---|---|---|---|"]
    sweep_lines = ["| model | crossbar | tiles | utilization | arrays on chip | fits? | ADC conv./img "
                   "| energy/img (6-bit) | latency/img | acc 6-bit ADC | acc 4-bit ADC |",
                   "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in ("mlp", "cnn"):
        model = load(name)
        base = fp32_acc(model, xte, yte)
        gm = symbolic_trace(model)
        nfold = fxpasses.fold_conv_bn(gm)
        rows = [("fp32 (PyTorch eager)", base),
                (f"fp32, {nfold} conv+BN folded", fp32_acc(gm, xte, yte))]

        def run(label, cfg):
            t0 = time.time()
            a = quant.evaluate(gm, quant.calibrate(gm, calib, cfg), cfg.rows, xte, yte)
            print(f"{name} {label}: {a:.4f} ({time.time() - t0:.0f}s)", flush=True)
            rows.append((label, a))
            return a

        run("int8, per-tensor weights, min-max activations", quant.Config(per_channel=False))
        run("int8, per-channel weights, min-max activations", quant.Config())
        run("int8, per-channel weights, 99.99th-pct activations", quant.Config(act_calib="pct"))
        for bits in (8, 6, 4):
            for adc_cal in ("minmax", "pct"):
                run(f"+ {bits}-bit ADC ({adc_cal} range), 256-row crossbar",
                    quant.Config(act_calib="pct", adc_bits=bits, adc_calib=adc_cal))
        for label, a in rows:
            acc_lines.append(f"| {name} | {label} | {100 * a:.2f}% | {100 * (a - base):+.2f} pp |")

        ls = crossbar.layers(gm, (1, 3, 32, 32))
        for R, C in ((64, 64), (128, 128), (256, 256), (512, 512), (256, 64)):
            c = crossbar.network_cost(ls, R, C, 6)
            a6 = quant.evaluate(gm, quant.calibrate(gm, calib, quant.Config(act_calib="pct", adc_bits=6, adc_calib="pct", rows=R)), R, xte, yte)
            a4 = quant.evaluate(gm, quant.calibrate(gm, calib, quant.Config(act_calib="pct", adc_bits=4, adc_calib="pct", rows=R)), R, xte, yte)
            print(f"{name} {R}x{C}: {c} acc6 {a6:.4f} acc4 {a4:.4f}", flush=True)
            sweep_lines.append(
                f"| {name} | {R}x{C} | {c['tiles']} | {100 * c['util']:.0f}% | {c['arrays']} | "
                f"{'yes' if c['fits'] else 'no: reprogram'} | {c['adc']:,} | {c['energy_uj'] * 1e3:.1f} nJ | "
                f"{c['latency_us']:.1f} us | {100 * a6:.2f}% | {100 * a4:.2f}% |")
        if name == "cnn":
            per = ["| layer | K x N | vectors/img | tiles @256x256 | utilization | ADC conv./img |", "|---|---|---|---|---|---|"]
            for l in ls:
                t = crossbar.tile(l, 256, 256)
                per.append(f"| {l.name} | {l.K} x {l.N} | {l.M} | {t['tiles']} | {100 * t['util']:.1f}% | {t['adc']:,} |")
    with open(os.path.join(OUT, "accuracy.md"), "w") as f:
        f.write("Calibration: 1000 training images. Accuracy: 10,000 test images. pp = percentage points.\n\n"
                + "\n".join(acc_lines) + "\n")
    with open(os.path.join(OUT, "tiling.md"), "w") as f:
        f.write("Analytical model (imc/crossbar.py), NOT hardware. Chip = 4M int8 cells. ADC range calibrated "
                "at the 99.99th percentile.\nAccuracy columns are measured (emulator, 10k test images).\n\n"
                + "\n".join(sweep_lines) + "\n\nCNN per layer:\n\n" + "\n".join(per) + "\n")


if __name__ == "__main__":
    main()
