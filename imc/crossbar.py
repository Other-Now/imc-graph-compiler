"""Analytical crossbar mapping and cost model. NOT a hardware measurement:
every constant below is an assumption, chosen to be order-of-magnitude
plausible for a charge-domain SRAM in-memory-compute macro. Use the outputs
to compare mappings with each other, not as absolute numbers."""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.fx.passes.shape_prop import ShapeProp

# ---- assumptions ---------------------------------------------------------
CHIP_CELLS = 64 * 256 * 256     # int8 weight cells on chip (64 arrays of 256x256)
T_MVM_NS = 100.0                # one analog MVM on one array, incl. ADC readout
E_ADC8_PJ = 2.0                 # one 8-bit ADC conversion; scales as 2^bits (Walden FoM)
E_CELL_FJ = 5.0                 # analog multiply-accumulate per active cell
E_ADD_PJ = 0.03                 # one 32-bit digital partial-sum add
E_WRITE_PJ = 1.0                # writing one int8 weight cell
E_DRAM_PJ_PER_B = 20.0          # DRAM traffic


@dataclass
class Layer:
    name: str
    K: int          # rows of the weight matrix (in-features, or C*KH*KW)
    N: int          # columns (out-features / out-channels)
    M: int          # input vectors per image (1 for linear, OH*OW for conv)


def layers(gm, input_shape):
    ShapeProp(gm).propagate(torch.zeros(1, *input_shape[1:]))
    out = []
    for n in gm.graph.nodes:
        if n.op != "call_module":
            continue
        m = gm.get_submodule(n.target)
        if isinstance(m, nn.Linear):
            out.append(Layer(n.target, m.in_features, m.out_features, 1))
        elif isinstance(m, nn.Conv2d):
            _, _, oh, ow = n.meta["tensor_meta"].shape
            out.append(Layer(n.target, m.in_channels * m.kernel_size[0] * m.kernel_size[1], m.out_channels, oh * ow))
    return out


def tile(layer, R, C):
    kt, nt = math.ceil(layer.K / R), math.ceil(layer.N / C)
    return dict(tiles=kt * nt, util=layer.K * layer.N / (kt * nt * R * C), mvms=kt * nt * layer.M,
                adc=kt * layer.N * layer.M, adds=(kt - 1) * layer.N * layer.M,
                cell_macs=layer.K * layer.N * layer.M)


def network_cost(ls, R, C, adc_bits, batch=1):
    """Per-image cost of one weight-stationary mapping."""
    arrays = CHIP_CELLS // (R * C)
    per = [tile(l, R, C) for l in ls]
    tiles = sum(p["tiles"] for p in per)
    fits = tiles <= arrays
    weights = sum(l.K * l.N for l in ls)
    e_adc = E_ADC8_PJ * 2.0 ** (adc_bits - 8)
    # If the network does not fit, every tile is rewritten once per batch.
    writes = 0 if fits else weights / batch
    energy_pj = sum(p["adc"] for p in per) * e_adc + sum(p["cell_macs"] for p in per) * E_CELL_FJ / 1e3 \
        + sum(p["adds"] for p in per) * E_ADD_PJ + writes * (E_WRITE_PJ + E_DRAM_PJ_PER_B)
    # Layers run one after another; a layer's tiles run in parallel, in waves
    # of `arrays` when the layer alone has more tiles than the chip.
    latency_ns = sum(l.M * T_MVM_NS * math.ceil(p["tiles"] / arrays) for l, p in zip(ls, per))
    return dict(arrays=arrays, tiles=tiles, fits=fits,
                util=sum(l.K * l.N for l in ls) / (tiles * R * C),
                adc=sum(p["adc"] for p in per), mvms=sum(p["mvms"] for p in per),
                reprogram_cells_per_img=writes, energy_uj=energy_pj / 1e6, latency_us=latency_ns / 1e3)


def reprogram_table(models_and_layers, R=256, C=256, batches=(1, 16, 256)):
    """When the chip holds fewer arrays than the network needs, every tile is
    rewritten once per batch; batching amortizes that over more images."""
    global CHIP_CELLS
    saved, lines = CHIP_CELLS, ["| model | arrays on chip (256x256) | tiles needed | fits? | "
                                + " | ".join(f"weight writes/img, batch {b}" for b in batches)
                                + " | energy/img batch 1 -> 256 (6-bit) |", "|---|---|---|---|" + "---|" * (len(batches) + 1)]
    for name, ls in models_and_layers:
        for arrays in (4, 16, 64):
            CHIP_CELLS = arrays * R * C
            cs = [network_cost(ls, R, C, 6, b) for b in batches]
            lines.append(f"| {name} | {arrays} | {cs[0]['tiles']} | {'yes' if cs[0]['fits'] else 'no'} | "
                         + " | ".join(f"{c['reprogram_cells_per_img']:,.0f}" for c in cs)
                         + f" | {cs[0]['energy_uj'] * 1e3:,.0f} -> {cs[-1]['energy_uj'] * 1e3:,.1f} nJ |")
    CHIP_CELLS = saved
    return "\n".join(lines)


if __name__ == "__main__":
    import os
    from torch.fx import symbolic_trace
    from . import fxpasses, models
    out = []
    for n in ("mlp", "cnn"):
        gm = symbolic_trace(models.build(n).eval())
        fxpasses.fold_conv_bn(gm)
        out.append((n, layers(gm, (1, 3, 32, 32))))
    t = reprogram_table(out)
    print(t)
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "reprogram.md"), "w") as f:
        f.write("Analytical model only (imc/crossbar.py). Weight-stationary: if the network's tiles exceed "
                "the arrays on chip, all weights are rewritten once per batch.\n\n" + t + "\n")
