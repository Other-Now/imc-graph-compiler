"""Post-training int8 quantization for crossbar execution, and a PyTorch
emulator of exactly what the compiled MLIR computes.

Per layer (linear, or conv via im2col):
  xq  = clamp(roundeven(x / sx), -128, 127)          activations, per-tensor
  wq  = clamp(roundeven(W / sw[n]), -127, 127)       weights, per-channel or per-tensor
  acc = sum over K-tiles of ADC(xq[:, tile] @ wq[tile, :])   (ADC per crossbar)
  y   = float(acc) * (sx * sw[n]) + bias[n]

The emulator uses the same float32 operations in the same order as the
emitted MLIR, so the two agree bit for bit (checked by difftest.py).
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .importer import QLayer


@dataclass
class Config:
    per_channel: bool = True
    act_calib: str = "minmax"         # "minmax" | "pct" (99.99th percentile of |x|)
    adc_bits: int | None = None       # None = ideal (exact digital accumulation)
    adc_calib: str = "minmax"         # range of the ADC: "minmax" | "pct"
    rows: int = 256                   # crossbar rows: the K-tile size the ADC sees
    pct: float = 99.99


def anchors(gm):
    return {n.target: gm.get_submodule(n.target) for n in gm.graph.nodes
            if n.op == "call_module" and isinstance(gm.get_submodule(n.target), (nn.Linear, nn.Conv2d))}


def weight_matrix(m):
    """[K, N] in im2col order (c, kh, kw) for conv."""
    w = m.weight.detach().float()
    return w.reshape(w.shape[0], -1).T.contiguous()


def _absmax(t, mode, pct):
    t = t.abs().flatten().double()
    if mode == "minmax":
        return t.max().item()
    k = max(1, int(round(len(t) * (1 - pct / 100))))
    return t.topk(k).values[-1].item()


def round_half_up_shift(v, s):
    return torch.floor((v + (2 ** (s - 1) if s > 0 else 0)) / 2 ** s)


def adc(p, bits, shift):
    """Same as the imc.mvm lowering, on integer-valued float64 tensors."""
    q = round_half_up_shift(p, shift)
    q = q.clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    return q * 2 ** shift


def shift_for(maxabs, bits):
    s = 0
    while np.floor((maxabs + (2 ** (s - 1) if s > 0 else 0)) / 2 ** s) > 2 ** (bits - 1) - 1:
        s += 1
    return s


class Emulator(torch.fx.Interpreter):
    """Runs the folded FX graph with every linear/conv replaced by its crossbar
    emulation. Records per-tile partial sums when `record` is set."""

    def __init__(self, gm, qparams, rows, record=None):
        super().__init__(gm)
        self.q, self.rows, self.record, self.last_acc = qparams, rows, record, None

    def call_module(self, target, args, kwargs):
        if target not in self.q:
            return super().call_module(target, args, kwargs)
        m, q, x = self.module.get_submodule(target), self.q[target], args[0]
        sx = torch.tensor(q.sx, dtype=torch.float32)
        xq = torch.clamp(torch.round(x / sx), -128, 127)
        if isinstance(m, nn.Conv2d):
            N, _, H, W = x.shape
            cols = F.unfold(xq, m.kernel_size, padding=m.padding, stride=m.stride)   # [N, CKK, L]
            cols = cols.transpose(1, 2).reshape(-1, cols.shape[1])
        else:
            cols = xq
        cols, w = cols.double(), torch.from_numpy(q.wq).double()
        acc = torch.zeros(cols.shape[0], w.shape[1], dtype=torch.float64)
        for k0 in range(0, w.shape[0], self.rows):
            p = cols[:, k0:k0 + self.rows] @ w[k0:k0 + self.rows]
            if self.record is not None:
                self.record.setdefault(target, []).append(p.abs().flatten().float())
            acc += adc(p, q.adc_bits, q.adc_shift) if q.adc_bits else p
        acc = acc.to(torch.int32)
        self.last_acc = acc
        y = acc.to(torch.float32) * torch.from_numpy(q.scale) + torch.from_numpy(q.bias)
        if isinstance(m, nn.Conv2d):
            OH = (H + 2 * m.padding[0] - m.kernel_size[0]) // m.stride[0] + 1
            OW = (W + 2 * m.padding[1] - m.kernel_size[1]) // m.stride[1] + 1
            y = y.reshape(N, OH, OW, -1).permute(0, 3, 1, 2)
        return y


def calibrate(gm, calib_x, cfg: Config):
    """gm must already have BN folded. Returns {module name: QLayer}."""
    mods = anchors(gm)
    # 1. activation scales from the fp32 model's layer inputs
    seen = {}
    hooks = [m.register_forward_pre_hook(lambda mod, a, name=name: seen.setdefault(name, []).append(a[0].detach()))
             for name, m in mods.items()]
    with torch.no_grad():
        for i in range(0, len(calib_x), 250):
            gm(calib_x[i:i + 250])
    for h in hooks:
        h.remove()
    q = {}
    for name, m in mods.items():
        sx = np.float32(_absmax(torch.cat([t.flatten() for t in seen[name]]), cfg.act_calib, cfg.pct) / 127)
        w = weight_matrix(m)
        wmax = w.abs().amax(0) if cfg.per_channel else w.abs().max().expand(w.shape[1])
        sw = (wmax / 127).clamp_min(1e-12).numpy().astype(np.float32)
        wq = torch.clamp(torch.round(w / torch.from_numpy(sw)), -127, 127).numpy().astype(np.int8)
        bias = m.bias.detach().numpy().astype(np.float32) if m.bias is not None else np.zeros(w.shape[1], np.float32)
        q[name] = QLayer(wq=wq, sx=sx, scale=(np.float32(sx) * sw).astype(np.float32), bias=bias)
    if cfg.adc_bits is None:
        return q
    # 2. ADC ranges from per-crossbar partial sums of the exact int8 model
    rec = {}
    with torch.no_grad():
        Emulator(gm, q, cfg.rows, record=rec).run(calib_x[:200])
    for name, parts in rec.items():
        mx = _absmax(torch.cat(parts), cfg.adc_calib, cfg.pct)
        q[name].adc_bits, q[name].adc_shift = cfg.adc_bits, shift_for(mx, cfg.adc_bits)
    return q


def evaluate(gm, q, rows, x, y, bs=250):
    hit = 0
    with torch.no_grad():
        for i in range(0, len(x), bs):
            out = Emulator(gm, q, rows).run(x[i:i + bs])
            hit += (out.argmax(1) == y[i:i + bs]).sum().item()
    return hit / len(x)
