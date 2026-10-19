"""Torch-FX GraphModule -> MLIR text (func / linalg / tensor / arith / math, and
the custom imc dialect in quantized mode).

The importer does its own shape and dtype inference for every node and then
checks it against torch's ShapeProp, so a wrong rule fails loudly instead of
emitting IR that crashes later in mlir-opt.

Elementwise ops are built as lazy expressions. A value marked fusable by
fxpasses.mark_fusion stays lazy and is folded into its consumer; everything
else is materialized as one linalg.generic. That is the whole fusion
mechanism: with no fusion marks, every op becomes its own kernel.
"""
import math
import operator
import struct
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx.passes.shape_prop import ShapeProp


class UnsupportedOpError(Exception):
    pass


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def ttype(shape, dt="f32"):
    return f"tensor<{''.join(f'{d}x' for d in shape)}{dt}>"


def dense(arr: np.ndarray):
    """Exact constant: raw little-endian bytes as a hex DenseElementsAttr."""
    arr = np.ascontiguousarray(arr)
    dt = {np.dtype(np.float32): "f32", np.dtype(np.int8): "i8", np.dtype(np.int32): "i32"}[arr.dtype]
    return f'dense<"0x{arr.tobytes().hex().upper()}"> : {ttype(arr.shape, dt)}'


def fconst(x):
    return "0x" + struct.pack(">f", np.float32(x)).hex().upper()


def dims(r):
    return ", ".join(f"d{i}" for i in range(r))


def amap(r, results):
    return f"affine_map<({dims(r)}) -> ({results})>"


def idmap(r):
    return amap(r, dims(r))


ELT_BYTES = {"f32": 4, "i32": 4, "i8": 1}


@dataclass
class Mat:                      # a materialized tensor value
    ssa: str
    shape: tuple
    dt: str = "f32"


@dataclass
class Lazy:                     # an elementwise expression not yet emitted
    shape: tuple
    dt: str
    ins: list                   # [(ssa, tensor type, affine map, element type)]
    fn: Callable                # (body, [scalar ssa per input]) -> scalar ssa


@dataclass
class Stats:
    kernels: list = field(default_factory=list)     # (op name, output bytes)
    fills: int = 0
    qdq_folds: int = 0
    returned_bytes: int = 0

    @property
    def intermediate_bytes(self):
        return sum(b for _, b in self.kernels) - self.returned_bytes


class Body:
    """Emits the scalar ops of one linalg.generic region."""

    def __init__(self, imp):
        self.imp, self.lines = imp, []

    def emit(self, rhs):
        v = self.imp.fresh()
        self.lines.append(f"    {v} = {rhs}")
        return v

    def f32(self, x):
        return self.emit(f"arith.constant {fconst(x)} : f32")


# --------------------------------------------------------------------------
# the importer
# --------------------------------------------------------------------------

@dataclass
class QLayer:
    """Quantized parameters of one linear/conv layer (see quant.py)."""
    wq: np.ndarray              # int8 [K, N] (conv: K = C*KH*KW, im2col order)
    sx: np.float32              # input activation scale
    scale: np.ndarray           # float32 [N] = sx * sw[n]
    bias: np.ndarray            # float32 [N]
    adc_bits: int | None = None
    adc_shift: int | None = None


class Importer:
    def __init__(self, gm, example_inputs, quant: dict | None = None, expose_acc=False):
        self.gm, self.quant, self.expose_acc = gm, quant or {}, expose_acc
        self.lines, self.n, self.env, self.stats = [], 0, {}, Stats()
        self.last_acc = None
        for i, x in enumerate(example_inputs):
            if x.dtype != torch.float32:
                raise UnsupportedOpError(f"input {i} has dtype {x.dtype}; only float32 inputs are supported")
        ShapeProp(gm).propagate(*example_inputs)
        self.example_inputs = example_inputs

    # ---- emission primitives -------------------------------------------
    def fresh(self):
        self.n += 1
        return f"%v{self.n}"

    def emit(self, rhs):
        v = self.fresh()
        self.lines.append(f"  {v} = {rhs}")
        return v

    def kernel(self, name, shape, dt):
        self.stats.kernels.append((name, int(np.prod(shape)) * ELT_BYTES[dt]))

    def const(self, arr):
        return self.emit(f"arith.constant {dense(arr)}")

    def filled(self, shape, dt, value):
        e = self.emit(f"tensor.empty() : {ttype(shape, dt)}")
        c = self.emit(f"arith.constant {value} : {dt}")
        self.stats.fills += 1
        return self.emit(f"linalg.fill ins({c} : {dt}) outs({e} : {ttype(shape, dt)}) -> {ttype(shape, dt)}")

    def materialize(self, v):
        if isinstance(v, Mat):
            return v
        r = len(v.shape)
        out_t = ttype(v.shape, v.dt)
        e = self.emit(f"tensor.empty() : {out_t}")
        body = Body(self)
        args = [f"%a{i}" for i in range(len(v.ins))]
        res = v.fn(body, args)
        maps = ", ".join([m for _, _, m, _ in v.ins] + [idmap(r)])
        ins = ", ".join(s for s, _, _, _ in v.ins)
        ins_t = ", ".join(t for _, t, _, _ in v.ins)
        bb = ", ".join([f"{a}: {et}" for a, (_, _, _, et) in zip(args, v.ins)] + [f"%out: {v.dt}"])
        out = self.fresh()
        self.lines.append(
            f"  {out} = linalg.generic {{indexing_maps = [{maps}], iterator_types = "
            f"[{', '.join(['\"parallel\"'] * r)}]}} ins({ins} : {ins_t}) outs({e} : {out_t}) {{\n"
            f"  ^bb0({bb}):\n" + "\n".join(body.lines) + f"\n    linalg.yield {res} : {v.dt}\n"
            f"  }} -> {out_t}")
        self.kernel("generic", v.shape, v.dt)
        return Mat(out, v.shape, v.dt)

    @staticmethod
    def lazy(v):
        if isinstance(v, Lazy):
            return v
        r = len(v.shape)
        return Lazy(v.shape, v.dt, [(v.ssa, ttype(v.shape, v.dt), idmap(r), v.dt)], lambda b, a: a[0])

    @staticmethod
    def compose(shape, dt, parts, fn):
        """New lazy value over the inputs of several lazy parts."""
        ins, cuts = [], [0]
        for p in parts:
            ins += p.ins
            cuts.append(len(ins))

        def build(b, args):
            vals = [p.fn(b, args[cuts[i]:cuts[i + 1]]) for i, p in enumerate(parts)]
            return fn(b, vals)
        return Lazy(shape, dt, ins, build)

    # ---- graph walk -----------------------------------------------------
    def run(self, name="forward"):
        args, results = [], []
        for node in self.gm.graph.nodes:
            if node.op == "placeholder":
                shape = tuple(node.meta["tensor_meta"].shape)
                a = f"%arg{len(args)}"
                args.append(f"{a}: {ttype(shape)}")
                self.env[node] = Mat(a, shape)
                continue
            if node.op == "output":
                outs = node.args[0] if isinstance(node.args[0], (tuple, list)) else (node.args[0],)
                for o in outs:
                    m = self.materialize(self.env[o])
                    results.append(m)
                continue
            if node.op == "get_attr":
                t = getattr(self.gm, node.target).detach().numpy().astype(np.float32)
                self.env[node] = Mat(self.const(t), t.shape)
                continue
            v = self.visit(node)
            self.check_shape(node, v)
            if isinstance(v, Lazy) and not node.meta.get("fuse"):
                s = self.qdq_fold_scale(node)
                if s is not None:
                    v = self.quantize_lazy(v, s)
                    self.stats.qdq_folds += 1
                v = self.materialize(v)
            self.env[node] = v

        if self.expose_acc and self.last_acc is not None:
            results.append(self.last_acc)
        self.stats.returned_bytes = sum(int(np.prod(r.shape)) * ELT_BYTES[r.dt] for r in results)
        rt = ", ".join(ttype(r.shape, r.dt) for r in results)
        text = [f"func.func @{name}({', '.join(args)}) -> ({rt}) {{"] + self.lines
        text.append(f"  return {', '.join(r.ssa for r in results)} : {rt}")
        text.append("}")
        self.result_types = [(r.shape, r.dt) for r in results]
        return "\n".join(text)

    def check_shape(self, node, v):
        tm = node.meta.get("tensor_meta")
        if tm is None:
            raise UnsupportedOpError(f"node {node.name}: produces a non-tensor value")
        ref = tuple(tm.shape)
        if tuple(v.shape) != ref:
            raise AssertionError(f"shape inference bug at {node.name}: importer says {v.shape}, "
                                 f"torch says {ref}")
        if tm.dtype != torch.float32:
            raise UnsupportedOpError(f"node {node.name}: dtype {tm.dtype} is not supported")

    def visit(self, node):
        tgt = node.target
        if node.op == "call_module":
            m = self.gm.get_submodule(tgt)
            x = self.env[node.args[0]]
            if isinstance(m, nn.Linear):
                return self.linear(node, m, x)
            if isinstance(m, nn.Conv2d):
                return self.conv2d(node, m, x)
            if isinstance(m, nn.ReLU):
                return self.relu(x)
            if isinstance(m, nn.MaxPool2d):
                return self.maxpool(node, x, m.kernel_size, m.stride, m.padding, m.dilation, m.ceil_mode)
            if isinstance(m, nn.Flatten):
                return self.flatten(x, m.start_dim, m.end_dim)
            if isinstance(m, nn.LayerNorm):
                return self.layernorm(node, m, x)
            if isinstance(m, nn.BatchNorm2d):
                raise UnsupportedOpError(f"node {node.name}: BatchNorm2d must be folded into the "
                                         f"preceding conv first (fxpasses.fold_conv_bn)")
            raise UnsupportedOpError(f"node {node.name}: module {type(m).__name__} is not supported")

        a = [self.env[x] if isinstance(x, torch.fx.Node) else x for x in node.args]
        kw = node.kwargs
        if node.op == "call_function":
            if tgt in (F.relu, torch.relu):
                return self.relu(a[0])
            if tgt in (operator.add, torch.add):
                return self.binary(node, a[0], a[1], "add")
            if tgt in (operator.mul, torch.mul):
                return self.binary(node, a[0], a[1], "mul")
            if tgt is torch.flatten:
                return self.flatten(a[0], *(a[1:] or [kw.get("start_dim", 0)]), kw.get("end_dim", -1))
            if tgt in (operator.matmul, torch.matmul, torch.mm):
                return self.matmul(node, a[0], a[1])
            if tgt in (F.softmax, torch.softmax):
                return self.softmax(node, a[0], a[1] if len(a) > 1 else kw.get("dim"))
            if tgt is F.max_pool2d:
                k = a[1] if len(a) > 1 else kw["kernel_size"]
                return self.maxpool(node, a[0], k, kw.get("stride") or k, kw.get("padding", 0), 1, False)
        if node.op == "call_method":
            if tgt == "relu":
                return self.relu(a[0])
            if tgt == "t" or (tgt == "transpose" and sorted(x % 2 for x in a[1:3]) == [0, 1]):
                return self.transpose(node, a[0])
            if tgt == "flatten":
                return self.flatten(a[0], *(a[1:] or [0]), -1)
        name = getattr(tgt, "__name__", str(tgt))
        raise UnsupportedOpError(
            f"node {node.name}: {node.op} '{name}' is not supported (supported: linear, conv2d, "
            f"relu, add, mul, flatten, max_pool2d, matmul, t/transpose, softmax, layer_norm)")

    # ---- elementwise ----------------------------------------------------
    def relu(self, x):
        lx = self.lazy(x)
        return self.compose(x.shape, "f32", [lx], lambda b, v: b.emit(f"arith.maximumf {v[0]}, {b.f32(0.0)} : f32"))

    def binary(self, node, x, y, op):
        f = {"add": "arith.addf", "mul": "arith.mulf"}[op]
        if isinstance(x, (int, float)):           # scalar * tensor: both ops commute
            x, y = y, x
        if isinstance(y, (int, float)):
            c = float(y)
            return self.compose(x.shape, "f32", [self.lazy(x)], lambda b, v: b.emit(f"{f} {v[0]}, {b.f32(c)} : f32"))
        if tuple(x.shape) != tuple(y.shape):
            raise UnsupportedOpError(f"node {node.name}: {op} of {x.shape} and {y.shape}; "
                                     f"implicit broadcasting is not supported")
        return self.compose(x.shape, "f32", [self.lazy(x), self.lazy(y)],
                            lambda b, v: b.emit(f"{f} {v[0]}, {v[1]} : f32"))

    def bias_epilogue(self, acc: Mat, bias, rank, chan_dim):
        """acc + bias broadcast along `chan_dim`; lazy so it can fuse."""
        bv = self.const(bias.astype(np.float32))
        n = len(bias)
        parts = [self.lazy(acc), Lazy((n,), "f32", [(bv, ttype((n,)), amap(rank, f"d{chan_dim}"), "f32")],
                                      lambda b, a: a[0])]
        return self.compose(acc.shape, "f32", parts, lambda b, v: b.emit(f"arith.addf {v[0]}, {v[1]} : f32"))

    # ---- anchors: linear / conv ------------------------------------------
    def linear(self, node, m, x):
        x = self.materialize(x)
        if len(x.shape) != 2:
            raise UnsupportedOpError(f"node {node.name}: linear expects a 2-D input, got {x.shape}")
        M, K = x.shape
        if K != m.in_features:
            raise UnsupportedOpError(f"node {node.name}: input has {K} features, layer expects {m.in_features}")
        N = m.out_features
        if node.target in self.quant:
            return self.qanchor(node, self.quantize_mat(x, self.quant[node.target]), self.quant[node.target], (M, N))
        wt = self.const(m.weight.detach().numpy().T.astype(np.float32).copy())
        init = self.filled((M, N), "f32", fconst(0.0))
        out_t = ttype((M, N))
        mm = self.emit(f"linalg.matmul ins({x.ssa}, {wt} : {ttype(x.shape)}, {ttype((K, N))}) "
                       f"outs({init} : {out_t}) -> {out_t}")
        self.kernel("matmul", (M, N), "f32")
        acc = Mat(mm, (M, N))
        if m.bias is None:
            return acc
        return self.bias_epilogue(acc, m.bias.detach().numpy(), 2, 1)

    def conv_geometry(self, node, m, shape):
        if len(shape) != 4:
            raise UnsupportedOpError(f"node {node.name}: conv2d expects NCHW input, got {shape}")
        if m.groups != 1 or tuple(m.dilation) != (1, 1) or m.padding_mode != "zeros" or isinstance(m.padding, str):
            raise UnsupportedOpError(f"node {node.name}: only groups=1, dilation=1, numeric zero padding")
        N, C, H, W = shape
        if C != m.in_channels:
            raise UnsupportedOpError(f"node {node.name}: input has {C} channels, conv expects {m.in_channels}")
        (kh, kw), (sh, sw), (ph, pw) = m.kernel_size, m.stride, m.padding
        if sh != sw or ph != pw:
            raise UnsupportedOpError(f"node {node.name}: only square stride/padding")
        OH, OW = (H + 2 * ph - kh) // sh + 1, (W + 2 * pw - kw) // sw + 1
        return N, C, H, W, m.out_channels, kh, kw, sh, ph, OH, OW

    def pad(self, x: Mat, p, dt, zero):
        if p == 0:
            return x
        N, C, H, W = x.shape
        z = self.emit(f"arith.constant {zero} : {dt}")
        shape = (N, C, H + 2 * p, W + 2 * p)
        v = self.emit(f"tensor.pad {x.ssa} low[0, 0, {p}, {p}] high[0, 0, {p}, {p}] {{\n"
                      f"  ^bb0(%i0: index, %i1: index, %i2: index, %i3: index):\n"
                      f"    tensor.yield {z} : {dt}\n  }} : {ttype(x.shape, dt)} to {ttype(shape, dt)}")
        self.kernel("pad", shape, dt)
        return Mat(v, shape, dt)

    def conv2d(self, node, m, x):
        x = self.materialize(x)
        N, C, H, W, Fo, kh, kw, s, p, OH, OW = self.conv_geometry(node, m, x.shape)
        if node.target in self.quant:
            return self.qconv(node, m, x, self.quant[node.target])
        xp = self.pad(x, p, "f32", fconst(0.0))
        w = self.const(m.weight.detach().numpy().astype(np.float32))
        out_shape = (N, Fo, OH, OW)
        init = self.filled(out_shape, "f32", fconst(0.0))
        v = self.emit(f"linalg.conv_2d_nchw_fchw {{dilations = dense<1> : tensor<2xi64>, strides = dense<{s}> : "
                      f"tensor<2xi64>}} ins({xp.ssa}, {w} : {ttype(xp.shape)}, {ttype((Fo, C, kh, kw))}) "
                      f"outs({init} : {ttype(out_shape)}) -> {ttype(out_shape)}")
        self.kernel("conv", out_shape, "f32")
        acc = Mat(v, out_shape)
        if m.bias is None:
            return acc
        return self.bias_epilogue(acc, m.bias.detach().numpy(), 4, 1)

    # ---- quantized anchors (imc dialect) ---------------------------------
    def quantize_lazy(self, v: Lazy, sx):
        def q(b, a):
            d = b.emit(f"arith.divf {a[0]}, {b.f32(sx)} : f32")
            r = b.emit(f"math.roundeven {d} : f32")
            c = b.emit(f"arith.maximumf {r}, {b.f32(-128.0)} : f32")
            c = b.emit(f"arith.minimumf {c}, {b.f32(127.0)} : f32")
            return b.emit(f"arith.fptosi {c} : f32 to i8")
        return self.compose(v.shape, "i8", [v], q)

    def quantize_mat(self, x: Mat, q: QLayer):
        if x.dt == "i8":                  # producer already requantized (QDQ fold)
            return x
        return self.materialize(self.quantize_lazy(self.lazy(x), q.sx))

    def imc_matmul(self, xq: Mat, q: QLayer):
        M, K = xq.shape
        N = q.wq.shape[1]
        w = self.const(q.wq)
        adc = f" {{adc_bits = {q.adc_bits} : i64, adc_shift = {q.adc_shift} : i64}}" if q.adc_bits else ""
        v = self.emit(f"imc.matmul {xq.ssa}, {w}{adc} : {ttype((M, K), 'i8')}, {ttype((K, N), 'i8')} "
                      f"-> {ttype((M, N), 'i32')}")
        self.kernel("imc.matmul", (M, N), "i32")
        self.last_acc = Mat(v, (M, N), "i32")
        return self.last_acc

    def dequant_epilogue(self, acc: Mat, q: QLayer, out_shape, acc_map, chan_dim):
        """y = float(acc) * (sx * sw[n]) + bias[n], lazy so relu/requant fuse in."""
        r = len(out_shape)
        sc, bi = self.const(q.scale.astype(np.float32)), self.const(q.bias.astype(np.float32))
        n = len(q.scale)
        ins = [(acc.ssa, ttype(acc.shape, "i32"), acc_map, "i32"),
               (sc, ttype((n,)), amap(r, f"d{chan_dim}"), "f32"),
               (bi, ttype((n,)), amap(r, f"d{chan_dim}"), "f32")]

        def fn(b, a):
            f = b.emit(f"arith.sitofp {a[0]} : i32 to f32")
            f = b.emit(f"arith.mulf {f}, {a[1]} : f32")
            return b.emit(f"arith.addf {f}, {a[2]} : f32")
        return Lazy(tuple(out_shape), "f32", ins, fn)

    def qanchor(self, node, xq, q, out_shape):
        acc = self.imc_matmul(xq, q)
        return self.dequant_epilogue(acc, q, out_shape, idmap(2), 1)

    def qconv(self, node, m, x, q):
        N, C, H, W, Fo, kh, kw, s, p, OH, OW = self.conv_geometry(node, m, x.shape)
        xq = self.pad(self.quantize_mat(x, q), p, "i8", "0")
        # im2col: [N, OH, OW, C, KH, KW] gathered with one affine map, then a free reshape.
        col_shape = (N, OH, OW, C, kh, kw)
        e = self.emit(f"tensor.empty() : {ttype(col_shape, 'i8')}")
        src = amap(6, f"d0, d3, d1 * {s} + d4, d2 * {s} + d5")
        col = self.emit(
            f"linalg.generic {{indexing_maps = [{src}, {idmap(6)}], iterator_types = "
            f"[{', '.join(['\"parallel\"'] * 6)}]}} ins({xq.ssa} : {ttype(xq.shape, 'i8')}) "
            f"outs({e} : {ttype(col_shape, 'i8')}) {{\n  ^bb0(%in: i8, %out: i8):\n    linalg.yield %in : i8\n"
            f"  }} -> {ttype(col_shape, 'i8')}")
        self.kernel("im2col", col_shape, "i8")
        rows, K = N * OH * OW, C * kh * kw
        col2 = self.emit(f"tensor.collapse_shape {col} [[0, 1, 2], [3, 4, 5]] : {ttype(col_shape, 'i8')} "
                         f"into {ttype((rows, K), 'i8')}")
        acc = self.imc_matmul(Mat(col2, (rows, K), "i8"), q)
        acc4 = self.emit(f"tensor.expand_shape {acc.ssa} [[0, 1, 2], [3]] output_shape [{N}, {OH}, {OW}, {Fo}] "
                         f": {ttype((rows, Fo), 'i32')} into {ttype((N, OH, OW, Fo), 'i32')}")
        # NHWC accumulator read through a permuted map: the transpose back to NCHW is free.
        return self.dequant_epilogue(Mat(acc4, (N, OH, OW, Fo), "i32"), q, (N, Fo, OH, OW),
                                     amap(4, "d0, d2, d3, d1"), 1)

    def qdq_fold_scale(self, node):
        """If every path from `node` reaches a quantized layer through only
        max-pool / flatten (which commute with a monotone quantizer), return
        that layer's input scale so the producer can emit int8 directly."""
        if not self.quant:
            return None
        scales, stack = set(), list(node.users)
        while stack:
            u = stack.pop()
            if u.op == "call_module" and u.target in self.quant:
                scales.add(float(self.quant[u.target].sx))
            elif (u.op == "call_module" and isinstance(self.gm.get_submodule(u.target), (nn.MaxPool2d, nn.Flatten))) \
                    or (u.op == "call_function" and u.target in (torch.flatten, F.max_pool2d)):
                stack.extend(u.users)
            else:
                return None
        return scales.pop() if len(scales) == 1 else None

    # ---- data movement / reductions --------------------------------------
    def maxpool(self, node, x, k, s, p, d, ceil):
        x = self.materialize(x)
        k = k if isinstance(k, int) else k[0]
        s = s if isinstance(s, int) else s[0]
        if p not in (0, (0, 0)) or d not in (1, (1, 1)) or ceil or len(x.shape) != 4:
            raise UnsupportedOpError(f"node {node.name}: max_pool2d supports only NCHW, no padding/dilation/ceil")
        N, C, H, W = x.shape
        out = (N, C, (H - k) // s + 1, (W - k) // s + 1)
        lowest = fconst(-math.inf) if x.dt == "f32" else "-128"
        init = self.filled(out, x.dt, lowest)
        win = self.emit(f"tensor.empty() : {ttype((k, k), x.dt)}")
        v = self.emit(f"linalg.pooling_nchw_max {{dilations = dense<1> : tensor<2xi64>, strides = dense<{s}> : "
                      f"tensor<2xi64>}} ins({x.ssa}, {win} : {ttype(x.shape, x.dt)}, {ttype((k, k), x.dt)}) "
                      f"outs({init} : {ttype(out, x.dt)}) -> {ttype(out, x.dt)}")
        self.kernel("maxpool", out, x.dt)
        return Mat(v, out, x.dt)

    def flatten(self, x, start=0, end=-1):
        x = self.materialize(x)
        r = len(x.shape)
        start, end = start % r, end % r
        if start == end:
            return x
        groups = [[i] for i in range(start)] + [list(range(start, end + 1))] + [[i] for i in range(end + 1, r)]
        shape = tuple(int(np.prod([x.shape[i] for i in g])) for g in groups)
        v = self.emit(f"tensor.collapse_shape {x.ssa} {groups} : {ttype(x.shape, x.dt)} into {ttype(shape, x.dt)}")
        return Mat(v, shape, x.dt)

    def matmul(self, node, a, b):
        a, b = self.materialize(a), self.materialize(b)
        if len(a.shape) != 2 or len(b.shape) != 2 or a.shape[1] != b.shape[0]:
            raise UnsupportedOpError(f"node {node.name}: matmul of {a.shape} and {b.shape} "
                                     f"(only 2-D with matching inner dims)")
        out = (a.shape[0], b.shape[1])
        init = self.filled(out, "f32", fconst(0.0))
        v = self.emit(f"linalg.matmul ins({a.ssa}, {b.ssa} : {ttype(a.shape)}, {ttype(b.shape)}) "
                      f"outs({init} : {ttype(out)}) -> {ttype(out)}")
        self.kernel("matmul", out, "f32")
        return Mat(v, out)

    def transpose(self, node, x):
        x = self.materialize(x)
        if len(x.shape) != 2:
            raise UnsupportedOpError(f"node {node.name}: transpose only for 2-D tensors")
        out = (x.shape[1], x.shape[0])
        e = self.emit(f"tensor.empty() : {ttype(out)}")
        v = self.emit(f"linalg.transpose ins({x.ssa} : {ttype(x.shape)}) outs({e} : {ttype(out)}) permutation = [1, 0]")
        self.kernel("transpose", out, "f32")
        return Mat(v, out)

    def row_reduce(self, ins, init_val, fn):
        """Reduce each row of an [M, N] iteration space into a [M] tensor."""
        M = ins[0][2]
        init = self.filled((M,), "f32", fconst(init_val))
        body = Body(self)
        args = [f"%a{i}" for i in range(len(ins))]
        res = fn(body, args, "%out")
        maps = ", ".join([m for _, _, _, m in ins] + [amap(2, "d0")])
        names = ", ".join(s for s, _, _, _ in ins)
        types = ", ".join(t for _, t, _, _ in ins)
        bb = ", ".join([f"{a}: f32" for a in args] + ["%out: f32"])
        v = self.fresh()
        self.lines.append(
            f"  {v} = linalg.generic {{indexing_maps = [{maps}], iterator_types = [\"parallel\", \"reduction\"]}} "
            f"ins({names} : {types}) outs({init} : {ttype((M,))}) {{\n  ^bb0({bb}):\n" + "\n".join(body.lines)
            + f"\n    linalg.yield {res} : f32\n  }} -> {ttype((M,))}")
        self.kernel("reduce", (M,), "f32")
        return v

    def softmax(self, node, x, dim):
        x = self.materialize(x)
        if len(x.shape) != 2 or dim not in (1, -1):
            raise UnsupportedOpError(f"node {node.name}: softmax only over the last dim of a 2-D tensor")
        M, N = x.shape
        xin = (x.ssa, ttype(x.shape), M, idmap(2))
        mx = self.row_reduce([xin], -math.inf, lambda b, a, o: b.emit(f"arith.maximumf {a[0]}, {o} : f32"))
        row = (ttype((M,)), amap(2, "d0"), "f32")

        def ex(b, a):
            return b.emit(f"math.exp {b.emit(f'arith.subf {a[0]}, {a[1]} : f32')} : f32")
        e = self.materialize(Lazy(x.shape, "f32", [(x.ssa, ttype(x.shape), idmap(2), "f32"), (mx, *row)], ex))
        sm = self.row_reduce([(e.ssa, ttype(x.shape), M, idmap(2))], 0.0,
                             lambda b, a, o: b.emit(f"arith.addf {a[0]}, {o} : f32"))
        return self.materialize(Lazy(x.shape, "f32", [(e.ssa, ttype(x.shape), idmap(2), "f32"), (sm, *row)],
                                     lambda b, a: b.emit(f"arith.divf {a[0]}, {a[1]} : f32")))

    def layernorm(self, node, m, x):
        x = self.materialize(x)
        if len(x.shape) != 2 or tuple(m.normalized_shape) != (x.shape[1],):
            raise UnsupportedOpError(f"node {node.name}: layer_norm only over the last dim of a 2-D tensor")
        M, N = x.shape
        xin = (x.ssa, ttype(x.shape), M, idmap(2))
        row = (ttype((M,)), amap(2, "d0"), "f32")
        sm = self.row_reduce([xin], 0.0, lambda b, a, o: b.emit(f"arith.addf {a[0]}, {o} : f32"))

        def centered(b, xv, sv):
            mean = b.emit(f"arith.divf {sv}, {b.f32(N)} : f32")
            return b.emit(f"arith.subf {xv}, {mean} : f32")

        def sq(b, a, o):
            d = centered(b, a[0], a[1])
            return b.emit(f"arith.addf {b.emit(f'arith.mulf {d}, {d} : f32')}, {o} : f32")
        var = self.row_reduce([xin, (sm, ttype((M,)), M, amap(2, "d0"))], 0.0, sq)
        g = self.const(m.weight.detach().numpy().astype(np.float32))
        bb = self.const(m.bias.detach().numpy().astype(np.float32))
        col = (ttype((N,)), amap(2, "d1"), "f32")

        def norm(b, a):
            d = centered(b, a[0], a[1])
            v = b.emit(f"arith.divf {a[2]}, {b.f32(N)} : f32")
            v = b.emit(f"math.rsqrt {b.emit(f'arith.addf {v}, {b.f32(m.eps)} : f32')} : f32")
            y = b.emit(f"arith.mulf {d}, {v} : f32")
            return b.emit(f"arith.addf {b.emit(f'arith.mulf {y}, {a[3]} : f32')}, {a[4]} : f32")
        return self.materialize(Lazy(x.shape, "f32", [(x.ssa, ttype(x.shape), idmap(2), "f32"), (sm, *row),
                                                      (var, *row), (g, *col), (bb, *col)], norm))


def import_graph(gm, example_inputs, **kw):
    """Returns (forward function text, Importer) so callers can read stats."""
    imp = Importer(gm, example_inputs, **kw)
    text = imp.run()
    return text, imp
