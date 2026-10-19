# Study notes: imc-graph-compiler

The sheet to learn this project from in 1–2 days. Read the README first for
the results. This file explains the theory behind each piece, walks the code
in the order data flows through it, and lists the questions an interviewer is
likely to ask, with answers and the file that backs each one.

---

## Day 1 morning: the pipeline end to end

```
nn.Module
  │ torch.fx.symbolic_trace                       (imc/difftest.py: traced)
  ▼
FX GraphModule ── fold_conv_bn ── mark_fusion      (imc/fxpasses.py)
  │ Importer: own shape/dtype inference, checked against ShapeProp
  ▼                                               (imc/importer.py)
MLIR: func + linalg + tensor + arith + math       fp32 path
      (+ imc.matmul with int8 tensors)            quantized path (imc/quant.py gives QLayer)
  │ imc-opt -imc-tile-crossbars                    (lib/ImcPasses.cpp)
  ▼
imc.program_weights / imc.mvm / imc.accumulate
  │ imc-opt -imc-lower-to-linalg
  ▼
linalg on tensors
  │ mlir-opt: one-shot-bufferize → linalg-to-loops → scf-to-cf → *-to-llvm
  ▼                                               (scripts/lower.sh)
LLVM dialect ── mlir-runner (JIT) ── printed i32 bits ── compared with PyTorch
                                                  (imc/mlirrun.py, imc/difftest.py)
```

### What Torch-FX is
`torch.fx.symbolic_trace` runs `forward` with *Proxy* objects instead of
tensors and records every operation into a `Graph` of `Node`s. Each node has
an `op`:
- `placeholder`: a function input.
- `get_attr`: a parameter or buffer read.
- `call_module`: e.g. `conv1`, an `nn.Conv2d` submodule.
- `call_function`: e.g. `F.relu`, `operator.add`.
- `call_method`: e.g. `x.t()`.
- `output`.

Symbolic tracing cannot follow data-dependent Python control flow (`if
x.sum() > 0`). `torch.export` handles more cases and gives an ATen-level
graph, but symbolic_trace keeps `nn.Linear`/`nn.Conv2d` as single nodes with
their parameters attached. That is what this importer wants: one node per
crossbar layer.

### The MLIR dialects used, and why
| dialect | used for |
|---|---|
| `func` | the `@forward` function and `@main` harness |
| `tensor` | `empty`, `pad`, `extract_slice`/`insert_slice` (tiling), `collapse_shape`/`expand_shape` (free reshapes) |
| `linalg` | every compute kernel: named ops (`matmul`, `conv_2d_nchw_fchw`, `pooling_nchw_max`, `transpose`, `fill`, `add`) and `linalg.generic` for elementwise chains, reductions and im2col |
| `arith` / `math` | scalar ops inside generic bodies (`addf`, `maximumf`, `roundeven`, `exp`, `rsqrt`, shifts for the ADC) |
| `imc` (mine) | the crossbar model: `matmul`, `program_weights`, `mvm`, `accumulate`, type `!imc.xbar<R, C>` |

**linalg.generic, in one paragraph.** A loop nest described by an iteration
space and one *indexing map* per operand: an affine map from the loop indices
to that operand's indices. `iterator_types` mark each loop as `parallel` or
`reduction`. The region is the scalar body. Broadcasting a bias `[N]` over
`[M, N]` is just the map `(d0, d1) -> (d1)`. A row reduction writes through
`(d0, d1) -> (d0)` with `d1` marked `reduction`. im2col is a gather whose input
map is `(n, oh, ow, c, kh, kw) -> (n, c, oh*s + kh, ow*s + kw)`.

**Tensors vs memrefs.** Everything is emitted on *tensors* (immutable SSA
values), which makes rewriting easy: no aliasing questions. `one-shot-bufferize`
then turns tensors into buffers (`memref`), reusing buffers in place where its
analysis proves that is safe. `tensor.empty` only carries a shape: "a buffer
of this shape will be needed here".

---

## Day 1 afternoon: the FX passes and the importer

### Conv + BatchNorm folding (`fxpasses.fold_conv_bn`)
In eval mode BN is an affine per-channel map:
`y = γ (x − μ) / √(σ² + ε) + β`. With `s = γ / √(σ² + ε)`, a conv followed by
BN is a conv with weights `W·s` (scaled per output channel) and bias
`(b − μ)·s + β`. The pass:
1. pattern-matches `call_module(BatchNorm2d)` whose input is a
   `call_module(Conv2d)` with exactly one user;
2. rewrites the conv's parameters;
3. redirects BN's users to the conv node and erases the BN node.

Only valid in eval mode, and only if nothing else reads the pre-BN value
(hence the single-user check).

### Shape and dtype inference (`Importer.visit` + `check_shape`)
Every handler computes its output shape from its input shapes: conv
`OH = (H + 2p − k)/s + 1`, flatten products, matmul `(M, N)`, and so on. It
raises `UnsupportedOpError` with the node name on mismatches such as wrong
channel count, broadcasting, or non-2-D linear input. Before import,
`ShapeProp` runs the real graph once to record torch's shapes, and
`check_shape` asserts the two agree. That turns an importer bug into an
immediate, named failure instead of IR that breaks later in mlir-opt.

### Fusion (`fxpasses.mark_fusion` + `Lazy` in the importer)
- **Rule:** a value is fused into its consumer iff it is elementwise (relu,
  add, mul, or the bias add of a linear/conv), has exactly **one** user, and
  that user is elementwise. A value with two users must be materialized, or it
  would be recomputed.
- **Mechanism:** elementwise handlers return a `Lazy` expression: a list of
  input tensors with their indexing maps plus a function that emits the scalar
  body. Composing two Lazy values concatenates their inputs and nests the body
  functions. `materialize()` emits the whole chain as one `linalg.generic`. So
  `matmul → +bias → relu` becomes `matmul` plus one generic, instead of
  `matmul`, a bias generic, and a relu generic, and the `[M, N]` tensor between
  bias and relu never exists.
- **Cross-check:** running upstream `mlir-opt -linalg-fuse-elementwise-ops` on
  the *unfused* IR gives exactly the same kernel count as my fused IR on all
  three models (results/fusion.md).
- **Limits:** this is producer-consumer fusion of elementwise ops only. There
  is no fusion into the matmul itself, no reduction fusion, and no tiling-level
  fusion. TorchInductor emits softmax and layernorm as one kernel each; I emit
  4 and 3 (see the README).

### The quantized path
- **Quantization scheme:** symmetric int8. Activations are per-tensor (scale
  `sx`); weights are per-output-channel (`sw[n]`) or per-tensor.
  `q = clamp(roundeven(x / s), −128, 127)`. Calibration is min-max or the
  99.99th percentile of |x| over 1000 training images.
- **Why per-channel helps:** one channel with large weights forces a large
  per-tensor scale, and every small channel then loses resolution.
- **Requantization folding (`qdq_fold_scale`):** after a layer, the fp32
  epilogue is `y = float(acc)·sx·sw[n] + b` followed by relu, and the next
  layer quantizes `y` again. Max-pool and flatten commute with a monotone
  quantizer: `q(max(a, b)) = max(q(a), q(b))`. So when every path to the next
  quantized layer goes only through pool/flatten, the importer appends that
  layer's quantize to the epilogue. The generic then maps
  **i32 → i8 directly**, and max-pool runs on int8. No fp32 activation tensor
  exists between crossbar layers: 2 folds in the MLP, 5 in the CNN. This is
  exact, not an approximation: the bit-exact difftest covers it.
- **Conv on crossbars (im2col):** an `[N·OH·OW, C·KH·KW]` matrix is gathered
  by one `linalg.generic` with the map above, then reshaped by a free
  `collapse_shape`. The int32 result is `expand_shape`d to NHWC. The epilogue
  reads it through the permuted map `(n, f, h, w) -> (n, h, w, f)`, so the
  NHWC→NCHW transpose costs nothing.

---

## Day 2 morning: the imc dialect and its passes

### ODS / TableGen (`include/Imc/ImcOps.td`)
`mlir-tblgen` generates C++ for the dialect class, the `!imc.xbar<rows, cols>`
type (a `TypeDef` with two `int64_t` parameters), the op classes, their
builders, accessors (`getAdcBits()`), parsers and printers (from
`assemblyFormat`). `hasVerifier = 1` declares `verify()`, which I write by hand
in `lib/ImcDialect.cpp`.

Operand types are deliberately `AnyRankedTensor`, so the verifiers can give
hardware-level messages ("a 300x64 weight tile does not fit a 256x256
crossbar") instead of ODS's generic "operand #0 must be...".
`qualified(type(...))` in the format makes the printer write
`!imc.xbar<256, 256>` instead of the elided `<256, 256>`.

### Verifiers (tested in `test/verify.mlir`, 9 cases)
- **program_weights:** int8, 2-D static, the tile fits the crossbar.
- **matmul:** int8 × int8 → int32, contraction dims match, `adc_bits` and
  `adc_shift` given together, bits in [1, 16].
- **mvm:** the input rows fit the crossbar. If the crossbar SSA value comes
  from a visible `program_weights`, the shapes must match that tile exactly.
- **accumulate:** int32.

### `-imc-tile-crossbars{rows, cols}` (`lib/ImcPasses.cpp`)
For `imc.matmul` `[M, K] × [K, N]`:

```
for each column block n0 (size ≤ cols):
  acc = none
  for each row block k0 (size ≤ rows):
    w  = extract_slice W[k0:k0+kr, n0:n0+nc]
    xb = imc.program_weights w            ← programmed once
    x  = extract_slice X[:, k0:k0+kr]
    p  = imc.mvm xb, x {adc}              ← all M input vectors stream through
    acc = acc ? imc.accumulate acc, p : p ← digital partial-sum add
  out = insert_slice acc into out[:, n0:n0+nc]
```

That is **weight-stationary**: each weight tile is written once and every
input vector reuses it.

Shapes are static, so the loops are unrolled at compile time. That is fine for
these models (the CNN's largest layer needs 16 tiles), but a production pass
would emit `scf.for` or use the `TilingInterface`.

**Key semantic point:** the pass refuses an `imc.matmul` without ADC
attributes, and the lowering refuses an *untiled* matmul *with* them. The ADC
rounds each crossbar's partial sum separately, so `ADC(a) + ADC(b) ≠ ADC(a + b)`.
**Tiling changes the numerics.** Tile size is a model-accuracy decision, not
just a performance one, and the experiments show it.

### `-imc-lower-to-linalg`
- `mvm` → `linalg.fill(0)` + `linalg.matmul` (i8 × i8 → i32; linalg
  sign-extends inputs to the accumulator type) + the ADC as a `linalg.generic`:
  `((v + 2^(s−1)) >> s)`, clamp to `[−2^(b−1), 2^(b−1)−1]`, `<< s`.
- `accumulate` → `linalg.add`.
- `program_weights` → erased (its tile feeds the matmul directly).
- An exact `imc.matmul` (no ADC) → a plain `linalg.matmul`.

### Why `imc-opt` links `libMLIR.so`
The Ubuntu `libmlir-20-dev` ships both ~370 static archives and one shared
`libMLIR.so`. The WSL VM here has 2 GB of RAM, and linking against the shared
library takes seconds. `imc-opt` registers only the dialects it needs, plus
the generic transform passes. Everything past linalg uses upstream `mlir-opt`.

---

## Day 2 afternoon: testing, the ADC model, the cost model

### Three levels of testing
1. **lit + FileCheck** (`test/`, 4 files):
   - verifier diagnostics with `-verify-diagnostics -split-input-file`;
   - the tiling structure;
   - the lowering;
   - one hand-checked end-to-end run (`e2e-adc.mlir`). A partial sum of 20
     saturates a 4-bit ADC at 14.
2. **Differential vs PyTorch** (`imc/difftest.py`):
   - fp32 models against eager PyTorch, with a tolerance because summation
     order differs;
   - the int8 crossbar path against `quant.Emulator`, **bit-exact**.
3. **Random-graph fuzzer** (`imc/fuzz.py`): random chains of
   linear / relu / add / mul / t / matmul / softmax / layernorm with random
   shapes, compiled fused and unfused.

**How results come back exactly.** The harness bitcasts every f32 result to
i32 inside MLIR and prints integers, so no decimal rounding gets in the way.
The emulator uses the same float32 operations in the same order (divide, then
roundeven, clamp, int conversion; `float(acc)·scale + bias` with no FMA). Its
integer matmul runs in float64, which is exact for these magnitudes.

**Is the bit-exact check vacuous?** No. `scripts/mutants.py` changes the
emulator's ADC rounding (half-up → floor) or its clamp bound by one code; both
are reported as mismatches.

### The ADC model
A crossbar column produces an analog sum. Its integer equivalent is
`v = Σ x_i·w_i`. The ADC has `b` bits and a power-of-two step `2^s`:
`q = clamp(round(v / 2^s), −2^(b−1), 2^(b−1)−1) · 2^s`.

- **Calibration:** `s` is the smallest shift such that the largest (or
  99.99th-percentile) |partial sum| seen on calibration data fits in b bits.
- **Two errors trade off:**
  - **rounding error** (step `2^s`), which grows when the range is large;
  - **clipping error**, which grows when the range is set too small.

  Percentile calibration accepts a little clipping to get a finer step.
- **Why crossbar rows matter:** a tile with R rows sums R products, so its
  partial-sum range grows with R. With a fixed number of ADC bits, a taller
  crossbar means a coarser step. Smaller crossbars mean more ADC conversions
  and more digital adds, but each one is more precise. This is the core
  analog-IMC trade-off, and the tiling sweep measures it.

### The cost model (`imc/crossbar.py`)
Analytical only, with every constant labelled as an assumption:
- 4M weight cells on chip;
- 100 ns per MVM;
- 2 pJ per 8-bit ADC conversion, scaling as 2^b;
- 5 fJ per cell MAC;
- 1 pJ per weight write;
- 20 pJ per DRAM byte.

Per layer:
- tiles = ⌈K/R⌉·⌈N/C⌉;
- utilization = K·N / (tiles·R·C);
- ADC conversions per image = ⌈K/R⌉·N·M.

If the whole network's tiles exceed the chip, the weights must be
reprogrammed every batch. Use it to *compare* mappings, never as a hardware
number.

---

## Likely interview questions

**Q: Why write your own importer instead of using torch-mlir?**
To own the semantic analysis: shape inference, clear errors, and a custom
lowering into my dialect. torch-mlir's FX importer emits the `torch` dialect,
which then needs torch-mlir's own lowering to linalg. I did *not* cross-check
against torch-mlir; there are no stable wheels for this torch version. Upstream
MLIR's elementwise fusion and TorchInductor are the comparisons I did run.

**Q: What does your verifier catch that ODS type constraints would not?**
Relations between operands. Examples: the tile fits *this* crossbar type's
parameters; an mvm's input matches the tile its crossbar was programmed with
(by looking through the SSA def); `adc_bits` and `adc_shift` are given
together.

**Q: Is your fusion legal in general?**
Elementwise producer-consumer fusion is always legal when the producer has a
single use: same iteration space, no reductions. With multiple uses it is
still legal, but it recomputes the value, which I chose not to do. Fusion
across a reduction (softmax) needs the reduction's result first, which is why
Inductor's single softmax kernel is a different, loop-level kind of fusion.

**Q: Why is quantize-after-maxpool equal to maxpool-after-quantize?**
`q` is monotone non-decreasing (divide by a positive scale, round, clamp), and
max commutes with any monotone non-decreasing function. It is exact, and
verified bit-exact.

**Q: Why symmetric int8 for post-ReLU activations?**
It is simple, and it keeps zero exactly representable, so zero-padding is
still zero. The cost is one bit: post-ReLU values are ≥ 0, so half of int8's
codes are unused. Asymmetric or uint8 activations would recover it; that is
listed as not done.

**Q: How would this map onto real hardware?**
`program_weights` would become DMA plus a write sequence. `mvm` would become
an array activation plus ADC readout. Tiles would be scheduled across arrays
with double-buffered weight loads when the model does not fit. The digital
`accumulate` would run in near-array SIMD units. None of that is modelled
beyond the analytical cost model.
