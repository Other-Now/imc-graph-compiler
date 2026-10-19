# imc-graph-compiler

A small graph compiler from **Torch-FX to MLIR** for a model of an **analog
in-memory-compute (IMC) accelerator**. In such a chip, int8 weights sit in
crossbar arrays and each matrix-vector product is computed in the analog
domain, then digitized by an ADC.

```
nn.Module ─ torch.fx ─► FX passes ─► my importer ─► MLIR (linalg / tensor / arith / imc)
            (conv+BN fold, fusion)   (shape/dtype      │ imc-opt: -imc-tile-crossbars, -imc-lower-to-linalg
                                     inference)        ▼ mlir-opt: bufferize → loops → LLVM
                                                  mlir-runner (CPU) ─► compared with PyTorch
```

- **Frontend (Python, [`imc/`](imc/)):**
  - FX capture of an MLP, a CNN trained on CIFAR-10, and a transformer block.
  - A hand-written importer to `func`/`linalg`/`tensor`/`arith`/`math`. It
    does its own shape and dtype inference (checked against torch's ShapeProp)
    and raises named errors for unsupported ops.
  - Conv+BatchNorm folding, producer-consumer fusion, and post-training int8
    quantization.
- **`imc` dialect (C++, ODS/TableGen, out-of-tree `imc-opt`,
  [`include/Imc`](include/Imc/ImcOps.td), [`lib/`](lib/)):**
  - the type `!imc.xbar<rows, cols>`;
  - the ops `imc.matmul`, `imc.program_weights`, `imc.mvm` (with an ADC model)
    and `imc.accumulate`;
  - verifiers that reject e.g. a 300×64 tile on a 256×256 crossbar, or
    non-int8 operands;
  - a **crossbar tiling pass** (weight-stationary, partial-sum accumulation)
    and a **lowering to linalg**.
- **Built against Ubuntu's `libmlir-20-dev` (MLIR 20.1.2).** No MLIR source
  build; `imc-opt` links `libMLIR.so`.

Machine: i5-9300H, WSL2 Ubuntu 24.04, GTX 1050 (training only). CI runs the
same build and tests on GitHub's ubuntu-24.04.

## Results

### 1. Correctness: the compiled code matches PyTorch, and the int8 path is bit-exact
From [`results/difftest.txt`](results/difftest.txt), produced by
`scripts/test.sh` (also run in CI):

| test | result |
|---|---|
| lit / FileCheck (verifier diagnostics, tiling, lowering, hand-checked e2e ADC run) | 4/4 files pass |
| fp32: MLP, CNN, transformer block, fused and unfused, vs eager PyTorch | max \|diff\| ≤ 9.5e-6 (6/6) |
| int8 crossbar path (tile + lower + run) vs PyTorch emulator, 2 models × 4 ADC/tile configs | **bit-identical** logits and int32 accumulators (8/8) |
| random-graph fuzzer: 200 random FX graphs × fused/unfused | 400/400 |
| unsupported input (sigmoid, unfolded BN, float64, broadcasting) | named error, 4/4 |

Results come back exact because the test harness bitcasts every f32 result
to i32 inside MLIR before printing. Planted bugs in the emulator (ADC
rounding, clamp bound) are both caught ([`scripts/mutants.py`](scripts/mutants.py)).

### 2. Fusion: kernel count and intermediate bytes
From [`results/fusion.md`](results/fusion.md). Batch 1. Kernels = linalg ops
other than fill.

| model | kernels unfused → fused | intermediate bytes unfused → fused | upstream `-linalg-fuse-elementwise-ops` | TorchInductor (CPU) |
|---|---|---|---|---|
| MLP | 8 → **6** | 4,648 → 3,112 (−33%) | 6 | 2 fused + 3 extern |
| CNN | 19 → **14** | 1.52 MB → 1.12 MB (−26%) | 14 | 6 fused + 6 extern |
| transformer | 29 → **26** | 90,496 → 74,112 (−18%) | 26 | 4 fused + 8 extern |

- My FX-level fusion gives **exactly the same kernel count** as upstream MLIR's
  elementwise fusion run on my unfused IR.
- Inductor is much better on the transformer (12 vs 26). It fuses reductions:
  it emits softmax and layernorm as one kernel each, where I emit 4 and 3.
  My pass does elementwise producer-consumer fusion only.

### 3. Quantization and the ADC: accuracy vs ADC bits
From [`results/accuracy.md`](results/accuracy.md). CIFAR-10, 10,000 test
images; calibration on 1,000 *training* images; 256-row crossbars; int8 with
per-channel weights.

| | MLP | CNN |
|---|---|---|
| fp32 | 58.71% | 85.36% |
| fp32 after conv+BN folding | 58.71% | 85.36% |
| int8 (ideal digital accumulation) | 58.65% | 85.38% |
| + 8-bit ADC per crossbar | 58.51% (−0.2 pp) | 85.14% (−0.2 pp) |
| + 6-bit ADC, min-max range | 56.17% | 82.47% |
| + 6-bit ADC, 99.99th-pct range | 57.13% (−1.6 pp) | 84.17% (−1.2 pp) |
| + 4-bit ADC, min-max range | 33.65% | **15.28%** |
| + 4-bit ADC, 99.99th-pct range | 33.65% | **52.26%** |

- **int8 itself costs nothing here** (≤ 0.07 pp). Per-channel weights did not
  beat per-tensor on these two models (85.29% vs 85.30%).
- **The ADC is where accuracy goes.** 8 bits is free, 6 bits costs about
  1–1.6 pp, and 4 bits breaks both models.
- **At 4 bits, how the ADC range is calibrated decides between 15% and 52%
  on the CNN.** Min-max picks the step from the single largest partial sum.
  Clipping the top 0.01% gives a 4× finer step.

### 4. Tiling: the crossbar size changes accuracy, not just utilization
From [`results/tiling.md`](results/tiling.md). Accuracy is measured with the
emulator; energy and arrays come from the **analytical model, not hardware**
([`imc/crossbar.py`](imc/crossbar.py)). Columns: tiles = arrays programmed;
util = cell utilization; ADC = ADC conversions per image.

| CNN crossbar | tiles | util | ADC | energy/img (model) | acc, 6-bit ADC | acc, 4-bit ADC |
|---|---|---|---|---|---|---|
| 64×64 | 150 | 96% | 434k | 352 nJ | 84.68% | **66.12%** |
| 128×128 | 45 | 80% | 266k | 263 nJ | 84.65% | 58.76% |
| 256×256 | 25 | 36% | 182k | 219 nJ | 84.17% | 52.26% |
| 512×512 | 14 | 16% | 116k | 183 nJ | 84.41% | 59.56% |

- The ADC rounds **each crossbar's partial sum separately**, so tiling is a
  numerics decision. The tiling pass is semantics-*changing* by design: the
  verifier refuses to lower an untiled `imc.matmul` that carries an ADC.
- Smaller crossbars sum fewer products per partial sum, so a given ADC step
  covers less range. At 4 bits, 64×64 beats 256×256 by 14 pp, at 2.4× the ADC
  conversions.
- **The curve is a staircase, not monotone.** Going from 256 to 512 rows, the
  calibrated ADC shift of the MLP's fc1 stays at 16 (the range did not cross a
  power of two). Meanwhile the number of separately rounded partial sums
  halves, 12 → 6, so 512 rows is *better* than 256 (MLP 6-bit: 58.03% vs
  57.13%).
- **Low utilization can force weight reprogramming**
  ([`results/reprogram.md`](results/reprogram.md)). The CNN has 0.59M weights,
  9 arrays' worth, but its conv layers use 1–19% of a 256×256 array, so it
  needs 25 arrays. On a 1M-cell chip, 256×256 crossbars therefore don't fit and
  the weights are rewritten every batch: 12,628 nJ/img at batch 1 in the model.
  128×128 crossbars fit: 263 nJ/img.

### 5. Requantization folding: int8 flows between crossbar layers
- **The fold:** a layer's dequantize epilogue (`float(acc)·sx·sw[n] + b`,
  relu) is fused with the *next* layer's quantize whenever only max-pool or
  flatten sits between them. Both commute exactly with a monotone quantizer.
- **The result:** each epilogue is one `linalg.generic` from i32 to i8, and
  max-pool runs on int8. No fp32 activation tensor exists between crossbar
  layers: 2 folds in the MLP, 5 in the CNN.
- **Checked:** covered by the bit-exact test.

## Layout

| path | what |
|---|---|
| [`include/Imc/ImcOps.td`](include/Imc/ImcOps.td), [`ImcPasses.td`](include/Imc/ImcPasses.td) | dialect, type, ops and passes in ODS |
| [`lib/ImcDialect.cpp`](lib/ImcDialect.cpp) | verifiers |
| [`lib/ImcPasses.cpp`](lib/ImcPasses.cpp) | `-imc-tile-crossbars`, `-imc-lower-to-linalg` |
| [`test/`](test/) | lit/FileCheck tests |
| [`imc/importer.py`](imc/importer.py) | FX → MLIR, fusion via lazy elementwise expressions, quantized emission |
| [`imc/fxpasses.py`](imc/fxpasses.py) | conv+BN folding, fusion marking |
| [`imc/quant.py`](imc/quant.py) | calibration, and an emulator of the exact compiled semantics |
| [`imc/crossbar.py`](imc/crossbar.py) | tiling statistics, analytical cost model |
| [`imc/difftest.py`](imc/difftest.py), [`imc/fuzz.py`](imc/fuzz.py) | differential tests, random graphs |
| [`docs/NOTES.md`](docs/NOTES.md) | study notes |

## Running it

```sh
# Linux/WSL with Ubuntu's llvm-20-dev, libmlir-20-dev, mlir-20-tools (or scripts/fetch-mlir-debs.sh, no sudo)
bash scripts/test.sh                    # build imc-opt, lit, difftest (+200 fuzz graphs)
bash scripts/py.sh -m imc.fusion_report # results/fusion.md
python scripts/get_cifar.py             # data/cifar10.npz
python -m imc.train                     # checkpoints/ (committed, so this is optional)
python -m imc.experiments               # results/accuracy.md, results/tiling.md (~40 min on CPU)
python -m imc.crossbar                  # results/reprogram.md
```

## Honest scope

- **The hardware is a model.**
  - The ADC is ideal apart from quantization: no noise, no nonlinearity, no
    device variation, and no IR drop in the crossbar.
  - Energy and latency come from an analytical model with assumed constants.
    Use them to compare mappings, never as chip numbers. In that model latency
    only depends on input vectors per image, so it is identical across tile
    shapes.
- **Accuracy is measured with a PyTorch emulator, not the compiled code.** The
  compiled code is bit-identical to the emulator on the tested inputs (2
  images × 8 configurations); running 10k images through mlir-runner's naive
  loops was not worth the time.
- **Small models.** The CNN is 85% on CIFAR-10. The transformer block is only
  compiled in fp32 (random weights) and is not mapped onto crossbars.
- **Op coverage is narrow.** It is listed in the importer's error message.
  There are no dynamic shapes and no batching beyond the static batch.
- **The tiling pass unrolls at compile time** (static shapes), rather than
  emitting `scf.for` or using MLIR's `TilingInterface`.
- **Fusion is elementwise producer-consumer only.** There is no reduction
  fusion and no fusion into the matmul.
- **Quantization is symmetric only.** Post-ReLU activations waste one bit of
  int8. There is no quantization-aware training.
- **Not done:**
  - no cross-check against torch-mlir's FX importer (no stable wheel for torch
    2.6);
  - no `torch.export` frontend;
  - no GPU codegen.
