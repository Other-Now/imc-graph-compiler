//===- ImcOps.td - analog in-memory-compute dialect ----------*- tablegen -*-===//
//
// A model of a weight-stationary analog in-memory-compute (IMC) accelerator.
// A crossbar holds an int8 weight tile in its cells; driving its rows with an
// int8 input vector produces one analog dot product per column, which an ADC
// digitizes to `adc_bits`. Partial sums from crossbars that share output
// columns are added digitally (`imc.accumulate`).
//
//   imc.matmul         (untiled)  MxK i8  x  KxN i8  ->  MxN i32
//     | -imc-tile-crossbars
//   imc.program_weights  KtxNt i8 tile -> !imc.xbar<R, C>
//   imc.mvm              xbar, MxKt i8 -> MxNt i32 (ADC applied per column)
//   imc.accumulate       i32 + i32 partial sums across K tiles
//     | -imc-lower-to-linalg
//   linalg.matmul / linalg.generic (ADC) / linalg.add
//
//===----------------------------------------------------------------------===//

#ifndef IMC_OPS
#define IMC_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Imc_Dialect : Dialect {
  let name = "imc";
  let cppNamespace = "::mlir::imc";
  let summary = "Weight-stationary analog in-memory-compute crossbars";
  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

def Imc_CrossbarType : TypeDef<Imc_Dialect, "Crossbar"> {
  let mnemonic = "xbar";
  let summary = "a crossbar array of `rows` x `cols` int8 cells";
  let parameters = (ins "int64_t":$rows, "int64_t":$cols);
  let assemblyFormat = "`<` $rows `,` $cols `>`";
}

//===----------------------------------------------------------------------===//
// Ops. Operand types are deliberately loose (any ranked tensor) so that the
// hand-written verifiers can give hardware-level error messages.
//===----------------------------------------------------------------------===//

class Imc_Op<string mnemonic, list<Trait> traits = []>
    : Op<Imc_Dialect, mnemonic, traits>;

def Imc_MatmulOp : Imc_Op<"matmul", [Pure]> {
  let summary = "untiled int8 x int8 -> int32 matrix multiply";
  let description = [{
    `lhs` (MxK) are activations, `rhs` (KxN) are weights. Without ADC
    attributes this is an exact integer matmul. With `adc_bits`/`adc_shift`
    the result is only defined once the op is tiled onto crossbars, because
    the ADC rounds each crossbar's partial sum separately: tiling changes the
    numerics, so it is a semantic choice, not an optimization.
  }];
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs,
                       OptionalAttr<I64Attr>:$adc_bits,
                       OptionalAttr<I64Attr>:$adc_shift);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
  let hasVerifier = 1;
}

def Imc_ProgramWeightsOp : Imc_Op<"program_weights", [Pure]> {
  let summary = "write an int8 weight tile into a crossbar";
  let arguments = (ins AnyRankedTensor:$weights);
  let results = (outs Imc_CrossbarType:$crossbar);
  let assemblyFormat = "$weights attr-dict `:` type($weights) `->` qualified(type($crossbar))";
  let hasVerifier = 1;
}

def Imc_MvmOp : Imc_Op<"mvm", [Pure]> {
  let summary = "analog matrix-vector multiply on a programmed crossbar";
  let description = [{
    Each of the M rows of `input` is one input vector driven onto the
    crossbar. Every column's integer dot product `v` is digitized by an ADC
    with `adc_bits` of resolution and a power-of-two step `2^adc_shift`:

      q = clamp((v + 2^(shift-1)) >> shift, -2^(bits-1), 2^(bits-1)-1) << shift
  }];
  let arguments = (ins Imc_CrossbarType:$crossbar, AnyRankedTensor:$input,
                       I64Attr:$adc_bits, I64Attr:$adc_shift);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    $crossbar `,` $input attr-dict `:` qualified(type($crossbar)) `,` type($input) `->` type($result)
  }];
  let hasVerifier = 1;
}

def Imc_AccumulateOp : Imc_Op<"accumulate", [Pure, Commutative,
                                             SameOperandsAndResultType]> {
  let summary = "digital int32 partial-sum accumulation";
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let hasVerifier = 1;
}

#endif // IMC_OPS
