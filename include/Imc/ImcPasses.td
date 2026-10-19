#ifndef IMC_PASSES
#define IMC_PASSES

include "mlir/Pass/PassBase.td"

def ImcTileCrossbars : Pass<"imc-tile-crossbars", "func::FuncOp"> {
  let summary = "Map imc.matmul onto fixed-size weight-stationary crossbars";
  let description = [{
    Splits the KxN weight matrix into ceil(K/rows) x ceil(N/cols) tiles. Each
    tile is programmed once and every input row is streamed through it
    (weight-stationary). Partial sums of tiles that share output columns are
    combined with imc.accumulate; column blocks are reassembled with
    tensor.insert_slice.
  }];
  let options = [
    Option<"rows", "rows", "int64_t", "256", "crossbar rows (K per tile)">,
    Option<"cols", "cols", "int64_t", "256", "crossbar columns (N per tile)">,
  ];
  let dependentDialects = ["tensor::TensorDialect"];
}

def ImcLowerToLinalg : Pass<"imc-lower-to-linalg", "func::FuncOp"> {
  let summary = "Lower imc ops to linalg so they run on a CPU";
  let description = [{
    imc.mvm becomes linalg.matmul (i8 x i8 -> i32) followed by a
    linalg.generic that models the ADC; imc.accumulate becomes linalg.add;
    imc.program_weights disappears (its tile feeds the matmul directly).
  }];
  let dependentDialects = ["linalg::LinalgDialect", "tensor::TensorDialect",
                           "arith::ArithDialect"];
}

#endif // IMC_PASSES
