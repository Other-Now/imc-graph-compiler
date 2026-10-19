//===- imc-opt.cpp - mlir-opt with the imc dialect -------------------------===//
//
// Registers only the dialects the pipeline produces, plus the generic
// transform passes (canonicalize, cse, ...). Lowering past linalg is left to
// upstream mlir-opt.

#include "Imc/ImcDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::imc::ImcDialect, mlir::func::FuncDialect,
                  mlir::arith::ArithDialect, mlir::tensor::TensorDialect,
                  mlir::linalg::LinalgDialect, mlir::math::MathDialect>();
  mlir::registerTransformsPasses();
  mlir::imc::registerImcPasses();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "imc optimizer driver\n", registry));
}
