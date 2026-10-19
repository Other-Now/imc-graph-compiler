#ifndef IMC_IMCDIALECT_H
#define IMC_IMCDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "Imc/ImcDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "Imc/ImcTypes.h.inc"

#define GET_OP_CLASSES
#include "Imc/ImcOps.h.inc"

namespace mlir::imc {
#define GEN_PASS_DECL
#include "Imc/ImcPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "Imc/ImcPasses.h.inc"
} // namespace mlir::imc

#endif // IMC_IMCDIALECT_H
