//===- ImcDialect.cpp - dialect, type and op verifiers ---------------------===//

#include "Imc/ImcDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::imc;

#include "Imc/ImcDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Imc/ImcTypes.cpp.inc"

#define GET_OP_CLASSES
#include "Imc/ImcOps.cpp.inc"

void ImcDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Imc/ImcTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "Imc/ImcOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Shared checks
//===----------------------------------------------------------------------===//

// A static 2-D tensor whose elements are signless integers of `bits` width.
static LogicalResult checkIntMatrix(Operation *op, Value v, StringRef what,
                                    unsigned bits) {
  auto t = dyn_cast<RankedTensorType>(v.getType());
  if (!t || t.getRank() != 2 || !t.hasStaticShape())
    return op->emitOpError() << what << " must be a static 2-D tensor, got "
                             << v.getType();
  if (!t.getElementType().isSignlessInteger(bits))
    return op->emitOpError()
           << what << " must have i" << bits << " elements (crossbar cells "
           << "and ADC outputs are integers), got " << t.getElementType();
  return success();
}

static LogicalResult checkAdc(Operation *op, int64_t bits, int64_t shift) {
  if (bits < 1 || bits > 16)
    return op->emitOpError() << "adc_bits must be in [1, 16], got " << bits;
  if (shift < 0 || shift > 30)
    return op->emitOpError() << "adc_shift must be in [0, 30], got " << shift;
  return success();
}

static ArrayRef<int64_t> shapeOf(Value v) {
  return cast<RankedTensorType>(v.getType()).getShape();
}

//===----------------------------------------------------------------------===//
// Verifiers
//===----------------------------------------------------------------------===//

LogicalResult MatmulOp::verify() {
  if (failed(checkIntMatrix(*this, getLhs(), "lhs", 8)) ||
      failed(checkIntMatrix(*this, getRhs(), "rhs", 8)) ||
      failed(checkIntMatrix(*this, getResult(), "result", 32)))
    return failure();
  auto l = shapeOf(getLhs()), r = shapeOf(getRhs()), o = shapeOf(getResult());
  if (l[1] != r[0])
    return emitOpError() << "contraction mismatch: lhs is " << l[0] << "x"
                         << l[1] << " but rhs is " << r[0] << "x" << r[1];
  if (o[0] != l[0] || o[1] != r[1])
    return emitOpError() << "result must be " << l[0] << "x" << r[1];
  if (getAdcBits().has_value() != getAdcShift().has_value())
    return emitOpError() << "adc_bits and adc_shift must be given together";
  if (getAdcBits())
    return checkAdc(*this, *getAdcBits(), *getAdcShift());
  return success();
}

LogicalResult ProgramWeightsOp::verify() {
  if (failed(checkIntMatrix(*this, getWeights(), "weights", 8)))
    return failure();
  auto w = shapeOf(getWeights());
  CrossbarType xb = getCrossbar().getType();
  if (w[0] > xb.getRows() || w[1] > xb.getCols())
    return emitOpError() << "a " << w[0] << "x" << w[1]
                         << " weight tile does not fit a " << xb.getRows()
                         << "x" << xb.getCols() << " crossbar";
  return success();
}

LogicalResult MvmOp::verify() {
  if (failed(checkIntMatrix(*this, getInput(), "input", 8)) ||
      failed(checkIntMatrix(*this, getResult(), "result", 32)) ||
      failed(checkAdc(*this, getAdcBits(), getAdcShift())))
    return failure();
  auto in = shapeOf(getInput()), out = shapeOf(getResult());
  CrossbarType xb = getCrossbar().getType();
  if (in[0] != out[0])
    return emitOpError() << "result must have " << in[0] << " rows";
  if (in[1] > xb.getRows())
    return emitOpError() << "input drives " << in[1] << " rows but the crossbar has "
                         << xb.getRows();
  if (out[1] > xb.getCols())
    return emitOpError() << "result reads " << out[1]
                         << " columns but the crossbar has " << xb.getCols();
  // When the programming op is visible, the shapes must agree exactly.
  if (auto pw = getCrossbar().getDefiningOp<ProgramWeightsOp>()) {
    auto w = shapeOf(pw.getWeights());
    if (in[1] != w[0] || out[1] != w[1])
      return emitOpError() << "crossbar was programmed with a " << w[0] << "x"
                           << w[1] << " tile but is used as " << in[1] << "x"
                           << out[1];
  }
  return success();
}

LogicalResult AccumulateOp::verify() {
  return checkIntMatrix(*this, getResult(), "result", 32);
}
