//===- ImcPasses.cpp - crossbar tiling and lowering to linalg --------------===//

#include "Imc/ImcDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"

namespace mlir::imc {
#define GEN_PASS_DEF_IMCTILECROSSBARS
#define GEN_PASS_DEF_IMCLOWERTOLINALG
#include "Imc/ImcPasses.h.inc"
} // namespace mlir::imc

using namespace mlir;
using namespace mlir::imc;

static ArrayRef<int64_t> shapeOf(Value v) {
  return cast<RankedTensorType>(v.getType()).getShape();
}

static Value slice2d(OpBuilder &b, Location loc, Value src, int64_t r0,
                     int64_t c0, int64_t nr, int64_t nc) {
  SmallVector<OpFoldResult> off{b.getIndexAttr(r0), b.getIndexAttr(c0)};
  SmallVector<OpFoldResult> size{b.getIndexAttr(nr), b.getIndexAttr(nc)};
  SmallVector<OpFoldResult> one{b.getIndexAttr(1), b.getIndexAttr(1)};
  return b.create<tensor::ExtractSliceOp>(loc, src, off, size, one);
}

//===----------------------------------------------------------------------===//
// -imc-tile-crossbars
//===----------------------------------------------------------------------===//

namespace {
struct TileCrossbars : imc::impl::ImcTileCrossbarsBase<TileCrossbars> {
  using ImcTileCrossbarsBase::ImcTileCrossbarsBase;

  void runOnOperation() override {
    if (rows <= 0 || cols <= 0) {
      getOperation().emitError("crossbar rows and cols must be positive");
      return signalPassFailure();
    }
    SmallVector<MatmulOp> work;
    getOperation().walk([&](MatmulOp op) { work.push_back(op); });

    for (MatmulOp op : work) {
      if (!op.getAdcBits()) {
        op.emitError("imc.matmul needs adc_bits/adc_shift before it can be "
                     "mapped onto crossbars");
        return signalPassFailure();
      }
      OpBuilder b(op);
      Location loc = op.getLoc();
      int64_t M = shapeOf(op.getLhs())[0], K = shapeOf(op.getLhs())[1],
              N = shapeOf(op.getRhs())[1];
      Type i32 = b.getI32Type();
      auto xbarTy = CrossbarType::get(b.getContext(), rows, cols);

      Value out = b.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{M, N}, i32);
      for (int64_t n0 = 0; n0 < N; n0 += cols) {
        int64_t nc = std::min<int64_t>(cols, N - n0);
        Value acc;
        for (int64_t k0 = 0; k0 < K; k0 += rows) {
          int64_t kr = std::min<int64_t>(rows, K - k0);
          Value w = slice2d(b, loc, op.getRhs(), k0, n0, kr, nc);
          Value xb = b.create<ProgramWeightsOp>(loc, xbarTy, w);
          Value x = slice2d(b, loc, op.getLhs(), 0, k0, M, kr);
          Value p = b.create<MvmOp>(loc, RankedTensorType::get({M, nc}, i32),
                                    xb, x, op.getAdcBitsAttr(),
                                    op.getAdcShiftAttr());
          acc = acc ? b.create<AccumulateOp>(loc, acc, p).getResult() : p;
        }
        SmallVector<OpFoldResult> off{b.getIndexAttr(0), b.getIndexAttr(n0)};
        SmallVector<OpFoldResult> size{b.getIndexAttr(M), b.getIndexAttr(nc)};
        SmallVector<OpFoldResult> one{b.getIndexAttr(1), b.getIndexAttr(1)};
        out = b.create<tensor::InsertSliceOp>(loc, acc, out, off, size, one);
      }
      op.replaceAllUsesWith(out);
      op.erase();
    }
  }
};

//===----------------------------------------------------------------------===//
// -imc-lower-to-linalg
//===----------------------------------------------------------------------===//

// Exact i8 x i8 -> i32 matmul. linalg.matmul sign-extends its operands to the
// accumulator type.
static Value intMatmul(OpBuilder &b, Location loc, Value x, Value w) {
  int64_t M = shapeOf(x)[0], N = shapeOf(w)[1];
  Type i32 = b.getI32Type();
  Value empty = b.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{M, N}, i32);
  Value zero = b.create<arith::ConstantIntOp>(loc, 0, 32);
  Value init = b.create<linalg::FillOp>(loc, zero, empty).getResult(0);
  return b.create<linalg::MatmulOp>(loc, ValueRange{x, w}, ValueRange{init})
      .getResult(0);
}

// The ADC model from imc.mvm's description, as one elementwise linalg.generic.
static Value adc(OpBuilder &b, Location loc, Value v, int64_t bits,
                 int64_t shift) {
  auto ty = cast<RankedTensorType>(v.getType());
  Value empty = b.create<tensor::EmptyOp>(loc, ty.getShape(), ty.getElementType());
  AffineMap id = b.getMultiDimIdentityMap(2);
  SmallVector<utils::IteratorType> par(2, utils::IteratorType::parallel);
  auto gen = b.create<linalg::GenericOp>(
      loc, TypeRange{ty}, ValueRange{v}, ValueRange{empty},
      ArrayRef<AffineMap>{id, id}, par,
      [&](OpBuilder &nb, Location l, ValueRange args) {
        auto c = [&](int64_t k) {
          return nb.create<arith::ConstantIntOp>(l, k, 32).getResult();
        };
        Value x = args[0];
        if (shift > 0) {
          x = nb.create<arith::AddIOp>(l, x, c(int64_t(1) << (shift - 1)));
          x = nb.create<arith::ShRSIOp>(l, x, c(shift));
        }
        x = nb.create<arith::MaxSIOp>(l, x, c(-(int64_t(1) << (bits - 1))));
        x = nb.create<arith::MinSIOp>(l, x, c((int64_t(1) << (bits - 1)) - 1));
        if (shift > 0)
          x = nb.create<arith::ShLIOp>(l, x, c(shift));
        nb.create<linalg::YieldOp>(l, x);
      });
  return gen.getResult(0);
}

struct LowerToLinalg : imc::impl::ImcLowerToLinalgBase<LowerToLinalg> {
  void runOnOperation() override {
    bool failedAny = false;
    getOperation().walk([&](Operation *op) {
      OpBuilder b(op);
      Location loc = op->getLoc();
      if (auto mm = dyn_cast<MatmulOp>(op)) {
        if (mm.getAdcBits()) {
          mm.emitError("imc.matmul with an ADC has no meaning until it is "
                       "tiled; run -imc-tile-crossbars first");
          failedAny = true;
          return;
        }
        mm.replaceAllUsesWith(intMatmul(b, loc, mm.getLhs(), mm.getRhs()));
        mm.erase();
      } else if (auto mvm = dyn_cast<MvmOp>(op)) {
        auto pw = mvm.getCrossbar().getDefiningOp<ProgramWeightsOp>();
        if (!pw) {
          mvm.emitError("crossbar is not defined by imc.program_weights");
          failedAny = true;
          return;
        }
        Value v = intMatmul(b, loc, mvm.getInput(), pw.getWeights());
        mvm.replaceAllUsesWith(adc(b, loc, v, mvm.getAdcBits(), mvm.getAdcShift()));
        mvm.erase();
      } else if (auto acc = dyn_cast<AccumulateOp>(op)) {
        auto ty = cast<RankedTensorType>(acc.getType());
        Value empty = b.create<tensor::EmptyOp>(loc, ty.getShape(), ty.getElementType());
        acc.replaceAllUsesWith(
            b.create<linalg::AddOp>(loc, ValueRange{acc.getLhs(), acc.getRhs()},
                                    ValueRange{empty})
                .getResult(0));
        acc.erase();
      }
    });
    // Every program_weights is now dead: its tile was folded into a matmul.
    getOperation().walk([&](ProgramWeightsOp pw) {
      if (pw->use_empty())
        pw.erase();
    });
    if (failedAny)
      signalPassFailure();
  }
};
} // namespace
