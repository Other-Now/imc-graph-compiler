#!/usr/bin/env bash
# linalg-on-tensors -> LLVM dialect, for CPU execution with mlir-runner.
# Reads stdin, writes stdout. Extra args are inserted before bufferization.
exec mlir-opt "$@" \
  --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
  --convert-linalg-to-loops --lower-affine --expand-strided-metadata \
  --convert-scf-to-cf --convert-math-to-llvm --convert-math-to-libm \
  --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
  --convert-cf-to-llvm --reconcile-unrealized-casts
