#!/usr/bin/env bash
# Configure + build imc-opt. Build dir lives on the Linux side (fast, survives).
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
. "$here/scripts/env.sh"
BUILD=${BUILD:-$HOME/igc-build}
cmake -S "$here" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR="$LLVM_ROOT/lib/cmake/mlir" -DLLVM_DIR="$LLVM_ROOT/lib/cmake/llvm" \
  -DCMAKE_CXX_COMPILER=${CXX:-g++} >/dev/null
ninja -C "$BUILD" -j ${JOBS:-2}
