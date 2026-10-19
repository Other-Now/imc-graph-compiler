#!/usr/bin/env bash
# Run the lit/FileCheck suite against the built imc-opt.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
. "$here/scripts/env.sh"
export IMC_BUILD=${BUILD:-$HOME/igc-build}
LIT=${LIT:-"python3 $LLVM_ROOT/build/utils/lit/lit.py"}
$LIT -v "$here/test"
