#!/usr/bin/env bash
# Build imc-opt, run the lit/FileCheck suite, then the Python differential tests.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
bash "$here/scripts/build.sh"
bash "$here/scripts/lit.sh"
bash "$here/scripts/py.sh" -m imc.difftest --fuzz "${FUZZ:-200}"
