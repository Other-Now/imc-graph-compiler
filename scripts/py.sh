#!/usr/bin/env bash
# Run python in the WSL venv from the repo root, with LLVM/MLIR on the env.
here=$(cd "$(dirname "$0")/.." && pwd)
. "$here/scripts/env.sh"
cd "$here"
export PYTHONPATH=$here${PYTHONPATH:+:$PYTHONPATH}
exec ${PYTHON:-$HOME/igc-venv/bin/python} "$@"
