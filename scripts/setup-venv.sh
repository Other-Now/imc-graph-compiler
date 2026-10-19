#!/usr/bin/env bash
# One-time, no-sudo: a venv with CPU torch for the MLIR side (WSL / CI-like Linux).
set -euo pipefail
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
uv venv --python 3.12 ~/igc-venv
uv pip install --python ~/igc-venv/bin/python numpy torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
~/igc-venv/bin/python -c "import torch, numpy; print(torch.__version__, numpy.__version__)"
