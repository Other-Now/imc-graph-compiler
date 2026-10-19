#!/usr/bin/env bash
# One-time, no-sudo: unpack Ubuntu noble's MLIR 20 debs next to the LLVM 20 ones.
set -euo pipefail
mkdir -p ~/llvm20/debs && cd ~/llvm20/debs
apt-get download libmlir-20 libmlir-20-dev mlir-20-tools
for f in libmlir-20_*.deb libmlir-20-dev_*.deb mlir-20-tools_*.deb; do
  dpkg-deb -x "$f" ~/llvm20/root
done
ls ~/llvm20/root/usr/lib/llvm-20/bin | grep -i mlir
ls ~/llvm20/root/usr/lib/llvm-20/lib/cmake
