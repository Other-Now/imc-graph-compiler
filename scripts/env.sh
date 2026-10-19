# Source from WSL: user-space LLVM/MLIR 20 unpacked from Ubuntu debs (no sudo; see fetch-mlir-debs.sh).
# In CI the same packages are apt-installed and LLVM_ROOT=/usr/lib/llvm-20.
export LLVM_ROOT=${LLVM_ROOT:-$HOME/llvm20/root/usr/lib/llvm-20}
if [ -d "$HOME/llvm20/root/usr/lib/x86_64-linux-gnu" ]; then
  export LD_LIBRARY_PATH=$HOME/llvm20/root/usr/lib/x86_64-linux-gnu:$LLVM_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi
export PATH=$LLVM_ROOT/bin:$PATH
# Python.h for TorchInductor's C++ backend, unpacked without sudo (fetch-pydev.sh).
if [ -d "$HOME/pydev/root/usr/include/python3.12" ]; then
  export CPATH=$HOME/pydev/root/usr/include/python3.12:$HOME/pydev/root/usr/include${CPATH:+:$CPATH}
fi
