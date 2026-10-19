"""Compile an imported function with imc-opt + mlir-opt and run it on the CPU
with mlir-runner. Inputs are baked in as constants; every result is printed
as raw i32 bits (floats are bitcast first), so values come back bit-exact."""
import os
import re
import subprocess

import numpy as np

from .importer import dense, idmap, ttype

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLVM_ROOT = os.environ.get("LLVM_ROOT", os.path.expanduser("~/llvm20/root/usr/lib/llvm-20"))
IMC_BUILD = os.environ.get("IMC_BUILD", os.path.expanduser("~/igc-build"))


def _tool(name):
    return os.path.join(IMC_BUILD if name == "imc-opt" else os.path.join(LLVM_ROOT, "bin"), name)


def harness(forward, inputs, result_types):
    lines = ["module {", "func.func private @printMemrefI32(tensor<*xi32>)", forward, "func.func @main() {"]
    names = []
    for i, x in enumerate(inputs):
        lines.append(f"  %in{i} = arith.constant {dense(x.astype(np.float32))}")
        names.append(f"%in{i}")
    n = len(result_types)
    res = "%r" if n == 1 else f"%r:{n}"
    tys = ", ".join(ttype(s, d) for s, d in result_types)
    in_tys = ", ".join(ttype(x.shape) for x in inputs)
    lines.append(f"  {res} = call @forward({', '.join(names)}) : ({in_tys}) -> ({tys})")
    for i, (shape, dt) in enumerate(result_types):
        v = "%r" if n == 1 else f"%r#{i}"
        it = ttype(shape, "i32")
        if dt == "f32":
            r = len(shape)
            lines += [f"  %e{i} = tensor.empty() : {it}",
                      f"  %b{i} = linalg.generic {{indexing_maps = [{idmap(r)}, {idmap(r)}], iterator_types = "
                      f"[{', '.join(['\"parallel\"'] * r)}]}} ins({v} : {ttype(shape)}) outs(%e{i} : {it}) {{",
                      "  ^bb0(%a: f32, %o: i32):",
                      "    %c = arith.bitcast %a : f32 to i32",
                      "    linalg.yield %c : i32",
                      f"  }} -> {it}"]
            v = f"%b{i}"
        lines += [f"  %u{i} = tensor.cast {v} : {it} to tensor<*xi32>",
                  f"  call @printMemrefI32(%u{i}) : (tensor<*xi32>) -> ()"]
    lines += ["  return", "}", "}"]
    return "\n".join(lines)


def compile_and_run(forward, inputs, result_types, imc_passes=None, pre_lower=(), keep=None):
    """imc_passes: e.g. ["-imc-tile-crossbars=rows=256 cols=256", "-imc-lower-to-linalg"]."""
    src = harness(forward, inputs, result_types)
    if keep:
        with open(keep, "w") as f:
            f.write(src)
    if imc_passes:
        src = _run([_tool("imc-opt"), *imc_passes], src)
    llvm = _run(["bash", os.path.join(REPO, "scripts", "lower.sh"), *pre_lower], src)
    libs = ",".join(os.path.join(LLVM_ROOT, "lib", l) for l in ("libmlir_runner_utils.so", "libmlir_c_runner_utils.so"))
    out = _run([_tool("mlir-runner"), "-e", "main", "-entry-point-result=void", f"-shared-libs={libs}"], llvm)
    return parse(out, result_types)


def _run(cmd, stdin):
    env = dict(os.environ, PATH=os.path.join(LLVM_ROOT, "bin") + os.pathsep + os.environ["PATH"])
    p = subprocess.run(cmd, input=stdin, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"{os.path.basename(cmd[0])} failed:\n{p.stderr[:3000]}")
    return p.stdout


def parse(out, result_types):
    blocks = out.split("Unranked Memref")[1:]
    if len(blocks) != len(result_types):
        raise RuntimeError(f"expected {len(result_types)} printed tensors, got {len(blocks)}")
    res = []
    for blk, (shape, dt) in zip(blocks, result_types):
        nums = np.array(re.findall(r"-?\d+", blk.split("data =", 1)[1]), dtype=np.int64).astype(np.int32)
        a = nums.view(np.float32) if dt == "f32" else nums
        res.append(a.reshape(shape))
    return res


def count_linalg_ops(text):
    """Kernels in a module: linalg ops other than fill/yield/index."""
    ops = re.findall(r"=\s*linalg\.(\w+)", text)
    return sum(1 for o in ops if o not in ("fill", "yield", "index"))


def mlir_opt(text, *passes):
    return _run([_tool("mlir-opt"), *passes], text)
