import os
import lit.formats

config.name = "imc"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(os.environ["IMC_BUILD"], "test")

llvm = os.environ["LLVM_ROOT"]
repo = os.path.dirname(config.test_source_root)
config.environment["PATH"] = os.pathsep.join(
    [os.environ["IMC_BUILD"], os.path.join(llvm, "bin"), os.environ["PATH"]])
if "LD_LIBRARY_PATH" in os.environ:
    config.environment["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]

libs = ",".join(os.path.join(llvm, "lib", l) for l in
                ("libmlir_runner_utils.so", "libmlir_c_runner_utils.so"))
config.substitutions.append(("%lower", "bash " + os.path.join(repo, "scripts", "lower.sh")))
config.substitutions.append(
    ("%run", f"mlir-runner -e main -entry-point-result=void -shared-libs={libs}"))
