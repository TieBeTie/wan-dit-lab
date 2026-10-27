"""Builds the operator libraries live_w4a4.dll and live_sage.dll: the CUDA kernels of Nunchaku (W4A4) and of
SageAttention (sm89) behind TORCH_LIBRARY operators live::*, linked to LibTorch and not to Python. The Python
model loads them with torch.ops.load_library (wan_dit_lab.kernels.operators), the Rust host before its
AOTInductor packages.

    python native/ops/build.py [live_w4a4 live_sage]

Needs MSVC (vcvars64) and nvcc: artifacts/cuda-12.8 by default, or CUDA_HOME. The flags of each library are
those of its upstream setup.py, for sm_89 only."""
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ.setdefault("CUDA_HOME", str(ROOT / "artifacts" / "cuda-12.8"))
from torch.utils import cpp_extension  # noqa: E402  (reads CUDA_HOME on import)

N = ROOT / "reference" / "nunchaku"
S = ROOT / "reference" / "SageAttention" / "csrc"
OVERRIDE = HERE / "override"  # torch/extension.h without pybind11
FIX = ROOT / "artifacts" / "torch_header_fix"  # nvcc and torch/csrc/dynamo/compiled_autograd.h

LIBRARIES = {
    # Nunchaku's setup.py: BF16, C++20, assertions kept
    "live_w4a4": dict(
        sources=[HERE / "w4a4_ops.cpp", N / "src/interop/torch.cpp",
                 *(N / "src/kernels/zgemm" / f for f in (
                     "gemm_w4a4.cu", "gemm_w4a4_launch_fp16_int4.cu", "gemm_w4a4_launch_fp16_int4_fasteri2f.cu",
                     "gemm_w4a4_launch_fp16_fp4.cu", "gemm_w4a4_launch_bf16_int4.cu", "gemm_w4a4_launch_bf16_fp4.cu",
                     "gemm_w4a4_test.cu", "attention.cu")),
                 N / "src/kernels/awq/gemm_awq.cu", N / "src/kernels/awq/gemv_awq.cu"],
        include=[OVERRIDE, FIX, N / "src", N / "nunchaku/csrc",
                 *(N / "third_party" / d for d in ("cutlass/include", "json/include", "mio/include", "spdlog/include"))],
        cxx=["/DENABLE_BF16=1", "/DBUILD_NUNCHAKU=1", "/std:c++20", "/UNDEBUG", "/Zc:__cplusplus", "/FS"],
        nvcc=["-DENABLE_BF16=1", "-DBUILD_NUNCHAKU=1", "-std=c++20", "-UNDEBUG", "-Xcudafe", "--diag_suppress=20208",
              "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
              "-U__CUDA_NO_HALF2_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_OPERATORS__",
              "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_BFLOAT162_OPERATORS__",
              "-U__CUDA_NO_BFLOAT162_CONVERSIONS__", "--threads=1", "--expt-relaxed-constexpr",
              "--expt-extended-lambda", "--ptxas-options=--allow-expensive-optimizations=true",
              "-gencode", "arch=compute_89,code=sm_89",
              "-Xcompiler", "/Zc:__cplusplus", "-Xcompiler", "/FS", "-Xcompiler", "/bigobj"]),
    # SageAttention's setup.py for its sm89 kernels; only the one Live runs
    "live_sage": dict(
        sources=[HERE / "sage_ops.cpp", S / "qattn/sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.cu"],
        include=[OVERRIDE, FIX, S / "qattn", S],
        cxx=["/O2", "/std:c++17", "/Zc:__cplusplus", "/FS"],
        nvcc=["-O3", "-std=c++17", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
              "--use_fast_math", "--threads=8", "-diag-suppress=174", "--expt-relaxed-constexpr",
              "-gencode", "arch=compute_89,code=sm_89", "-Xcompiler", "/Zc:__cplusplus", "-Xcompiler", "/FS"]),
}


def build(name):
    """Builds `name` with ninja in build/<name>/ and puts it at build/<name>.dll (cpp_extension names it .pyd)."""
    spec = LIBRARIES[name]
    directory = HERE / "build" / name
    directory.mkdir(parents=True, exist_ok=True)
    cpp_extension.load(name, [str(s) for s in spec["sources"]], extra_cflags=spec["cxx"],
                       extra_cuda_cflags=spec["nvcc"], extra_include_paths=[str(i) for i in spec["include"]],
                       build_directory=str(directory), is_python_module=False, with_cuda=True, verbose=True)
    target = HERE / "build" / f"{name}.dll"
    shutil.copyfile(directory / f"{name}.pyd", target)
    return target


if __name__ == "__main__":
    import sys
    for name in sys.argv[1:] or LIBRARIES:
        print(name, build(name), flush=True)
    sys.path.insert(0, str(ROOT / "src"))
    from wan_dit_lab.kernels.operators import load
    print([op for op in dir(load()) if not op.startswith("_")])
