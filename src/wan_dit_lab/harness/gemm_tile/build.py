"""Builds the tile bench (gemm_tile_bench.cu) with the flags of live_w4a4 (ops/build.py) plus ptxas -v, into
ops/build/gemm_tile_bench/. Bench only: Live never loads it.

    python -m wan_dit_lab.harness.gemm_tile.build
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "ops"))
import build as ops  # noqa: E402  (sets CUDA_HOME before torch.utils.cpp_extension is imported)
from torch.utils import cpp_extension  # noqa: E402


def build():
    spec = ops.LIBRARIES["live_w4a4"]
    directory = ROOT / "ops" / "build" / "gemm_tile_bench"
    directory.mkdir(parents=True, exist_ok=True)
    cpp_extension.load("gemm_tile_bench", [str(Path(__file__).with_name("gemm_tile_bench.cu"))],
                       extra_cflags=spec["cxx"], extra_cuda_cflags=spec["nvcc"] + ["-Xptxas", "-v"],
                       extra_include_paths=[str(i) for i in spec["include"]], build_directory=str(directory),
                       is_python_module=False, with_cuda=True, verbose=True)
    return directory / "gemm_tile_bench.pyd"


if __name__ == "__main__":
    print(build())
