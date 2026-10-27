"""The same native CUDA toolchain for tests and live child processes."""

import os
from pathlib import Path

from wan_dit_lab.harness.config import ROOT


def gpu_environment():
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONIOENCODING="utf-8")
    if os.name == "nt":
        import triton

        toolkit = ROOT / "artifacts/cuda_toolkit"
        env.update(CUDA_HOME=str(toolkit), CUDA_PATH=str(toolkit),
                   CC=str(Path(triton.__file__).parent / "runtime/tcc/tcc.exe"))
        env["PATH"] = str(toolkit / "bin") + os.pathsep + env["PATH"]
    return env
