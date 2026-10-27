"""The operators live::* of native/ops (built by native/ops/build.py): the W4A4 layers of Nunchaku and the attention
kernel of SageAttention on sm89, C++ libraries without Python that both this package and the Rust host load."""

from pathlib import Path

import torch

BUILD = Path(__file__).resolve().parents[3] / "native" / "ops" / "build"
LIBRARIES = ("live_w4a4", "live_sage")


def library(name):
    path = BUILD / f"{name}.dll"
    if not path.exists():
        raise FileNotFoundError(f"{path} is not built: python native/ops/build.py")
    return path


def load():
    """Registers live::* in this process, once."""
    if not hasattr(torch.ops.live, "sage_sm89"):
        for name in LIBRARIES:
            torch.ops.load_library(str(library(name)))
    return torch.ops.live
