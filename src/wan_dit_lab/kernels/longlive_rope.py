"""AOT adapter for LongLive 2.0's original FP32 Triton kernel.

The kernel stays in the reference checkout; only its input table is adapted
from the native host's real cos/sin table. No duplicated numerical kernel.
"""
import importlib.util
from pathlib import Path

import torch
import triton
from torch.library import triton_op, wrap_triton

_path = Path(__file__).resolve().parents[3] / "reference/LongLive/LongLive2.0/utils/rope_triton.py"
_spec = importlib.util.spec_from_file_location("longlive_original_rope", _path)
upstream = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(upstream)


@triton_op("wan_dit_lab::longlive_rope", mutates_args={})
def _rotate(x: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor, length: int) -> torch.Tensor:
    out = torch.empty_like(x)
    tokens, heads, dim = x.shape
    wrap_triton(upstream._rope_apply_kernel)[(tokens, triton.cdiv(heads, 8))](
        x, cosine, sine, out, length, heads, dim // 2,
        x.stride(0), x.stride(1), out.stride(0), out.stride(1), cosine.stride(0),
        BLOCK_N=8, BLOCK_D=triton.next_power_of_2(dim // 2))
    return out


def rope_apply(x, grid_sizes, freq, start_frame=0):
    pairs = x.shape[-1] // 2
    spatial = pairs // 3
    time, height, width = freq.split([pairs - 2 * spatial, spatial, spatial], -1)
    result = []
    for sample, (f, h, w) in zip(x.unbind(0), grid_sizes):
        cs = torch.cat([
            time[:, start_frame:start_frame + f].view(2, f, 1, 1, -1).expand(2, f, h, w, -1),
            height[:, :h].view(2, 1, h, 1, -1).expand(2, f, h, w, -1),
            width[:, :w].view(2, 1, 1, w, -1).expand(2, f, h, w, -1),
        ], -1).reshape(2, f * h * w, pairs).float()
        result.append(_rotate(sample, cs[0].contiguous(), cs[1].contiguous(), f * h * w))
    return torch.stack(result)
