"""Optional single-kernel 3D RoPE. The model's original rope.py is unchanged.

Input: (batch, token, heads, dim), frequencies: (cos/sin, position, pair).
Rotations use the input/frequency promoted precision, with fusion disabled.
Frequencies are fixed positional constants, not trainable parameters.
"""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.jit
def _rope_kernel(
    X, FREQ, OUT,
    LENGTH: tl.constexpr, HEADS: tl.constexpr, PAIRS: tl.constexpr,
    XS0: tl.constexpr, XS1: tl.constexpr, XS2: tl.constexpr, XS3: tl.constexpr,
    FS0: tl.constexpr, FS1: tl.constexpr, FS2: tl.constexpr,
    GRIDS: tl.constexpr, START: tl.constexpr, INVERSE: tl.constexpr,
    TOTAL: tl.constexpr, BLOCK: tl.constexpr, USE_FP64: tl.constexpr,
):
    # A logical lane owns a pair, so both outputs stay in registers until store.
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < TOTAL
    pair = index % PAIRS
    head = index // PAIRS % HEADS
    token = index // (PAIRS * HEADS) % LENGTH
    batch = index // (PAIRS * HEADS * LENGTH)
    frames = tl.full((BLOCK,), GRIDS[0][0], tl.int32)
    height = tl.full((BLOCK,), GRIDS[0][1], tl.int32)
    width = tl.full((BLOCK,), GRIDS[0][2], tl.int32)
    for b in tl.static_range(1, len(GRIDS)):
        frames = tl.where(batch == b, GRIDS[b][0], frames)
        height = tl.where(batch == b, GRIDS[b][1], height)
        width = tl.where(batch == b, GRIDS[b][2], width)
    spatial = PAIRS // 3
    time_pairs = PAIRS - 2 * spatial
    position = tl.where(pair < time_pairs, START + token // (height * width),
                        tl.where(pair < time_pairs + spatial,
                                 token // width % height, token % width))
    rotate = valid & (token < frames * height * width)
    compute_dtype = tl.float64 if USE_FP64 else tl.float32
    cosine = tl.load(FREQ + position * FS1 + pair * FS2, rotate, other=1).to(compute_dtype)
    sine = tl.load(FREQ + FS0 + position * FS1 + pair * FS2,
                   rotate, other=0).to(compute_dtype)
    if INVERSE:
        sine = -sine
    address = batch * XS0 + token * XS1 + head * XS2 + (2 * pair) * XS3
    even = tl.load(X + address, valid, other=0).to(compute_dtype)
    odd = tl.load(X + address + XS3, valid, other=0).to(compute_dtype)
    out_even = even * cosine - odd * sine
    out_odd = odd * cosine + even * sine
    # Padding is copied verbatim, including signed zeros and exceptional values.
    out_even = tl.where(rotate, out_even, even)
    out_odd = tl.where(rotate, out_odd, odd)
    # PyTorch's FP64 -> FP16/BF16 cast goes through FP32. Preserve that
    # double-rounding contract inside registers, without a global temporary.
    if OUT.dtype.element_ty == tl.float16 or OUT.dtype.element_ty == tl.bfloat16:
        out_even = out_even.to(tl.float32)
        out_odd = out_odd.to(tl.float32)
    tl.store(OUT + 2 * index, out_even, valid)
    tl.store(OUT + 2 * index + 1, out_odd, valid)


@triton_op("wan_dit_lab::rope3d_rotate", mutates_args={})
def _rotate(x: torch.Tensor, freq: torch.Tensor, grid_key: str,
            start: int, inverse: bool, output_dtype: torch.dtype) -> torch.Tensor:
    output = torch.empty(x.shape, device=x.device, dtype=output_dtype)
    if x.numel():
        # A serialized static grid cannot be auto-lifted into SymInts by AOT.
        grids = [int(value) for value in grid_key.split(",")]
        batch, length, heads, dim = x.shape
        total = batch * length * heads * (dim // 2)
        wrap_triton(_rope_kernel)[(triton.cdiv(total, 256),)](
            x, freq, output, length, heads, dim // 2,
            *x.stride(), *freq.stride(),
            tuple(tuple(grids[b:b + 3]) for b in range(0, len(grids), 3)),
            start, inverse, total, 256, x.dtype == torch.float64 or freq.dtype == torch.float64,
            num_warps=4, enable_fp_fusion=False)
    return output


def _setup_context(ctx, inputs, output):
    x, freq, grid_key, start, inverse, _ = inputs
    ctx.save_for_backward(freq)
    ctx.grid_key, ctx.start, ctx.inverse = grid_key, start, inverse
    ctx.input_dtype = x.dtype


def _backward(ctx, grad_output):
    (freq,) = ctx.saved_tensors
    # The Jacobian of a rotation is its transpose. No input activation is saved.
    grad = _rotate(grad_output, freq, ctx.grid_key, ctx.start, not ctx.inverse, ctx.input_dtype)
    return grad, None, None, None, None, None


_rotate.register_autograd(_backward, setup_context=_setup_context)


def rope_apply(x, grid_sizes, freq, start_frame=0, *, output_dtype=None):
    """Rotate, preserving input dtype by default. CPU grids are host metadata.

    For fullgraph compilation pass Python grids, not data-dependent tensor sizes.
    Supports different grids per sample, padding and arbitrary positive strides.
    """
    # Triton constexpr grids describe a fixed video tile. Torch 2.10 may
    # otherwise correlate their integers with auto-dynamic batch dimensions.
    torch._dynamo.mark_static(x)
    torch._dynamo.mark_static(freq)
    if x.device.type != "cuda" or freq.device != x.device:
        raise ValueError("Input and frequencies must be on the same CUDA device")
    dtypes = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    if x.dtype not in dtypes or freq.dtype not in (torch.float32, torch.float64):
        raise TypeError("Floating input and real FP32/FP64 frequencies required")
    if x.ndim != 4 or x.shape[-1] < 2 or x.shape[-1] % 2:
        raise ValueError("Input must be (batch, token, heads, even dim)")
    if freq.ndim != 3 or freq.shape[0] != 2 or freq.shape[2] != x.shape[-1] // 2:
        raise ValueError("Frequencies must be (2, positions, dim/2)")
    if freq.requires_grad:
        raise ValueError("Trainable frequencies are not supported")
    if isinstance(grid_sizes, torch.Tensor):
        if grid_sizes.device.type != "cpu":
            raise ValueError("Grid sizes must be host metadata, not CUDA tensors")
        grid_sizes = grid_sizes.tolist()
    grids = [list(g) for g in grid_sizes]
    if len(grids) != x.shape[0] or not grids:
        raise ValueError("One grid is required per nonempty batch")
    if not isinstance(start_frame, int) or start_frame < 0:
        raise ValueError("start_frame must be a nonnegative integer")
    for g in grids:
        if len(g) != 3 or any(not isinstance(v, int) for v in g):
            raise ValueError("Each grid must contain three integers")
        f, h, w = g
        if f < 0 or h <= 0 or w <= 0 or f * h * w > x.shape[1]:
            raise ValueError("Grid exceeds sequence length or has invalid dimensions")
        spatial = (x.shape[-1] // 2) // 3
        if start_frame + f > freq.shape[1] or (spatial and max(h, w) > freq.shape[1]):
            raise ValueError("Grid or offset exceeds frequency bounds")
    output_dtype = x.dtype if output_dtype is None else output_dtype
    if output_dtype not in dtypes:
        raise TypeError("Floating output dtype required")
    grid_key = ",".join([str(value) for grid in grids for value in grid])
    return _rotate(x, freq, grid_key, start_frame, False, output_dtype)


def rope_apply_wan(x, grid_sizes, freqs, start_frame=0):
    """Upstream Wan adapter: complex frequencies and FP32 output, without a copy."""
    if freqs.dtype not in (torch.complex64, torch.complex128):
        raise TypeError("Wan adapter requires complex frequencies")
    real_freqs = torch.view_as_real(freqs).permute(2, 0, 1)
    return rope_apply(x, grid_sizes, real_freqs, start_frame, output_dtype=torch.float32)
