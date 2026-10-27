"""One launch: row quantization of K/V straight into persistent FP8 cache.

The original K/V remain available for current-chunk attention. No quantized
temporary or staging allocation is needed. This is an inference-only store.
"""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.jit
def _fp8_rn(x):
    # Direct FP32 -> E4M3 on SM89+, without Triton 3.4's FP16 intermediate.
    bits = tl.inline_asm_elementwise(
        "{ .reg .b16 p; cvt.rn.satfinite.e4m3x2.f32 p, 0f00000000, $1; cvt.u32.u16 $0, p; }",
        constraints="=r,f", args=[x], dtype=tl.uint32, is_pure=True, pack=1)
    return bits.to(tl.uint8).to(tl.float8e4nv, bitcast=True)


@triton.jit
def _store_rows(X, CACHE, SCALE, STRIDES: tl.constexpr,
                N: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                CAPACITY: tl.constexpr, AT: tl.constexpr,
                ROWS: tl.constexpr, BD: tl.constexpr, BR: tl.constexpr):
    row = tl.program_id(0) * BR + tl.arange(0, BR)
    dim = tl.arange(0, BD)
    head = row % H
    token = row // H % N
    batch = row // (H * N)
    address = (batch * STRIDES[0] + token * STRIDES[1] + head * STRIDES[2])[:, None]
    x = tl.load(X + address + dim[None, :] * STRIDES[3],
                (row[:, None] < ROWS) & (dim[None, :] < D), other=0).to(tl.float32)
    maximum = tl.maximum(tl.max(tl.abs(x), axis=1), 1.e-12)
    scale = maximum * (1. / 448.)
    quantized = _fp8_rn(tl.div_rn(x, scale[:, None]))
    target = (batch * CAPACITY + AT + token) * H + head
    tl.store(CACHE + target[:, None] * D + dim[None, :], quantized,
             (row[:, None] < ROWS) & (dim[None, :] < D))
    tl.store(SCALE + target, scale, row < ROWS)


@triton.jit
def _store_kernel(K, V, CK, KS, CV, VS,
                  KSTRIDES: tl.constexpr, VSTRIDES: tl.constexpr,
                  N: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                  CAPACITY: tl.constexpr, AT: tl.constexpr,
                  ROWS: tl.constexpr, BD: tl.constexpr, BR: tl.constexpr):
    # Separate branches preserve mixed K/V dtypes without pointer casts.
    if tl.program_id(1) == 0:
        _store_rows(K, CK, KS, KSTRIDES, N, H, D, CAPACITY, AT, ROWS, BD, BR)
    else:
        _store_rows(V, CV, VS, VSTRIDES, N, H, D, CAPACITY, AT, ROWS, BD, BR)


# Inductor 2.8 may reuse a custom operator's graph after its implementation changes.
# The transitive JIT key covers the helpers too, without clearing unrelated caches.
_OP_NAME = "wan_dit_lab::store_kv_fp8_" + _store_kernel.cache_key[:16]


@triton_op(_OP_NAME, mutates_args={"cache_k", "cache_ks", "cache_v", "cache_vs"})
def _store(k: torch.Tensor, v: torch.Tensor, cache_k: torch.Tensor,
           cache_ks: torch.Tensor, cache_v: torch.Tensor, cache_vs: torch.Tensor,
           at: int) -> None:
    b, n, h, d = k.shape
    if k.numel():
        wrap_triton(_store_kernel)[(triton.cdiv(b * n * h, 4), 2)](
            k, v, cache_k, cache_ks, cache_v, cache_vs,
            k.stride(), v.stride(), n, h, d, cache_k.shape[1], at,
            b * n * h, triton.next_power_of_2(d), 4,
            num_warps=4, enable_fp_fusion=False)


def store_kv_fp8(k, v, cache_k, cache_ks, cache_v, cache_vs, at):
    """Store [B,N,H,D] in cache[:, at:at+N], using original quantize_rows math.

    Inputs may be strided. Cache buffers are contiguous and mutually disjoint
    from inputs. As with KVCache.store, callers own those persistent buffers.
    """
    tensors = (k, v, cache_k, cache_ks, cache_v, cache_vs)
    if k.ndim != 4 or v.shape != k.shape or k.shape[-1] < 1:
        raise ValueError("K and V must have matching [B,N,H,D] shapes")
    if k.device.type != "cuda" or any(t.device != k.device for t in tensors):
        raise ValueError("All buffers must be on the same CUDA device")
    if any(t.requires_grad for t in tensors):
        raise ValueError("Cache store is inference-only")
    if k.dtype not in (torch.float16, torch.bfloat16, torch.float32) or v.dtype not in (
            torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("K and V must be FP16, BF16 or FP32")
    if cache_k.dtype != torch.float8_e4m3fn or cache_v.dtype != cache_k.dtype:
        raise TypeError("Cache values must be FP8 E4M3FN")
    if cache_ks.dtype != torch.float32 or cache_vs.dtype != torch.float32:
        raise TypeError("Cache scales must be FP32")
    b, n, h, d = k.shape
    if cache_k.ndim != 4 or cache_k.shape != cache_v.shape or (
            cache_k.shape[0], *cache_k.shape[2:]) != (b, h, d):
        raise ValueError("Cache dimensions must match inputs")
    if cache_ks.shape != (*cache_k.shape[:3], 1) or cache_vs.shape != cache_ks.shape:
        raise ValueError("One scale is required per cache row")
    if any(not t.is_contiguous() for t in tensors[2:]):
        raise ValueError("Cache buffers must be contiguous")
    if not isinstance(at, int) or at < 0 or at + n > cache_k.shape[1]:
        raise ValueError("Write exceeds cache bounds")
    _store(k, v, cache_k, cache_ks, cache_v, cache_vs, at)


def install_cache_store():
    """Opt in within this process only. Return the original method for restoration."""
    from wan_dit_lab.generation.causal import KVCache

    original = KVCache.store

    def store(cache, at, k, v):
        if cache.fp8:
            store_kv_fp8(k, v, cache.k, cache.ks, cache.v, cache.vs, at)
        else:
            original(cache, at, k, v)

    KVCache.store = store
    return original
