"""Sage2++ cross-attention: immutable, once-per-prompt text K/V and per-query RMS/INT8.

Two opaque byte tensors preserve the existing cross-package ABI. Each contains
quantized values followed by their FP32 scales; no packing copy is performed.
"""
import hashlib
import inspect

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from torch.library import triton_op, wrap_triton

from wan_dit_lab.kernels import operators, sage_kv
from wan_dit_lab.kernels.cache_fp8 import _fp8_rn


@triton.jit
def _cross_rms(X, R, N: tl.constexpr, C: tl.constexpr, SX: tl.constexpr,
               EPS: tl.constexpr, BC: tl.constexpr):
    row = tl.program_id(0)
    b, t = row // N, row % N
    d = tl.arange(0, BC)
    acc = tl.full((BC,), 0., tl.float32)
    for offset in range(triton.cdiv(C, BC)):
        c = offset * BC + d
        x = tl.load(X + b * SX[0] + t * SX[1] + c, c < C, other=0).to(tl.float32)
        acc = acc + x * x
    tl.store(R + row, libdevice.rsqrt(tl.sum(acc, 0) / C + EPS))


@triton.jit
def _normalized(X, R, W, b, h, t, d, N: tl.constexpr, D: tl.constexpr, SX: tl.constexpr):
    x = tl.load(X + b * SX[0] + t[:, None] * SX[1] + h * SX[2] + d[None, :],
                (t < N)[:, None], other=0).to(tl.float32)
    r = tl.load(R + b * N + t, t < N, other=0)
    w = tl.load(W + h * D + d).to(tl.float32)
    return (x * r[:, None]) * w[None, :]


@triton.jit
def _cross_moments(K, V, R, W, SUM, VSUM, VMIN, VMAX, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                   SK: tl.constexpr, SV: tl.constexpr):
    tile, h, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    t, d = tile * 64 + tl.arange(0, 64), tl.arange(0, D)
    k = _normalized(K, R, W, b, h, t, d, N, D, SK)
    v = tl.load(V + b * SV[0] + t[:, None] * SV[1] + h * SV[2] + d[None, :],
                (t < N)[:, None], other=0).to(tl.float32)
    address = ((b * H + h) * tl.num_programs(0) + tile) * D + d
    tl.store(SUM + address, tl.sum(k, 0))
    tl.store(VSUM + address, tl.sum(v, 0))
    tl.store(VMIN + address, tl.min(tl.where((t < N)[:, None], v, float("inf")), 0))
    tl.store(VMAX + address, tl.max(tl.where((t < N)[:, None], v, -float("inf")), 0))


@triton.jit
def _cross_key_int8(K, R, W, SUM, PACKED, B: tl.constexpr, N: tl.constexpr, H: tl.constexpr,
                    D: tl.constexpr, SK: tl.constexpr, TILES: tl.constexpr, BT: tl.constexpr):
    group, h, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i, d, tile = tl.arange(0, 16), tl.arange(0, D), tl.arange(0, BT)
    partial = tl.load(SUM + ((b * H + h) * TILES + tile[:, None]) * D + d[None, :],
                      (tile < TILES)[:, None], other=0.)
    mean = tl.sum(partial, 0) / N
    t = (group // 4) * 64 + (i // 2) * 8 + (group % 4) * 2 + i % 2
    x = _normalized(K, R, W, b, h, t, d, N, D, SK)
    x = tl.where((t < N)[:, None], x - mean[None, :], 0.)
    scale = tl.max(tl.max(tl.abs(x), 1), 0) / 127. + 1e-7
    y = x / scale
    y = (y + tl.where(y >= 0, .5, -.5)).to(tl.int8)
    out = PACKED.to(tl.pointer_type(tl.int8))
    scales = (PACKED + B * N * H * D).to(tl.pointer_type(tl.float32))
    tl.store(out + ((b * N + t[:, None]) * H + h) * D + d[None, :], y, (t < N)[:, None])
    tl.store(scales + (b * H + h) * (TILES * 4) + group, scale)


@triton.jit
def _cross_value_fp8(V, VSUM, VMIN, VMAX, PACKED, B: tl.constexpr, N: tl.constexpr, H: tl.constexpr,
                     D: tl.constexpr, SV: tl.constexpr, TILES: tl.constexpr, BT: tl.constexpr,
                     SCALE_MAX: tl.constexpr):
    block, h, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    tile, d = tl.arange(0, BT), tl.arange(0, D)
    address = ((b * H + h) * TILES + tile[:, None]) * D + d[None, :]
    partial = tl.load(VSUM + address, (tile < TILES)[:, None], other=0.)
    mean = tl.sum(partial, 0) / N
    minimum = tl.min(tl.load(VMIN + address, (tile < TILES)[:, None], other=float("inf")), 0)
    maximum = tl.max(tl.load(VMAX + address, (tile < TILES)[:, None], other=-float("inf")), 0)
    scale = tl.maximum(tl.maximum(tl.abs(minimum-mean), tl.abs(maximum-mean)), 1e-8) / SCALE_MAX
    t = block * 64 + tl.arange(0, 64)
    x = tl.load(V + b * SV[0] + t[:, None] * SV[1] + h * SV[2] + d[None, :],
                (t < N)[:, None], other=0).to(tl.float32)
    x = tl.where((t < N)[:, None], x - mean[None, :], 0.)
    y = tl.trans(_fp8_rn(tl.div_rn(x, scale[None, :])))
    i = t % 16
    to = t - i + ((i // 2) % 4) * 4 + ((i // 2) // 4) * 2 + i % 2
    out = PACKED.to(tl.pointer_type(tl.float8e4nv))
    tl.store(out + ((b * D + d[:, None]) * H + h) * (TILES * 64) + to[None, :], y)
    if block == 0:
        scales = (PACKED + B * D * H * (TILES * 64)).to(tl.pointer_type(tl.float32))
        tl.store(scales + (b * H + h) * D + d, scale)
        tl.store(scales + B * H * D + (b * H + h) * D + d, mean)


@triton.jit
def _cross_query_int8(Q, R, W, Q8, QS, N: tl.constexpr, H: tl.constexpr,
                      D: tl.constexpr, SQ: tl.constexpr):
    group, h, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    t = (group // 8) * 32 + group % 8 + tl.arange(0, 4) * 8
    d = tl.arange(0, D)
    x = _normalized(Q, R, W, b, h, t, d, N, D, SQ)
    scale = tl.max(tl.max(tl.abs(x), 1), 0) / 127. + 1e-7
    y = x / scale
    y = (y + tl.where(y >= 0, .5, -.5)).to(tl.int8)
    tl.store(Q8 + ((b * N + t[:, None]) * H + h) * D + d[None, :], y, (t < N)[:, None])
    tl.store(QS + (b * H + h) * tl.num_programs(0) + group, scale)


def _prepare_kv(k: torch.Tensor, v: torch.Tensor, weight: torch.Tensor, eps: float
                ) -> tuple[torch.Tensor, torch.Tensor]:
    b, n, h, d = k.shape
    tiles = triton.cdiv(n, 64)
    r = torch.empty(b, n, device=k.device, dtype=torch.float32)
    wrap_triton(_cross_rms)[(b * n,)](k, r, n, h*d, k.stride(), eps,
                                      min(1024, triton.next_power_of_2(h*d)), num_warps=8)
    sums, v_sums, minima, maxima = [torch.empty(b, h, tiles, d, device=k.device, dtype=torch.float32) for _ in range(4)]
    wrap_triton(_cross_moments)[(tiles, h, b)](k, v, r, weight, sums, v_sums, minima, maxima,
                                             n, h, d, k.stride(), v.stride())
    key = torch.empty(b*n*h*d + b*h*tiles*4*4, device=k.device, dtype=torch.uint8)
    value = torch.empty(b*d*h*tiles*64 + b*h*d*8, device=k.device, dtype=torch.uint8)
    wrap_triton(_cross_key_int8)[(tiles*4, h, b)](k, r, weight, sums, key, b, n, h, d,
                                                k.stride(), tiles, triton.next_power_of_2(tiles))
    wrap_triton(_cross_value_fp8)[(tiles, h, b)](v, v_sums, minima, maxima, value, b, n, h, d, v.stride(),
                                               tiles, triton.next_power_of_2(tiles), sage_kv.V_MAX)
    return key, value


def _prepare_query(q: torch.Tensor, weight: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    b, n, h, d = q.shape
    r = torch.empty(b, n, device=q.device, dtype=torch.float32)
    wrap_triton(_cross_rms)[(b*n,)](q, r, n, h*d, q.stride(), eps,
                                   min(1024, triton.next_power_of_2(h*d)), num_warps=8)
    q8 = torch.empty(q.shape, device=q.device, dtype=torch.int8)
    qs = torch.empty(b, h, triton.cdiv(n, 128)*32, device=q.device, dtype=torch.float32)
    wrap_triton(_cross_query_int8)[(qs.shape[-1], h, b)](q, r, weight, q8, qs, n, h, d, q.stride())
    return q8, qs


_KEY = "".join(kernel.cache_key for kernel in (_cross_rms, _cross_moments, _cross_key_int8,
                                               _cross_value_fp8, _cross_query_int8)) + str(sage_kv.V_MAX)
def _register(name, function):
    suffix = hashlib.sha256((_KEY + inspect.getsource(function)).encode()).hexdigest()[:16]
    return triton_op("wan_dit_lab::" + name + "_" + suffix, mutates_args={})(function)
_prepare_kv = _register("sage_cross_kv", _prepare_kv)
_prepare_query = _register("sage_cross_query", _prepare_query)


def prepare_kv(k, v, weight, eps):
    if k.shape != v.shape or k.ndim != 4 or k.shape[-1] != 128 or k.shape[1] < 1:
        raise ValueError("Sage cross requires matching nonempty [B,N,H,128] text K/V")
    return _prepare_kv(k, v, weight, eps)


def unpack(key, value, batch, tokens, heads, dim):
    """Views into the two immutable prompt buffers; no generation-time copy."""
    tiles = triton.cdiv(tokens, 64)
    kb, vb = batch*tokens*heads*dim, batch*dim*heads*tiles*64
    k8 = key[:kb].view(torch.int8).view(batch, tokens, heads, dim)
    ks = key[kb:].view(torch.float32).view(batch, heads, tiles*4)
    v8 = value[:vb].view(torch.float8_e4m3fn).view(batch, dim, heads, tiles*64)
    vs = value[vb:vb+batch*heads*dim*4].view(torch.float32).view(batch, heads, dim)
    return k8, v8, ks, vs


def _value_mean(value, batch, tokens, heads, dim):
    start = batch*dim*heads*triton.cdiv(tokens, 64)*64 + batch*heads*dim*4
    return value[start:].view(torch.float32).view(batch, 1, heads, dim)


def attend(q, weight, eps, key, value, tokens):
    b, _, h, d = q.shape
    if d != 128:
        raise ValueError("Sage sm89 cross requires head_dim=128")
    q8, qs = _prepare_query(q, weight, eps)
    k8, v8, ks, vs = unpack(key, value, b, tokens, h, d)
    centered = operators.load().sage_sm89(q8, k8, v8, qs, ks, vs, d ** -.5)
    return (centered.float() + _value_mean(value, b, tokens, h, d)).to(q.dtype)


def dequantize(key, value, batch, tokens, heads, dim):
    """Export/check-only numerical view of the opaque ABI, never in generation."""
    k8, v8, ks, vs = unpack(key, value, batch, tokens, heads, dim)
    t = torch.arange(tokens, device=key.device)
    groups = (t // 64)*4 + (t % 8)//2
    k = k8.float() * ks[:, :, groups].transpose(1, 2).unsqueeze(-1)
    i = t % 16
    permuted = t-i + ((i//2)%4)*4 + ((i//2)//4)*2 + i%2
    v = (v8.float()*vs[:, :, :, None].transpose(1, 2))[:, :, :, permuted].permute(0, 3, 2, 1)
    return k, v + _value_mean(value, batch, tokens, heads, dim)
