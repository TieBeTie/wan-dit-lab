"""Wan RMSNorm/RoPE directly into Sage2++ INT8 Q/K, over the live FP8 cache.

Q has no normalized/rotated global temporary. K rotation is recomputed during
the mean and quantization passes, avoiding a full floating-point K temporary.
Only new K are rotated; cached K already contain their original positions.
RMS statistics and rotations use FP32, as the existing compiled live block.
Current K is rounded to BF16 before centering and caching, preserving that ABI.
"""
import hashlib
import inspect

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from torch.library import triton_op, wrap_triton

from wan_dit_lab.kernels import operators, sage_kv
from wan_dit_lab.kernels.cache_fp8 import _fp8_rn, _store_rows


@triton.jit
def _rms(Q, K, QR, KR, N: tl.constexpr, C: tl.constexpr,
         SQ: tl.constexpr, SK: tl.constexpr, EQ: tl.constexpr, EK: tl.constexpr, BC: tl.constexpr):
    row = tl.program_id(0)
    b, t = row // N, row % N
    c = tl.arange(0, BC)
    qs = tl.full((BC,), 0., tl.float32)
    ks = tl.full((BC,), 0., tl.float32)
    # Preserve the live block's FP32 reduction order, including its 1024-wide
    # accumulation tiles. Tiny RMS differences can cross an INT8 rounding tie.
    for offset in range(triton.cdiv(C, BC)):
        index = offset * BC + c
        q = tl.load(Q + b * SQ[0] + t * SQ[1] + index, index < C, other=0).to(tl.float32)
        k = tl.load(K + b * SK[0] + t * SK[1] + index, index < C, other=0).to(tl.float32)
        qs = qs + q * q
        ks = ks + k * k
    tl.store(QR + row, libdevice.rsqrt(tl.sum(qs, 0) / C + EQ))
    tl.store(KR + row, libdevice.rsqrt(tl.sum(ks, 0) / C + EK))


@triton.jit
def _rotate_raw(X, R, W, F, b, h, t, valid, d,
                N: tl.constexpr, D: tl.constexpr, SX: tl.constexpr, SF: tl.constexpr,
                GH: tl.constexpr, GW: tl.constexpr, START: tl.constexpr, BF16: tl.constexpr,
                MATCH: tl.constexpr = False):
    pair = d // 2
    spatial: tl.constexpr = (D // 2) // 3
    temporal: tl.constexpr = D // 2 - 2 * spatial
    position = tl.where(pair[None, :] < temporal, START + t[:, None] // (GH * GW),
                        tl.where(pair[None, :] < temporal + spatial,
                                 t[:, None] // GW % GH, t[:, None] % GW))
    cosine = tl.load(F + position * SF[1] + pair[None, :] * SF[2], valid[:, None], other=1)
    sine = tl.load(F + SF[0] + position * SF[1] + pair[None, :] * SF[2], valid[:, None], other=0)
    base = b * SX[0] + t[:, None] * SX[1] + h * SX[2] + 2 * pair[None, :]
    r = tl.load(R + b * N + t, valid, other=0)
    even = tl.load(X + base, valid[:, None], other=0).to(tl.float32) * r[:, None]
    odd = tl.load(X + base + 1, valid[:, None], other=0).to(tl.float32) * r[:, None]
    if MATCH:  # RMSNorm returns the projection's BF16 before the weight (models/rms_norm.py)
        even = even.to(tl.bfloat16).to(tl.float32)
        odd = odd.to(tl.bfloat16).to(tl.float32)
    even = even * tl.load(W + h * D + 2 * pair)[None, :].to(tl.float32)
    odd = odd * tl.load(W + h * D + 2 * pair + 1)[None, :].to(tl.float32)
    if MATCH:
        # BF16 times the BF16 weight, then LongLive's RoPE kernel: its products as that kernel contracts them,
        # and its BF16 output
        even = even.to(tl.bfloat16).to(tl.float32)
        odd = odd.to(tl.bfloat16).to(tl.float32)
        rotated = tl.where((d % 2)[None, :] == 0,
                           tl.fma(even, cosine, -(odd * sine)), tl.fma(even, sine, odd * cosine))
    else:
        # Match the existing Inductor rotation's fused multiply-add orientation.
        # Letting the compiler contract the other product changes INT8 tie cases.
        rotated = tl.where((d % 2)[None, :] == 0,
                           tl.fma(even, cosine, -(odd * sine)), tl.fma(odd, cosine, even * sine))
    if BF16 or MATCH:
        rotated = rotated.to(tl.float32).to(tl.bfloat16).to(tl.float32)
    return tl.where(valid[:, None], rotated, 0.)


@triton.jit
def _q_rope_int8(Q, QR, W, F, Q8, QS, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                 SQ: tl.constexpr, SF: tl.constexpr, GH: tl.constexpr, GW: tl.constexpr, START: tl.constexpr,
                 MATCH: tl.constexpr, QP=None, POOL: tl.constexpr = False):
    group, h, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    t = (group // 8) * 32 + group % 8 + tl.arange(0, 4) * 8
    d = tl.arange(0, D)
    x = _rotate_raw(Q, QR, W, F, b, h, t, t < N, d, N, D, SQ, SF, GH, GW, START, False, MATCH)
    if POOL:  # block sparsity: the sum of this group's rotated queries (0 past N), before quantization
        tl.store(QP + ((b * H + h) * tl.num_programs(0) + group) * D + d, tl.sum(x, 0))
    scale = tl.max(tl.max(tl.abs(x), 1), 0) / 127. + 1e-7
    y = x / scale
    y = (y + tl.where(y >= 0, .5, -.5)).to(tl.int8)
    tl.store(Q8 + ((b * N + t[:, None]) * H + h) * D + d[None, :], y, (t < N)[:, None])
    tl.store(QS + (b * H + h) * tl.num_programs(0) + group, scale)


@triton.jit
def _keys(CK, KS, K, KR, W, F, b, h, t, d,
          AT: tl.constexpr, N: tl.constexpr, D: tl.constexpr, SC: tl.constexpr, SS: tl.constexpr,
          SK: tl.constexpr, SF: tl.constexpr, GH: tl.constexpr, GW: tl.constexpr, START: tl.constexpr,
          MATCH: tl.constexpr):
    past = t < AT
    current = (t >= AT) & (t < AT + N)
    cached = tl.load(CK + b * SC[0] + t[:, None] * SC[1] + h * SC[2] + d[None, :], past[:, None], other=0.).to(tl.float32)
    scale = tl.load(KS + b * SS[0] + t * SS[1] + h * SS[2], past, other=0)
    new = _rotate_raw(K, KR, W, F, b, h, t - AT, current, d, N, D, SK, SF, GH, GW, START, True, MATCH)
    return tl.where(past[:, None], cached * scale[:, None], new)


@triton.jit
def _k_rope_sum(CK, KS, K, KR, W, F, PARTIAL,
                AT: tl.constexpr, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                SC: tl.constexpr, SS: tl.constexpr, SK: tl.constexpr, SF: tl.constexpr,
                GH: tl.constexpr, GW: tl.constexpr, START: tl.constexpr, MATCH: tl.constexpr):
    tile, h, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    d = tl.arange(0, D)
    x = _keys(CK, KS, K, KR, W, F, b, h, tile * 64 + tl.arange(0, 64), d,
              AT, N, D, SC, SS, SK, SF, GH, GW, START, MATCH)
    tl.store(PARTIAL + ((b * H + h) * tl.num_programs(0) + tile) * D + d, tl.sum(x, 0))


@triton.jit
def _k_rope_int8(CK, KS, K, KR, W, F, MEAN, K8, KScale,
                 AT: tl.constexpr, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                 SC: tl.constexpr, SS: tl.constexpr, SK: tl.constexpr, SF: tl.constexpr,
                 GH: tl.constexpr, GW: tl.constexpr, START: tl.constexpr, MATCH: tl.constexpr):
    block, group = tl.program_id(0) // 4, tl.program_id(0) % 4
    h, b = tl.program_id(1), tl.program_id(2)
    i, d = tl.arange(0, 16), tl.arange(0, D)
    t = block * 64 + (i // 2) * 8 + group * 2 + i % 2
    x = _keys(CK, KS, K, KR, W, F, b, h, t, d, AT, N, D, SC, SS, SK, SF, GH, GW, START, MATCH)
    # Cache the uncentered current keys. Every token has exactly one owner.
    current = (t >= AT) & (t < AT + N)
    row_scale = tl.maximum(tl.max(tl.abs(x), 1), 1e-12) * (1. / 448.)
    packed = _fp8_rn(tl.div_rn(x, row_scale[:, None]))
    tl.store(CK + b * SC[0] + t[:, None] * SC[1] + h * SC[2] + d[None, :], packed, current[:, None])
    tl.store(KS + b * SS[0] + t * SS[1] + h * SS[2], row_scale, current)
    mean = tl.load(MEAN + (b * H + h) * D + d)
    x = tl.where((t < AT + N)[:, None], x - mean[None, :], 0.)
    scale = tl.max(tl.max(tl.abs(x), 1), 0) / 127. + 1e-7
    y = x / scale
    y = (y + tl.where(y >= 0, .5, -.5)).to(tl.int8)
    tl.store(K8 + ((b * (AT + N) + t[:, None]) * H + h) * D + d[None, :], y, (t < AT + N)[:, None])
    tl.store(KScale + (b * H + h) * tl.num_programs(0) + block * 4 + group, scale)


# Hash the transitive kernels so Inductor cannot reuse an older operator body.
_KEY = "".join(kernel.cache_key for kernel in (_rms, _q_rope_int8, _k_rope_sum, _k_rope_int8,
                                               sage_kv._sum_max, sage_kv._quant_v, _store_rows))


def _launch(q, k, v, wq, wk, freq, grid_key, start, ck, ks, cv, vs, at, eps_q, eps_k, match, pool):
    """The kernels of the preparation. pool (block sparsity) adds the sums of the rotated queries per 32-token group
    and of the keys per 64-token tile; without it not one instruction of them is compiled."""
    _, gh, gw = (int(x) for x in grid_key.split(","))
    b, n, h, d = q.shape
    dev, length = q.device, at + n
    tiles, blocks = triton.cdiv(length, 64), triton.cdiv(length, 64)
    qr, kr = [torch.empty(b, n, device=dev, dtype=torch.float32) for _ in range(2)]
    wrap_triton(_rms)[(b * n,)](q, k, qr, kr, n, h * d, q.stride(), k.stride(), eps_q, eps_k,
                                 min(1024, triton.next_power_of_2(h * d)), num_warps=8)
    q8 = torch.empty(b, n, h, d, device=dev, dtype=torch.int8)
    qs = torch.empty(b, h, triton.cdiv(n, 128) * 32, device=dev, dtype=torch.float32)
    if pool:
        qp = torch.empty(b, h, qs.shape[-1], d, device=dev, dtype=torch.float32)
        wrap_triton(_q_rope_int8)[(qs.shape[-1], h, b)](q, qr, wq, freq, q8, qs, n, h, d,
                                                         q.stride(), freq.stride(), gh, gw, start, match, qp, True)
    else:
        wrap_triton(_q_rope_int8)[(qs.shape[-1], h, b)](q, qr, wq, freq, q8, qs, n, h, d,
                                                         q.stride(), freq.stride(), gh, gw, start, match, None, False)
    partial = torch.empty(b, h, tiles, d, device=dev, dtype=torch.float32)
    k_args = (at, n, h, d, ck.stride(), ks.stride(), k.stride(), freq.stride(), gh, gw, start, match)
    wrap_triton(_k_rope_sum)[(tiles, h, b)](ck, ks, k, kr, wk, freq, partial, *k_args)
    mean = partial.sum(2) / length
    k8 = torch.empty(b, length, h, d, device=dev, dtype=torch.int8)
    kscale = torch.empty(b, h, blocks * 4, device=dev, dtype=torch.float32)
    wrap_triton(_k_rope_int8)[(blocks * 4, h, b)](ck, ks, k, kr, wk, freq, mean, k8, kscale, *k_args)
    # V retains the existing Sage2++ channel scales and permutation.
    v_max = torch.zeros(b, h, d, device=dev, dtype=torch.float32)
    unused = lambda: torch.empty(1, device=dev, dtype=torch.int32)
    maps = (unused(), unused())
    v_args = (*sage_kv._strides(cv), *sage_kv._strides(vs), *sage_kv._strides(v))
    wrap_triton(sage_kv._sum_max)[(tiles, h, b)](cv, vs, v, torch.empty_like(v_max), v_max, at, length,
                                                 *v_args, *maps, D=d, T=64, SUM=False, HMAP=False, TMAP=False)
    vscale = v_max.clamp(min=1e-8) / sage_kv.V_MAX
    v8 = torch.empty(b, d, h, blocks * 64, device=dev, dtype=torch.float8_e4m3fn)
    wrap_triton(sage_kv._quant_v)[(blocks, h, b)](cv, vs, v, vscale, v8, at, length, *v_args,
                                                v8.stride(0), v8.stride(1), v8.stride(2), *maps,
                                                D=d, T=64, HMAP=False, TMAP=False)
    wrap_triton(_store_rows)[(triton.cdiv(b * n * h, 4),)](v, cv, vs, v.stride(), n, h, d, cv.shape[1], at,
                                                           b * n * h, d, 4, enable_fp_fusion=False)
    if pool:  # the queries per 128-token block of Sage's kernel (32 groups each), the keys per 64-token block
        return q8, k8, v8, qs, kscale, vscale, qp.view(b, h, -1, 32, d).sum(3), partial
    return q8, k8, v8, qs, kscale, vscale


def _prepare(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor,
             freq: torch.Tensor, grid_key: str, start: int, ck: torch.Tensor, ks: torch.Tensor,
             cv: torch.Tensor, vs: torch.Tensor, at: int, eps_q: float, eps_k: float, match: bool
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _launch(q, k, v, wq, wk, freq, grid_key, start, ck, ks, cv, vs, at, eps_q, eps_k, match, False)


def _prepare_pooled(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor,
                    freq: torch.Tensor, grid_key: str, start: int, ck: torch.Tensor, ks: torch.Tensor,
                    cv: torch.Tensor, vs: torch.Tensor, at: int, eps_q: float, eps_k: float, match: bool
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                               torch.Tensor, torch.Tensor]:
    return _launch(q, k, v, wq, wk, freq, grid_key, start, ck, ks, cv, vs, at, eps_q, eps_k, match, True)


# Host launch options matter too: changing num_warps changes reduction rounding.
_HASH = hashlib.sha256((_KEY + inspect.getsource(_launch)).encode()).hexdigest()[:16]
_prepare = triton_op("wan_dit_lab::sage_rope_prepare_" + _HASH, mutates_args={"ck", "ks", "cv", "vs"})(_prepare)
_prepare_pooled = triton_op("wan_dit_lab::sage_rope_prepare_pooled_" + _HASH,
                            mutates_args={"ck", "ks", "cv", "vs"})(_prepare_pooled)


def prepare(q, k, v, wq, wk, freq, grid, start, ck, ks, cv, vs, at, eps_q, eps_k, longlive_rounding=False,
            pool=False):
    """Inference-only [B,N,H,128] preparation; persistent cache buffers are mutated. longlive_rounding: round where
    the separate path (RMSNorm, LongLive's RoPE kernel, then Sage) rounds to BF16, so the two agree. pool: also
    return the sums of the rotated queries per 128-token block [B, H, blocks, 128] and of the keys, cached and new,
    per 64-token block [B, H, blocks, 128] (block sparsity)."""
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or q.shape[-1] != 128:
        raise ValueError("Sage sm89 requires matching [B,N,H,128] Q/K/V")
    if any(x.stride(-1) != 1 or x.stride(-2) != q.shape[-1] for x in (q, k, v)):
        raise ValueError("Projection heads/channels must be contiguous")
    if len(grid) != 3 or grid[0] * grid[1] * grid[2] != q.shape[1] or min(grid) <= 0:
        raise ValueError("Grid must cover the complete current chunk")
    if start < 0 or start + grid[0] > freq.shape[1] or max(grid[1:]) > freq.shape[1]:
        raise ValueError("RoPE position exceeds the frequency table")
    if at < 0 or at + q.shape[1] > ck.shape[1]:
        raise ValueError("Write exceeds cache bounds")
    return (_prepare_pooled if pool else _prepare)(q, k, v, wq, wk, freq, ",".join(map(str, grid)), start, ck, ks,
                                                    cv, vs, at, eps_q, eps_k, longlive_rounding)


def attend(q, k, v, norm_q, norm_k, grid, freq, start, ck, ks, cv, vs, at, longlive_rounding=False,
           skip_mass=None, key_keep=None):
    """Sage over the cache and the chunk. skip_mass None: dense. A number: block-sparse (kernels/sparse_blocks.py),
    over the key blocks key_keep allows (int32 [blocks of 64 of the cache buffer], nonzero allowed) and the chunk's,
    less the weakest whose estimated shares add up to less than skip_mass."""
    scale = q.shape[-1] ** -.5
    if skip_mass is None:
        q8, k8, v8, qs, kscale, vscale = prepare(q, k, v, norm_q.weight, norm_k.weight, freq, grid, start,
                                                ck, ks, cv, vs, at, norm_q.eps, norm_k.eps, longlive_rounding)
        return operators.load().sage_sm89(q8, k8, v8, qs, kscale, vscale, scale)
    from wan_dit_lab.kernels import sparse_blocks as sparse
    q8, k8, v8, qs, kscale, vscale, q_sums, k_sums = prepare(q, k, v, norm_q.weight, norm_k.weight, freq, grid, start,
                                                             ck, ks, cv, vs, at, norm_q.eps, norm_k.eps,
                                                             longlive_rounding, pool=True)
    n, length = q.shape[1], at + q.shape[1]
    k_tokens = sparse.block_tokens(length, sparse.CTA_K, q.device)
    q_mean = q_sums / sparse.block_tokens(n, sparse.CTA_Q, q.device)[:, None]
    k_mean = k_sums / k_tokens[:, None]
    blocks = torch.arange(k_tokens.shape[0], device=q.device)
    forced = blocks >= at // sparse.CTA_K  # the chunk's blocks, the last of the keys among them
    kept = sparse.keep(q_mean, k_mean, k_tokens, key_keep[:k_tokens.shape[0]] != 0, forced, skip_mass, scale)
    lut, valid = sparse.table(kept)
    return operators.load().sage_sm89_sparse(q8, k8, v8, qs, kscale, vscale, lut, valid, scale)
