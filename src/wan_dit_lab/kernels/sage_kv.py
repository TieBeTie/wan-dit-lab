"""SageAttention 2 (INT8 Q K, FP8 P V, sm89) straight over the FP8 KV cache of causal.KVCache.

sageattn takes keys and values in 16 bits and quantizes them itself on every call: K - mean(K) to INT8 with a scale
per thread group of a block of 64 tokens, V to FP8 with a scale per channel, transposed. Over an FP8 cache this means
reading the cache back into a bfloat16 copy first (KVCache.finished) and quantizing that copy again: 10 ms and 7 ms
of a 79 ms block at 704x1280 (E23). Here the same quantization of Sage reads the cache itself, FP8 times its scale
per token, and the chunk's own keys and values in 16 bits (they attend to themselves unquantized by the cache, as
before), and writes the INT8 keys and FP8 values in the layouts of Sage's kernel, which then runs as sageattn runs
it (sageattn_qk_int8_pv_fp8_cuda, pv_accum_dtype "fp32+fp16"; live::sage_sm89 of native/ops). The scales stay
those of Sage: computed over all the keys and values of the call.

Tokens [0, at) come from the cache, [at, at + n) from the chunk. Layout NHD: [batch, token, head, head_dim].
"""

from typing import Optional

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from wan_dit_lab.kernels import operators

V_MAX = 2.25  # the largest |V| in FP8 for the FP16 accumulation of P V: 448 * 2.25 * 64 < 65504 (Sage, core.py)
BLKQ, WARPQ, BLKK, WARPK = 128, 32, 64, 64  # Sage's per-thread granularity on sm89


@triton.jit
def _load(Cache, Scale, Chunk, b, h, offs, at, L, d,
          scb, scn, sch, ssb, ssn, ssh, snb, snn, snh, Heads, Tokens, HMAP: tl.constexpr, TMAP: tl.constexpr):
    """Tokens offs of head h: FP8 cache times its scale before `at`, the 16-bit chunk from `at` on, 0 past L. With
    HMAP the cache head is Heads[h] (the chunk is already of the chosen heads); with TMAP cache token i is Tokens[i]."""
    in_cache = offs < at
    in_chunk = (offs >= at) & (offs < L)
    hc = tl.load(Heads + h) if HMAP else h
    tok = tl.load(Tokens + offs, mask=in_cache, other=0) if TMAP else offs
    c = tl.load(Cache + b * scb + tok[:, None] * scn + hc * sch + d[None, :], mask=in_cache[:, None], other=0.)
    s = tl.load(Scale + b * ssb + tok * ssn + hc * ssh, mask=in_cache, other=0.)
    x = tl.load(Chunk + b * snb + (offs - at)[:, None] * snn + h * snh + d[None, :], mask=in_chunk[:, None], other=0.)
    return tl.where(in_cache[:, None], c.to(tl.float32) * s[:, None], x.to(tl.float32))


@triton.jit
def _sum_max(Cache, Scale, Chunk, Sum, Max, at, L,
             scb, scn, sch, ssb, ssn, ssh, snb, snn, snh, Heads, Tokens,
             D: tl.constexpr, T: tl.constexpr, SUM: tl.constexpr, HMAP: tl.constexpr, TMAP: tl.constexpr):
    """Over a tile of T tokens of one head: the sum of each channel (SUM) or the largest |x|, added to [B, H, D]."""
    t, h, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    H = tl.num_programs(1)
    d = tl.arange(0, D)
    x = _load(Cache, Scale, Chunk, b, h, t * T + tl.arange(0, T), at, L, d,
              scb, scn, sch, ssb, ssn, ssh, snb, snn, snh, Heads, Tokens, HMAP, TMAP)
    if SUM:
        tl.atomic_add(Sum + (b * H + h) * D + d, tl.sum(x, axis=0))
    else:
        tl.atomic_max(Max + (b * H + h) * D + d, tl.max(tl.abs(x), axis=0))


@triton.jit
def _quant_k(Cache, Scale, Chunk, Mean, Out, OutScale, at, L,
             scb, scn, sch, ssb, ssn, ssh, snb, snn, snh, sob, son, soh, Heads, Tokens,
             D: tl.constexpr, BLK: tl.constexpr, HMAP: tl.constexpr, TMAP: tl.constexpr):
    """Sage's quant_key_per_thread_int8_kernel over the two sources, with the mean of the keys taken off: one scale
    for the tokens 8 i + 2 g and 8 i + 2 g + 1 of a block of BLK, g = 0..3."""
    blk, g = tl.program_id(0) // 4, tl.program_id(0) % 4
    h, b = tl.program_id(1), tl.program_id(2)
    H = tl.num_programs(1)
    d = tl.arange(0, D)
    mean = tl.load(Mean + (b * H + h) * D + d)
    n0 = blk * BLK + tl.arange(0, BLK // 8) * 8 + g * 2
    n1 = n0 + 1
    x0 = _load(Cache, Scale, Chunk, b, h, n0, at, L, d, scb, scn, sch, ssb, ssn, ssh, snb, snn, snh,
               Heads, Tokens, HMAP, TMAP)
    x1 = _load(Cache, Scale, Chunk, b, h, n1, at, L, d, scb, scn, sch, ssb, ssn, ssh, snb, snn, snh,
               Heads, Tokens, HMAP, TMAP)
    x0 = tl.where((n0 < L)[:, None], x0 - mean[None, :], 0.)
    x1 = tl.where((n1 < L)[:, None], x1 - mean[None, :], 0.)
    scale = tl.maximum(tl.max(tl.abs(x0)), tl.max(tl.abs(x1))) / 127. + 0.0000001
    q0 = x0 / scale
    q1 = x1 / scale
    q0 = (q0 + 0.5 * tl.where(q0 >= 0, 1, -1)).to(tl.int8)
    q1 = (q1 + 0.5 * tl.where(q1 >= 0, 1, -1)).to(tl.int8)
    tl.store(Out + b * sob + n0[:, None] * son + h * soh + d[None, :], q0, mask=(n0 < L)[:, None])
    tl.store(Out + b * sob + n1[:, None] * son + h * soh + d[None, :], q1, mask=(n1 < L)[:, None])
    tl.store(OutScale + (b * H + h) * tl.num_programs(0) + blk * 4 + g, scale)


@triton.jit
def _quant_v(Cache, Scale, Chunk, VScale, Out, at, L,
             scb, scn, sch, ssb, ssn, ssh, snb, snn, snh, sob, sod, soh, Heads, Tokens,
             D: tl.constexpr, T: tl.constexpr, HMAP: tl.constexpr, TMAP: tl.constexpr):
    """V / its channel's scale in FP8, transposed to [B, D, H, padded tokens] and, within every 16 tokens, in Sage's
    order [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15] (transpose_pad_permute_cuda); 0 past L."""
    t, h, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    H = tl.num_programs(1)
    d = tl.arange(0, D)
    i = t * T + tl.arange(0, T)
    x = _load(Cache, Scale, Chunk, b, h, i, at, L, d, scb, scn, sch, ssb, ssn, ssh, snb, snn, snh,
              Heads, Tokens, HMAP, TMAP)
    s = tl.load(VScale + (b * H + h) * D + d)
    y = tl.trans((x / s[None, :]).to(tl.float8e4nv))  # [D, T]: the store runs along the tokens, as the output does
    j = i % 16
    pair = j // 2
    to = i - j + (pair % 4) * 4 + (pair // 4) * 2 + j % 2
    tl.store(Out + b * sob + d[:, None] * sod + h * soh + to[None, :], y)


def _strides(x):
    return x.stride(0), x.stride(1), x.stride(2)


def quant_q(Q):
    """Sage's per-thread INT8 quantization of the queries (quant_query_per_thread_int8_kernel on sm89, BLKQ 128,
    WARPQ 32) in torch ops, bit for bit: in a block of 32 tokens, thread lane l of 8 holds the tokens l, l + 8,
    l + 16, l + 24 and they share one scale max|x| / 127 + 1e-7; x / scale is rounded half away from zero. Sage's
    Triton kernel compiled wrong under AOTInductor (q_int8 off by up to 127); this one Inductor compiles itself.
    Returns q [B, n, H, D] INT8 and the scales [B, H, blocks * 8]."""
    B, n, H, D = Q.shape
    blocks = triton.cdiv(n, BLKQ) * (BLKQ // WARPQ)  # of 32 tokens, as the kernel's grid has them
    x = torch.nn.functional.pad(Q.float(), (0, 0, 0, 0, 0, blocks * WARPQ - n)).view(B, blocks, 4, 8, H, D)
    scale = x.abs().amax(dim=(2, 5), keepdim=True) / 127. + 0.0000001  # [B, blocks, 1, 8, H, 1]
    y = x / scale
    q = (y + torch.where(y >= 0, 0.5, -0.5)).trunc().to(torch.int8).view(B, blocks * WARPQ, H, D)[:, :n]
    return q.contiguous(), scale.view(B, blocks, 8, H).permute(0, 3, 1, 2).reshape(B, H, blocks * 8).contiguous()


def attend(Q, cache_k, cache_ks, cache_v, cache_vs, at, K, V, sm_scale=None, heads=None, tokens=None):
    """softmax(Q K^T / sqrt(d)) V over the cached tokens [0, at) (FP8 cache_k, cache_v with scales per token and head
    cache_ks, cache_vs [B, tokens, H, 1]) and the chunk's own K, V [B, n, H, D]; Q [B, n_q, H, D]. Returns
    [B, n_q, H, D] in bfloat16. heads (int32 [H]): head h of Q, K, V is cache head heads[h], the cache keeping all of
    them; tokens (int32 [at]): cached token i is buffer token tokens[i]. Neither copies the cache. One operator for
    torch.compile and torch.export (_attend), whose Triton kernels AOTInductor compiles into its package."""
    return _attend(Q, cache_k, cache_ks, cache_v, cache_vs, K, V, at, Q.shape[-1] ** -0.5 if sm_scale is None else
                   sm_scale, heads, tokens)


@triton_op("wan_dit_lab::sage_kv_attend", mutates_args={})
def _attend(Q: torch.Tensor, cache_k: torch.Tensor, cache_ks: torch.Tensor, cache_v: torch.Tensor,
            cache_vs: torch.Tensor, K: torch.Tensor, V: torch.Tensor, at: int, sm_scale: float,
            heads: Optional[torch.Tensor], tokens: Optional[torch.Tensor]) -> torch.Tensor:
    B, nq, H, D = Q.shape
    n = K.shape[1]
    L = at + n
    dev = Q.device
    hmap, tmap = heads is not None, tokens is not None
    # an unused pointer gets a tensor of its own: a tensor passed for two arguments of a kernel is functionalized by
    # AOTInductor into two copies, and the kernel's writes to one of them are lost
    unused = lambda: torch.empty(1, device=dev, dtype=torch.int32)
    maps = (heads if hmap else unused(), tokens if tmap else unused())
    k_args = (*_strides(cache_k), *_strides(cache_ks), *_strides(K))
    v_args = (*_strides(cache_v), *_strides(cache_vs), *_strides(V))
    T = 64
    tiles = triton.cdiv(L, T)

    k_sum = torch.zeros(B, H, D, device=dev, dtype=torch.float32)
    wrap_triton(_sum_max)[(tiles, H, B)](cache_k, cache_ks, K, k_sum, torch.empty_like(k_sum), at, L, *k_args, *maps,
                                         D=D, T=T, SUM=True, HMAP=hmap, TMAP=tmap)
    k_mean = k_sum / L
    blocks = triton.cdiv(L, BLKK)
    k_int8 = torch.empty(B, L, H, D, device=dev, dtype=torch.int8)
    k_scale = torch.empty(B, H, blocks * (BLKK // WARPK) * 4, device=dev, dtype=torch.float32)
    wrap_triton(_quant_k)[(blocks * 4, H, B)](cache_k, cache_ks, K, k_mean, k_int8, k_scale, at, L, *k_args,
                                              *_strides(k_int8), *maps, D=D, BLK=BLKK, HMAP=hmap, TMAP=tmap)

    v_max = torch.zeros(B, H, D, device=dev, dtype=torch.float32)
    wrap_triton(_sum_max)[(tiles, H, B)](cache_v, cache_vs, V, torch.empty_like(v_max), v_max, at, L, *v_args, *maps,
                                         D=D, T=T, SUM=False, HMAP=hmap, TMAP=tmap)
    v_scale = v_max.clamp(min=1e-8) / V_MAX
    padded = blocks * BLKK
    v_fp8 = torch.empty(B, D, H, padded, device=dev, dtype=torch.float8_e4m3fn)
    wrap_triton(_quant_v)[(padded // T, H, B)](cache_v, cache_vs, V, v_scale, v_fp8, at, L, *v_args,
                                               v_fp8.stride(0), v_fp8.stride(1), v_fp8.stride(2), *maps, D=D, T=T,
                                               HMAP=hmap, TMAP=tmap)
    q_int8, q_scale = quant_q(Q)
    # Sage's kernel (sageattn_qk_int8_pv_fp8_cuda, pv_accum_dtype "fp32+fp16") as live::sage_sm89 of native/ops
    return operators.load().sage_sm89(q_int8, k_int8, v_fp8, q_scale, k_scale, v_scale, sm_scale)
