"""Arithmetic, traffic and verdicts behind docs/report-2026-11-16-bottlenecks.md (measurements only, no kernel here).

The cost model counts the bytes a launch must move at least once (weights, activations, scales, outputs) and the
multiply-adds it performs; the achieved GB/s and ops/s are those divided by a measured time, the shares are against
peaks measured on the same card (harness/bottleneck_bench.py).
"""

import math
from dataclasses import dataclass

GROUP = 64  # input channels per INT4 scale
PAD = 256  # rows of the activation, padded to a multiple, as Nunchaku does

GEMM_LAYERS = ("self QKV", "self O", "cross Q", "cross O", "FFN 1 (+GELU, quantize of FFN 2)", "FFN 2")


@dataclass(frozen=True)
class Cost:
    read_bytes: int = 0
    write_bytes: int = 0
    int4_macs: int = 0
    int8_macs: int = 0
    lora_macs: int = 0  # 16-bit multiply-adds of the low-rank branch

    @property
    def bytes(self):
        return self.read_bytes + self.write_bytes


def padded(m):
    return (m + PAD - 1) // PAD * PAD


def name_gemms(ms):
    """{layer: ms} from the GEMM launches of a block in launch order (qkv, o, cross q, cross o, ffn 1, ffn 2)."""
    if len(ms) != len(GEMM_LAYERS):
        raise ValueError(f"a block launches {len(GEMM_LAYERS)} W4A4 GEMMs, got {len(ms)}")
    return dict(zip(GEMM_LAYERS, ms))


def w4a4_gemm(tokens, k, n, rank=32, next_layer_epilogue=False):
    """One gemm_w4a4 launch of [tokens, k] x [n, k]: INT4 weights with a scale per 64 channels, a low-rank branch.
    next_layer_epilogue: the output leaves as INT4 with scales and the low-rank down projection of the next layer
    (the first layer of the gelu MLP)."""
    mp = padded(tokens)
    read = (mp * k // 2 + n * k // 2 + n * k // GROUP * 2 + k // GROUP * mp * 2 + mp * rank * 4 + n * rank * 2)
    if next_layer_epilogue:
        write = mp * n // 2 + n // GROUP * mp * 2 + mp * rank * 4
    else:
        write = tokens * n * 2
    return Cost(read, write, int4_macs=tokens * k * n, lora_macs=tokens * rank * n)


def w4a4_quantize(tokens, k, rank=32):
    """One quantize_w4a4_fuse_lora launch: bfloat16 to INT4 with scales, and the low-rank down projection."""
    mp = padded(tokens)
    return Cost(tokens * k * 2, mp * k // 2 + k // GROUP * mp * 2 + mp * rank * 4, lora_macs=tokens * k * rank)


def sage(tokens, keys, heads, head_dim):
    """The kernel of Sage 2 (INT8 Q K^T, FP8 P V) over `keys` keys: Q and K INT8, V FP8, output bfloat16. Bytes are
    the compulsory traffic; the kernel re-reads K and V once per block of queries, mostly from the L2 cache."""
    d = heads * head_dim
    return Cost(tokens * d + keys * d * 2, tokens * d * 2, int8_macs=tokens * keys * d * 2)


def verdict(bw_share, ops_share, ms, launch_ms=0.05):
    """compute bound, bandwidth bound, launch overhead or neither, from the shares of the measured peaks."""
    if ms < launch_ms:
        return "launch overhead"
    top = max(bw_share, ops_share)
    if top < 0.35:
        return f"neither near a peak (bandwidth {bw_share:.0%}, arithmetic {ops_share:.0%})"
    return "compute bound" if ops_share >= bw_share else "bandwidth bound"


def exponent(xs, ts):
    """Slope of log t against log x (least squares): 0 constant, 1 linear, 2 quadratic."""
    lx, lt = [math.log(x) for x in xs], [math.log(t) for t in ts]
    mx, mt = sum(lx) / len(lx), sum(lt) / len(lt)
    return sum((a - mx) * (b - mt) for a, b in zip(lx, lt)) / sum((a - mx) ** 2 for a in lx)


def scaling_class(e):
    if abs(e) < 0.1:
        return "constant"
    if abs(e - 1) < 0.15:
        return "linear"
    if abs(e - 2) < 0.2:
        return "quadratic"
    return f"between linear and quadratic ({e:.2f})"


def smi_summary(header, rows, busy=90):
    """{column: (min, mean, max), "samples": n} of the samples of nvidia-smi --format=csv at utilization.gpu >= busy;
    a column is named without its unit."""
    names = [h.split("[")[0].strip() for h in header]
    values = [[_number(c) for c in row] for row in rows]
    util = names.index("utilization.gpu")
    kept = [row for row in values if row[util] is not None and row[util] >= busy]
    out = {"samples": len(kept)}
    for i, name in enumerate(names):
        col = [row[i] for row in kept if row[i] is not None]
        if col:
            out[name] = (min(col), sum(col) / len(col), max(col))
    return out


def _number(cell):
    try:
        return float(cell.strip().split()[0])
    except (ValueError, IndexError):
        return None
