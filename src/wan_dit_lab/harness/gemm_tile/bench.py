"""Times Nunchaku's W4A4 GEMM main loop under other tilings (gemm_tile_bench.cu, built by build.py) on the six
layers of a block at 544x960, and forecasts the chunk if the block's GEMM ran that fast. Bench only, random inputs.

    python -m wan_dit_lab.harness.gemm_tile.bench [variant ...]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

# name: (M padded to the 256 rows of a tile, N, K, calls per block); 4080 tokens, dim 3072, feed forward 14336
LAYERS = {
    "qkv": (4096, 9216, 3072, 1), "o_self": (4096, 3072, 3072, 1), "q_cross": (4096, 3072, 3072, 1),
    "o_cross": (4096, 3072, 3072, 1), "ffn1": (4096, 14336, 3072, 1), "ffn2": (4096, 3072, 14336, 1),
}
# variant: (BLOCK_M, BLOCK_N, warps), as gemm_tile_bench.cu instantiates them
VARIANTS = {0: (256, 128, 8), 1: (256, 64, 8), 2: (128, 64, 4), 3: (256, 96, 8)}


def forecast(base, variant, gemm_ms, chunk_ms):
    """The chunk with its GEMM time gemm_ms scaled by the ratio of the block's GEMM time under `variant` to that under
    `base` ({layer: ms per call}, weighted by the calls of LAYERS)."""
    calls = {name: layer[3] for name, layer in LAYERS.items()}
    ratio = sum(variant[n] * calls[n] for n in LAYERS) / sum(base[n] * calls[n] for n in LAYERS)
    gemm = gemm_ms * ratio
    chunk = chunk_ms - gemm_ms + gemm
    return {"gemm_ms": gemm, "chunk_ms": chunk, "gain": chunk_ms / chunk - 1}


def time_layer(op, variant, M, N, K, repeats=50):
    import torch
    act = torch.randint(-128, 127, (M * K // 2,), dtype=torch.int8, device="cuda")
    wgt = torch.randint(-128, 127, (N * K // 2,), dtype=torch.int8, device="cuda")
    ascales = (torch.rand(M * K // 64, device="cuda") * 0.01 + 0.001).bfloat16()
    wscales = (torch.rand(N * K // 64, device="cuda") * 0.01 + 0.001).bfloat16()
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    run = lambda: op(act, wgt, ascales, wscales, out, variant, M, N, K)
    for _ in range(10):
        run()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(repeats):
        run()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats


def main(variants):
    import torch
    torch.ops.load_library(str(ROOT / "ops" / "build" / "gemm_tile_bench" / "gemm_tile_bench.pyd"))
    op = torch.ops.gemm_tile.run
    times = {}
    for v in variants:
        _, bn, _ = VARIANTS[v]
        times[v] = {n: time_layer(op, v, M, N, K) if N % bn == 0 else None for n, (M, N, K, _) in LAYERS.items()}
        print(v, VARIANTS[v], {n: None if t is None else round(t, 3) for n, t in times[v].items()}, flush=True)
    return times


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or list(VARIANTS))
