"""Measurements for docs/report-2026-11-16-bottlenecks.md: peaks of the card, the launches of a block package per
window fill, the W4A4 layers and the Sage kernel at scaled sizes. Measurement only: no kernel or config is changed.

    .venv\\Scripts\\python.exe -m wan_dit_lab.harness.bottleneck_bench <package folder> <out.json> [peaks|block|scaling]...

Every timing is taken after a burn-in that holds the card at its sustained, power-capped clock, and the SM clock and
power seen by nvidia-smi during a section are stored with it (the card's maximum clock is 3105 MHz, a short burst
runs near it, a long run does not).
"""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from wan_dit_lab.harness import block_profile as bp
from wan_dit_lab.harness import bottleneck as bn

SMI = "utilization.gpu,power.draw,clocks.sm,clocks.mem,temperature.gpu,clocks_event_reasons.sw_power_cap"


class Smi:
    """nvidia-smi samples while a section runs."""

    def __enter__(self):
        self.rows, self.stop = [], threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def run(self):
        while not self.stop.is_set():
            out = subprocess.run(["nvidia-smi", f"--query-gpu={SMI}", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True).stdout.strip()
            self.rows.append([c.strip() for c in out.split(",")])

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()

    def summary(self):
        header = ["utilization.gpu", "power.draw", "clocks.sm", "clocks.mem", "temperature.gpu"]
        rows = [r for r in self.rows if len(r) == len(header) + 1]
        cap = sum(r[5] == "Active" for r in rows)
        s = bn.smi_summary(header, [r[:5] for r in rows], busy=50)
        s["power_cap_active_share"] = cap / max(len(rows), 1)
        return s


def burn(fn, seconds):
    import torch
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        fn()
    torch.cuda.synchronize()


def timed(fn, runs=20):
    import torch
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(runs):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / runs


def launches(fn, runs=5):
    """[(name, mean ms)] of the kernels of one call of fn in launch order, memsets and copies left out. Every call is
    profiled in a session of its own (several calls in one session interleave their events); calls whose launch
    sequence differs from the most common one are dropped."""
    import torch
    from collections import Counter
    from torch.profiler import ProfilerActivity, profile
    seqs = []
    for _ in range(runs):
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            fn()
            torch.cuda.synchronize()
        ev = [e for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA
              and not e.name.startswith(("Memset", "Memcpy"))]
        ev.sort(key=lambda e: e.time_range.start)
        seqs.append([(e.name, e.time_range.elapsed_us() / 1000) for e in ev])
    common = Counter(tuple(n for n, _ in s) for s in seqs).most_common(1)[0][0]
    kept = [s for s in seqs if tuple(n for n, _ in s) == common]
    return [(common[i], sum(s[i][1] for s in kept) / len(kept)) for i in range(len(common))]


def peaks():
    import torch
    out = {}
    with Smi() as smi:
        a = torch.empty(512 * 2**20, dtype=torch.uint8, device="cuda")
        b = torch.empty_like(a)
        copy = lambda: b.copy_(a)
        burn(copy, 1.0)
        out["copy_GBps"] = 2 * a.numel() / (timed(copy, 20) / 1000) / 1e9  # read and write
        del a, b
        for name, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
            x = torch.randn(4080, 3072, device="cuda", dtype=dtype)
            w = torch.randn(14336, 3072, device="cuda", dtype=dtype)
            mm = lambda: x @ w.t()
            burn(mm, 3.0)
            out[f"{name}_matmul_TFLOPS"] = 2 * 4080 * 3072 * 14336 / (timed(mm, 20) / 1000) / 1e12
        x8 = torch.randint(-100, 100, (4096, 3072), device="cuda", dtype=torch.int8)
        w8 = torch.randint(-100, 100, (14336, 3072), device="cuda", dtype=torch.int8).t()  # column-major, as cuBLASLt needs
        mm8 = lambda: torch._int_mm(x8, w8)
        burn(mm8, 3.0)
        out["int8_mm_TOPS"] = 2 * 4096 * 3072 * 14336 / (timed(mm8, 20) / 1000) / 1e12
        a8 = torch.randn(4096, 3072, device="cuda").to(torch.float8_e4m3fn)
        b8 = torch.randn(14336, 3072, device="cuda").to(torch.float8_e4m3fn).t()
        one = torch.tensor(1.0, device="cuda")
        mmf = lambda: torch._scaled_mm(a8, b8, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
        burn(mmf, 3.0)
        out["fp8_scaled_mm_TFLOPS"] = 2 * 4096 * 3072 * 14336 / (timed(mmf, 20) / 1000) / 1e12
    out["smi"] = smi.summary()
    return out


def w4a4_inputs(k, n, rank=32, dev="cuda"):
    import torch
    qw = torch.randint(0, 255, (n, k // 2), device=dev, dtype=torch.uint8)
    ws = (torch.rand(k // bn.GROUP, n, device=dev) * 0.01 + 0.001).to(torch.bfloat16)
    down = (torch.randn(k, rank, device=dev) * 0.02).to(torch.bfloat16)
    up = (torch.randn(n, rank, device=dev) * 0.02).to(torch.bfloat16)
    smooth = torch.ones(k, device=dev, dtype=torch.bfloat16)
    bias = torch.zeros(n, device=dev, dtype=torch.bfloat16)
    return qw, ws, down, up, smooth, bias


def gemm_ms(tokens, k, n, rank=32):
    """ms of the W4A4 GEMM launch and of the quantize launch of live::w4a4_linear at this shape."""
    import torch
    from wan_dit_lab.kernels import operators
    operators.load()
    x = torch.randn(tokens, k, device="cuda", dtype=torch.bfloat16)
    qw, ws, down, up, smooth, bias = w4a4_inputs(k, n, rank)
    fn = lambda: torch.ops.live.w4a4_linear(x, qw, ws, down, up, smooth, bias)
    burn(fn, 2.0)
    kernels = launches(fn, 10)
    gemm = sum(ms for name, ms in kernels if "gemm_w4a4_kernel" in name)
    quant = sum(ms for name, ms in kernels if "quantize_w4a4" in name)
    return gemm, quant


def sage_core_ms(tokens, keys, heads=24, head_dim=128):
    import torch
    from wan_dit_lab.kernels import operators
    ops = operators.load()
    q = torch.randint(-100, 100, (1, tokens, heads, head_dim), device="cuda", dtype=torch.int8)
    k = torch.randint(-100, 100, (1, keys, heads, head_dim), device="cuda", dtype=torch.int8)
    padded = (keys + 63) // 64 * 64
    v = torch.randn(1, head_dim, heads, padded, device="cuda").to(torch.float8_e4m3fn)
    qs = torch.rand(1, heads, (tokens + 127) // 128 * 4 * 8, device="cuda") * 0.01
    ks = torch.rand(1, heads, (keys + 63) // 64 * 4, device="cuda") * 0.01
    vs = torch.rand(1, heads, head_dim, device="cuda") * 0.01
    fn = lambda: ops.sage_sm89(q, k, v, qs, ks, vs, head_dim ** -0.5)
    burn(fn, 2.0)
    return timed(fn, 20)


def block(package):
    from wan_dit_lab.kernels import operators
    operators.load()
    package = Path(package)
    result = {}
    for step in range(4):
        m, inputs = bp.block_inputs(package, step)
        blk = bp.load(package, f"block_{step}")
        call = lambda: blk(*inputs)
        with Smi() as smi:
            burn(call, 3.0)
            wall = timed(call, 30)
            seq = launches(call, 9)
        result[f"block_{step}"] = {"wall_ms": wall, "launches": seq, "parts": bp.parts(seq),
                                   "smi": smi.summary(), "x_dtype": str(inputs[1].dtype),
                                   "x_shape": list(inputs[1].shape)}
    return result


def delivery(package):
    """Host to device copy of one weight row (pinned RAM to VRAM) alone, and the block with a second stream copying
    rows without a pause (the worst case of the streamed weights): the block's wall time with and without."""
    import torch
    from wan_dit_lab.kernels import operators
    operators.load()
    package = Path(package)
    m, inputs = bp.block_inputs(package, 3)
    row_bytes = m["row_bytes"]
    host = torch.empty(row_bytes, dtype=torch.uint8).pin_memory()
    dev = torch.empty(row_bytes, dtype=torch.uint8, device="cuda")
    out = {"row_bytes": row_bytes}
    h2d = lambda: dev.copy_(host, non_blocking=True)
    burn(h2d, 1.0)
    ms = timed(h2d, 20)
    out["row_copy_ms"], out["row_copy_GBps"] = ms, row_bytes / (ms / 1000) / 1e9
    blk = bp.load(package, "block_3")
    call = lambda: blk(*inputs)
    burn(call, 3.0)
    out["block_alone_ms"] = timed(call, 30)
    side = torch.cuda.Stream()
    stop = threading.Event()
    copies = [0]

    def pump():
        with torch.cuda.stream(side):
            while not stop.is_set():
                dev.copy_(host, non_blocking=True)
                copies[0] += 1
                side.synchronize()

    thread = threading.Thread(target=pump, daemon=True)
    thread.start()
    time.sleep(0.5)
    burn(call, 1.0)
    copies[0] = 0
    start = time.perf_counter()
    out["block_with_copies_ms"] = timed(call, 60)
    elapsed = time.perf_counter() - start
    out["copies_during"] = copies[0]
    out["copy_GBps_during_block"] = copies[0] * row_bytes / elapsed / 1e9
    stop.set()
    thread.join()
    return out


def scaling():
    out = {"gemm": {}, "sage_tokens": {}, "sage_keys": {}}
    shapes = {"3072x9216 (QKV)": (3072, 9216), "3072x3072 (O, cross Q, cross O)": (3072, 3072),
              "3072x14336 (FFN 1)": (3072, 14336), "14336x3072 (FFN 2)": (14336, 3072)}
    for name, (k, n) in shapes.items():
        out["gemm"][name] = {str(t): gemm_ms(t, k, n) for t in (2040, 4080, 8160)}
    for t in (2040, 4080, 8160):
        out["sage_tokens"][str(t)] = sage_core_ms(t, 16320)
    for keys in (4080, 8160, 12240, 16320):
        out["sage_keys"][str(keys)] = sage_core_ms(4080, keys)
    return out


def main():
    package, dest = sys.argv[1], Path(sys.argv[2])
    which = sys.argv[3:] or ["peaks", "block", "scaling"]
    result = json.loads(dest.read_text()) if dest.exists() else {}
    for name in which:
        result[name] = {"peaks": peaks, "block": lambda: block(package), "scaling": scaling,
                          "delivery": lambda: delivery(package)}[name]()
        dest.write_text(json.dumps(result, indent=1))
        print(name, "done", flush=True)


if __name__ == "__main__":
    main()
