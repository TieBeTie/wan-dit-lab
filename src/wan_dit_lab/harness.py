import json
import statistics
import subprocess
from pathlib import Path

import torch

from wan_dit_lab.cli import environment

LIMIT_MEM_ALLOC_FRAC = 0.95
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks" / "results.jsonl"
GPU_FIELDS = ["name", "memory.used", "memory.total", "utilization.gpu", "temperature.gpu"]


def limit_memory(fraction=LIMIT_MEM_ALLOC_FRAC):
    torch.cuda.set_per_process_memory_fraction(fraction)


def bench_events(f, *arg, warmup, number, repeat):
    for _ in range(warmup):
        f(*arg)
    
    ts = []
    for _ in range(number):
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
        start.record()
        for _ in range(repeat):
            f(*arg)
        end.record()
        torch.cuda.synchronize()
        ts.append(start.elapsed_time(end) / repeat)
    return ts


def peak_memory(f, *arg):
    torch.cuda.reset_peak_memory_stats()
    f(*arg)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


def summarize(ts):
    q = statistics.quantiles(ts, n=100)
    return {"median_ms": statistics.median(ts), "p50_ms": q[49], "p95_ms": q[94], "p99_ms": q[98]}


def describe_inputs(*arg):
    return [
        {"shape": list(a.shape), "dtype": str(a.dtype)} if isinstance(a, torch.Tensor) else repr(a)
        for a in arg
    ]


def git_commit():
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def gpu_state(device=0):
    out = subprocess.run(
        ["nvidia-smi", f"--id={device}", f"--query-gpu={','.join(GPU_FIELDS)}",
         "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    )
    return dict(zip(GPU_FIELDS, out.stdout.strip().split(", ")))


def append_row(row, path=RESULTS):
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(label, f, *arg, warmup, number, repeat, device=0):
    limit_memory()
    gpu = gpu_state(device)  # before the measurement: shows what else is on the card
    ts = bench_events(f, *arg, warmup=warmup, number=number, repeat=repeat)
    row = {
        "label": label,
        "function": getattr(f, "__name__", type(f).__name__),
        "inputs": describe_inputs(*arg),
        "warmup": warmup,
        "number": number,
        "repeat": repeat,
        **summarize(ts),
        "peak_allocated_bytes": peak_memory(f, *arg),
        "memory_fraction": LIMIT_MEM_ALLOC_FRAC,
        "commit": git_commit(),
        "gpu": gpu,
        "environment": environment(),
    }
    append_row(row)
    return row


def check_memory_limit(max_gib=12):
    """Allocate 1 GiB chunks until OOM; True if the limit stopped the process."""
    limit_memory()
    chunks = []
    try:
        for _ in range(max_gib):
            chunks.append(torch.empty(2**28, device="cuda"))  # 2**28 float32 = 1 GiB
    except torch.cuda.OutOfMemoryError:
        print(f"OOM at chunk {len(chunks) + 1}: limit works")
        return True
    else:
        print(f"no OOM up to {max_gib} GiB: limit does NOT work")
        return False
    finally:
        del chunks
        torch.cuda.empty_cache()


if __name__ == "__main__":
    check_memory_limit()
    for n in (512, 1024, 2048, 4096):
        x = torch.randn(n, n, device="cuda")
        w = torch.randn(n, n, device="cuda")
        row = run(f"matmul_{n}", torch.matmul, x, w, warmup=50, number=200, repeat=5)
        print(row["label"], {k: round(v, 4) for k, v in row.items() if k.endswith("_ms")},
              row["peak_allocated_bytes"] // 2**20, "MiB")
