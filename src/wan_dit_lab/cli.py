"""Environment diagnostics and an explicitly synthetic GPU baseline."""

import argparse
import importlib.metadata
import json
import platform
import statistics
import time

import torch


def environment():
    packages = {}
    for name in ("torch", "triton", "triton-windows", "safetensors"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if info["cuda_available"]:
        device = torch.cuda.get_device_properties(0)
        info["gpu"] = device.name
        info["vram_bytes"] = device.total_memory
        info["compute_capability"] = [device.major, device.minor]
    return info


def benchmark():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for bench-smoke")
    torch.manual_seed(0)
    x = torch.randn((256, 512), device="cuda", dtype=torch.float16)
    w = torch.randn((512, 1024), device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        for _ in range(10):
            torch.mm(x, w)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(20):
            start.record()
            for _ in range(20):
                y = torch.mm(x, w)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) / 20)
        before = time.perf_counter()
        for _ in range(100):
            y = torch.mm(x, w)
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - before) * 1000 / 100
        finite = torch.isfinite(y).all().item()
    return {
        "environment": environment(),
        "operation": "synthetic FP16 matmul; not Wan performance",
        "shapes": [[256, 512], [512, 1024]],
        "gpu_median_ms": statistics.median(times),
        "gpu_min_ms": min(times),
        "gpu_max_ms": max(times),
        "wall_mean_ms": wall_ms,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "finite": finite,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("doctor", "bench-smoke"))
    args = parser.parse_args()
    result = environment() if args.command == "doctor" else benchmark()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
