"""The small operator harness: timing first, optional probes from config."""

from pathlib import Path

import torch

from .checks import regression_check
from .config import load_config
from .memory import memory_probe
from .profiler import profile_kernels
from .provenance import append_row, describe_inputs, metadata
from .statistics import summarize
from .timing import bench_events, bench_wall, paired_benchmarks


def limit_memory(fraction):
    torch.cuda.set_per_process_memory_fraction(fraction)


def run(label, f, *arg, config=None, warmup=None, number=None, repeat=None, device=0):
    cfg = load_config(config)
    cfg = cfg.model_copy(update={key: value for key, value in
        (("warmup", warmup), ("number", number), ("repeat", repeat)) if value is not None})
    cfg = type(cfg).model_validate(cfg.model_dump())
    if cfg.memory.vram_fraction is not None:
        limit_memory(cfg.memory.vram_fraction)
    row = dict(label=label, function=getattr(f, "__name__", type(f).__name__),
               inputs=describe_inputs(*arg), warmup=cfg.warmup, number=cfg.number,
               repeat=cfg.repeat, config=cfg.model_dump())
    if cfg.metrics.metadata:
        row.update(metadata(device))
    times = bench_events(f, *arg, warmup=cfg.warmup, number=cfg.number, repeat=cfg.repeat)
    if cfg.metrics.latency:
        row.update(summarize(times))
    if cfg.metrics.wall:
        row["wall"] = summarize(bench_wall(f, *arg, number=cfg.number, repeat=cfg.repeat))
    if cfg.metrics.memory:
        row["memory"] = memory_probe(lambda: f(*arg), config=cfg)
        row["peak_allocated_bytes"] = row["memory"]["peak_allocated_bytes"]
    if cfg.output.path:
        path = Path(cfg.output.path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        append_row(row, path)
    return row


def compare(functions, *, config=None):
    """Shared probes and paired timing for warmed stateless variants."""
    cfg = load_config(config)
    result = dict(config=cfg.model_dump(), memory={}, kernels={}, wall={})
    for name, fn in functions.items():
        if cfg.metrics.memory:
            result["memory"][name] = memory_probe(fn, config=cfg)
        if cfg.metrics.kernels:
            result["kernels"][name] = profile_kernels(fn, config=cfg)
    result["timing"] = paired_benchmarks(functions, config=cfg)
    if cfg.metrics.wall:
        for name, fn in functions.items():
            samples = bench_wall(fn, number=cfg.number, repeat=cfg.repeat)
            result["wall"][name] = dict(**summarize(samples), samples_ms=samples,
                                       number=cfg.number, repeat=cfg.repeat)
    result["performance_gate"] = regression_check(
        result["timing"]["original"]["samples_ms"], result["timing"]["triton"]["samples_ms"],
        cfg.checks.max_latency_regression)
    return result
