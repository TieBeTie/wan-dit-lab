"""CUDA events, native wall time and paired graph replay. No reporting policy."""

import random
import statistics
import time

import torch

from .config import load_config
from .statistics import summarize


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


def bench_wall(f, *arg, number, repeat):
    """Warmed native call path: Python submission through GPU completion.

    Synchronize at batch boundaries, not every invocation. Returned samples
    are batch-averaged wall milliseconds and are distinct from GPU event time.
    """
    if min(number, repeat) <= 0:
        raise ValueError("number and repeat must be positive")
    torch.cuda.synchronize()
    samples = []
    for _ in range(number):
        before = time.perf_counter()
        for _ in range(repeat):
            f(*arg)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - before) * 1000 / repeat)
    return samples


def peak_memory(f, *arg):
    torch.cuda.reset_peak_memory_stats()
    f(*arg)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


def paired_benchmarks(functions, *, config=None, warmup=None, number=None, repeat=None, seed=None):
    """Stateless operators on identical inputs, shuffled within every paired round.

    Compile before calling. Capture each warmed operator in its own CUDA graph
    to remove Python submission gaps. Times are per invocation, not per replay.
    Graph-private pools remain alive until all measurements are complete.
    """
    cfg = load_config(config)
    warmup = cfg.warmup if warmup is None else warmup
    number = cfg.number if number is None else number
    repeat = cfg.repeat if repeat is None else repeat
    seed = cfg.seed if seed is None else seed
    if not functions:
        raise ValueError("At least one of the functions must be supplied")
    if min(warmup, number, repeat) <= 0:
        raise ValueError("warmup, number and repeat must be positive")
    graphs, outputs = {}, {}
    with torch.inference_mode():
        # Warm all variants before capturing or timing any variant.
        for fn in functions.values():
            for _ in range(warmup):
                fn()
        torch.cuda.synchronize()
        for name, fn in functions.items():
            if cfg.timing.mode != "cuda_graph":
                continue
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(repeat):
                    outputs[name] = fn()
            graphs[name] = graph
        runners = {name: graphs[name].replay if name in graphs else fn for name, fn in functions.items()}
        for run in runners.values():
            for _ in range(warmup):
                run()
            # Short kernels otherwise finish warmup before GPU clocks ramp up.
            until = time.perf_counter() + cfg.timing.settle_ms / 1000
            while time.perf_counter() < until:
                run()
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        samples = {name: [] for name in functions}
        rng = random.Random(seed)
        for _ in range(number):
            order = list(functions)
            rng.shuffle(order)
            for name in order:
                if cfg.timing.mode == "cuda_graph":
                    elapsed = bench_events(graphs[name].replay, warmup=0, number=1, repeat=1)[0] / repeat
                else:
                    elapsed = bench_events(functions[name], warmup=0, number=1, repeat=repeat)[0]
                samples[name].append(elapsed)
    return {name: dict(**summarize(times), min_ms=min(times), max_ms=max(times),
                       mean_ms=statistics.mean(times), samples_ms=times,
                       number=number, repeat=repeat, warmup=warmup,
                       settle_ms_per_variant=cfg.timing.settle_ms,
                       timing=cfg.timing.mode)
            for name, times in samples.items()}
