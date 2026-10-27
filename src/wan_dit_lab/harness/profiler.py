"""Kernel launches of one warmed call, outside timing."""

import torch

from .config import load_config
from .cuda_graph import graph_kernels


def profile_kernels(fn, *, config=None):
    for attempt in range(load_config(config).checks.profiler_attempts):
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as profile:
            fn()
            torch.cuda.synchronize()
        names = [event.name for event in profile.events()
                 if event.device_type == torch.autograd.DeviceType.CUDA
                 and not event.is_user_annotation
                 and not event.name.startswith(("Memcpy", "Memset"))]
        if names:
            return dict(count=len(names), names=names, attempts=attempt + 1,
                        source="cupti_timeline",
                        scope="one warmed forward, outside timing")
    diagnostic = [(event.name, str(event.device_type), getattr(event, "is_user_annotation", None))
                  for event in profile.events()]
    try:
        names = graph_kernels(fn)
    except Exception as error:
        raise RuntimeError(f"Profiler did not record CUDA events for the GPU workload: {diagnostic[-30:]}") from error
    return dict(count=len(names), names=names, attempts=attempt + 1,
                source="cuda_graph_nodes", scope="one warmed forward, outside timing")
