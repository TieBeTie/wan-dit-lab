"""Sample process/device memory without synchronizing every CUDA block."""

import ctypes
import functools
import os
import threading
import time
import types
from collections import deque

import psutil
import torch


def summarize_memory(rows):
    """Byte counts. RAM peak/free are sampled, GPU allocated peaks are allocator counters."""
    def value(key, mode):
        values = [row[key] for row in rows if row.get(key) is not None]
        return mode(values) if values else None

    return dict(vram=dict(baseline_bytes=value("allocated", lambda x: x[0]),
                          current_bytes=value("allocated", lambda x: x[-1]),
                          peak_bytes=value("peak_allocated", max),
                          reserved_peak_bytes=value("peak_reserved", max),
                          free_min_bytes=value("device_free", min)),
                ram=dict(baseline_bytes=value("rss", lambda x: x[0]),
                         current_bytes=value("rss", lambda x: x[-1]),
                         peak_bytes=value("rss", max), free_min_bytes=value("ram_available", min)))


def memory_probe(fn, *, config=None):
    """One stateless operator call. Stateful streams use MemorySampler in that same run."""
    from .config import load_config

    cfg = load_config(config)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    sampler = MemorySampler(cfg.memory.sample_interval_s)
    sampler.start()
    before = sampler.samples[0]
    try:
        output = fn()
        torch.cuda.synchronize()
    finally:
        sampler.close()
    after = sampler.samples[-1]
    result = dict(allocated_before_bytes=before["allocated"], reserved_before_bytes=before["reserved"],
                  peak_allocated_bytes=after["peak_allocated"], peak_reserved_bytes=after["peak_reserved"],
                  additional_peak_allocated_bytes=after["peak_allocated"] - before["allocated"],
                  device_free_before_bytes=before["device_free"], device_free_after_bytes=after["device_free"],
                  device_total_bytes=before["device_total"], **summarize_memory(sampler.samples),
                  sample_interval_s=cfg.memory.sample_interval_s,
                  scope="one warmed call; RAM and free-memory extrema are sampled")
    del output
    return result


class Nvml:
    def __init__(self):
        self.dll = (
            ctypes.WinDLL("nvml.dll") if os.name == "nt" else ctypes.CDLL("libnvidia-ml.so.1")
        )
        self.check(self.dll.nvmlInit_v2())
        self.handle = ctypes.c_void_p()
        self.check(self.dll.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)))

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f"NVML error {code}")

    def memory(self):
        class Memory(ctypes.Structure):
            _fields_ = [
                ("total", ctypes.c_ulonglong),
                ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong),
            ]

        info = Memory()
        self.check(self.dll.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(info)))
        return dict(device_total=info.total, device_free=info.free, device_used=info.used)


class MemorySampler:
    def __init__(self, interval=0.1):
        self.interval, self.samples = interval, []
        self.process, self.nvml = psutil.Process(), Nvml()
        self.stop_event, self.failure = threading.Event(), None
        self.started = time.perf_counter()
        self.active_stage = "startup"

    def snapshot(self, stage=None):
        stage = stage or self.active_stage
        process = self.process.memory_info()
        row = dict(
            stage=stage,
            t=time.perf_counter() - self.started,
            rss=process.rss,
            private=getattr(process, "private", None),
            ram_available=psutil.virtual_memory().available,
            **self.nvml.memory(),
        )
        row.update(
            allocated=torch.cuda.memory_allocated(),
            reserved=torch.cuda.memory_reserved(),
            peak_allocated=torch.cuda.max_memory_allocated(),
            peak_reserved=torch.cuda.max_memory_reserved(),
        )
        host_stats = getattr(torch.cuda.memory, "host_memory_stats", None)
        pinned = host_stats() if host_stats else {}
        row.update(
            pinned_active=pinned.get("allocated_bytes.current"),
            pinned_peak=pinned.get("allocated_bytes.peak"),
            pinned_reserved=pinned.get("reserved_bytes.current"),
        )
        self.samples.append(row)
        return row

    def start(self):
        self.snapshot("before_model_load")

        def sample():
            try:
                while not self.stop_event.wait(self.interval):
                    self.snapshot()
            except Exception as error:
                self.failure = error

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()

    def close(self):
        self.stop_event.set()
        self.thread.join()
        self.snapshot("after_final_resource_release_before_exit")
        if self.failure:
            raise RuntimeError("Memory sampling failed") from self.failure


def inventory(roots):
    """Deduplicate physical storage globally, keeping tensor paths for audit.

    Code, CUDA contexts and transient kernel workspaces are not tensor roots.
    Their unknown bytes remain untracked overhead, never a fabricated zero.
    """
    seen_objects, seen_storage, rows = set(), set(), []

    def walk(value, path, category):
        from wan_dit_lab.offloading.resource_store import TensorStore

        if isinstance(value, TensorStore):
            return  # backing entries are separate roots with their actual resource categories
        if id(value) in seen_objects:
            return
        seen_objects.add(id(value))
        if isinstance(value, torch.Tensor):
            if value.device.type == "meta" or not value.numel():
                return
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr(), storage.nbytes())
            if key in seen_storage:
                return
            seen_storage.add(key)
            rows.append(
                dict(
                    resource_id=path,
                    category=category,
                    storage_id=list(key),
                    shape=list(value.shape),
                    dtype=str(value.dtype),
                    device=str(value.device),
                    bytes=storage.nbytes(),
                    pinned=value.is_pinned() if value.device.type == "cpu" else False,
                )
            )
        elif isinstance(value, dict):
            for k, v in value.items():
                walk(v, f"{path}.{k}", category)
        elif isinstance(value, (list, tuple, deque)):
            for i, v in enumerate(value):
                walk(v, f"{path}.{i}", category)
        elif hasattr(value, "__dict__") and not isinstance(
            value, (types.FunctionType, types.MethodType, types.ModuleType, type, functools.partial)
        ):
            walk(vars(value), path, category)

    for category, name, root in roots:
        walk(root, name, category)
    return rows
