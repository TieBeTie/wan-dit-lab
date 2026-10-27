"""Lossless backing for explicitly owned inference tensors.

Transfers deliberately finish before returning. This is the correctness backend
for resource placement, not an asynchronous prefetch scheduler. RAM/disk budgets
cover retained backing; one resource's temporary serialization/restore buffers
and the bounded pinned staging buffer must be budgeted separately.
"""

import functools
import math
import os
import sys
import time
import uuid
from collections.abc import MutableSequence
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch


def validate_placements(options):
    """Reject misspelled/unsupported policies before loading any model weights."""
    managed = options.get("mode", "fixed") == "managed"
    if options.get("mode", "fixed") not in ("fixed", "managed"):
        raise ValueError("Unknown resource policy mode")
    roles = ("self_kv", "cross_kv", "decoder_weights", "decoder_state", "history", "latents")
    if managed and any(role in options for role in roles):
        raise ValueError("Managed policy uses common budgets, not per-resource placements")
    defaults = dict(self_kv="vram", cross_kv="vram", decoder_weights="vram",
                    decoder_state="vram", history="vram", latents="ram",
                    backing_ram_gib=6.0, backing_disk_gib=32.0, staging_mib=16,
                    mode="fixed", vram_cache_gib=5.5, working_ram_mib=256,
                    working_vram_mib=512, system_ram_reserve_gib=2.,
                    process_rss_limit_gib=8., disk_reserve_gib=3., prefetch=True)
    if managed:
        defaults.update({role: "auto" for role in roles})
        defaults.update(backing_ram_gib=1., backing_disk_gib=8.)
    unknown = set(options) - defaults.keys()
    if unknown:
        raise ValueError(f"Unknown resource policy fields: {sorted(unknown)}")
    result = defaults | options
    for name in roles:
        allowed = ("auto",) if managed else (
            ("ram", "disk") if name == "latents" else ("vram", "ram", "disk")
        )
        if result[name] not in allowed:
            raise ValueError(f"Unsupported placement for {name}: {result[name]}")
    if not isinstance(result["prefetch"], bool):
        raise ValueError("prefetch must be boolean")
    for name in ("backing_ram_gib", "backing_disk_gib", "staging_mib", "vram_cache_gib",
                 "working_ram_mib", "working_vram_mib", "system_ram_reserve_gib",
                 "process_rss_limit_gib", "disk_reserve_gib"):
        value = result[name]
        if name == "process_rss_limit_gib" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Invalid budget {name}")
        if value < 0 or (name == "staging_mib" and value == 0):
            raise ValueError(f"Invalid budget {name}")
    return result


def map_tensors(value, function, memo=None):
    """Small tensor pytrees only; repeated references to one tensor stay shared."""
    memo = {} if memo is None else memo
    if isinstance(value, torch.Tensor):
        if id(value) not in memo:
            memo[id(value)] = function(value)
        return memo[id(value)]
    if isinstance(value, dict):
        return {key: map_tensors(item, function, memo) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(map_tensors(item, function, memo) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Unsupported resource value: {type(value).__name__}")


def tensor_bytes(value):
    storages = {}

    def visit(tensor):
        if tensor.layout != torch.strided or tensor.device.type == "meta":
            raise ValueError("Only materialized strided tensors can be offloaded")
        storage = tensor.untyped_storage()
        key = (tensor.device, storage.data_ptr())
        if storage.nbytes() and key in storages and storages[key] is not tensor:
            raise ValueError("Distinct views sharing storage need an explicit ownership adapter")
        storages[key] = tensor
        return tensor

    map_tensors(value, visit)
    return sum(t.numel() * t.element_size() for t in storages.values())


class TensorTransfer:
    """Synchronous boundary, CUDA copies through one bounded pinned byte buffer."""

    def __init__(self, staging_bytes=16 * 1024**2):
        if isinstance(staging_bytes, bool) or not isinstance(staging_bytes, int) or staging_bytes < 1:
            raise ValueError("staging_bytes must be positive")
        self.staging_bytes = staging_bytes
        self.buffer = None

    @torch.no_grad()
    def tensor(self, tensor, device):
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if tensor.device.type != "cuda" and device.type != "cuda":
            return tensor.detach().to(device, copy=True).contiguous()
        if tensor.device.type == "cuda" and device.type == "cuda":
            raise ValueError("Backing transfers must have a RAM endpoint")
        if self.buffer is None:
            self.buffer = torch.empty(self.staging_bytes, dtype=torch.uint8, pin_memory=True)
        source = tensor.detach().contiguous()
        target = torch.empty(source.shape, dtype=source.dtype, device=device)
        src, dst = source.reshape(-1).view(torch.uint8), target.reshape(-1).view(torch.uint8)
        cuda_device = source.device if source.is_cuda else target.device
        with torch.cuda.device(cuda_device):
            stream = torch.cuda.current_stream(cuda_device)
            for start in range(0, src.numel(), self.staging_bytes):
                n = min(self.staging_bytes, src.numel() - start)
                if source.is_cuda:
                    self.buffer[:n].copy_(src[start:start + n], non_blocking=True)
                    stream.synchronize()  # host read and buffer reuse require completion
                    dst[start:start + n].copy_(self.buffer[:n])
                else:
                    self.buffer[:n].copy_(src[start:start + n])
                    dst[start:start + n].copy_(self.buffer[:n], non_blocking=True)
                    stream.synchronize()
        return target

    def tree(self, value, device):
        return map_tensors(value, lambda tensor: self.tensor(tensor, device))

    def close(self):
        self.buffer = None  # every transfer above has already completed


class TensorStore:
    """Atomic replacement: a failed save never invalidates the previous backing."""

    def __init__(self, directory, ram_bytes, disk_bytes, staging_bytes=16 * 1024**2):
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in (ram_bytes, disk_bytes)):
            raise ValueError("Backing budgets must be nonnegative integer bytes")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.limits = {"ram": ram_bytes, "disk": disk_bytes}
        self.used = {"ram": 0, "disk": 0}
        self.peak = dict(self.used)
        self.entries = {}
        self.bindings = {}
        self.transfer = TensorTransfer(staging_bytes)
        self.events = []
        self.closed = False

    def _open(self):
        if self.closed:
            raise RuntimeError("TensorStore is closed")

    def put(self, name, value, tier):
        self._open()
        if tier not in self.limits:
            raise ValueError(f"Unknown backing tier: {tier}")
        started = time.perf_counter()
        size = tensor_bytes(value)
        # RAM retains old and new versions during replacement. Disk does too until rename.
        if self.used[tier] + size > self.limits[tier]:
            raise MemoryError(f"{tier} backing budget exceeded for {name}")
        host = self.transfer.tree(value, "cpu")
        if tier == "ram":
            entry = dict(tier=tier, bytes=size, value=host)
        else:
            temporary = self.directory / f"{uuid.uuid4().hex}.tmp"
            final = temporary.with_suffix(".pt")
            try:
                torch.save(host, temporary)
                size = temporary.stat().st_size
                if self.used[tier] + size > self.limits[tier]:
                    raise MemoryError(f"disk backing budget exceeded for {name}")
                os.replace(temporary, final)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
            entry = dict(tier=tier, bytes=size, path=final)
        # New data are complete before old backing or the live GPU owner can be freed.
        self.delete(name)
        self.entries[name] = entry
        self.used[tier] += size
        self.peak[tier] = max(self.peak[tier], self.used[tier])
        self.events.append(dict(event="save", resource=name, tier=tier,
                                bytes=size, duration_s=time.perf_counter() - started))

    def get(self, name, device="cpu"):
        self._open()
        started = time.perf_counter()
        entry = self.entries[name]
        value = entry["value"] if entry["tier"] == "ram" else torch.load(
            entry["path"], map_location="cpu", weights_only=True, mmap=True
        )
        result = self.transfer.tree(value, device)
        self.events.append(dict(event="restore", resource=name, tier=entry["tier"],
                                bytes=entry["bytes"], duration_s=time.perf_counter() - started))
        return result

    def delete(self, name):
        entry = self.entries.get(name)
        if entry is not None:
            if entry["tier"] == "disk":
                entry["path"].unlink(missing_ok=True)
            self.used[entry["tier"]] -= entry["bytes"]
            del self.entries[name]

    def export(self, name, destination):
        """Move disk backing into the permanent archive without materializing it."""
        self._open()
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        entry = self.entries[name]
        if destination.exists():
            raise FileExistsError(destination)
        if entry["tier"] == "disk":
            os.replace(entry["path"], destination)
            self.used["disk"] -= entry["bytes"]
            del self.entries[name]
        else:
            temporary = destination.with_suffix(f".{uuid.uuid4().hex}.tmp")
            try:
                torch.save(entry["value"], temporary)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            self.delete(name)

    def close(self):
        if any(binding.depth for binding in self.bindings.values()):
            raise RuntimeError("Cannot close a store with active resources")
        error = None
        for binding in list(self.bindings.values()):
            try:
                binding.close()
            except Exception as failure:
                error = error or failure
        for name in list(self.entries):
            try:
                self.delete(name)
            except Exception as failure:
                error = error or failure
        self.transfer.close()
        self.closed = not self.bindings and not self.entries
        if error is not None:
            raise error


class StateBinding:
    """An exclusive owner supplies capture/install/clear; nested uses share a lease."""

    def __init__(self, store, name, capture, install, clear, device, tier, mutable=True,
                 adopt_backing=False):
        store._open()
        if tier not in ("ram", "disk", "auto"):
            raise ValueError("State backing must be ram, disk or auto")
        if tier == "auto" and not hasattr(store, "release_binding"):
            raise ValueError("Automatic placement requires a ResourceCache")
        if name in store.bindings or (name in store.entries and not adopt_backing):
            raise ValueError(f"Resource already has an owner: {name}")
        if adopt_backing and name not in store.entries:
            raise ValueError("Cannot adopt a missing backing resource")
        self.store, self.name = store, name
        self.capture, self.install, self.clear = capture, install, clear
        self.device, self.tier, self.mutable = device, tier, mutable
        self.dirty = mutable
        self.offloaded, self.depth = adopt_backing, 0
        self.closed = False
        store.bindings[name] = self

    def _open(self):
        if self.closed:
            raise RuntimeError("Resource binding is closed")

    def evict(self):
        self._open()
        if self.depth:
            raise RuntimeError("Cannot evict an active resource")
        if self.offloaded:
            return
        if self.dirty or self.name not in self.store.entries:
            self.store.put(self.name, self.capture(), self.tier)
            self.dirty = False
        else:
            # Read-only data can be discarded only after its current CUDA readers finish.
            map_tensors(self.capture(), self._finish_reader)
        self.clear()
        self.offloaded = True
        if self.tier == "auto":
            self.store.forget_gpu(self)

    @staticmethod
    def _finish_reader(tensor):
        if tensor.is_cuda:
            torch.cuda.current_stream(tensor.device).synchronize()
        return tensor

    @contextmanager
    def resident(self, readonly=False, consumer="resident"):
        self._open()
        requested = time.perf_counter()
        ready = not self.offloaded
        if self.tier == "auto":
            ready = self.store.prepare_binding(self)
        if self.offloaded:
            self.install(self.store.get(self.name, self.device))
            self.offloaded = False
        self.depth += 1
        try:
            if self.mutable and not readonly:
                if not self.dirty and self.tier == "auto":
                    self.store.discard_stale(self)
                self.dirty = True
            if self.tier == "auto":
                self.store.activate_binding(self)
                self.store.events.append(dict(event="consumer_ready", resource=self.name,
                    consumer=consumer, node=getattr(self.store, "node", None),
                    ready_at_request=ready, duration_s=time.perf_counter() - requested))
            yield
        finally:
            self.depth -= 1
            if not self.depth:
                active_error = sys.exc_info()[1]
                try:
                    if self.tier == "auto":
                        self.store.release_binding(self)
                    else:
                        self.evict()
                except Exception:
                    if active_error is None:
                        raise
                    # The live owner remains valid when persistence fails.

    def close(self):
        if self.closed:
            return
        if self.depth:
            raise RuntimeError("Cannot close an active resource")
        if self.tier == "auto":
            self.store.wait_device(self.name, install=False)
        self.clear()
        if self.tier == "auto":
            self.store.forget_gpu(self)
        self.store.delete(self.name)
        self.offloaded = True
        self.closed = True
        self.store.bindings.pop(self.name, None)


def resident_method(function):
    @functools.wraps(function)
    def wrapped(self, *args, **kwargs):
        binding = getattr(self, "_residency", None)
        with binding.resident(consumer=function.__name__) if binding is not None else nullcontext():
            return function(self, *args, **kwargs)
    return wrapped


def bind_tensor_attributes(owner, store, name, device, tier):
    """KV tensor fields; scalar counters stay on the owner, small RoPE table is shared."""
    fields = set()

    def capture():
        values = {key: value for key, value in vars(owner).items()
                  if key != "time_freqs" and isinstance(value, torch.Tensor)}
        fields.update(values)
        return {key: getattr(owner, key, None) for key in fields}

    def install(values):
        for key, value in values.items():
            setattr(owner, key, value)

    def clear():
        for key in fields:
            setattr(owner, key, None)

    return StateBinding(store, name, capture, install, clear, device, tier)


def bind_module(module, store, name, device, tier):
    """Exclusive inference module; clear drops parameters onto the meta device."""
    return StateBinding(
        store, name, module.state_dict,
        lambda values: module.load_state_dict(values, assign=True),
        lambda: module.to("meta"), device, tier, mutable=False,
    )


class StoredSequence(MutableSequence):
    """CPU-facing sequence with RAM/disk backing, suitable for cross KV and archives."""

    def __init__(self, store, prefix, tier, values=(), maxlen=None):
        if maxlen is not None and maxlen < 0:
            raise ValueError("maxlen must be nonnegative")
        self.store, self.prefix, self.tier = store, prefix, tier
        self.maxlen = maxlen
        self.names = []
        try:
            self.extend(values)
        except BaseException:
            self.close()
            raise

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        return self.store.get(self.names[index])

    def on_device(self, index, device):
        return self.store.get(self.names[index], device)

    @contextmanager
    def device_lease(self, index, device):
        if hasattr(self.store, "lease"):
            with self.store.lease(self.names[index], device) as value:
                yield value
        else:
            yield self.on_device(index, device)

    def prefetch(self, index):
        if 0 <= index < len(self) and hasattr(self.store, "prefetch"):
            self.store.prefetch(self.names[index])

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            raise TypeError("Slice assignment is not supported")
        self.store.put(self.names[index], value, self.tier)

    def __delitem__(self, index):
        name = self.names[index]
        self.store.delete(name)
        del self.names[index]

    def insert(self, index, value):
        name = f"{self.prefix}/{uuid.uuid4().hex}"
        self.store.put(name, value, self.tier)
        self.names.insert(index, name)
        if self.maxlen is not None and len(self.names) > self.maxlen:
            del self[0]

    def close(self):
        for name in self.names:
            self.store.delete(name)
        self.names.clear()
