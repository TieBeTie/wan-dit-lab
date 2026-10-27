"""One bounded cache policy for owned inference tensors, independent of tensor role.

VRAM retains idle bindings under its cache budget. RAM retains backing copies
under a separate budget; eviction writes an explicit file, or reuses a clean one.
Only active transfer workspaces may exceed cache budgets. No OS paging is used
as a placement operation. OS file-cache residency remains outside our control.
"""

import math
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import psutil
import torch

from wan_dit_lab.offloading.resource_store import (
    StateBinding,
    TensorStore,
    TensorTransfer,
    map_tensors,
    tensor_bytes,
)


def memory_probe():
    return psutil.Process().memory_info().rss, psutil.virtual_memory().available


def gpu_bytes(value):
    sizes = []
    map_tensors(value, lambda tensor: sizes.append(tensor.numel() * tensor.element_size())
                if tensor.is_cuda else None)
    return sum(sizes)


class ResourceCache(TensorStore):
    def __init__(self, directory, ram_bytes, disk_bytes, vram_bytes,
                 workspace_bytes=256 * 1024**2, vram_workspace_bytes=512 * 1024**2,
                 staging_bytes=16 * 1024**2, min_system_ram_bytes=0,
                 rss_limit_bytes=None, min_disk_free_bytes=0, probe=memory_probe,
                 prefetch=True, cost_model=None):
        super().__init__(directory, ram_bytes, disk_bytes, staging_bytes)
        for value in (vram_bytes, workspace_bytes, vram_workspace_bytes,
                      min_system_ram_bytes, min_disk_free_bytes):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Cache budgets must be nonnegative integer bytes")
        self.vram_limit, self.workspace_limit = vram_bytes, workspace_bytes
        self.vram_workspace_limit = vram_workspace_bytes
        self.min_system_ram, self.rss_limit = min_system_ram_bytes, rss_limit_bytes
        self.min_disk_free, self.probe = min_disk_free_bytes, probe
        self.gpu_entries, self.gpu_used, self.gpu_peak = {}, 0, 0
        self.clock, self.accessed, self.cycles = 0, {}, []
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1) if prefetch else None
        self.pending = {}
        self.device_executor = ThreadPoolExecutor(max_workers=1) if prefetch else None
        self.device_pending, self.device_inflight = {}, {}
        self.device_timing = {}
        # Separate bounded staging: a speculative H2D copy must not serialize a
        # consumer's D2H eviction or reuse its pinned buffer.
        self.device_transfer = TensorTransfer(staging_bytes)
        self.prefetch_stream = None
        self.sources = {}  # read-only mmap sources, not private RAM tensor replicas
        self.host_reservations, self.device_reservations = {}, {}
        self.host_staging_peak = 0
        self.pinned_peak = 0
        self.value_owners = {}
        self.cost_model = cost_model

    def placement_victims(self, names, required, tier, incoming=None):
        if self.cost_model is None:
            return [self._victim(names)]
        rows = []
        for name in names:
            binding = self.bindings.get(name)
            entry = self.entries.get(name, {})
            dirty = binding.dirty if tier == "gpu" and binding else "path" not in entry
            size = self.gpu_entries[name] if tier == "gpu" else entry["payload_bytes"]
            distance = self._priority(name)[0]
            if not math.isfinite(distance):
                distance = 1  # unknown reuse is not assumed to be infinitely far away
            cost = self.cost_model.eviction(name, tier, dirty, distance)
            rows.append((name, size, cost))
        if incoming is not None:
            name, size = incoming
            distance = self._priority(name)[0]
            if not math.isfinite(distance):
                distance = 1
            # The alternative is to store the incoming value directly on disk.
            # Mandatory RAM admission can otherwise expel a hotter resource on
            # every GPU eviction and turn a cache into disk-write churn.
            rows.append((name, size, self.cost_model.eviction(name, tier, True, distance)))
        selected = self.cost_model.victims(rows, required)
        self.events.append(dict(event="placement", tier=tier, required_bytes=required,
                                victims=selected, profile=self.cost_model.profile))
        return selected

    def _value_binding(self, name, device):
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        with self.lock:
            self._open()
            if name not in self.bindings:
                owner = {"value": None}
                binding = StateBinding(
                    self, name, lambda: owner["value"],
                    lambda value: owner.update(value=value), lambda: owner.update(value=None),
                    device, "auto", mutable=False, adopt_backing=True,
                )
                self.value_owners[name] = owner
            else:
                binding = self.bindings[name]
                owned_device = torch.device(binding.device)
                if owned_device.type == "cuda" and owned_device.index is None:
                    owned_device = torch.device("cuda", torch.cuda.current_device())
                if name not in self.value_owners or owned_device != device:
                    raise ValueError("Stored value already has a different owner/device")
        return binding

    @contextmanager
    def lease(self, name, device="cuda"):
        """Read-only GPU reuse for any stored value, independent of its resource type."""
        binding = self._value_binding(name, device)
        with binding.resident(readonly=True):
            yield self.value_owners[name]["value"]

    def _invalidate_value(self, name, close=False):
        if name not in self.value_owners:
            return
        binding = self.bindings[name]
        if binding.depth:
            raise RuntimeError("Cannot replace a resource with active readers")
        if not binding.offloaded:
            map_tensors(binding.capture(), binding._finish_reader)
            binding.clear()
            binding.offloaded = True
            self.forget_gpu(binding)
        if close:
            binding.closed = True
            self.bindings.pop(name, None)
            self.value_owners.pop(name, None)

    def reserve_host(self, name, size):
        with self.lock:
            if name in self.host_reservations:
                raise ValueError(f"Duplicate host reservation {name}")
            self._headroom(size)
            self.host_reservations[name] = size
            self.host_staging_peak = max(self.host_staging_peak, sum(self.host_reservations.values()))
            self._pinned_usage()

    def _pinned_usage(self):
        buffers = sum(transfer.staging_bytes for transfer in (self.transfer, self.device_transfer)
                      if transfer.buffer is not None)
        self.pinned_peak = max(self.pinned_peak, sum(self.host_reservations.values()) + buffers)

    def release_host(self, name):
        with self.lock:
            self.host_reservations.pop(name, None)

    def reserve_device(self, name, size):
        with self.lock:
            if name in self.device_reservations:
                raise ValueError(f"Duplicate GPU reservation {name}")
            self._gpu_room(size)
            self.device_reservations[name] = size
            self.gpu_peak = max(self.gpu_peak, self.gpu_used + sum(self.device_reservations.values()))

    def release_device(self, name):
        with self.lock:
            self.device_reservations.pop(name, None)

    def _gpu_room(self, size, exclude=()):
        while self.gpu_used + sum(self.device_reservations.values()) + sum(self.device_inflight.values()) + size > self.vram_limit + self.vram_workspace_limit:
            victims = [name for name in self.gpu_entries
                       if name not in exclude and not self.bindings[name].depth]
            if not victims:
                raise MemoryError("Active GPU resources exceed cache plus working reserve")
            required = self.gpu_used + sum(self.device_reservations.values()) + sum(self.device_inflight.values()) + size - self.vram_limit - self.vram_workspace_limit
            for name in self.placement_victims(victims, required, "gpu"):
                self.bindings[name].evict()

    def register_source(self, name, read, payload_bytes):
        """A pre-existing immutable file mapping; never reserialize the checkpoint."""
        with self.lock:
            if name in self.sources or name in self.entries:
                raise ValueError(f"Duplicate source {name}")
            self.sources[name] = dict(read=read, payload_bytes=payload_bytes)

    def source(self, name):
        with self.lock:
            self._open()
            # The source returns views of an existing mapping, not a private
            # block allocation. Pinned destination memory is reserved separately.
            self._headroom(0)
            value = self.sources[name]["read"]()
            self.events.append(dict(event="source_read", resource=name,
                                    bytes=self.sources[name]["payload_bytes"], duration_s=0))
            return value

    def unregister_source(self, name):
        with self.lock:
            self.sources.pop(name, None)

    def register_cycle(self, names):
        names = tuple(names)
        if not names or len(set(names)) != len(names):
            raise ValueError("Access cycle must contain distinct resources")
        with self.lock:
            self.cycles.append(dict(positions={name: i for i, name in enumerate(names)},
                                    cursor=-1, length=len(names)))

    def touch(self, name):
        self.clock += 1
        self.accessed[name] = self.clock
        for cycle in self.cycles:
            if name in cycle["positions"]:
                cycle["cursor"] = cycle["positions"][name]

    def _priority(self, name):
        distance = float("inf")
        for cycle in self.cycles:
            if name in cycle["positions"] and cycle["cursor"] >= 0:
                distance = (cycle["positions"][name] - cycle["cursor"]) % cycle["length"]
                distance = distance or cycle["length"]
                break
        clean_file = "path" in self.entries.get(name, {})
        return distance, clean_file, -self.accessed.get(name, 0)

    def _victim(self, names):
        return max(names, key=self._priority)

    def _workspace(self, size):
        if size > self.workspace_limit:
            raise MemoryError("Resource exceeds bounded transfer workspace; split the resource")

    def _file(self, name, host, size):
        # Reserve old + new during atomic replacement, including serializer overhead.
        if self.used["disk"] + size > self.limits["disk"]:
            raise MemoryError("Disk cache budget exceeded")
        if shutil.disk_usage(self.directory).free < self.min_disk_free + size + 1024**2:
            raise OSError("Insufficient free disk space for safe resource spill")
        temporary = self.directory / f"{uuid.uuid4().hex}.tmp"
        final = temporary.with_suffix(".pt")
        started = time.perf_counter()
        try:
            torch.save(host, temporary)
            file_bytes = temporary.stat().st_size
            if self.used["disk"] + file_bytes > self.limits["disk"]:
                raise MemoryError("Disk cache budget exceeded")
            os.replace(temporary, final)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        self.peak["disk"] = max(self.peak["disk"], self.used["disk"] + file_bytes)
        self.events.append(dict(event="write", resource=name, bytes=file_bytes,
                                duration_s=time.perf_counter() - started))
        return final, file_bytes

    def _spill(self, name):
        entry = self.entries[name]
        if entry["tier"] != "ram":
            return
        if "path" not in entry:
            path, size = self._file(name, entry["value"], entry["payload_bytes"])
            entry.update(path=path, file_bytes=size)
            self.used["disk"] += size
        self.used["ram"] -= entry["payload_bytes"]
        del entry["value"]
        entry.update(tier="disk", bytes=entry["file_bytes"])
        self.events.append(dict(event="spill", resource=name, bytes=entry["payload_bytes"],
                                duration_s=0))

    def _ram_candidates(self, exclude=()):
        return [name for name, entry in self.entries.items()
                if entry["tier"] == "ram" and name not in exclude and name not in self.device_pending]

    def _headroom(self, workspace, exclude=()):
        if not self.min_system_ram and self.rss_limit is None:
            return
        while True:
            rss, available = self.probe()
            if available >= self.min_system_ram + workspace and (
                self.rss_limit is None or rss + workspace <= self.rss_limit
            ):
                return
            candidates = self._ram_candidates(exclude)
            if not candidates:
                raise MemoryError("Insufficient physical RAM headroom; stopped before allocation")
            for name in self.placement_victims(candidates, 1, "ram"):
                self._spill(name)

    def _make_ram_room(self, size, exclude=(), incoming=None):
        if size > self.limits["ram"]:
            return False
        while self.used["ram"] + size > self.limits["ram"]:
            candidates = self._ram_candidates(exclude)
            if not candidates:
                return False
            if self.cost_model is not None:
                required = self.used["ram"] + size - self.limits["ram"]
                selected = self.placement_victims(candidates, required, "ram",
                                                  incoming=(incoming, size) if incoming else None)
                if incoming in selected:
                    return False
                for name in selected:
                    self._spill(name)
                continue
            victim = self._victim(candidates)
            if incoming is not None and self._priority(incoming) > self._priority(victim):
                return False  # bypass a cold incoming resource rather than evict a nearer use
            self._spill(victim)
        return True

    def _wait(self, name):
        future = self.pending.pop(name, None)
        if future is not None:
            ready = future.done()
            started = time.perf_counter()
            try:
                future.result()  # readiness before a consumer or a newer version writes
            finally:
                self.events.append(dict(event="prefetch_wait", resource=name, target="ram",
                    ready_at_request=ready, duration_s=time.perf_counter() - started))

    def put(self, name, value, tier="auto"):
        if tier != "auto":
            raise ValueError("Managed resources use one automatic policy")
        self.wait_device(name, install=False)
        self._wait(name)
        with self.lock:
            self._open()
            started = time.perf_counter()
            if name in self.value_owners and self.bindings[name].depth:
                raise RuntimeError("Cannot replace a resource with active readers")
            size = tensor_bytes(value)
            self.touch(name)
            self._workspace(size)
            self._headroom(size, (name,))
            previous = self.entries.get(name)
            credit = previous["payload_bytes"] if previous and previous["tier"] == "ram" else 0
            in_ram = self._make_ram_room(max(0, size - credit), (name,), name) and size <= self.limits["ram"]
            host = self.transfer.tree(value, "cpu")
            self._pinned_usage()
            entry = dict(tier="ram", bytes=size, payload_bytes=size, value=host)
            if not in_ram:
                path, file_bytes = self._file(name, host, size)
                entry = dict(tier="disk", bytes=file_bytes, payload_bytes=size,
                             path=path, file_bytes=file_bytes)
            try:
                self._delete(name)
            except BaseException:
                if "path" in entry:
                    entry["path"].unlink(missing_ok=True)
                raise
            self.entries[name] = entry
            self._invalidate_value(name)
            if entry["tier"] == "ram":
                self.used["ram"] += size
            if "path" in entry:
                self.used["disk"] += entry["file_bytes"]
            for place in self.used:
                self.peak[place] = max(self.peak[place], self.used[place])
            self.touch(name)
            self.events.append(dict(event="save", resource=name, tier=entry["tier"], bytes=size,
                                    duration_s=time.perf_counter() - started))

    def _read(self, name, entry):
        if entry["tier"] == "ram":
            return entry["value"]
        return torch.load(entry["path"], map_location="cpu", weights_only=True, mmap=True)

    def get(self, name, device="cpu"):
        self._wait(name)
        with self.lock:
            self._open()
            started = time.perf_counter()
            entry = self.entries[name]
            self._workspace(entry["payload_bytes"])
            # GPU restore allocates its destination in VRAM. The CPU source is
            # already resident or mmap-backed; only a missing pinned staging
            # buffer requires a new private host allocation.
            if torch.device(device).type == "cuda":
                host_allocation = self.transfer.staging_bytes if self.transfer.buffer is None else 0
            else:
                host_allocation = entry["payload_bytes"]
            self._headroom(host_allocation, (name,))
            source = entry["tier"]
            result = self.transfer.tree(self._read(name, entry), device)
            self._pinned_usage()
            if source == "disk":
                # Includes page faults and destination copy; never a cold SSD benchmark.
                self.events.append(dict(event="read", resource=name, bytes=entry["file_bytes"],
                                        duration_s=time.perf_counter() - started,
                                        timing="file restore including memory transfer; cache state unknown"))
            self.touch(name)
            self.events.append(dict(event="restore", resource=name, tier=source,
                                    bytes=entry["payload_bytes"], duration_s=time.perf_counter() - started))
            return result

    def _promote(self, name):
        with self.lock:
            started = time.perf_counter()
            entry = self.entries.get(name)
            binding = self.bindings.get(name)
            if entry is None or entry["tier"] == "ram" or (binding and not binding.offloaded):
                return False
            size = entry["payload_bytes"]
            # Speculative I/O never evicts useful data just to create a prefetch slot.
            if self.used["ram"] + size > self.limits["ram"] or size > self.workspace_limit:
                return False
            if self.min_system_ram or self.rss_limit is not None:
                rss, available = self.probe()
                if available < self.min_system_ram + size or (
                    self.rss_limit is not None and rss + size > self.rss_limit
                ):
                    return False
            value = self.transfer.tree(self._read(name, entry), "cpu")
            self.events.append(dict(event="read", resource=name, bytes=entry["file_bytes"],
                                    duration_s=time.perf_counter() - started,
                                    timing="file restore including memory transfer; cache state unknown"))
            entry.update(tier="ram", bytes=size, value=value)
            self.used["ram"] += size
            self.peak["ram"] = max(self.peak["ram"], self.used["ram"])
            self.events.append(dict(event="prefetch", resource=name, bytes=size,
                                    duration_s=time.perf_counter() - started))
            return True

    def prefetch(self, name):
        with self.lock:
            if self.closed or self.executor is None or name in self.pending or name in self.device_pending:
                return
            if name not in self.entries or self.entries[name]["tier"] == "ram":
                return
            binding = self.bindings.get(name)
            if binding is not None and not binding.offloaded:
                return
            # One in-flight CPU load, bounded by the same workspace and cache budgets.
            if any(not future.done() for future in self.pending.values()):
                return
            self.pending[name] = self.executor.submit(self._promote, name)

    def _device_load(self, name, device):
        """Worker owns its staging and stream; never changes a live model owner."""
        self._wait(name)
        with self.lock:
            ticket = self.device_timing[name]
            ticket["started_s"] = time.perf_counter()
            entry = self.entries[name]
            source, size = entry["tier"], entry["payload_bytes"]
            value = self._read(name, entry)
        started = time.perf_counter()
        with torch.cuda.device(device), torch.no_grad():
            if self.prefetch_stream is None:
                self.prefetch_stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(self.prefetch_stream):
                result = self.device_transfer.tree(value, device)
                # TensorTransfer completes copies before returning. The future
                # therefore means ready in VRAM, not merely enqueued on CUDA.
        elapsed = time.perf_counter() - started
        with self.lock:
            self._pinned_usage()
        self.events.append(dict(event="device_prefetch", resource=name, tier=source,
                                bytes=size, duration_s=elapsed,
                                queue_s=ticket["started_s"] - ticket["submitted_s"]))
        if source == "disk":
            self.events.append(dict(event="read", resource=name, bytes=entry["file_bytes"],
                duration_s=elapsed, timing="prefetch file restore including H2D; cache state unknown"))
        return result

    def prefetch_device(self, name):
        """Admit a GPU load only into unused cache/working space; never evict speculatively."""
        with self.lock:
            binding = self.bindings.get(name)
            if self.closed or self.device_executor is None or name not in self.entries:
                return False
            if binding is None:
                binding = self._value_binding(name, "cuda")
            if (binding is None
                    or binding.closed or not binding.offloaded or name in self.device_pending
                    or torch.device(binding.device).type != "cuda"):
                return False
            entry = self.entries[name]
            size = entry["payload_bytes"]
            if size > self.workspace_limit:
                return False
            used = self.gpu_used + sum(self.device_reservations.values()) + sum(self.device_inflight.values())
            if used + size > self.vram_limit + self.vram_workspace_limit:
                self.events.append(dict(event="prefetch_deferred", resource=name, reason="vram_budget"))
                return False
            self._headroom(self.device_transfer.staging_bytes if self.device_transfer.buffer is None else 0,
                           (name,))
            self.device_inflight[name] = size
            self.device_timing[name] = dict(submitted_s=time.perf_counter(), started_s=None)
            try:
                self.device_pending[name] = self.device_executor.submit(self._device_load, name, binding.device)
            except BaseException:
                self.device_inflight.pop(name, None)
                self.device_timing.pop(name, None)
                raise
            self.gpu_peak = max(self.gpu_peak, used + size)
            return True

    def wait_device(self, name, install=True):
        # Call outside lock: a queued worker needs that lock to snapshot backing.
        future = self.device_pending.get(name)
        if future is None:
            return
        ready, started = future.done(), time.perf_counter()
        try:
            value = future.result()
            with self.lock:
                binding = self.bindings.get(name)
                if install and binding is not None and not binding.closed and binding.offloaded:
                    def record_reader(tensor):
                        if tensor.is_cuda:
                            tensor.record_stream(torch.cuda.current_stream(tensor.device))
                        return tensor
                    map_tensors(value, record_reader)
                    binding.install(value)
                    binding.offloaded = False
                    self._account_gpu(binding)
        finally:
            with self.lock:
                self.device_pending.pop(name, None)
                self.device_inflight.pop(name, None)
                self.device_timing.pop(name, None)
            self.events.append(dict(event="prefetch_wait", resource=name, target="vram",
                ready_at_request=ready, duration_s=time.perf_counter() - started))

    def prefetch_schedule(self, upcoming, next_boundary_s=0.):
        """Start by the last block boundary before the measured load deadline."""
        if self.cost_model is None:
            for name, _ in upcoming[:1]:
                if not self.prefetch_device(name):
                    self.prefetch(name)
            return
        for name, lead_s in upcoming:
            entry = self.entries.get(name)
            binding = self.bindings.get(name)
            if entry is None or name in self.device_pending or (binding and not binding.offloaded):
                continue
            kind = "disk_restore" if entry["tier"] == "disk" else "gpu_restore"
            load_s = self.cost_model.seconds(name, kind)
            # There is one transfer worker. A short copy behind a large KV load
            # needs the latter's remaining service time, not just its own copy.
            backlog_s = self.device_backlog()
            if lead_s - next_boundary_s <= backlog_s + load_s:
                self.events.append(dict(event="prefetch_due", resource=name,
                                        lead_s=lead_s, predicted_load_s=load_s, backlog_s=backlog_s,
                                        profile=self.cost_model.profile))
                if self.prefetch_device(name):
                    continue
                self.prefetch(name)

    def device_backlog(self):
        if self.cost_model is None:
            return 0.
        now, total = time.perf_counter(), 0.
        with self.lock:
            for name, future in self.device_pending.items():
                if future.done():
                    continue
                entry, ticket = self.entries[name], self.device_timing[name]
                kind = "disk_restore" if entry["tier"] == "disk" else "gpu_restore"
                service = self.cost_model.seconds(name, kind)
                elapsed = 0. if ticket["started_s"] is None else now - ticket["started_s"]
                total += max(0., service - elapsed)
        return total

    def calibrate_idle_bindings(self):
        """Exercise actual write/restore paths even for otherwise hot resources.

        Calibration only, outside timed evaluation. Preserve values via readonly
        leases; missing samples must never be filled with invented costs.
        """
        if self.cost_model is not None:
            raise ValueError("Cannot change calibration during frozen evaluation")
        for name in list(self.bindings):
            self._wait(name)
            binding = self.bindings.get(name)
            if binding is None or binding.closed or binding.depth:
                continue
            if not binding.offloaded:
                binding.evict()
            if name not in self.entries:
                continue
            with self.lock:
                self._spill(name)
            with binding.resident(readonly=True):
                pass
        for name in list(self.entries):
            if name in self.bindings:
                continue
            self._wait(name)
            with self.lock:
                self._spill(name)
            # Unbound data (context/latent history) also needs measured restore
            # costs before the optimizer can compare its legal placements.
            value = self.get(name, "cuda" if torch.cuda.is_available() else "cpu")
            del value

    def trim_ram(self, target_bytes=0):
        with self.lock:
            if target_bytes < 0:
                raise ValueError("RAM target must be nonnegative")
            while self.used["ram"] > target_bytes:
                for name in self.placement_victims(self._ram_candidates(), self.used["ram"] - target_bytes, "ram"):
                    self._spill(name)

    def _account_gpu(self, binding):
        size = gpu_bytes(binding.capture()) if not binding.offloaded else 0
        if size:
            self.gpu_entries[binding.name] = size
        else:
            self.gpu_entries.pop(binding.name, None)
        self.gpu_used = sum(self.gpu_entries.values())
        self.gpu_peak = max(self.gpu_peak, self.gpu_used + sum(self.device_reservations.values()))

    def forget_gpu(self, binding):
        with self.lock:
            self.gpu_entries.pop(binding.name, None)
            self.gpu_used = sum(self.gpu_entries.values())

    def discard_stale(self, binding):
        """A write lease makes the old backing version unusable for future reads."""
        with self.lock:
            if binding.offloaded or not binding.depth:
                raise RuntimeError("Cannot retire backing before the live owner is acquired")
            entry = self.entries.get(binding.name)
            if entry is not None:
                self.events.append(dict(event="retire_backing", resource=binding.name,
                                        bytes=entry["payload_bytes"], duration_s=0.))
                self._delete(binding.name)

    def prepare_binding(self, binding):
        future = self.device_pending.get(binding.name)
        ready = not binding.offloaded or (future is not None and future.done() and future.exception() is None)
        self.wait_device(binding.name)
        with self.lock:
            self.touch(binding.name)
            self._headroom(0)
            if binding.offloaded and torch.device(binding.device).type == "cuda":
                size = self.entries[binding.name]["payload_bytes"]
                self._gpu_room(size)
        return ready

    def activate_binding(self, binding):
        with self.lock:
            self._account_gpu(binding)

    def reserve_gpu(self, binding, size):
        """Dynamic resources announce their size before allocating a new GPU buffer."""
        with self.lock:
            extra = max(0, size - self.gpu_entries.get(binding.name, 0))
            self._gpu_room(extra, (binding.name,))

    def release_binding(self, binding):
        with self.lock:
            self.touch(binding.name)
            self._account_gpu(binding)
            if binding.name not in self.gpu_entries:
                binding.evict()  # a CPU owner must not bypass the bounded RAM cache
            while sum(size for name, size in self.gpu_entries.items()
                      if not self.bindings[name].depth) > max(0, self.vram_limit - sum(self.device_reservations.values())):
                victims = [name for name in self.gpu_entries if not self.bindings[name].depth]
                required = sum(self.gpu_entries[name] for name in victims) - max(0, self.vram_limit - sum(self.device_reservations.values()))
                for name in self.placement_victims(victims, required, "gpu"):
                    self.bindings[name].evict()

    def _delete(self, name):
        entry = self.entries.get(name)
        if entry is None:
            return
        if "path" in entry:
            entry["path"].unlink(missing_ok=True)
            self.used["disk"] -= entry["file_bytes"]
        if entry["tier"] == "ram":
            self.used["ram"] -= entry["payload_bytes"]
        del self.entries[name]
        self.accessed.pop(name, None)

    def delete(self, name):
        self.wait_device(name, install=False)
        self._wait(name)
        with self.lock:
            self._invalidate_value(name, close=True)
            self._delete(name)

    def export(self, name, destination):
        self.wait_device(name, install=False)
        self._wait(name)
        with self.lock:
            self._open()
            self._invalidate_value(name, close=True)
            entry = self.entries[name]
            if "path" not in entry:
                self._spill(name)
            # Drop a clean RAM cache copy and reuse base's disk-file move.
            if entry["tier"] == "ram":
                self._spill(name)
            super().export(name, destination)

    def close(self):
        if any(binding.depth for binding in self.bindings.values()):
            raise RuntimeError("Cannot close a store with active resources")
        if self.device_executor is not None:
            self.device_executor.shutdown(wait=True)
            self.device_executor = None
        error = None
        for name in list(self.device_pending):
            try:
                self.wait_device(name, install=False)
            except Exception as failure:
                error = error or failure
        self.device_transfer.close()
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
        for name in list(self.pending):
            try:
                self._wait(name)
            except Exception as failure:
                error = error or failure
        with self.lock:
            try:
                super().close()
            finally:
                self.sources.clear()
                self.host_reservations.clear()
                self.device_reservations.clear()
        if error is not None:
            raise error
