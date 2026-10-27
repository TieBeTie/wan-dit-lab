"""Automatic cache admission/eviction, dirty persistence and memory headroom."""

import copy

import pytest
import torch

from wan_dit_lab.offloading.resource_cache import ResourceCache
from wan_dit_lab.offloading.resource_store import StateBinding, validate_placements


def cache_at(path, ram=8, **kwargs):
    return ResourceCache(path, ram_bytes=ram, disk_bytes=1024**2, vram_bytes=0,
                         workspace_bytes=1024**2, staging_bytes=257, **kwargs)


def test_one_policy_applies_to_all_resource_names(tmp_path):
    store = cache_at(tmp_path)
    resources = {"weights/a": 11, "kv/a": 12, "vae/state": 13, "data/z": 14}
    for name, number in resources.items():
        store.put(name, torch.tensor(number, dtype=torch.int32))
        assert store.used["ram"] <= 8
    assert store.used["disk"] > 0
    for name, number in resources.items():
        assert store.get(name).item() == number
    store.close()
    assert not list(tmp_path.iterdir())


def test_dirty_version_is_saved_once_and_clean_cache_reuses_file(tmp_path):
    store = cache_at(tmp_path, ram=64)
    store.put("x", torch.tensor([1.]))
    store.trim_ram()
    first_path = store.entries["x"]["path"]
    store.prefetch("x")
    assert store.get("x").item() == 1
    assert store.entries["x"]["tier"] == "ram"
    store.trim_ram()
    assert store.entries["x"]["path"] == first_path
    assert len([event for event in store.events if event["event"] == "write"]) == 1
    current = store.get("x")
    current.add_(4)
    store.put("x", current)
    assert not first_path.exists()
    store.trim_ram()
    assert store.get("x").item() == 5
    assert len([event for event in store.events if event["event"] == "write"]) == 2
    store.close()


def test_known_cycle_bypasses_farthest_use_instead_of_thrashing(tmp_path):
    store = cache_at(tmp_path)
    store.register_cycle(["a", "b", "c"])
    for name in ("a", "b", "c"):
        store.put(name, torch.tensor(1, dtype=torch.int32))
    assert store.entries["a"]["tier"] == "ram"
    assert store.entries["b"]["tier"] == "ram"
    assert store.entries["c"]["tier"] == "disk"
    for _ in range(3):
        for name in ("a", "b", "c"):
            value = store.get(name)
            store.put(name, value + 1)
            assert store.used["ram"] <= 8
    assert store.entries["a"]["tier"] == "ram"
    assert store.entries["b"]["tier"] == "ram"
    assert all(store.get(name).item() == 4 for name in ("a", "b", "c"))
    store.close()


@pytest.mark.parametrize("failure", ["save", "rename", "quota"])
def test_failed_spill_preserves_resident_version(tmp_path, monkeypatch, failure):
    store = cache_at(tmp_path)
    store.put("x", torch.tensor([9.]))

    def fail(*args, **kwargs):
        raise OSError("injected I/O error")

    if failure == "save":
        monkeypatch.setattr(torch, "save", fail)
    elif failure == "rename":
        monkeypatch.setattr("wan_dit_lab.offloading.resource_cache.os.replace", fail)
    else:
        store.limits["disk"] = 1
    with pytest.raises((OSError, MemoryError)):
        store.trim_ram()
    assert store.entries["x"]["tier"] == "ram"
    assert store.get("x").item() == 9
    assert not list(tmp_path.glob("*.tmp"))
    assert store.used["disk"] == 0
    store.close()


def test_memory_pressure_evicts_before_allocation(tmp_path):
    store = cache_at(tmp_path, min_system_ram_bytes=4,
                      probe=lambda: (store.used["ram"], 8 - store.used["ram"]))
    store.put("a", torch.tensor(1, dtype=torch.int32))
    assert store.entries["a"]["tier"] == "ram"
    store.put("b", torch.tensor(2, dtype=torch.int32))
    assert store.entries["a"]["tier"] == "disk"
    assert store.entries["b"]["tier"] == "ram"
    assert store.used["ram"] == 4
    store.close()


def test_impossible_memory_headroom_stops_before_copy(tmp_path, monkeypatch):
    store = cache_at(tmp_path, min_system_ram_bytes=8, probe=lambda: (0, 4))
    monkeypatch.setattr(store.transfer, "tree", lambda *args: pytest.fail("must not allocate"))
    with pytest.raises(MemoryError, match="physical RAM"):
        store.put("x", torch.ones(2))
    assert not store.entries
    store.close()


def test_transfer_workspace_is_bounded(tmp_path, monkeypatch):
    store = cache_at(tmp_path)
    store.workspace_limit = 4
    monkeypatch.setattr(store.transfer, "tree", lambda *args: pytest.fail("must not allocate"))
    with pytest.raises(MemoryError, match="workspace"):
        store.put("x", torch.ones(2))
    store.close()


def test_low_disk_space_keeps_latest_owner_alive(tmp_path, monkeypatch):
    from types import SimpleNamespace

    store = cache_at(tmp_path, ram=0)
    owner = {"x": torch.tensor([7.])}
    binding = StateBinding(store, "state", lambda: owner["x"],
                           lambda value: owner.update(x=value),
                           lambda: owner.update(x=None), "cpu", "auto")
    monkeypatch.setattr("wan_dit_lab.offloading.resource_cache.shutil.disk_usage",
                        lambda path: SimpleNamespace(free=0))
    with pytest.raises(OSError, match="disk space"):
        binding.evict()
    assert owner["x"].item() == 7 and not binding.offloaded
    store.close()


def test_source_read_does_not_make_a_private_backing_or_write(tmp_path):
    store = cache_at(tmp_path)
    source = torch.arange(1000)
    store.register_source("weights/0", lambda: source, source.numel() * source.element_size())
    assert store.source("weights/0") is source
    assert store.used == {"ram": 0, "disk": 0}
    assert [event["event"] for event in store.events] == ["source_read"]
    store.close()


def test_managed_policy_has_only_shared_budgets():
    policy = validate_placements(dict(mode="managed", backing_ram_gib=1))
    assert all(policy[role] == "auto" for role in
               ("self_kv", "cross_kv", "decoder_weights", "decoder_state", "history", "latents"))
    with pytest.raises(ValueError, match="per-resource"):
        validate_placements(dict(mode="managed", self_kv="disk"))


def test_cpu_owner_mutations_and_readonly_scope(tmp_path):
    store = cache_at(tmp_path, ram=0)
    owner = {"x": torch.tensor([1.])}
    binding = StateBinding(store, "x", lambda: owner["x"],
                           lambda value: owner.update(x=value),
                           lambda: owner.update(x=None), "cpu", "auto")
    binding.evict()
    for _ in range(3):
        with binding.resident(readonly=True):
            assert owner["x"].item() == 1
    assert len([e for e in store.events if e["event"] == "write"]) == 1
    with binding.resident():
        owner["x"].add_(1)
    assert store.get("x").item() == 2
    assert len([e for e in store.events if e["event"] == "write"]) == 2
    store.close()


def test_readonly_value_lease_cannot_be_updated_while_active(tmp_path):
    store = cache_at(tmp_path, ram=64)
    store.put("x", torch.tensor([1.]))
    with store.lease("x", "cpu") as value:
        assert value.item() == 1
        with pytest.raises(RuntimeError, match="active readers"):
            store.put("x", torch.tensor([2.]))
        with pytest.raises(RuntimeError, match="active readers"):
            store.delete("x")
    store.put("x", torch.tensor([2.]))
    with store.lease("x", "cpu") as value:
        assert value.item() == 2
    store.delete("x")
    assert not store.bindings and not store.value_owners
    store.close()


def test_speculative_prefetch_skips_memory_pressure(tmp_path):
    store = cache_at(tmp_path, ram=0)
    store.put("x", torch.tensor([3.]))
    store.limits["ram"] = 64
    store.min_system_ram = 8
    store.probe = lambda: (0, 4)
    store.prefetch("x")
    assert store.pending["x"].result(timeout=5) is False
    assert store.used["ram"] == 0
    store.close()


def test_checkpoint_packing_casts_without_materializing_all_blocks():
    from wan_dit_lab.offloading.offload import module_on

    block = torch.nn.Linear(3, 2)
    source = copy.deepcopy(block.state_dict())
    row = torch.empty(sum(p.numel() for p in block.parameters()), dtype=torch.bfloat16)
    torch.cat([p.detach().flatten() for p in block.parameters()], out=row)
    slot = module_on(row, block)
    assert slot.weight.dtype == torch.bfloat16 and block.weight.dtype == torch.float32
    assert slot.weight.untyped_storage().data_ptr() == row.untyped_storage().data_ptr()
    assert all(torch.equal(value, source[key]) for key, value in block.state_dict().items())


@pytest.mark.gpu
def test_hot_gpu_state_is_not_written_until_real_eviction(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    store = cache_at(tmp_path, ram=0)
    store.vram_limit, store.vram_workspace_limit = 1024, 1024
    owner = {"x": torch.zeros(8, device="cuda")}
    binding = StateBinding(store, "state", lambda: owner["x"],
                           lambda value: owner.update(x=value),
                           lambda: owner.update(x=None), "cuda", "auto")
    for _ in range(5):
        with binding.resident():
            owner["x"].add_(1)
    assert not store.entries and not binding.offloaded
    assert not [event for event in store.events if event["event"] == "write"]
    store.reserve_device("other/active", 2048)
    assert owner["x"] is None and binding.offloaded
    assert len([event for event in store.events if event["event"] == "write"]) == 1
    store.release_device("other/active")
    with binding.resident(readonly=True):
        assert torch.equal(owner["x"], torch.full((8,), 5., device="cuda"))
    binding.evict()
    assert len([event for event in store.events if event["event"] == "write"]) == 1
    store.close()


@pytest.mark.gpu
def test_active_gpu_lease_is_never_an_eviction_victim(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    store = cache_at(tmp_path)
    store.vram_limit, store.vram_workspace_limit = 1024, 1024
    owner = {"x": torch.ones(8, device="cuda")}
    binding = StateBinding(store, "state", lambda: owner["x"],
                           lambda value: owner.update(x=value),
                           lambda: owner.update(x=None), "cuda", "auto")
    with binding.resident():
        with pytest.raises(MemoryError, match="Active GPU"):
            store.reserve_device("too_big", 2020)
        assert owner["x"] is not None
    store.close()


@pytest.mark.gpu
def test_readonly_weight_slots_use_the_common_reservation_and_source_registry(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from wan_dit_lab.offloading.offload import BlockOffloader

    torch.manual_seed(12)
    blocks = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(3)])
    original = copy.deepcopy(blocks.state_dict())
    reference = copy.deepcopy(blocks).to("cuda", torch.bfloat16).requires_grad_(False)
    expected = torch.ones(1, 8, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        for _ in range(3):
            for block in reference:
                expected = block(expected)
    store = cache_at(tmp_path)
    store.vram_limit = 1024**2
    offloader = BlockOffloader(blocks, 2, "cuda", ram_window=1,
                               dtype=torch.bfloat16, resource_cache=store)
    try:
        offloader.start()
        actual = torch.ones(1, 8, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            for _ in range(9):
                block = offloader.get_next()
                actual = block(actual)
                offloader.release(block)
        assert torch.equal(actual, expected)
        assert all(value.dtype == torch.float32 and torch.equal(value, original[key])
                   for key, value in blocks.state_dict().items())
        assert store.sources and store.host_reservations and store.device_reservations
        assert not [event for event in store.events if event["event"] == "write"]
    finally:
        offloader.close()
        assert not store.sources and not store.host_reservations and not store.device_reservations
        store.close()


def test_gpu_restore_does_not_require_full_resource_free_ram(tmp_path, monkeypatch):
    store = cache_at(tmp_path, ram=0)
    value = torch.arange(1000, dtype=torch.float32)
    store.put("kv", value)
    store.min_system_ram = 0
    store.rss_limit = 10000
    store.probe = lambda: (0, store.transfer.staging_bytes)
    # Destination is mocked: this test checks admission independently of CUDA.
    monkeypatch.setattr(store.transfer, "tree", lambda source, device: source.clone())
    assert torch.equal(store.get("kv", "cuda"), value)
    with pytest.raises(MemoryError, match="physical RAM"):
        store.get("kv", "cpu")
    store.close()


def test_existing_pinned_buffer_needs_no_second_host_reservation(tmp_path, monkeypatch):
    store = cache_at(tmp_path, ram=0)
    value = torch.arange(1000, dtype=torch.float32)
    store.put("kv", value)
    store.rss_limit = 10000
    store.probe = lambda: (0, 0)
    store.transfer.buffer = torch.empty(257, dtype=torch.uint8)
    monkeypatch.setattr(store.transfer, "tree", lambda source, device: source.clone())
    assert torch.equal(store.get("kv", "cuda"), value)
    store.close()


def test_calibration_probes_hot_binding_and_preserves_value(tmp_path):
    from wan_dit_lab.offloading.resource_cost import calibration_from_events
    store = cache_at(tmp_path, ram=1024)
    owner = {"x": torch.tensor([7.])}
    binding = StateBinding(store, "kv/0", lambda: owner["x"],
                           lambda value: owner.update(x=value),
                           lambda: owner.update(x=None), "cpu", "auto")
    store.calibrate_idle_bindings()
    samples = calibration_from_events(store.events)["kv/0"]
    assert samples["disk_write"] and samples["disk_restore"] and samples["gpu_restore"]
    with binding.resident(readonly=True):
        assert owner["x"].item() == 7
    store.close()


@pytest.mark.gpu
@pytest.mark.parametrize("disk", [False, True])
def test_gpu_prefetch_is_ready_before_consumer_and_preserves_mutations(tmp_path, disk):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    store = cache_at(tmp_path, ram=0 if disk else 65536)
    store.vram_limit, store.vram_workspace_limit = 65536, 65536
    owner = {"x": torch.arange(4096, device="cuda", dtype=torch.float32)}
    binding = StateBinding(store, "kv", lambda: owner["x"],
        lambda value: owner.update(x=value), lambda: owner.update(x=None), "cuda", "auto")
    try:
        for iteration in range(3):
            binding.evict()
            assert store.prefetch_device("kv")
            # Future readiness, unlike merely submitting a CUDA copy, guarantees
            # that the following consumer needs no transfer.
            store.device_pending["kv"].result(timeout=15)
            assert binding.offloaded and owner["x"] is None
            with binding.resident():
                assert torch.equal(owner["x"], torch.arange(4096, device="cuda") + iteration)
                owner["x"].add_(1)
            assert not store.device_inflight
        consumers = [r for r in store.events if r["event"] == "consumer_ready"]
        assert all(r["ready_at_request"] for r in consumers)
        assert not [r for r in store.events if r["event"] == "restore"]
    finally:
        store.close()
    assert not list(tmp_path.iterdir())


@pytest.mark.gpu
def test_prefetch_failure_reaches_consumer_and_can_retry(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    store = cache_at(tmp_path, ram=65536)
    store.vram_limit = 65536
    store.put("cross", torch.arange(16))
    original = store.device_transfer.tree
    def fail(*args, **kwargs):
        raise OSError("injected GPU prefetch error")
    monkeypatch.setattr(store.device_transfer, "tree", fail)
    try:
        assert store.prefetch_device("cross")
        with pytest.raises(OSError, match="injected GPU"):
            with store.lease("cross"):
                pytest.fail("must not run with a missing resource")
        assert not store.device_pending and not store.device_inflight
        monkeypatch.setattr(store.device_transfer, "tree", original)
        with store.lease("cross") as value:
            assert torch.equal(value.cpu(), torch.arange(16))
    finally:
        store.close()


@pytest.mark.gpu
def test_new_version_and_delete_wait_for_gpu_prefetch(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    store = cache_at(tmp_path, ram=65536)
    store.vram_limit = 65536
    try:
        store.put("cross", torch.zeros(4096))
        assert store.prefetch_device("cross")
        store.put("cross", torch.ones(4096))
        with store.lease("cross", "cuda:0") as value:
            assert torch.equal(value.cpu(), torch.ones(4096))
        store.bindings["cross"].evict()
        assert store.prefetch_device("cross")
        store.delete("cross")
        assert "cross" not in store.entries and "cross" not in store.bindings
        assert not store.device_pending and not store.device_inflight
    finally:
        store.close()


def test_prefetch_reserves_vram_and_never_evicts_to_make_speculative_room(tmp_path):
    store = cache_at(tmp_path, ram=65536)
    store.vram_limit = store.vram_workspace_limit = 0
    store.put("cross", torch.ones(4096))
    assert not store.prefetch_device("cross")
    assert not store.device_pending and not store.device_inflight
    assert store.events[-1]["reason"] == "vram_budget"
    store.close()


@pytest.mark.parametrize("profile,expected", [("median", []), ("guarded", ["kv"])])
def test_guarded_calibration_starts_before_the_last_safe_boundary(tmp_path, monkeypatch, profile, expected):
    from wan_dit_lab.offloading.resource_cost import ResourceCosts
    costs = ResourceCosts({"kv": {"gpu_restore": [.01, .02, .1]},
                           "dit/block/0": {"compute": [.02, .05, .08]}}, profile)
    store = cache_at(tmp_path, ram=65536, cost_model=costs)
    store.put("kv", torch.ones(16))
    calls = []
    monkeypatch.setattr(store, "prefetch_device", lambda name: calls.append(name) or True)
    try:
        # 3 remaining compute blocks: median lead .15, guarded lead .06.
        compute = costs.seconds("dit/block/0", "compute")
        store.prefetch_schedule([("kv", 3 * compute)], next_boundary_s=compute)
        assert calls == expected
    finally:
        store.close()


def test_short_transfer_deadline_includes_queued_large_transfer(tmp_path, monkeypatch):
    import time
    from concurrent.futures import Future
    from wan_dit_lab.offloading.resource_cost import ResourceCosts
    costs = ResourceCosts({"long": {"gpu_restore": [.1]}, "short": {"gpu_restore": [.01]}})
    store = cache_at(tmp_path, ram=65536, cost_model=costs)
    store.put("long", torch.ones(8))
    store.put("short", torch.ones(8))
    pending = Future()
    store.device_pending["long"] = pending
    store.device_timing["long"] = dict(submitted_s=time.perf_counter(), started_s=None)
    calls = []
    monkeypatch.setattr(store, "prefetch_device", lambda name: calls.append(name) or True)
    try:
        store.prefetch_schedule([("short", .08)], next_boundary_s=.02)
        assert calls == ["short"]  # .01 alone would incorrectly defer it
    finally:
        pending.set_result(None)
        store.close()


@pytest.mark.gpu
@pytest.mark.parametrize("ram", [0, 65536])
def test_mutable_gpu_owner_retires_only_the_obsolete_backing(tmp_path, ram):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    store = cache_at(tmp_path, ram=ram)
    store.vram_limit = 65536
    owner = {"x": torch.ones(16, device="cuda")}
    binding = StateBinding(store, "kv", lambda: owner["x"],
        lambda value: owner.update(x=value), lambda: owner.update(x=None), "cuda", "auto")
    try:
        binding.evict()
        with binding.resident(readonly=True):
            assert "kv" in store.entries  # still a valid clean version
        with binding.resident():
            assert "kv" not in store.entries
            assert store.used == {"ram": 0, "disk": 0}
            owner["x"].add_(1)
        binding.evict()
        with binding.resident(readonly=True):
            assert torch.equal(owner["x"], torch.full((16,), 2., device="cuda"))
    finally:
        store.close()


@pytest.mark.gpu
def test_late_prefetch_is_counted_and_waited_before_reading(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    import threading
    store = cache_at(tmp_path, ram=65536)
    store.vram_limit = 65536
    store.put("cross", torch.arange(16))
    release, demand = threading.Event(), threading.Event()
    load, wait = store._device_load, store.wait_device
    def delayed(*args):
        assert release.wait(10)
        return load(*args)
    def consumer_wait(*args, **kwargs):
        demand.set()
        return wait(*args, **kwargs)
    monkeypatch.setattr(store, "_device_load", delayed)
    monkeypatch.setattr(store, "wait_device", consumer_wait)
    def release_at_demand():
        if demand.wait(10):
            release.set()
    helper = threading.Thread(target=release_at_demand)
    try:
        assert store.prefetch_device("cross")
        helper.start()
        with store.lease("cross", "cuda:0") as value:
            assert torch.equal(value.cpu(), torch.arange(16))
        row = next(r for r in store.events if r["event"] == "consumer_ready")
        assert row["ready_at_request"] is False
        assert row["duration_s"] > 0
    finally:
        release.set()
        helper.join(timeout=10)
        store.close()
