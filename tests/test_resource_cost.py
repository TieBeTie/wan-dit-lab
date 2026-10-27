import pytest
import torch

from wan_dit_lab.offloading.resource_cache import ResourceCache
from wan_dit_lab.offloading.resource_cost import ResourceCosts, calibration_from_events


def test_profiles_change_real_placement():
    samples = {"a": {"disk_restore": [1, 1, 100]},
               "b": {"disk_restore": [5, 5, 5]}}
    def victims(profile):
        model = ResourceCosts(samples, profile)
        rows = [(name, 8, model.eviction(name, "ram", False, 1)) for name in samples]
        return model.victims(rows, 8)
    assert victims("median") == ["a"]
    assert victims("guarded") == ["b"]


def test_optimizer_counts_write_and_multiple_resource_capacity():
    model = ResourceCosts({"a": {"disk_restore": [1], "disk_write": [100]}})
    assert model.eviction("a", "ram", True, 1) == 101
    assert model.eviction("a", "ram", False, 1) == 1
    assert set(model.victims([("a", 3, 1), ("b", 4, 1), ("c", 8, 10)], 7)) == {"a", "b"}
    with pytest.raises(MemoryError):
        model.victims([("a", 3, 1)], 4)


def test_missing_cost_is_not_silently_replaced():
    with pytest.raises(ValueError, match="Missing calibration"):
        ResourceCosts({}).seconds("kv/0", "disk_write")
    with pytest.raises(ValueError):
        ResourceCosts({"a": {"disk_write": [float("nan")]}})


def test_common_cache_executes_cost_selected_spill(tmp_path):
    costs = ResourceCosts({name: {"disk_restore": [read], "disk_write": [0]}
                           for name, read in (("a", 100), ("b", 1), ("c", 3))})
    store = ResourceCache(tmp_path, 8, 1024**2, 0, min_system_ram_bytes=0,
                          cost_model=costs, prefetch=False)
    for name in ("a", "b", "c"):
        store.put(name, torch.tensor(1, dtype=torch.int32))
    assert store.entries["a"]["tier"] == "ram"
    assert store.entries["b"]["tier"] == "disk"
    assert store.get("b").item() == 1
    assert any(row["event"] == "placement" for row in store.events)
    store.close()


def test_cold_incoming_value_does_not_displace_a_more_valuable_ram_copy(tmp_path):
    costs = ResourceCosts({name: {"disk_restore": [read], "disk_write": [.1]}
                           for name, read in (("hot", 100), ("cold", 1))})
    store = ResourceCache(tmp_path, 8, 1024**2, 0, cost_model=costs, prefetch=False)
    try:
        store.put("hot", torch.tensor([1., 2.]))
        store.put("cold", torch.tensor([3., 4.]))
        assert store.entries["hot"]["tier"] == "ram"
        assert store.entries["cold"]["tier"] == "disk"
        assert torch.equal(store.get("cold"), torch.tensor([3., 4.]))
        assert store.used["ram"] == 8
    finally:
        store.close()


def test_prefetch_deadline_changes_with_profile(tmp_path, monkeypatch):
    samples = {"a": {"disk_restore": [1, 1, 10]}}
    store = ResourceCache(tmp_path, 0, 1024**2, 0, prefetch=False)
    store.put("a", torch.tensor(1))
    called = []
    monkeypatch.setattr(store, "prefetch", called.append)
    store.cost_model = ResourceCosts(samples, "median")
    store.prefetch_schedule([("a", 5)])
    assert not called
    store.cost_model = ResourceCosts(samples, "guarded")
    store.prefetch_schedule([("a", 5)])
    assert called == ["a"]
    store.cost_model = None
    store.close()


def test_calibration_keeps_measured_end_to_end_costs():
    samples = calibration_from_events([
        dict(event="restore", resource="cross/" + "a" * 32, tier="disk", duration_s=2),
        dict(event="write", resource="a", duration_s=3)])
    assert samples["cross/item"]["disk_restore"] == [2]
    assert samples["a"]["disk_write"] == [3]
