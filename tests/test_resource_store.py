"""Persistence correctness, ownership failures, and real CUDA resource round trips."""

import copy
from pathlib import Path

import pytest
import torch

from wan_dit_lab.generation.causal import CHUNK, KVCache, Stream
from wan_dit_lab.models.model import WanModel
from wan_dit_lab.offloading.offload import OnDevice
from wan_dit_lab.offloading.resource_store import (
    StateBinding,
    StoredSequence,
    TensorStore,
    bind_module,
    validate_placements,
)
from wan_dit_lab.models.rope import rope


def store_at(path, **kwargs):
    return TensorStore(path, ram_bytes=32 * 1024**2, disk_bytes=64 * 1024**2,
                       staging_bytes=257, **kwargs)


@pytest.mark.parametrize("tier", ["ram", "disk"])
def test_tensor_tree_exact_round_trip_and_independent_backing(tmp_path, tier):
    store = store_at(tmp_path)
    tensor = torch.arange(24, dtype=torch.bfloat16).reshape(4, 6).T
    data = dict(pair=(tensor, tensor), integer=torch.arange(4), scalar=torch.tensor(3.),
                empty=torch.empty(0), flag=True, optional=None)
    store.put("state", data, tier)
    tensor.add_(1)
    restored = store.get("state")
    assert restored["pair"][0] is restored["pair"][1]
    assert torch.equal(restored["pair"][0], tensor - 1)
    assert restored["scalar"].shape == torch.Size([])
    restored["pair"][0].zero_()
    assert store.get("state")["pair"][0].count_nonzero() > 0
    store.close()
    assert store.used == {"ram": 0, "disk": 0}
    assert not list(tmp_path.iterdir())
    with pytest.raises(RuntimeError, match="closed"):
        store.get("state")


def test_overlapping_views_require_explicit_ownership(tmp_path):
    store = store_at(tmp_path)
    tensor = torch.ones(8)
    with pytest.raises(ValueError, match="views sharing storage"):
        store.put("alias", [tensor[:4], tensor[2:]], "ram")
    assert not store.entries


def test_budget_failure_keeps_only_current_copy_alive(tmp_path):
    store = TensorStore(tmp_path, ram_bytes=8, disk_bytes=0)
    owner = {"x": torch.ones(4)}
    binding = StateBinding(store, "state", lambda: owner["x"],
                           lambda value: owner.update(x=value),
                           lambda: owner.update(x=None), "cpu", "ram")
    with pytest.raises(MemoryError, match="budget"):
        binding.evict()
    assert torch.equal(owner["x"], torch.ones(4))
    assert not binding.offloaded


@pytest.mark.parametrize("failure", ["save", "rename", "quota"])
def test_failed_disk_replacement_preserves_previous_version(tmp_path, monkeypatch, failure):
    store = store_at(tmp_path)
    store.put("state", torch.tensor([1]), "disk")
    before = store.used.copy()

    def fail(*args, **kwargs):
        raise OSError("injected disk failure")

    if failure == "save":
        monkeypatch.setattr(torch, "save", fail)
    elif failure == "rename":
        monkeypatch.setattr("wan_dit_lab.offloading.resource_store.os.replace", fail)
    else:
        store.limits["disk"] = store.used["disk"] + 1
    with pytest.raises((OSError, MemoryError)):
        store.put("state", torch.tensor([2]), "disk")
    assert store.get("state").item() == 1
    assert store.used == before
    assert not list(tmp_path.glob("*.tmp"))
    assert len(list(tmp_path.glob("*.pt"))) == 1


def test_nested_leases_preserve_mutations_and_do_not_release_active_owner(tmp_path):
    store = store_at(tmp_path)
    owner = {"x": torch.tensor([1])}
    binding = StateBinding(store, "x", lambda: owner["x"],
                           lambda value: owner.update(x=value),
                           lambda: owner.update(x=None), "cpu", "ram")
    binding.evict()
    assert owner["x"] is None
    with binding.resident():
        owner["x"].add_(1)
        with binding.resident():
            owner["x"].add_(2)
            with pytest.raises(RuntimeError, match="active"):
                binding.evict()
        assert owner["x"].item() == 4
    assert owner["x"] is None
    assert store.get("x").item() == 4
    with pytest.raises(ValueError, match="node failed"), binding.resident():
        owner["x"].add_(3)
        raise ValueError("node failed")
    assert store.get("x").item() == 7
    binding.close()
    assert not store.entries


def test_original_compute_error_survives_persistence_error(tmp_path):
    store = TensorStore(tmp_path, ram_bytes=0, disk_bytes=0)
    binding = StateBinding(store, "x", lambda: torch.ones(2), lambda value: None,
                           lambda: pytest.fail("must not clear"), "cpu", "ram")
    with pytest.raises(ValueError, match="compute failed"), binding.resident():
        raise ValueError("compute failed")
    assert not binding.offloaded


def test_binding_ownership_and_close_are_safe_to_reuse(tmp_path):
    store = store_at(tmp_path)

    def create():
        return StateBinding(store, "x", lambda: torch.tensor(3), lambda value: None,
                            lambda: None, "cpu", "ram")

    first = create()
    with pytest.raises(ValueError, match="owner"):
        create()
    with first.resident():
        with pytest.raises(RuntimeError, match="active"):
            store.close()
    first.close()
    second = create()
    second.evict()
    first.close()
    assert store.get("x").item() == 3
    with pytest.raises(RuntimeError, match="closed"), first.resident():
        pass
    store.close()
    assert second.closed and not store.bindings


@pytest.mark.parametrize("tier", ["ram", "disk"])
def test_immutable_module_backing_is_written_once(tmp_path, tier):
    store = store_at(tmp_path)
    module = torch.nn.Linear(3, 2).requires_grad_(False)
    x = torch.ones(1, 3)
    expected = module(x)
    binding = bind_module(module, store, "weights", "cpu", tier)
    binding.evict()
    assert module.weight.is_meta
    for _ in range(3):
        with binding.resident():
            assert torch.equal(module(x), expected)
        assert module.weight.is_meta
    assert sum(event["event"] == "save" for event in store.events) == 1
    binding.close()


@pytest.mark.parametrize("tier", ["ram", "disk"])
def test_bounded_sequence_updates_and_exports_archive(tmp_path, tier):
    store = store_at(tmp_path / "cache")
    seq = StoredSequence(store, "history", tier, maxlen=2)
    for i in range(5):
        seq.append(torch.tensor(i))
    assert [int(x) for x in seq] == [3, 4]
    seq[0] = torch.tensor(30)
    assert [int(x) for x in seq] == [30, 4]
    names = list(seq.names)
    for i, name in enumerate(names):
        store.export(name, tmp_path / "archive" / f"{i}.pt")
    seq.close()
    store.close()
    assert torch.load(tmp_path / "archive/0.pt", weights_only=True).item() == 30
    assert not list((tmp_path / "cache").iterdir())


@pytest.mark.parametrize("options", [
    {"kv": "ram"}, {"self_kv": "cpu"}, {"latents": "vram"},
    {"staging_mib": 0}, {"backing_ram_gib": float("nan")}, {"backing_disk_gib": -1},
])
def test_invalid_policy_rejected_before_allocation(options):
    with pytest.raises(ValueError):
        validate_placements(options)


def test_cuda_graph_with_moving_state_is_rejected_before_model_access(tmp_path):
    store = store_at(tmp_path)
    with pytest.raises(ValueError, match="CUDA graph"):
        Stream(None, None, None, 32, 32, resource_store=store, state_tier="ram", cuda_graph=True)


def frequencies(dim=64, device="cpu"):
    return torch.cat([rope(128, dim - 4 * (dim // 6)),
                      rope(128, 2 * (dim // 6)), rope(128, 2 * (dim // 6))], dim=-1).to(device)


@pytest.mark.parametrize("tier", ["ram", "disk", "auto"])
@pytest.mark.parametrize("fp8", [False, True])
def test_kv_ring_pc_decay_reset_round_trip(tmp_path, tier, fp8):
    torch.manual_seed(17)
    freqs = frequencies()
    reference = KVCache(18 * 4, 10 * 4, time_freqs=freqs[:, :22, :12],
                        fp8=fp8, pc_recent=1)
    actual = copy.deepcopy(reference)
    if tier == "auto":
        from wan_dit_lab.offloading.resource_cache import ResourceCache

        store = ResourceCache(tmp_path, ram_bytes=4096, disk_bytes=16 * 1024**2,
                              vram_bytes=0, workspace_bytes=1024**2)
    else:
        store = store_at(tmp_path)
    actual.configure_offloading(store, "self_kv", "cpu", tier)
    for chunk in range(12):
        if chunk == 8:
            reference.forget(.1, .7)
            actual.forget(.1, .7)
        for _ in range(2):
            q, k, v = [torch.randn(1, CHUNK * 4, 2, 64) for _ in range(3)]
            expected = reference.attend(q, k, v)
            result = actual.attend(q, k, v)
            assert torch.equal(result, expected)
            assert actual.k is None and actual.v is None
        reference.commit()
        actual.commit()
        assert actual.saved == reference.saved and actual.frames == reference.frames
        with actual._residency.resident():
            for field in ("k", "v", "frame_of", "written", "place_in_frame", "cos", "sin"):
                a, b = getattr(actual, field), getattr(reference, field)
                if field in ("k", "v"):
                    a, b = a[:, :reference.saved], b[:, :reference.saved]
                torch.testing.assert_close(a.view(torch.uint8), b.view(torch.uint8), rtol=0, atol=0)
    actual.reset()
    assert actual.saved == 0 and actual.frames == 0
    actual._residency.close()
    assert not store.entries


@pytest.mark.gpu
def test_cuda_transfer_waits_for_producer_and_reuses_small_pinned_buffer(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    store = store_at(tmp_path)
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        tensor = torch.arange(1025, device="cuda", dtype=torch.bfloat16)
        tensor.mul_(3)
        store.put("x", tensor, "disk")
        restored = store.get("x", "cuda")
        assert torch.equal(restored, tensor)
    assert store.transfer.buffer.numel() == 257
    store.close()
    assert store.transfer.buffer is None


def tiny_stream(device, **options):
    model = WanModel(dim=128, eps=1e-6, ffn_dim=256, freq_dim=256, in_dim=16,
                     num_heads=2, num_layers=2, out_dim=16, text_len=8, text_dim=32)
    model.requires_grad_(False).eval()
    model.block_offloader = OnDevice(model.blocks, device)
    model.to(device)
    context = torch.randn(1, 8, 32, device=device)
    stream = Stream(model, context, frequencies(device=device), 32, 32,
                    torch.Generator(device).manual_seed(12),
                    max_frames=12, sink_frames=3, pc_recent=1,
                    cross_cache=True, positions="window", **options)
    return stream


@pytest.mark.gpu
@pytest.mark.parametrize("tier", ["ram", "disk", "auto", "auto_inline"])
def test_cuda_stream_matches_resident_through_prompt_changes(tmp_path, tier):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    inline = tier == "auto_inline"
    tier = "auto" if inline else tier
    if tier == "auto":
        from wan_dit_lab.offloading.resource_cache import ResourceCache

        store = ResourceCache(tmp_path, ram_bytes=8192, disk_bytes=64 * 1024**2,
                              vram_bytes=0, workspace_bytes=1024**2,
                              vram_workspace_bytes=4 * 1024**2, staging_bytes=1024**2)
    else:
        store = TensorStore(tmp_path, ram_bytes=32 * 1024**2, disk_bytes=64 * 1024**2,
                            staging_bytes=1024**2)
    torch.manual_seed(37)
    reference = tiny_stream("cuda")
    torch.manual_seed(37)
    actual = tiny_stream("cuda", resource_store=store, state_tier=tier,
                         cross_tier=tier, history_tier=tier, commit_in_attention=inline)
    for block in actual.model.blocks:
        block.forward = torch.compile(block.forward, backend="eager")
    try:
        for chunk in range(8):
            if chunk in (4, 6):
                context = torch.randn_like(reference.context)
                mode = "blend" if chunk == 4 else "recache+decay"
                reference.set_prompt(context, mode, blend_chunks=2)
                actual.set_prompt(context, mode, blend_chunks=2)
            with torch.no_grad():
                expected, result = reference.next_chunk(), actual.next_chunk()
            assert torch.equal(result, expected)
            assert all(cache.k is None for cache in actual.caches)
        assert any(event["event"] == "restore" for event in store.events)
    finally:
        actual.close()
        reference.close()
        actual.model.block_offloader.close()
        reference.model.block_offloader.close()
        store.close()


@pytest.mark.gpu
@pytest.mark.parametrize("tier", ["ram", "disk"])
def test_real_streaming_vae_offloads_weights_and_temporal_state(tmp_path, tier):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from wan_dit_lab.models.checkpoint import REFERENCE
    from wan_dit_lab.models.vae import TAEHVStream

    checkpoint = REFERENCE / "checkpoints/taew2_1.pth"
    if not checkpoint.exists():
        pytest.skip("TAEHV checkpoint unavailable")
    store = TensorStore(tmp_path, ram_bytes=512 * 1024**2, disk_bytes=1024**3,
                        staging_bytes=1024**2)
    reference = TAEHVStream("cuda")
    actual = TAEHVStream("cpu")
    actual.configure_offloading(store, "cuda", tier, tier)
    try:
        for _ in range(3):
            x = torch.randn(16, 3, 4, 4, device="cuda")
            expected, result = reference(x), actual(x)
            assert torch.equal(result, expected)
            assert next(actual.decoder.parameters()).is_meta
            assert not actual.last
        writes = [e for e in store.events if e["event"] == "save" and e["resource"] == "vae/weights"]
        assert len(writes) == 1
    finally:
        actual.close()
        reference.close()
        store.close()


def test_inventory_does_not_count_unrelated_store_entries_as_state(tmp_path):
    from wan_dit_lab.harness.memory import inventory

    store = store_at(tmp_path)
    store.put("weights", torch.ones(4), "ram")
    assert not inventory([("state", "store", store)])
    assert inventory([("weights", "weights", store.entries["weights"]["value"])])[0]["bytes"] == 16


@pytest.mark.gpu
@pytest.mark.parametrize("placement", ["disk", "managed"])
def test_complete_pipeline_writes_video_and_chunked_archive(tmp_path, monkeypatch, placement):
    """Real tiny DiT + real TAEHV + ffmpeg, including ResourcePlan cleanup."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    import importlib
    import json
    from types import SimpleNamespace

    from wan_dit_lab.models.checkpoint import REFERENCE

    if not (REFERENCE / "checkpoints/taew2_1.pth").exists():
        pytest.skip("TAEHV checkpoint unavailable")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/experiments"))
    script = importlib.import_module("offloading_generate")
    spec = dict(dim=128, eps=1e-6, ffn_dim=256, freq_dim=256, in_dim=16,
                num_heads=2, num_layers=2, out_dim=16, text_len=8)
    torch.manual_seed(32)
    weights = WanModel(**spec, text_dim=32).state_dict()

    class Text:
        def __init__(self, *args, **kwargs):
            self.encoder = SimpleNamespace(blocks=[torch.nn.Linear(2, 2)])

        def __call__(self, prompts):
            return torch.ones(1, 8, 32)

        def close(self):
            self.encoder = None

    monkeypatch.setattr(script, "TextEncoder", Text)
    monkeypatch.setattr(script, "model_config", lambda: spec)
    monkeypatch.setattr(script, "self_forcing_weights", lambda: weights)
    monkeypatch.setattr(script, "WanModel", lambda **kw: WanModel(**kw, text_dim=32))
    common = dict(latent_frames=6, saved_video_frames=21, compile=None,
                  height=32, width=32, context_frames=12, sink_frames=3,
                  positions="window", attention="cudnn", context_noise=0,
                  cross_cache=True, pc_recent=1, shift=5., fps=16)
    config = dict(common=common, execution=dict(resource_budgets=dict(vram_gib=7.25),
                                               require_compilation=False, record_metadata=False))
    policy = dict(t5=dict(ram=2, gpu=2), dit=dict(ram=2, gpu=2))
    script.run(config, dict(prompt="test", seed=4), policy, tmp_path / "baseline")
    if placement == "managed":
        policy["resources"] = dict(mode="managed", backing_ram_gib=16 * 1024 / 2**30,
                                   vram_cache_gib=0.0002, working_ram_mib=32,
                                   working_vram_mib=32, system_ram_reserve_gib=0,
                                   process_rss_limit_gib=8, disk_reserve_gib=0, staging_mib=1)
    else:
        policy["resources"] = dict(self_kv="disk", cross_kv="disk", decoder_weights="disk",
                                   decoder_state="disk", history="disk", latents="disk",
                                   staging_mib=1)
    script.run(config, dict(prompt="test", seed=4), policy, tmp_path / "offloaded")
    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))
    original = read(tmp_path / "baseline/record.json")
    actual = read(tmp_path / "offloaded/record.json")
    assert [row["latent_hash"] for row in actual["chunks"]] == [row["latent_hash"] for row in original["chunks"]]
    assert [row["decoded_hash"] for row in actual["chunks"]] == [row["decoded_hash"] for row in original["chunks"]]
    assert actual["written_frames"] == 21
    assert read(tmp_path / "offloaded/complete.json")["status"] == "complete"
    manifest = read(tmp_path / "offloaded/latents.json")
    restored = torch.cat([torch.load(tmp_path / "offloaded" / name, weights_only=True)
                          for name in manifest["chunks"]], dim=manifest["dim"])
    assert torch.equal(restored, torch.load(tmp_path / "baseline/latents.pt", weights_only=True))
    assert not list((tmp_path / "offloaded/resource_backing").iterdir())
    assert actual["backing_peak_bytes"]["disk"] > 0
    if placement == "managed":
        assert actual["backing_peak_bytes"]["ram"] <= 16 * 1024
        assert any(row["event"] == "spill" for row in actual["tensor_resource_events"])
    assert any(row["resource"].startswith("self_kv/") for row in actual["tensor_resource_events"])

    class FailingDecoder(script.TAEHVStream):
        def _decode(self, latents):
            raise ValueError("injected decoder failure")

    monkeypatch.setattr(script, "TAEHVStream", FailingDecoder)
    with pytest.raises(ValueError, match="injected decoder failure"):
        script.run(config, dict(prompt="test", seed=4), policy, tmp_path / "failed")
    assert not list((tmp_path / "failed/resource_backing").iterdir())
    assert not (tmp_path / "failed/complete.json").exists()
    assert (tmp_path / "failed/record.json").exists()
