"""Pipeline behavior, independent of Rust's internal types and of a CUDA installation."""
import copy
import json
from pathlib import Path

import pytest

from wan_dit_lab.runtime.live_pipeline import export_identity, load_pipeline, prepare, publish, reusable

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def pipeline():
    return json.loads((ROOT / "configs/pipeline.proposed.json").read_text(encoding="utf-8"))


def save(tmp_path, pipeline):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(pipeline), encoding="utf-8")
    return path


def test_the_approved_config_is_the_supported_pipeline(pipeline, tmp_path):
    export, run = load_pipeline(save(tmp_path, pipeline))
    assert export == pipeline[0] and run == pipeline[1]


@pytest.mark.parametrize("debug", [False, True, "fps", ["fps"], "memory", "transfers", "prompts", ["fps", "memory", "transfers", "prompts"]])
@pytest.mark.parametrize("every", ["5s", "10s", "0.5s", "end"])
def test_debug_and_drain_are_generation_settings(pipeline, tmp_path, debug, every):
    pipeline[1].update(debug=debug, drain={"every": every})
    _, run = load_pipeline(save(tmp_path, pipeline))
    assert run["debug"] == debug and run["drain"]["every"] == every


def test_generation_settings_do_not_change_the_export_identity(pipeline, tmp_path):
    # Use tiny real files, so invalidation tests exercise content identity without any checkpoints.
    source = tmp_path / "weights"
    source.write_bytes(b"weights")
    export = pipeline[0]
    export["model"] = {"architecture": str(source), "checkpoint": str(source), "block_weights": str(source),
                       "decoder": str(source), "text_encoder": {"checkpoint": str(source), "block_weights": str(source), "tokenizer": str(source)}}
    before = export_identity(export, tmp_path, sources=[source], versions={"torch": "2.10"})
    pipeline[1]["seed"] = 9
    pipeline[1]["prompts"]["initial"] = "another scene"
    pipeline[1]["offloading"]["calibration"] = "always"
    assert export_identity(export, tmp_path, sources=[source], versions={"torch": "2.10"}) == before
    source.write_bytes(b"changed")
    assert export_identity(export, tmp_path, sources=[source], versions={"torch": "2.10"}) != before


def test_kernel_and_compiler_changes_invalidate_the_packages(pipeline, tmp_path):
    export = pipeline[0]
    export["model"] = {}
    before = export_identity(export, tmp_path, sources=[], versions={"torch": "2.10"})
    export["kernels"]["cross_attention"] = ["efficient"]
    assert export_identity(export, tmp_path, sources=[], versions={"torch": "2.10"}) != before
    export["kernels"]["cross_attention"] = ["cudnn", "efficient"]
    assert export_identity(export, tmp_path, sources=[], versions={"torch": "2.11"}) != before


def test_prepare_reuses_only_complete_matching_packages_and_propagates_failure(pipeline, tmp_path):
    export = pipeline[0]
    export["output"] = str(tmp_path / "packages")
    calls = []

    def compile_ok(out):
        calls.append("export")
        out.mkdir(exist_ok=True)
        (out / "text.pt2").write_bytes(b"compiled")
        (out / "text.safetensors").write_bytes(b"weights")
        publish(out, {"kernels": export["kernels"], "checks": {"text": {}}}, "identity")

    prepare(export, "identity", compile_ok)
    prepare(export, "identity", compile_ok)
    assert calls == ["export"]
    (Path(export["output"]) / "text.pt2").unlink()
    prepare(export, "identity", compile_ok)
    assert calls == ["export", "export"]
    # Corruption of the same byte length must not be accepted as a cache hit.
    (Path(export["output"]) / "text.pt2").write_bytes(b"corrupt!")
    prepare(export, "identity", compile_ok)
    assert calls == ["export", "export", "export"]

    def fail(out):
        raise RuntimeError("export failed")

    with pytest.raises(RuntimeError, match="export failed"):
        prepare(export, "different identity", fail)


def test_a_failed_export_leaves_the_working_packages_intact(pipeline, tmp_path):
    # On 2026-11-23 an export that failed at block_2 had already overwritten text..block_1 of the working packages:
    # the packages are built aside and replace the working ones only once complete.
    export = pipeline[0]
    export["output"] = str(tmp_path / "packages")

    def compile_as(identity, body):
        def compile_packages(out):
            out.mkdir(exist_ok=True)
            (out / "text.pt2").write_bytes(body)
            (out / "text.safetensors").write_bytes(b"weights")
            if body == b"half":
                raise RuntimeError("block_2 did not compile")
            publish(out, {"kernels": export["kernels"], "checks": {"text": {}}}, identity)
        return compile_packages

    prepare(export, "old", compile_as("old", b"working"))
    with pytest.raises(RuntimeError, match="block_2"):
        prepare(export, "new", compile_as("new", b"half"))
    assert reusable(export["output"], "old")
    assert (Path(export["output"]) / "text.pt2").read_bytes() == b"working"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["packages"], "no half-built folder is left behind"
    prepare(export, "new", compile_as("new", b"rebuilt"))
    assert reusable(export["output"], "new")
    assert (Path(export["output"]) / "text.pt2").read_bytes() == b"rebuilt"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["packages"]


def test_comments_and_docstrings_do_not_change_the_export_identity(pipeline, tmp_path):
    # On 2026-11-23 a changed comment in multi_head_attention.py made every package stale.
    export = pipeline[0]
    export["model"] = {}
    source = tmp_path / "attention.py"
    versions = {"torch": "2.10"}
    source.write_text('def f(q, k):\n    # softmax(QK^T / sqrt(d)) V\n    return q @ k\n', encoding="utf-8")
    before = export_identity(export, tmp_path, sources=[source], versions=versions)
    source.write_text('def f(q, k):\n    """Scores."""\n    # softmax((QK^T)/sqrt(d))\n\n    return q @ k  # scores\n',
                      encoding="utf-8")
    assert export_identity(export, tmp_path, sources=[source], versions=versions) == before
    source.write_text('def f(q, k):\n    return q @ k.T\n', encoding="utf-8")
    assert export_identity(export, tmp_path, sources=[source], versions=versions) != before


def test_read_only_preparation_never_launches_an_export(pipeline, tmp_path):
    export = pipeline[0]
    export["output"] = str(tmp_path / "missing")
    export["rebuild"] = "never"
    with pytest.raises(ValueError, match="export"):
        prepare(export, "identity", lambda: pytest.fail("unexpected export"))


def test_text_encoder_uses_the_checkpoint_and_tokenizer_requested_by_export(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from wan_dit_lab.models import text_encoder as module
    seen = {}
    monkeypatch.setattr(module, "HuggingfaceTokenizer", lambda **kw: seen.update(tokenizer=kw["name"]))
    monkeypatch.setattr(module, "mapped_torch_checkpoint", lambda path: seen.update(checkpoint=path) or {})
    model = SimpleNamespace(norm=SimpleNamespace(to=lambda device: "norm"),
                            load_state_dict=lambda *a, **kw: None, eval=lambda: None)
    monkeypatch.setattr(module, "umt5_xxl", lambda **kw: model)
    module.TextEncoder("cpu", checkpoint=tmp_path / "t5.pth", tokenizer=tmp_path / "tokenizer")
    assert seen == {"checkpoint": tmp_path / "t5.pth", "tokenizer": str(tmp_path / "tokenizer")}


def test_dense_t5_rows_preserve_original_parameters_and_embeddings(monkeypatch, tmp_path):
    import torch
    from types import SimpleNamespace
    from wan_dit_lab.models import text_encoder as module
    torch.manual_seed(11)
    blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4).to(torch.bfloat16) for _ in range(2)])
    base = SimpleNamespace(tokenizer=object(), norm=torch.nn.Identity(),
                           encoder=SimpleNamespace(blocks=blocks, token_embedding=torch.nn.Embedding(5, 4)))
    monkeypatch.setattr(module, "TextEncoder", lambda *a, **kw: base)
    checkpoint = tmp_path / "t5.pth"
    checkpoint.write_bytes(b"T5 identity")
    import unittest.mock
    with unittest.mock.patch("shutil.disk_usage", return_value=SimpleNamespace(free=10**12)):
        encoder = module.FastTextEncoder("cpu", weights_file=tmp_path / "t5.bf16", checkpoint=checkpoint,
                                         precision="bfloat16")
    rows = encoder._map().view(encoder.blocks, encoder.per_block)
    x = torch.randn(2, 4, dtype=torch.bfloat16)
    for original, row in zip(blocks, rows):
        actual = encoder.slot(row)
        assert type(actual) is torch.nn.Linear
        assert torch.equal(actual(x), original(x))
        assert actual.weight.dtype == torch.bfloat16
    checkpoint.write_bytes(b"another T5")
    with pytest.raises(ValueError, match="checkpoint"):
        module.FastTextEncoder("cpu", weights_file=tmp_path / "t5.bf16", checkpoint=checkpoint, precision="bfloat16")


def test_text_precision_is_an_export_setting_not_a_file_path(pipeline, tmp_path):
    pipeline[0]["model"]["text_encoder"]["precision"] = "bfloat16"
    assert load_pipeline(save(tmp_path, pipeline))[0]["model"]["text_encoder"]["precision"] == "bfloat16"
    source = tmp_path / "weights"
    source.write_bytes(b"weights")
    export = pipeline[0]
    export["model"] = {"checkpoint": str(source), "text_encoder": {"precision": "bfloat16", "checkpoint": str(source)}}
    before = export_identity(export, tmp_path, sources=[], versions={})
    export["model"]["text_encoder"]["precision"] = "int8"
    assert export_identity(export, tmp_path, sources=[], versions={}) != before


@pytest.mark.skipif(not Path("C:/BuildTools/2022/VC/Auxiliary/Build/vcvars64.bat").exists(), reason="MSVC integration")
def test_preparation_bootstraps_the_compiler_without_importing_torch():
    import runpy
    module = runpy.run_path(str(ROOT / "scripts/conversion/prepare_live.py"))
    env = module["compiler_environment"]()
    assert any((Path(p) / "cl.exe").exists() for p in env["PATH"].split(";"))
    assert Path(env["CUDA_HOME"]).exists()
    assert "cudart.lib" in env["LINK"]


def test_disk_kv_and_physical_ram_reserve(pipeline, tmp_path):
    pipeline[1]["offloading"]["ram_margin_fraction"] = 0.125
    pipeline[1]["offloading"]["resources"]["kv_cache"] = "disk"
    _, run = load_pipeline(save(tmp_path, pipeline))
    assert run["offloading"]["ram_margin_fraction"] == 0.125


@pytest.mark.parametrize("profile", [{"stretch": .2992905106483072}, {"stretch": .3, "ntk_aware": True}, {"stretch": .3, "by_parts": {"alpha": .25, "beta": 2}}])
def test_spatial_rope_profiles_round_trip(pipeline, tmp_path, profile):
    pipeline[0]["rope"].update(interpolate=True, **profile)
    export, _ = load_pipeline(save(tmp_path, pipeline))
    assert export["rope"] == pipeline[0]["rope"]


def test_a_file_unchanged_in_size_and_time_is_not_hashed_again(tmp_path, monkeypatch):
    # Like a layer cache: the digest of a file is remembered with its size and modification time, and read again
    # only when either changed. Without it every start of Live read every weight file from the disk.
    from wan_dit_lab.runtime import live_pipeline
    monkeypatch.setattr(live_pipeline, "DIGEST_CACHE", tmp_path / "digests.json")
    file = tmp_path / "weights.bin"
    file.write_bytes(b"a" * 1000)
    first = live_pipeline.digest(file)
    opened = []
    real_open = Path.open
    monkeypatch.setattr(Path, "open", lambda self, *a, **k: (opened.append(self), real_open(self, *a, **k))[1])
    assert live_pipeline.digest(file) == first and opened == []
    # a changed content (size) is read again
    file.write_bytes(b"b" * 1001)
    opened.clear()
    assert live_pipeline.digest(file) != first and [p for p in opened if p == file] == [file]
    # the same size with the time moved on: read again, since the content may differ
    opened.clear()
    old = live_pipeline.digest(file)
    file.write_bytes(b"c" * 1001)
    opened.clear()
    import os
    os.utime(file, ns=(file.stat().st_atime_ns, file.stat().st_mtime_ns + 5_000_000_000))
    assert live_pipeline.digest(file) != old and [p for p in opened if p == file] == [file]


def test_a_broken_digest_cache_is_ignored_and_rebuilt(tmp_path, monkeypatch):
    from wan_dit_lab.runtime import live_pipeline
    cache = tmp_path / "digests.json"
    monkeypatch.setattr(live_pipeline, "DIGEST_CACHE", cache)
    cache.write_text("{not json", encoding="utf-8")
    file = tmp_path / "x.bin"
    file.write_bytes(b"x")
    expected = __import__("hashlib").sha256(b"x").hexdigest()
    assert live_pipeline.digest(file) == expected
    assert json.loads(cache.read_text(encoding="utf-8"))[str(file.resolve())]["sha256"] == expected
