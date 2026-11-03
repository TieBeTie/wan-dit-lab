"""Pipeline behavior, independent of Rust's internal types and of a CUDA installation."""
import copy
import json
from pathlib import Path

import pytest

from wan_dit_lab.runtime.live_pipeline import export_identity, load_pipeline, prepare, publish

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


@pytest.mark.parametrize("debug", [False, True, "fps", ["fps"]])
@pytest.mark.parametrize("every", ["5s", "10s", "0.5s", "end"])
def test_debug_and_drain_are_generation_settings(pipeline, tmp_path, debug, every):
    pipeline[1].update(debug=debug, drain={"every": every})
    _, run = load_pipeline(save(tmp_path, pipeline))
    assert run["debug"] == debug and run["drain"]["every"] == every


@pytest.mark.parametrize("setting", [
    {"debug": "unknown"}, {"debug": ["fps", "unknown"]}, {"debug": []},
    {"debug": ["fps", "fps"]}, {"debug": 1}, {"debug": None},
    *({"drain": {"every": v}} for v in
      ("never", "0s", "-5s", "NaNs", "infs", "1e999s", "", "5", "0.00000000001s", "0.0000000009s", "1e20s")),
    {"drain": {"every": "5s", "typo": True}},
])
def test_invalid_debug_or_drain_is_rejected(pipeline, tmp_path, setting):
    pipeline[1].update(setting)
    with pytest.raises(ValueError):
        load_pipeline(save(tmp_path, pipeline))


@pytest.mark.parametrize("change", [
    lambda p: p.reverse(),
    lambda p: p.append(copy.deepcopy(p[0])),
    lambda p: p[1].update(packages_from="absent"),
    lambda p: p[0]["kernels"].pop("rope_dtype"),
    lambda p: p[1]["offloading"].update(recalibrate=True),
    lambda p: p[1]["offloading"].update(calibration="unknown"),
    lambda p: p[1].update(prompts={"source": "script", "file": "a.json", "initial": "wrong"}),
    lambda p: p[0]["sampling"].update(steps=3),
    lambda p: p[0]["context"].update(chunk_frames=0),
])
def test_invalid_pipelines_fail_before_starting_processes(pipeline, tmp_path, change):
    change(pipeline)
    with pytest.raises(ValueError):
        load_pipeline(save(tmp_path, pipeline))


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

    def compile_ok():
        calls.append("export")
        out = Path(export["output"])
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

    def fail():
        raise RuntimeError("export failed")

    with pytest.raises(RuntimeError, match="export failed"):
        prepare(export, "different identity", fail)


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


def test_text_precision_rejects_unknown_values(pipeline, tmp_path):
    pipeline[0]["model"]["text_encoder"]["precision"] = "bf116"
    with pytest.raises(ValueError, match="precision"):
        load_pipeline(save(tmp_path, pipeline))


@pytest.mark.skipif(not Path("C:/BuildTools/2022/VC/Auxiliary/Build/vcvars64.bat").exists(), reason="MSVC integration")
def test_preparation_bootstraps_the_compiler_without_importing_torch():
    import runpy
    module = runpy.run_path(str(ROOT / "scripts/conversion/prepare_live.py"))
    env = module["compiler_environment"]()
    assert any((Path(p) / "cl.exe").exists() for p in env["PATH"].split(";"))
    assert Path(env["CUDA_HOME"]).exists()
    assert "cudart.lib" in env["LINK"]


@pytest.mark.parametrize("fraction", [-0.1, 1, float("nan"), float("inf")])
def test_invalid_ram_margin_rejected(pipeline, tmp_path, fraction):
    pipeline[1]["offloading"]["ram_margin_fraction"] = fraction
    with pytest.raises(ValueError):
        load_pipeline(save(tmp_path, pipeline))


def test_disk_kv_and_physical_ram_reserve(pipeline, tmp_path):
    pipeline[1]["offloading"]["ram_margin_fraction"] = 0.125
    pipeline[1]["offloading"]["resources"]["kv_cache"] = "disk"
    _, run = load_pipeline(save(tmp_path, pipeline))
    assert run["offloading"]["ram_margin_fraction"] == 0.125
