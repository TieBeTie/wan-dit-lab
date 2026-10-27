"""A small explicit plan, with invalid orders rejected before model setup."""

import pytest

from wan_dit_lab.runtime.pipeline import resolve_experiment, install


def test_mode_and_harness_are_independent_and_defaults_are_resolved():
    raw = {"run": {"mode": "live"}, "pipeline": [{"name": "compile", "mode": "default"}]}
    actual = resolve_experiment(raw)
    assert actual["run"]["mode"] == "live"
    assert actual["harness"]["metrics"]["memory"]
    assert "harness" not in raw
    assert resolve_experiment({})["run"]["mode"] == "benchmark"


@pytest.mark.parametrize("pipeline", [
    [{"name": "compile"}, {"name": "rope", "backend": "triton"}],
    [{"name": "rope", "backend": "triton"}] * 2,
    [{"name": "unknown"}],
    [{"name": "compile", "backend": "triton"}],
    [{"name": "rope", "backend": "invalid"}],
])
def test_bad_pipeline_is_rejected_before_any_install(pipeline):
    with pytest.raises(ValueError):
        resolve_experiment({"pipeline": pipeline})


def test_pipeline_installs_and_restores_even_when_user_code_fails():
    from wan_dit_lab.models import multi_head_attention as attention
    original = attention.rope_apply
    config = resolve_experiment({"pipeline": [
        {"name": "rope", "backend": "triton"}, {"name": "compile", "mode": "default"}]})
    with pytest.raises(RuntimeError, match="example"):
        with install(config["pipeline"]) as compilation:
            assert attention.rope_apply is not original
            assert compilation["mode"] == "default"
            raise RuntimeError("example")
    assert attention.rope_apply is original


def test_compile_step_does_not_send_both_mode_and_options(monkeypatch):
    import torch

    def compiler(model, **options):
        assert not ("mode" in options and "options" in options)
        return model
    monkeypatch.setattr(torch, "compile", compiler)
    for step in ({"name": "compile"}, {"name": "compile", "options": {"max_autotune": True}}):
        with install([step]) as settings:
            torch.compile(lambda: None, **settings)


def test_shared_workload_inheritance_and_cycle_detection(tmp_path):
    import json

    base = tmp_path / "base.json"
    base.write_text(json.dumps({"height": 512, "harness": {"number": 12}}))
    actual = resolve_experiment({"base": str(base), "harness": {"repeat": 5}})
    assert actual["height"] == 512
    assert actual["harness"]["number"] == 12 and actual["harness"]["repeat"] == 5
    base.write_text(json.dumps({"base": str(base)}))
    with pytest.raises(ValueError, match="cycle"):
        resolve_experiment({"base": str(base)})
