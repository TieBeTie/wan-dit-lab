"""JSON selects execution mode: the Live window or Live timed without one, both the Rust host."""

import json
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("mode,command", [("live", ["native/target/release/live.exe"]),
                                         ("benchmark", ["native/target/release/live.exe", "bench"])])
def test_run_dispatches_by_explicit_mode_without_a_harness_section(monkeypatch, tmp_path, mode, command):
    from wan_dit_lab.cli import main

    path = tmp_path / "experiment.json"
    path.write_text(json.dumps({"run": {"mode": mode}}))
    monkeypatch.setattr(sys, "argv", ["wan-lab", "run", "--config", str(path)])
    monkeypatch.setattr("wan_dit_lab.runtime.environment.gpu_environment", lambda: {})
    monkeypatch.setattr("wan_dit_lab.harness.checks.run_tests", lambda *a, **kw: {"status": "passed"})
    calls = []
    monkeypatch.setattr("wan_dit_lab.cli.subprocess.run", lambda args, **kw:
                        calls.append(args) or SimpleNamespace(returncode=0))
    with pytest.raises(SystemExit) as exit:
        main()
    assert exit.value.code == 0
    got = [str(a).replace("\\", "/") for a in calls[0]]
    assert got[0].endswith(command[0]) and got[1:len(command)] == command[1:]


def test_synthetic_smoke_reuses_the_harness_instead_of_its_own_loop(monkeypatch):
    import torch
    from wan_dit_lab.cli import benchmark

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "manual_seed", lambda seed: None)
    monkeypatch.setattr(torch, "randn", lambda *a, **kw: object())
    calls = []
    monkeypatch.setattr("wan_dit_lab.harness.runner.run", lambda *a, **kw:
                        calls.append(kw) or {"p50_ms": 1.})
    assert benchmark({"number": 3})["p50_ms"] == 1.
    assert calls[0]["config"] == {"number": 3}
