"""Configuration and package boundaries, specified before the harness refactor."""

import pytest

from wan_dit_lab.harness.config import HarnessConfig, load_config
from wan_dit_lab.harness.streaming import stream_summary
from wan_dit_lab.harness import runner


def test_base_config_is_typed_and_override_does_not_mutate_input():
    override = {"number": 4, "metrics": {"wall": False}}
    cfg = load_config(override)
    assert cfg.number == 4 and cfg.repeat > 0
    assert cfg.metrics.memory and cfg.metrics.latency
    assert not cfg.metrics.wall
    assert cfg.memory.vram_fraction is None
    assert override == {"number": 4, "metrics": {"wall": False}}


@pytest.mark.parametrize("config", [
    {"number": 0}, {"repeat": -1}, {"warmup": -1},
    {"metrics": {"typo": True}}, {"memory": {"sample_interval_s": 0}},
    {"memory": {"vram_fraction": 1.1}}, {"unrecognized": True},
])
def test_bad_configuration_fails_before_gpu_access(config):
    with pytest.raises(ValueError):
        load_config(config)


def test_disabled_probes_are_not_executed_and_result_is_not_written(monkeypatch):
    monkeypatch.setattr(runner, "bench_events", lambda *a, **kw: [2., 4.])
    def forbidden(*a, **kw):
        raise AssertionError("Disabled probe was executed")
    for name in ("metadata", "memory_probe", "append_row", "limit_memory", "bench_wall"):
        monkeypatch.setattr(runner, name, forbidden)
    result = runner.run("test", lambda: None, config={
        "warmup": 0, "number": 2, "repeat": 1,
        "metrics": {"memory": False, "metadata": False, "wall": False},
        "output": {"path": None}})
    assert result["p50_ms"] == 3.
    assert "memory" not in result and "wall" not in result and "gpu" not in result
    assert result["config"]["number"] == 2


def test_stream_metric_groups_can_be_disabled_without_losing_deliveries():
    cfg = HarnessConfig(metrics={"memory": False, "disk_io": False, "readiness": False})
    record = {"chunks": [{"index": 0, "frames": 12, "delivery_s": 2.}]}
    result = stream_summary(record, warmup=0, config=cfg)
    assert result["steady_fps"] == 6.
    assert "rss_peak_gib" not in result
    assert "disk_read_gib" not in result
    assert "resource_readiness" not in result


def test_memory_names_separate_baseline_current_peak_and_global_free():
    from wan_dit_lab.harness.memory import summarize_memory

    rows = [dict(allocated=1, reserved=4, peak_allocated=3, peak_reserved=5,
                 device_free=12, device_used=20, rss=10, ram_available=50),
            dict(allocated=2, reserved=6, peak_allocated=5, peak_reserved=7,
                 device_free=8, device_used=24, rss=15, ram_available=40)]
    result = summarize_memory(rows)
    assert result["vram"]["baseline_bytes"] == 1
    assert result["vram"]["current_bytes"] == 2
    assert result["vram"]["peak_bytes"] == 5
    assert result["vram"]["reserved_peak_bytes"] == 7
    assert result["vram"]["free_min_bytes"] == 8
    assert result["ram"]["baseline_bytes"] == 10
    assert result["ram"]["peak_bytes"] == 15
    assert result["ram"]["free_min_bytes"] == 40
    assert summarize_memory([])["ram"]["peak_bytes"] is None


def test_compare_obeys_the_same_probe_switches(monkeypatch):
    monkeypatch.setattr(runner, "paired_benchmarks", lambda *a, **kw: {
        label: {"samples_ms": [time] * 30, "p50_ms": time}
        for label, time in (("original", 2.), ("triton", 1.))})
    def forbidden(*a, **kw):
        raise AssertionError("Disabled probe was executed")
    for name in ("memory_probe", "profile_kernels", "bench_wall"):
        monkeypatch.setattr(runner, name, forbidden)
    result = runner.compare({"original": lambda: None, "triton": lambda: None}, config={
        "metrics": {"memory": False, "kernels": False, "wall": False}})
    assert result["performance_gate"]["passed"]
    assert result["performance_gate"]["speedup"] == 2.
    assert result["memory"] == result["kernels"] == result["wall"] == {}


def test_empty_test_list_disables_subprocess_and_failed_checks_stop_run(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from wan_dit_lab.harness.checks import run_tests

    calls = []
    monkeypatch.setattr("wan_dit_lab.harness.checks.subprocess.run", lambda *a, **kw:
                        calls.append(a) or SimpleNamespace(returncode=1, stdout="1 failed"))
    result = run_tests({"checks": {"tests": []}}, tmp_path / "empty.log")
    assert result["status"] == "disabled" and not calls
    with pytest.raises(RuntimeError, match="tests"):
        run_tests({"checks": {"tests": ["tests/test_harness.py"]}}, tmp_path / "failed.log")


def test_missing_cuda_profiler_events_are_retried_and_never_reported_as_zero(monkeypatch):
    import torch
    from types import SimpleNamespace
    from wan_dit_lab.harness.profiler import profile_kernels

    attempts = []
    class Profile:
        def __enter__(self):
            attempts.append(True)
            return self
        def __exit__(self, *args):
            pass
        def events(self):
            return [] if len(attempts) == 1 else [SimpleNamespace(
                device_type=torch.autograd.DeviceType.CUDA, is_user_annotation=False, name="real_kernel")]
    monkeypatch.setattr(torch.profiler, "profile", lambda **kwargs: Profile())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    result = profile_kernels(lambda: None)
    assert result["count"] == 1 and result["attempts"] == 2
    monkeypatch.setattr(Profile, "events", lambda self: [])
    def unavailable(*args):
        raise RuntimeError("capture unavailable")
    monkeypatch.setattr("wan_dit_lab.harness.profiler.graph_kernels", unavailable)
    with pytest.raises(RuntimeError, match="CUDA events"):
        profile_kernels(lambda: None)


def test_missing_timeline_can_use_measured_cuda_graph_nodes(monkeypatch):
    import torch
    from wan_dit_lab.harness.profiler import profile_kernels

    class Profile:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def events(self): return []
    monkeypatch.setattr(torch.profiler, "profile", lambda **kw: Profile())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr("wan_dit_lab.harness.profiler.graph_kernels", lambda fn: ["kernel", "kernel"])
    result = profile_kernels(lambda: None)
    assert result["count"] == 2 and result["source"] == "cuda_graph_nodes"


@pytest.mark.gpu
def test_cuda_graph_counts_actual_kernel_nodes():
    import torch
    from wan_dit_lab.harness.cuda_graph import graph_kernels

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x = torch.randn(1024, device="cuda")
    fn = lambda: x.sin().cos()
    fn()
    assert len(graph_kernels(fn)) == 2


def test_events_mode_does_not_capture_cuda_graphs(monkeypatch):
    import torch
    from wan_dit_lab.harness import timing

    def forbidden():
        raise AssertionError("Graph capture was disabled")
    monkeypatch.setattr(torch.cuda, "CUDAGraph", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(timing, "bench_events", lambda *a, **kw: [1.])
    result = timing.paired_benchmarks({"one": lambda: None}, config={
        "warmup": 1, "number": 12, "repeat": 1, "timing": {"mode": "cuda_events", "settle_ms": 0}})
    assert result["one"]["timing"] == "cuda_events"
