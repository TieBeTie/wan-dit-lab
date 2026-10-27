from collections import Counter

import pytest
import torch

from wan_dit_lab import harness


def test_working_mode_defaults_to_compile_without_mutating_input():
    config = {"common": {"width": 832}, "execution": {}}
    actual = harness.workload_config(config)
    assert actual["common"]["compile"] == "default"
    assert actual["execution"]["require_compilation"] is True
    assert "compile" not in config["common"]


@pytest.mark.parametrize("mode", [None, False, True, "invalid"])
def test_working_mode_rejects_silent_eager_or_invalid_modes(mode):
    with pytest.raises(ValueError, match="requires"):
        harness.workload_config({"common": {"compile": mode}})


def test_eager_diagnostic_must_opt_out_explicitly():
    config = harness.workload_config({"common": {"compile": None},
                                      "execution": {"require_compilation": False}})
    assert config["common"]["compile"] is None


def test_stream_metrics_use_actual_deliveries_and_do_not_execute_work():
    record = dict(chunks=[dict(index=0, frames=9, delivery_s=None),
                          dict(index=1, frames=12, delivery_s=4.),
                          dict(index=2, frames=12, delivery_s=6.)],
                  memory_samples=[dict(rss=2**30, private=2 * 2**30, peak_allocated=3 * 2**30)],
                  tensor_resource_events=[dict(event="write", bytes=2**30, duration_s=.25)])
    result = harness.stream_summary(record, warmup=1)
    assert result["steady_mean_s"] == result["steady_median_s"] == 5
    assert result["steady_fps"] == 2.4
    assert result["steady_p90_s"] == 5.8
    assert result["rss_peak_gib"] == 1
    assert result["disk_write_gib"] == 1
    assert result["disk_write_wall_s"] == .25
    assert harness.stream_summary(record, warmup=5) == {}


def test_single_stream_does_not_invent_quantile_samples():
    assert harness.summarize([7.])["p99_ms"] == 7
    with pytest.raises(ValueError):
        harness.summarize([])


def test_resource_readiness_distinguishes_missing_telemetry_from_zero_misses():
    record = dict(chunks=[dict(index=0, frames=12, delivery_s=1.)])
    assert harness.stream_summary(record, warmup=0)["resource_readiness"] is None
    assert harness.stream_summary(record, warmup=0)["disk_read_gib"] is None
    record["tensor_resource_events"] = [
        dict(event="consumer_ready", resource="kv/0", ready_at_request=True, duration_s=.01),
        dict(event="consumer_ready", resource="kv/1", ready_at_request=False, duration_s=.2),
        dict(event="prefetch_wait", resource="kv/1", duration_s=.15),
    ]
    result = harness.stream_summary(record, warmup=0)["resource_readiness"]
    assert result["consumers"] == 2 and result["misses"] == 1
    assert result["late_acquisition_wall_s"] == .2
    assert result["prefetch_wait_wall_s"] == .15


def test_pinned_allocator_peak_is_not_confused_with_buffer_payload():
    record = dict(chunks=[dict(index=0, frames=12, delivery_s=1.)],
        memory_samples=[dict(pinned_peak=2**30)], pinned_staging_peak_bytes=2**29)
    result = harness.stream_summary(record, warmup=0)
    assert result["pinned_peak_gib"] == 1
    assert result["pinned_payload_peak_gib"] == .5


def test_weight_gaps_separate_startup_warmup_and_steady_generation():
    def trace(node, gap):
        return dict(node=node, pack_wait_s=0., copy_ready_at_request=True,
                    dependency_wait_s=.01, compute_s=1., compute_stream_gap_s=gap)
    record = dict(chunks=[dict(index=3, frames=12, delivery_s=2.)],
                  t5_trace=[trace("text", 6.)],
                  dit_trace=[trace("text", 1.), trace("denoise_0", .5),
                             trace("denoise_3", .1), trace("denoise_4", None)])
    result = harness.stream_summary(record, warmup=3)["weight_readiness"]
    phases = result["phases"]
    assert phases["initialization"]["compute_stream_gap_s"] == 7.
    assert phases["warmup_dit"]["compute_stream_gap_s"] == .5
    assert phases["steady_dit"]["compute_stream_gap_s"] == .1
    assert phases["steady_dit"]["consumers"] == 2
    assert sum(p["compute_stream_gap_s"] for p in phases.values()) == pytest.approx(
        result["compute_stream_gap_s"])


def test_stream_harness_requires_evidence_of_actual_compilation(monkeypatch):
    stats = Counter()
    monkeypatch.setitem(torch._dynamo.utils.counters, "stats", stats)
    monkeypatch.setattr("wan_dit_lab.harness.workload.metadata", lambda device: dict(commit="test", harness_version=2))
    run = harness.WorkloadRun("stream", {"common": {}, "execution": {}},
                               {"prompt": "test", "seed": 4}, {})
    record = dict(chunks=[], memory_samples=[])
    run.finish(record)
    with pytest.raises(RuntimeError, match="no compiled graphs"):
        run.validate(record)
    stats.update(unique_graphs=2, calls_captured=12)
    run.finish(record)
    run.validate(record)
    assert record["harness"]["commit"] == "test"
    assert record["harness"]["settings"]["compile"] == "default"
    assert record["harness"]["number"] == record["harness"]["repeat"] == 1
    assert record["torch_compile"]["unique_graphs"] == 2
