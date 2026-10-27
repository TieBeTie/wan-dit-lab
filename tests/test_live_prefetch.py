"""Fresh calibration of the two-slot Live offloader, separate from steady FPS."""

import pytest
import torch

from wan_dit_lab.offloading.live_prefetch import calibration_key, summarize_prefetch, TracedStreamedRows


def test_source_workload_and_runtime_changes_invalidate_profile():
    first = calibration_key({"kernel": "v1"}, {"height": 512}, {"torch": "2.10"})
    assert first == calibration_key({"kernel": "v1"}, {"height": 512}, {"torch": "2.10"})
    assert first != calibration_key({"kernel": "v2"}, {"height": 512}, {"torch": "2.10"})
    assert first != calibration_key({"kernel": "v1"}, {"height": 480}, {"torch": "2.10"})
    assert first != calibration_key({"kernel": "v1"}, {"height": 512}, {"torch": "2.8"})


def test_summary_distinguishes_gpu_lateness_from_wait_event_overhead():
    rows = [dict(block=0, copy_ms=3., compute_ms=10., dependency_wait_ms=.002, late_ms=-7.),
            dict(block=1, copy_ms=5., compute_ms=12., dependency_wait_ms=1., late_ms=1.)]
    result = summarize_prefetch(rows)
    assert result["consumers"] == 2 and result["late_consumers"] == 1
    assert result["late_ms_total"] == 1.
    assert result["copy"]["p50_ms"] == 4.
    assert result["compute"]["max_ms"] == 12.
    assert result["per_block"]["0"]["late_consumers"] == 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_two_slot_calibration_reuses_correct_rows_and_stops_before_measurement():
    rows = torch.stack([torch.full((256,), i, dtype=torch.uint8) for i in range(4)])
    blocks = [None] * 4
    offload = TracedStreamedRows(blocks, "cuda", range(4), rows, lambda row: row)
    offload.start()
    seen = []
    for i in range(12):
        block = offload.get_next()
        seen.append(block.clone())
        offload.release(block)
    torch.cuda.synchronize()
    profile = offload.finish_calibration()
    for i, block in enumerate(seen):
        assert torch.equal(block.cpu(), rows[i % 4])
    assert profile["consumers"] == 12
    assert len(profile["samples"]) == 12
    assert all(r["copy_ms"] >= 0 and r["compute_ms"] >= 0 for r in profile["samples"])
    for i in range(4):
        block = offload.get_next()
        seen = block.clone()
        offload.release(block)
        assert torch.equal(seen.cpu(), rows[i])
    assert len(offload.samples) == 12


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tracing_restarted_mid_stream_skips_untimed_copies_and_drains_per_chunk():
    rows = torch.stack([torch.full((256,), i, dtype=torch.uint8) for i in range(4)])
    offload = TracedStreamedRows([None] * 4, "cuda", range(4), rows, lambda row: row)
    offload.start()
    offload.tracing = False
    for _ in range(3):  # copies of blocks 3, 0, 1 issued untimed
        offload.release(offload.get_next())
    offload.samples.clear()
    offload.tracing = True
    for _ in range(8):
        block = offload.get_next()
        offload.release(block)
    torch.cuda.synchronize()
    drained = offload.drain()
    assert 0 < len(drained) < 8 and not offload.samples
    assert all(r["copy_ms"] >= 0 for r in drained)
