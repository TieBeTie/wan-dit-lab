"""trace_chunks: the chunks, the three passes of a chunk and the idle time of the card read from the kernel list of an
Nsight Systems trace (docs/report-2026-11-16-bottlenecks.md, section on the counters)."""

import pytest

from wan_dit_lab.harness import trace_chunks as tc


def test_busy_time_is_the_union_of_overlapping_intervals():
    assert tc.busy([(0, 10), (5, 12), (20, 25)]) == 17


def test_busy_time_of_nothing_is_zero():
    assert tc.busy([]) == 0


def test_idle_time_is_the_span_minus_the_busy_time():
    assert tc.idle([(0, 10), (15, 20)], 0, 20) == 5


def test_idle_time_only_counts_the_asked_window():
    assert tc.idle([(0, 10), (15, 20), (30, 40)], 5, 25) == 10


def test_anchors_are_split_into_chunks_and_passes_in_order():
    anchors = list(range(2 * 3 * 4))  # 2 chunks, 3 passes, 4 blocks
    parts = tc.split(anchors, blocks=4, passes=3)
    assert len(parts) == 2 and len(parts[0]) == 3
    assert parts[0][0] == [0, 1, 2, 3]
    assert parts[1][2] == [20, 21, 22, 23]


def test_a_number_of_anchors_that_does_not_fit_is_refused():
    with pytest.raises(ValueError):
        tc.split(list(range(25)), blocks=4, passes=3)


def test_chunk_and_pass_times_come_from_the_first_anchors_of_the_next_unit():
    # anchors every 10 ns, 4 blocks, 3 passes, 2 chunks; the second chunk starts 5 ns late
    anchors = [i * 10 + (5 if i >= 12 else 0) for i in range(24)]
    parts = tc.split(anchors, blocks=4, passes=3)
    chunk_ns, pass_ns = tc.durations(parts, end=300)
    assert chunk_ns == [125, 300 - 125]
    assert pass_ns[0] == [40, 40, 45]
    assert pass_ns[1] == [40, 40, 300 - 205]


def test_kernel_time_by_name_counts_only_kernels_that_start_inside_the_window():
    kernels = [(0, 10, "a"), (12, 15, "b"), (20, 30, "a"), (40, 45, "a")]
    assert tc.by_name(kernels, 10, 40) == {"b": (1, 3), "a": (1, 10)}


# The moments wan.exe calls Probe::ready, read from the driver API calls of the generating thread
# (docs/report-2026-11-16-bottlenecks.md, "Причина среднего fps 17.0").

def _decode(t, copies=3):
    """start_decode: `copies` cuMemcpyDtoHAsync, then the stream sync of debug "transfers" from t + 10 to t + 20."""
    return [(t + i, t + i + 1, "cuMemcpyDtoHAsync_v2") for i in range(copies)] + [(t + 10, t + 20, "cuStreamSynchronize")]


def test_ready_follows_the_stream_sync_after_each_decode_and_the_flush_is_the_last_event_sync():
    calls = ([(-50, -49, "cuMemcpyDtoHAsync_v2"), (-40, -30, "cuStreamSynchronize")]  # a lone copy is no decode
             + _decode(100) + [(195, 196, "cuEventSynchronize")] + _decode(200)
             + [(295, 296, "cuEventSynchronize")] + _decode(300) + [(330, 331, "cuEventSynchronize")])
    moments, flush = tc.ready_moments(calls, copies=3)
    assert moments == [120, 220, 320]
    assert flush == 331


def test_without_a_stream_sync_ready_follows_the_last_copy_call():
    calls = [(0, 1, "cuMemcpyDtoHAsync_v2"), (2, 3, "cuMemcpyDtoHAsync_v2"), (50, 60, "cuEventSynchronize")]
    assert tc.ready_moments(calls, copies=2) == ([3], 60)


def test_the_metric_counts_the_flush_frames_over_the_flush_gap():
    # 4 chunks of 10 s, the first one trimmed to 7 frames; the flush follows the last decode by 1 s
    moments, flush = [10, 20, 30, 40], 41
    m = tc.metric_samples(moments, flush, [7, 8, 8, 8])
    assert m["warmup"] == {"frames": 7, "seconds": None}
    assert m["samples"] == [(8, 10), (8, 10), (8, 1)]
    assert m["metric"] == {"frames": 24, "seconds": 21, "fps": 24 / 21}
    # the honest mean: the frames of the chunks computed in the counted time, the flush dropped
    assert m["computed"] == {"frames": 16, "seconds": 20, "fps": 16 / 20}
    assert m["last_chunk"] == {"frames": 8, "seconds": 10, "fps": 0.8}


def test_the_mean_over_full_windows_starts_at_the_asked_chunk():
    m = tc.metric_samples([10, 20, 32, 44], 45, [7, 8, 8, 8], full_from=3)
    assert m["full_window"] == {"frames": 8, "seconds": 12, "fps": 8 / 12}
