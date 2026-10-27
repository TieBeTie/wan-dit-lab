import ast

import pytest

from wan_dit_lab.experiments.live_rope_metrics import LiveMetrics, instrument_live_source


def test_live_fps_counts_actual_frames_excludes_warmup_and_times_the_display():
    now = [10.]
    metrics = LiveMetrics(1, 2, "same", clock=lambda: now[0])
    metrics.record_chunk(10., 29, "same")
    for stamp in (10.5, 11., 12.5):
        metrics.record_frame(stamp)
    now[0] = 12.
    metrics.record_chunk(2., 32, "same")
    now[0] = 14.
    metrics.record_chunk(2., 32, "same")
    result = metrics.summary()
    assert metrics.complete
    assert result["generated_fps"] == 16.
    assert result["displayed_fps"] == .75
    assert result["frames"] == 64 and result["measured_chunks"] == 2
    assert result["prompt_unchanged"] is True


def test_prompt_change_is_recorded_and_incomplete_runs_cannot_claim_success():
    metrics = LiveMetrics(1, 2, "initial")
    metrics.record_chunk(1., 32, "changed")
    assert not metrics.complete
    with pytest.raises(ValueError, match="incomplete"):
        metrics.summary()


def test_instrumentation_has_exact_anchors_and_does_not_edit_the_original_file():
    source = '''
def generate():
    state["chunk_s"] = 2.
def tick():
    shown["frame"] = frames.get_nowait()
'''
    tree = instrument_live_source(source)
    calls = [node.func.id for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert "__live_chunk" in calls and "__live_frame" in calls and "__live_done" in calls
    with pytest.raises(ValueError, match="anchors"):
        instrument_live_source("def tick():\n    pass\n")


def test_tick_timing_wraps_the_work_and_not_the_rescheduling():
    source = '''
def generate():
    state["chunk_s"] = 2.
def tick():
    shown["frame"] = frames.get_nowait()
    work()
    root.after(10, tick)
'''
    tree = instrument_live_source(source, ticks=True)
    tick = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "tick")
    lines = [ast.unparse(statement) for statement in tick.body]
    assert lines[1] == "__tick_begin = __live_clock()"
    assert lines[-2] == "__live_tick(__tick_begin, frames.qsize())"
    assert lines[-1] == "root.after(10, tick)"


def test_display_summary_reads_tick_work_and_queue():
    now = [0.]
    metrics = LiveMetrics(1, 1, "p", clock=lambda: now[0])
    now[0] = 1.
    metrics.record_chunk(1., 32, "p")
    metrics.record_tick(1.5, 0)
    metrics.record_tick(1.6, 3)
    for stamp in (1.5, 1.6, 1.7):
        metrics.record_frame(stamp)
    now[0] = 3.
    metrics.record_chunk(2., 32, "p")
    display = metrics.display_summary()
    assert display["ticks"] == 2 and display["queue_empty_ticks"] == 1 and display["queue_mean"] == 1.5
