"""Pilot reporting must not mislabel weight traffic or pool unpaired results."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def pilot(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/experiments"))
    return importlib.import_module("offloading_resource_pilot")


def test_aggregate_preserves_worst_peaks_and_minimum_free_memory(pilot):
    rows = [dict(time_s=2, ram_rss_peak_gib=3, vram_device_min_free_gib=2),
            dict(time_s=8, ram_rss_peak_gib=9, vram_device_min_free_gib=.5)]
    assert pilot.aggregate(rows) == dict(time_s=5, ram_rss_peak_gib=9,
                                        vram_device_min_free_gib=.5)


def test_resource_metrics_keep_weight_trace_separate(pilot, monkeypatch):
    monkeypatch.setattr(pilot, "summarize", lambda record: dict(
        resource_deadline_misses=2, resource_wait_total_s=1,
        resource_wait_max_s=.5, ram_to_vram_gib=5,
    ))
    record = dict(tensor_resource_events=[
        dict(event="save", bytes=1024**3, duration_s=2),
        dict(event="restore", bytes=2 * 1024**3, duration_s=3),
    ], backing_peak_bytes=dict(ram=1024**3, disk=2 * 1024**3))
    row = pilot.metrics(record)
    assert row["weights_ram_to_vram_gib"] == 5
    assert "ram_to_vram_gib" not in row
    assert row["resource_save_restore_wall_s"] == 5
    assert row["resource_restore_logical_gib"] == 2
    assert row["backing_ram_peak_gib"] == 1


def test_report_rejects_mismatched_pairs_before_writing_comparison(pilot, monkeypatch, tmp_path):
    config = dict(policies=dict(a={}, b={}), evaluation=[dict(id="01")])
    for method in ("a", "b"):
        folder = tmp_path / "evaluation/01" / method
        folder.mkdir(parents=True)
        (folder / "complete.json").write_text('{}', encoding="utf-8")
        (folder / "record.json").write_text('{}', encoding="utf-8")

    def mismatch(a, b):
        raise RuntimeError("mismatch")

    monkeypatch.setattr(pilot, "matched", mismatch)
    with pytest.raises(RuntimeError, match="mismatch"):
        pilot.report(config, tmp_path)
    assert not (tmp_path / "comparison.csv").exists()


def test_low_disk_space_blocks_launch(pilot, monkeypatch, tmp_path):
    from types import SimpleNamespace

    monkeypatch.setattr(pilot.shutil, "disk_usage", lambda path: SimpleNamespace(free=512 * 1024**2))
    with pytest.raises(RuntimeError, match="Low disk space"):
        pilot.check_disk_space(tmp_path, 3)
    pilot.check_disk_space(tmp_path, .25)


def test_stop_worker_tree_kills_only_its_descendants(pilot, monkeypatch):
    from types import SimpleNamespace

    events = []
    child = SimpleNamespace(kill=lambda: events.append("child"))
    root = SimpleNamespace(children=lambda recursive: [child], kill=lambda: events.append("root"))
    monkeypatch.setattr(pilot.psutil, "Process", lambda pid: root)
    worker = SimpleNamespace(pid=123, wait=lambda: events.append("wait"))
    pilot.stop_worker_tree(worker)
    assert events == ["root", "child", "wait"]
