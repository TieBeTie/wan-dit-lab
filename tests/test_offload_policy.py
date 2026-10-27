import importlib.util
from pathlib import Path

import pytest

from wan_dit_lab.offloading.offload_policy import select_policy


def calibration():
    record = dict(
        resource_events=[],
        node_times=[dict(node="decode_8", duration_s=2)],
        memory_samples=[
            dict(stage="text", allocated=100, peak_allocated=100, rss=100),
            dict(stage="decode_8", allocated=100, peak_allocated=100, rss=100),
        ],
    )
    for phase, blocks in [("t5", 24), ("dit", 30)]:
        record[f"{phase}_block_bytes"] = 10
        record[f"{phase}_trace"] = [
            dict(
                block=b,
                node="text" if phase == "t5" else "denoise_8",
                pack_s=0.1,
                copy_s=h,
                compute_s=c,
            )
            for b in range(blocks)
            for h, c in [(0.2, 1), (0.4, 0.5)]
        ]
    return record


def test_profiles_use_same_samples_but_opposite_tail_directions():
    median = select_policy([calibration()], "median", 130, 130)
    guarded = select_policy([calibration()], "guarded", 130, 130)
    assert median["dit"]["costs"]["copy_s"][0] == pytest.approx(0.3)
    assert median["dit"]["costs"]["compute_s"][0] == 0.75
    assert guarded["dit"]["costs"]["copy_s"][0] == 0.4
    assert guarded["dit"]["costs"]["compute_s"][0] == 0.5
    assert guarded["dit"]["estimated_peak_gpu"] <= 130
    assert guarded["dit"]["estimated_peak_ram"] <= 130


def test_memory_constraint_can_reject_all_candidates():
    with pytest.raises(RuntimeError, match="fits"):
        select_policy([calibration()], "median", 1, 130)


def test_missing_calibration_is_not_silently_replaced():
    with pytest.raises(ValueError):
        select_policy([], "median", 100, 100)


def workflow_module():
    path = Path(__file__).resolve().parents[1] / "scripts/experiments/offloading_experiment.py"
    spec = importlib.util.spec_from_file_location("offloading_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_holm_and_exact_output_gate():
    module = workflow_module()
    assert module.holm(dict(a=0.01, b=0.04)) == dict(a=0.02, b=0.04)
    original = dict(
        context_hash="x", written_frames=9, chunks=[dict(latent_hash="a", decoded_hash="b")]
    )
    module.matched(original, original)
    changed = original | dict(chunks=[dict(latent_hash="z", decoded_hash="b")])
    with pytest.raises(RuntimeError, match="changed"):
        module.matched(original, changed)


def test_report_waits_for_all_independent_pairs(tmp_path, monkeypatch):
    import csv
    import json

    module = workflow_module()
    methods = ["fixed_2x2", "median", "guarded"]
    config = dict(
        execution=dict(order=methods),
        evaluation=[dict(id=str(i)) for i in range(24)],
        inventory=dict(categories=["weights", "state"]),
    )
    monkeypatch.setattr(module, "summarize", lambda r: dict(full_run_s=r["full_run_s"]))
    with (tmp_path / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "label", *methods])
        writer.writerow(["full_run_s", "time", "", "", ""])

    def add(i):
        for method, duration in [("fixed_2x2", 100), ("median", 90), ("guarded", 100)]:
            folder = tmp_path / "evaluation" / str(i) / method
            folder.mkdir(parents=True)
            record = dict(
                full_run_s=duration,
                context_hash="same",
                written_frames=480,
                chunks=[dict(latent_hash="same", decoded_hash="same")],
                resource_events=[],
                inventory=[],
            )
            (folder / "record.json").write_text(json.dumps(record))
            (folder / "complete.json").write_text("{}")

    for i in range(23):
        add(i)
    module.report(config, tmp_path)
    assert not (tmp_path / "statistics.json").exists()
    add(23)
    module.report(config, tmp_path)
    results = json.loads((tmp_path / "statistics.json").read_text())
    assert results["median"]["reject_for_benefit"]
    assert results["guarded"]["reject_null"] and not results["guarded"]["reject_for_benefit"]
