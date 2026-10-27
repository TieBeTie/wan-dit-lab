"""Tests of compare_live_rope.py, archived with it (they were in tests/test_entrypoint.py)."""

import json


def test_standalone_benchmark_creates_its_own_report(tmp_path):
    from scripts.checks.compare_live_rope import write_comparison

    records = {"original": {"summary": {"generated_fps": 14.}}}
    write_comparison(tmp_path, {"height": 512, "width": 928, "harness": {"number": 2}}, records)
    assert (tmp_path / "Report.html").exists()
    report = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert report["live"]["original"]["summary"]["generated_fps"] == 14.


def test_existing_report_sections_are_preserved(tmp_path):
    from scripts.checks.compare_live_rope import write_comparison

    root = tmp_path / "report"
    root.mkdir()
    (root / "results.json").write_text(json.dumps({"status": "passed", "existing": "keep"}))
    raw = tmp_path / "raw"
    write_comparison(raw, {"report_dir": str(root), "report_key": "new_measurement"},
                     {"original": {"summary": {"generated_fps": 14.}}})
    report = json.loads((root / "results.json").read_text(encoding="utf-8"))
    assert report["existing"] == "keep" and "new_measurement" in report


