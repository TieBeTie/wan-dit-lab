"""The local report is compact, honest about missing data, and safe to open."""

from wan_dit_lab.reports.rope_report import render_html, render_markdown, write_latest


def sample_report():
    row = dict(
        case=dict(name="video_b1_bf16", batch=1, grid=[21, 30, 52], dtype="bfloat16"),
        timing={label: dict(p50_ms=p50, p90_ms=p50 + .1, p99_ms=p50 + .2)
                for label, p50 in (("original", 4.), ("triton", 2.))},
        wall={"original": dict(p50_ms=4.2), "triton": dict(p50_ms=2.2)},
        memory={"original": dict(additional_peak_allocated_bytes=480 * 2**20),
                "triton": dict(additional_peak_allocated_bytes=96 * 2**20)},
        kernels={"original": dict(count=2), "triton": dict(count=1)},
        performance_gate=dict(passed=True, speedup=2.), numerical={})
    return dict(status="passed", config=dict(rounds=30, summary_case="video_b1_bf16"),
                metadata=dict(environment=dict(gpu="RTX 4060")),
                tests=dict(passed=80, failed=0, skipped=0), operators=[row])


def test_compact_report_has_two_comparable_columns_and_consistent_speedup():
    report = sample_report()
    actual = render_markdown(report)
    assert "| Метрика | PyTorch | Triton |" in actual
    assert "4.000" in actual and "2.000" in actual
    assert "×2.00" in actual
    assert "480.0" in actual and "96.0" in actual
    assert "80 тестов" in actual
    assert "results.json" in actual


def test_unmeasured_counters_never_become_zero_and_partial_failures_stay_visible():
    report = sample_report()
    report.update(status="failed", error="A numerical check failed")
    actual = render_markdown(report)
    assert "Ошибка" in actual and "A numerical check failed" in actual
    assert "не измерены" in actual
    assert "нет данных" in actual
    assert "Пройдено" not in actual


def test_html_escapes_error_text_and_has_no_remote_runtime_dependencies():
    report = sample_report()
    report.update(status="failed", error="<script>alert('x')</script>")
    actual = render_html(report)
    assert "<script>" not in actual
    assert "&lt;script&gt;" in actual
    assert "<html lang=\"ru\"" in actual
    assert "<table>" in actual and "<details>" in actual
    assert "<script src" not in actual and "@import" not in actual


def test_latest_points_to_the_last_run_even_if_it_failed(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    write_latest(first, "passed")
    write_latest(second, "failed")
    import json

    pointer = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert pointer["status"] == "failed"
    assert pointer["run_dir"] == str(second)


def test_live_table_uses_measured_fps_instead_of_extrapolating_block_speed():
    report = sample_report()
    report["live"] = dict(status="passed", settings=dict(height=512, width=928),
                          original=dict(summary=dict(generated_fps=12., displayed_fps=11.)),
                          triton=dict(summary=dict(generated_fps=13., displayed_fps=12.)))
    actual = render_markdown(report)
    assert "Live" in actual and "512×928" in actual
    assert "12.000" in actual and "13.000" in actual


def test_cache_store_section_keeps_previous_live_and_distinguishes_its_baseline():
    report = sample_report()
    report["cache_store"] = dict(timing={"original": {"p50_ms": 1.}, "triton": {"p50_ms": .2}},
                                kernels={"original": {"count": 18}, "triton": {"count": 1}},
                                memory={}, wall={}, numerical={})
    report["cache_store_live"] = dict(settings=dict(height=512, width=928),
        original=dict(summary=dict(generated_fps=13., displayed_fps=9.),
                      calibration=dict(profiles=[dict(consumers=216, late_consumers=0)])),
        triton=dict(summary=dict(generated_fps=14., displayed_fps=9.),
                    calibration=dict(profiles=[dict(consumers=216, late_consumers=0)])))
    for actual in (render_html(report), render_markdown(report)):
        assert "FP8 KV" in actual and "×5.00" in actual
        assert "RoPE Triton" in actual
        assert "13.000" in actual and "14.000" in actual
        assert "0 / 216" in actual


def test_common_memory_metrics_are_visible_when_actually_measured():
    from wan_dit_lab.reports.rope_report import live_rows

    memory = dict(vram=dict(baseline_bytes=2**30, current_bytes=2**30, peak_bytes=2 * 2**30,
                           reserved_peak_bytes=3 * 2**30, free_min_bytes=4 * 2**30),
                  ram=dict(baseline_bytes=5 * 2**30, current_bytes=6 * 2**30,
                           peak_bytes=7 * 2**30, free_min_bytes=8 * 2**30))
    rows = live_rows({"original": {"memory": memory}})
    labels = [row[0] for row in rows]
    assert any("RAM" in label and "исходная" in label for label in labels)
    assert any("RAM" in label and "свободная" in label for label in labels)
    assert any("RSS" in label for label in labels)
