import math

import pytest
import torch

from wan_dit_lab import harness
from wan_dit_lab.harness.checks import block_accuracy_gate, error_metrics, regression_check


def test_error_metrics_are_finite_even_at_zero_reference():
    zero = torch.zeros(4)
    exact = error_metrics(zero, zero)
    assert exact["max_abs"] == exact["relative_l2"] == exact["rmse"] == 0
    assert exact["mismatched_elements"] == 0
    changed = error_metrics(torch.ones(4), zero)
    assert changed["max_abs"] == changed["rmse"] == 1
    assert changed["mismatched_elements"] == 4
    assert all(math.isfinite(changed[k]) for k in ("max_abs", "relative_l2", "rmse"))


def test_nonfinite_or_shape_mismatch_never_passes_silently():
    assert error_metrics(torch.tensor([float("nan")]), torch.ones(1))["finite"] is False
    with pytest.raises(ValueError, match="shape"):
        error_metrics(torch.zeros(2), torch.zeros(3))


def test_regression_gate_uses_paired_samples_and_requires_enough_rounds():
    assert regression_check([10.] * 12, [9.] * 12, .05)["passed"]
    assert not regression_check([10.] * 12, [11.] * 12, .05)["passed"]
    with pytest.raises(ValueError, match="paired"):
        regression_check([1.] * 12, [1.], .05)
    with pytest.raises(ValueError, match="rounds"):
        regression_check([1.], [1.], .05)


def test_paired_harness_rejects_empty_or_invalid_measurement_before_cuda():
    with pytest.raises(ValueError, match="functions"):
        harness.paired_benchmarks({}, warmup=1, number=12, repeat=1)
    with pytest.raises(ValueError, match="positive"):
        harness.paired_benchmarks({"a": lambda: None}, warmup=1, number=0, repeat=1)


def test_block_accuracy_gate_has_an_independent_reference_and_fails_bad_outputs():
    reference = torch.tensor([1., 2.])
    baseline = reference + .01
    assert block_accuracy_gate(reference + .009, baseline, reference, .05)["passed"]
    assert not block_accuracy_gate(reference + .1, baseline, reference, .05)["passed"]
    assert not block_accuracy_gate(reference + .001, reference, reference, .05)["passed"]
    assert not block_accuracy_gate(torch.full_like(reference, float("nan")),
                                   baseline, reference, .05)["passed"]


def test_paired_ratios_cannot_hide_regression_of_reported_medians():
    # Pairwise ratios can cancel while the marginal medians show a slowdown.
    previous = [1., 1., 100., 100.] * 3
    candidate = [.9, .9, 110., 110.] * 3
    result = regression_check(previous, candidate, .05)
    assert not result["passed"]
    assert result["speedup"] == pytest.approx(50.5 / 55.45)


def test_wall_timer_measures_submission_and_completion_separately_from_gpu_events(monkeypatch):
    timestamps = iter([0., .006, .010, .018])
    synchronizations, calls = [], []
    monkeypatch.setattr("wan_dit_lab.harness.timing.time.perf_counter", lambda: next(timestamps))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: synchronizations.append(True))
    actual = harness.bench_wall(lambda: calls.append(True), number=2, repeat=3)
    assert actual == pytest.approx([2., 8. / 3.])
    assert len(calls) == 6
    assert len(synchronizations) == 3
