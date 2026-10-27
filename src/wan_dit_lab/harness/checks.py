"""Numerical metrics and deterministic paired regression gate for operator tests."""

import math
import os
import random
import re
import statistics
import subprocess
import sys
from pathlib import Path

import torch


def run_tests(config, log, *, env=None):
    """One configured local gate. A disabled gate is not a passing test suite."""
    from .config import ROOT, load_config

    cfg = load_config(config)
    if not cfg.checks.tests:
        return dict(status="disabled", passed=None)
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONIOENCODING="utf-8")
    environment.update(env or {})
    result = subprocess.run([sys.executable, "-m", "pytest", *cfg.checks.tests, "-q", "--tb=short"],
                            cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Local tests failed: {log}")
    return dict(status="passed", **{name: int(m[1]) if (m := re.search(rf"(\d+) {name}", result.stdout)) else 0
                                    for name in ("passed", "failed", "skipped")})


def error_metrics(actual, reference):
    if actual.shape != reference.shape:
        raise ValueError("Compared tensors must have the same shape")
    actual, reference = actual.detach().double(), reference.detach().double()
    finite = bool((torch.isfinite(actual).all() & torch.isfinite(reference).all()).item())
    if not finite:
        return dict(finite=False, max_abs=None, rmse=None, relative_l2=None,
                    mismatched_elements=int((actual != reference).sum().item()),
                    elements=actual.numel())
    difference = actual - reference
    norm = reference.norm().item()
    return dict(finite=True,
                max_abs=difference.abs().max().item() if difference.numel() else 0.,
                rmse=difference.square().mean().sqrt().item() if difference.numel() else 0.,
                relative_l2=difference.norm().item() / max(norm, 1e-30),
                mismatched_elements=int((actual != reference).sum().item()),
                elements=actual.numel())


def regression_check(reference_ms, candidate_ms, max_regression, min_rounds=12):
    if len(reference_ms) != len(candidate_ms):
        raise ValueError("Regression gate requires paired samples")
    if len(reference_ms) < min_rounds:
        raise ValueError(f"At least {min_rounds} measured rounds required")
    if any(not math.isfinite(v) or v <= 0 for v in reference_ms + candidate_ms):
        raise ValueError("Durations must be finite and positive")
    ratios = [candidate / reference for reference, candidate in zip(reference_ms, candidate_ms)]
    rng = random.Random(0)
    bootstrap = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(2000))
    median = statistics.median(ratios)
    ratio_of_medians = statistics.median(candidate_ms) / statistics.median(reference_ms)
    return dict(passed=max(median, ratio_of_medians) <= 1 + max_regression,
                max_latency_regression=max_regression, paired_median_ratio=median,
                ratio_of_medians=ratio_of_medians, speedup=1 / ratio_of_medians,
                paired_speedup=1 / median, ratio_bootstrap_95=[bootstrap[49], bootstrap[1949]],
                rounds=len(ratios), method="paired ratios, deterministic percentile bootstrap",
                scope="one local run, not an independent multi-machine study")


def block_accuracy_gate(candidate, baseline, fp32_reference, max_relative_increase):
    """Check integration against an independent precision reference, not bitwise BF16.

    Permit a configured relative increase of existing L2 error, with an FP32
    numerical floor. Bound maximum error separately by one low-precision step
    at the reference's largest magnitude. Never hide NaN/Inf behind an average.
    """
    actual = error_metrics(candidate, fp32_reference)
    previous = error_metrics(baseline, fp32_reference)
    if not actual["finite"] or not previous["finite"]:
        return dict(passed=False, candidate=actual, baseline=previous)
    l2_limit = previous["relative_l2"] * (1 + max_relative_increase) + 1e-7
    peak_limit = previous["max_abs"] + torch.finfo(candidate.dtype).eps * (
        fp32_reference.abs().max().item())
    return dict(passed=actual["relative_l2"] <= l2_limit and actual["max_abs"] <= peak_limit,
                candidate=actual, baseline=previous, relative_l2_limit=l2_limit,
                max_abs_limit=peak_limit, max_relative_error_increase=max_relative_increase,
                scope="engineering integration tolerance, not a threshold from RoFormer")
