"""Local RoPE correctness/performance gate, sharing the project's GPU harness."""

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

from wan_dit_lab import harness
from wan_dit_lab.harness.checks import block_accuracy_gate, error_metrics
from wan_dit_lab.kernels.rope_triton import rope_apply as fused_rope
from wan_dit_lab.models.rope import rope
from wan_dit_lab.models.rope import rope_apply as original_rope
from wan_dit_lab.reports.rope_report import render_html, render_markdown, write_latest
from wan_dit_lab.runtime.pipeline import compile_settings, resolve_experiment

ROOT = Path(__file__).resolve().parents[3]


def frequencies(dim, positions=128):
    pairs = dim // 2
    spatial = pairs // 3
    return torch.cat([rope(positions, 2 * n) for n in
                      (pairs - 2 * spatial, spatial, spatial)], dim=-1).cuda()


def make_block_pair(dim, ffn_dim, heads, dtype):
    """Load two isolated attention namespaces. Never patch the user's module."""
    from wan_dit_lab.models.block import WanBlock

    blocks = {}
    for label, apply in (("original", original_rope), ("triton", fused_rope)):
        name = f"_rope_validation_attention_{label}"
        spec = importlib.util.spec_from_file_location(
            name, ROOT / "src/wan_dit_lab/models/multi_head_attention.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        module.rope_apply = apply
        blocks[label] = WanBlock(dim=dim, ffn_dim=ffn_dim, num_heads=heads, eps=1e-6,
                                 attention=module.MultiHeadAttention).cuda().to(dtype).eval()
        blocks[label].requires_grad_(False)
    blocks["triton"].load_state_dict(blocks["original"].state_dict(), strict=True)
    return blocks








def compile_functions(functions, settings):
    compiled, records = {}, {}
    for name, fn in functions.items():
        before = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        compiled[name] = torch.compile(fn, fullgraph=True, dynamic=False, **settings)
        start = time.perf_counter()
        compiled[name]()
        torch.cuda.synchronize()
        records[name] = dict(
            first_call_wall_s=time.perf_counter() - start,
            unique_graphs=torch._dynamo.utils.counters["stats"]["unique_graphs"] - before,
            fullgraph=True, settings=settings,
            cache="existing compiler cache may be reused")
        if records[name]["unique_graphs"] < 1:
            raise RuntimeError(f"No new compiled graph for {name}")
    return compiled, records


def precision_limit(dtype):
    return dict(rtol=2 * torch.finfo(dtype).eps, atol=2 * torch.finfo(dtype).eps)


def gradient_probe(x, freq, grids, start):
    # A fixed small grid makes this safe on every performance case, including B2.
    value = x[:, :8].detach().requires_grad_()
    small_grids = ((1, 2, 4),) * value.shape[0]
    actual = fused_rope(value, small_grids, freq, start)
    reference = original_rope(value, small_grids, freq, start)
    probe = torch.randn_like(actual)
    grad = torch.autograd.grad(actual, value, probe)[0]
    grad_ref = torch.autograd.grad(reference, value, probe)[0]
    value64 = value.detach().double().requires_grad_()
    output64 = original_rope(value64, small_grids, freq.double(), start)
    grad64 = torch.autograd.grad(output64, value64, probe.double())[0]
    # Original low-precision autograd rounds each branch before accumulation.
    tolerance = dict(rtol=4 * torch.finfo(x.dtype).eps, atol=4 * torch.finfo(x.dtype).eps)
    torch.testing.assert_close(grad, grad_ref, **tolerance)
    return dict(against_original=error_metrics(grad, grad_ref),
                against_fp64=error_metrics(grad, grad64),
                original_against_fp64=error_metrics(grad_ref, grad64), shape=list(value.shape),
                tolerance=tolerance, scope="small gradient probe, not training throughput")


def measure(functions, config):
    cfg = harness.load_config(config.get("harness"))
    compiled, compilation = compile_functions(functions, compile_settings(config["pipeline"]))
    before = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    result = harness.compare(compiled, config=cfg)
    new_graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"] - before
    if new_graphs:
        raise RuntimeError("Recompilation occurred during warmed measurement")
    return result | dict(compilation=compilation, measured_new_graphs=new_graphs)


def operator_case(case, config):
    torch.manual_seed(config["seed"])
    dtype = getattr(torch, case["dtype"])
    grid = tuple(case["grid"])
    grids = (grid,) * case["batch"]
    x = torch.randn(case["batch"], math.prod(grid), case["heads"], case["dim"],
                    device="cuda", dtype=dtype)
    freq = frequencies(case["dim"], max(128, max(grid) + 16))
    start = 7
    with torch.inference_mode():
        functions = dict(original=lambda: original_rope(x, grids, freq, start),
                         triton=lambda: fused_rope(x, grids, freq, start))
        actual, original = functions["triton"](), functions["original"]()
        # FP64 real oracle is the unchanged original evaluated on FP64 inputs.
        real64 = original_rope(x.double(), grids, freq, start)
        torch.testing.assert_close(actual, original, **precision_limit(dtype))
        metrics = dict(against_original=error_metrics(actual, original),
                       against_fp64=error_metrics(actual, real64),
                       original_against_fp64=error_metrics(original, real64),
                       norm_drift=error_metrics(actual.double().norm(dim=-1),
                                                x.double().norm(dim=-1)),
                       original_norm_drift=error_metrics(original.double().norm(dim=-1),
                                                         x.double().norm(dim=-1)))
        del actual, original, real64
        result = measure(functions, config | {"harness": config["harness"] | {"repeat": case.get("repeat", config["harness"]["repeat"])}})
        compiled_reference = torch.compile(functions["original"], fullgraph=True,
                                           dynamic=False, **compile_settings(config["pipeline"]))
        compiled_candidate = torch.compile(functions["triton"], fullgraph=True,
                                           dynamic=False, **compile_settings(config["pipeline"]))
        actual, original = compiled_candidate(), compiled_reference()
        torch.testing.assert_close(actual, original, **precision_limit(dtype))
        metrics["compiled_against_compiled_original"] = error_metrics(actual, original)
        real64 = original_rope(x.double(), grids, freq, start)
        metrics["compiled_against_fp64"] = error_metrics(actual, real64)
        metrics["compiled_original_against_fp64"] = error_metrics(original, real64)
        del actual, original, real64
    with torch.enable_grad(), torch.inference_mode(False):
        # Input/frequencies created under inference_mode cannot be saved for backward.
        result["gradient"] = gradient_probe(x.clone(), freq.clone(), grids, start)
    result["numerical"] = metrics
    result["case"] = case
    useful_bytes = 2 * x.numel() * x.element_size()
    for value in result["timing"].values():
        value["tokens_per_s"] = case["batch"] * math.prod(grid) * 1000 / value["p50_ms"]
        value["useful_io_gb_per_s"] = useful_bytes / (value["p50_ms"] * 1e6)
        value["bandwidth_scope"] = "input+output bytes only, not measured DRAM traffic"
    if result["kernels"]["triton"]["count"] != 1:
        raise RuntimeError("Single-kernel forward contract failed")
    return result


def block_case(case, config):
    torch.manual_seed(config["seed"])
    dtype, grid = getattr(torch, case["dtype"]), tuple(case["grid"])
    with torch.inference_mode():
        blocks = make_block_pair(case["dim"], case["ffn_dim"], case["heads"], dtype)
        x = torch.randn(1, math.prod(grid), case["dim"], device="cuda", dtype=dtype)
        time_input = torch.randn(1, 6, case["dim"], device="cuda", dtype=dtype)
        context = torch.randn(1, 16, case["dim"], device="cuda", dtype=dtype)
        freq = frequencies(case["dim"] // case["heads"])
        functions = {label: (lambda block=block: block(
            x, time_input, grid_sizes=(grid,), freqs=freq, context=context, start_frame=7))
                     for label, block in blocks.items()}
        # Force the same cuDNN-independent SDPA math choice for both variants.
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            result = measure(functions, config)
            compiled = {name: torch.compile(fn, fullgraph=True, dynamic=False, **compile_settings(config["pipeline"]))
                        for name, fn in functions.items()}
            actual, original = compiled["triton"](), compiled["original"]()
            result["numerical"] = error_metrics(actual, original)
        eager_original, eager_candidate = functions["original"](), functions["triton"]()
        torch.testing.assert_close(eager_candidate, eager_original, rtol=0, atol=0)
        result["eager_numerical"] = error_metrics(eager_candidate, eager_original)
        float_block = copy.deepcopy(blocks["original"]).float()
        precision = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("highest")
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                fp32 = float_block(x.float(), time_input.float(), grid_sizes=(grid,),
                                   freqs=freq, context=context.float(), start_frame=7)
        finally:
            torch.set_float32_matmul_precision(precision)
        result["accuracy_gate"] = block_accuracy_gate(
            actual, original, fp32, config["harness"]["checks"]["max_block_relative_error_increase"])
        result["fp32_reference"] = "same quantized weights promoted to FP32, math SDPA, highest matmul precision"
    result["case"] = case
    result["scope"] = "local WanBlock, identical random weights, PyTorch Flash SDPA, no checkpoint/video"
    return result


def local_path(value):
    if os.name != "nt" and len(value) > 2 and value[1] == ":":
        return Path("/mnt") / value[0].lower() / value[3:].replace("\\", "/")
    return Path(value)


def write_report(report, directory):
    (directory / "results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    (directory / "Report.md").write_text(render_markdown(report), encoding="utf-8")
    (directory / "Report.html").write_text(render_html(report), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/rope_triton.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tests-only", action="store_true")
    args = parser.parse_args()
    config = resolve_experiment(json.loads(args.config.read_text(encoding="utf-8")))
    directory = args.output or local_path(config["output_dir"]) / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", config=config, operators=[],
                  started_utc=datetime.now(timezone.utc).isoformat())
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required. A skipped GPU suite is not a passing validation.")
        report["metadata"] = harness.metadata()
        try:
            import resource

            report["process_peak_rss_bytes_at_start"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        except ImportError:
            pass
        sources = [str(path.relative_to(ROOT)).replace("\\", "/")
                   for path in sorted((ROOT / "src/wan_dit_lab").rglob("*.py"))]
        sources += ["configs/harness.json", "tests/test_rope_triton.py", "tests/test_harness_config.py",
                    "tests/test_pipeline.py", "configs/experiments/rope_triton.json"]
        report["source_sha256"] = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                                    for path in sources}
        print(f"Results: {directory}", flush=True)
        from wan_dit_lab.harness.checks import run_tests

        report["tests"] = run_tests(config["harness"], directory / "tests.log")
        report["tests_exit_code"] = 0
        print("Correctness tests passed", flush=True)
        if not args.tests_only:
            for case in config["cases"]:
                print(f"Measuring {case['name']}", flush=True)
                report["operators"].append(operator_case(case, config))
                write_report(report, directory)
                torch._dynamo.reset()
                torch.cuda.empty_cache()
            print("Measuring complete local WanBlock", flush=True)
            report["block"] = block_case(config["block"], config)
            all_rows = report["operators"] + [report["block"]]
            report["performance_passed"] = all(row["performance_gate"]["passed"] for row in all_rows)
        # Hash before/after establishes that the original source was never edited.
        assert report["source_sha256"]["src/wan_dit_lab/models/rope.py"] == hashlib.sha256(
            (ROOT / "src/wan_dit_lab/models/rope.py").read_bytes()).hexdigest()
        if "block" in report and not report["block"]["accuracy_gate"]["passed"]:
            report["status"] = "accuracy_regression"
        else:
            report["status"] = "passed" if report.get("performance_passed", True) else "regression"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        print(report["error"], file=sys.stderr, flush=True)
    finally:
        try:
            import resource

            report["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            report["ram_scope"] = "whole runner process high-water mark, not per-operator RAM"
        except ImportError:
            pass
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_report(report, directory)
        write_latest(directory, report["status"])
    print(f"Status: {report['status']}\nReport: {directory / 'Report.html'}", flush=True)
    return 0 if report["status"] == "passed" else 1
