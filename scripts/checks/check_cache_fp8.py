"""Local TDD gate, paired cache-store benchmark, then optional real Live A/B."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
import hashlib
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests-only", action="store_true")
    args = parser.parse_args()
    config_path = ROOT / "configs/experiments/cache_fp8_live.json"
    from wan_dit_lab.runtime.pipeline import resolve_experiment

    config = resolve_experiment(json.loads(config_path.read_text(encoding="utf-8")))
    report_dir = Path(config["report_dir"])
    directory = report_dir / "cache_store" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory.mkdir(parents=True)
    from wan_dit_lab.harness.checks import run_tests

    tests = run_tests(config.get("harness"), directory / "tests.log")
    if args.tests_only:
        print(tests, flush=True)
        return 0

    import torch
    from wan_dit_lab import harness
    from wan_dit_lab.kernels.cache_fp8 import store_kv_fp8
    from wan_dit_lab.generation.causal import KVCache
    from wan_dit_lab.reports.rope_report import render_html, render_markdown

    torch.manual_seed(23)
    # LongLive2 5B, eight latent frames, VAE stride 16, patch size 2.
    tokens = 8 * (config["height"] // 32) * (config["width"] // 32)
    k = torch.randn(1, tokens, 24, 128, device="cuda", dtype=torch.float32)
    v = torch.randn(k.shape, device="cuda", dtype=torch.bfloat16)
    caches = {name: KVCache(4 * tokens, fp8=True) for name in ("original", "triton")}
    for cache in caches.values():
        cache.allocate(v)
    previous, candidate = caches.values()
    functions = dict(original=lambda: previous.store(tokens, k, v),
                     triton=lambda: store_kv_fp8(k, v, candidate.k, candidate.ks,
                                               candidate.v, candidate.vs, tokens))
    for function in functions.values():
        function()
    for name in ("k", "ks", "v", "vs"):
        a, b = (getattr(cache, name)[:, tokens:2*tokens] for cache in caches.values())
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), name
    row = dict(case=dict(shape=list(k.shape), k_dtype=str(k.dtype), v_dtype=str(v.dtype)), bitwise=True,
               metadata=harness.metadata(),
               sources={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                        ("src/wan_dit_lab/generation/causal.py", "src/wan_dit_lab/kernels/cache_fp8.py")})
    row["tests"] = tests
    row["artifacts"] = {name: str((directory / filename).relative_to(report_dir)).replace("\\", "/")
                        for name, filename in (("tests", "tests.log"), ("results", "results.json"))}
    row.update(harness.compare(functions, config=config.get("harness")))
    row["status"] = "passed" if row["performance_gate"]["passed"] else "regression"
    (directory / "results.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    report = json.loads((report_dir / "results.json").read_text(encoding="utf-8"))
    report["cache_store"] = row
    (report_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (report_dir / "Report.html").write_text(render_html(report), encoding="utf-8")
    (report_dir / "Report.md").write_text(render_markdown(report), encoding="utf-8")
    for name, result in row["timing"].items():
        print(f"{name}: {result['p50_ms']:.4f} ms, {row['kernels'][name]['count']} kernels", flush=True)
    if row["status"] != "passed":
        return 1
    # the store inside Live is now a kernel of its AOTInductor block packages: Live itself is timed by live.exe bench
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
