"""Prepare reproducible results and PR bodies from the measured upstream runs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import json
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/experiments/2026-10-24/01_Upstream_RoPE"


def table(report):
    grouped = {}
    for row in report["rows"]:
        grouped.setdefault(row["case"], {})[row["variant"]] = row
    lines = ["| Case | Original eager, ms | Patched eager, ms | Original compiled, ms | Patched compiled, ms | Compiled speedup |",
             "|---|---:|---:|---:|---:|---:|"]
    for case, rows in grouped.items():
        a, b = rows["baseline_compiled"], rows["candidate_compiled"]
        values = [rows[k]["median_ms"] for k in ("baseline","candidate","baseline_compiled","candidate_compiled")]
        match = re.match(r"rope_\((\d+), (\d+), (\d+)\)_torch\.(\w+)", case)
        label = ("RoPE, " + "×".join(match.groups()[:3]) + ", " +
                 {"float16":"FP16","bfloat16":"BF16","float32":"FP32"}[match.group(4)]
                 if match else "Upstream DiT block, 3×30×52")
        lines.append("| " + label + " | " + " | ".join(f"{v:.3f}" for v in values)
                     + f" | {a['median_ms']/b['median_ms']:.3f}x |")
    return "\n".join(lines)


def main():
    for version in ("21", "22"):
        repo = ROOT / "reference/upstream" / ("Wan2.1" if version == "21" else "Wan2.2")
        windows = OUT / ("wan21_final.json" if version == "21" else "wan22_windows.json")
        linux = OUT / f"wan{version}_linux.json"
        reports = {"Windows / SDPA": json.loads(windows.read_text(encoding="utf-8"))}
        if linux.exists():
            reports = {"WSL2 Linux / FlashAttention": json.loads(linux.read_text(encoding="utf-8")), **reports}
        for report in reports.values():
            if report["errors"]:
                raise ValueError(f"Unresolved benchmark errors: {report['errors']}")
            assert all(r["compiled_graphs"] > 0 for r in report["rows"] if r["variant"].endswith("_compiled"))
        lines = ["# Uniform-grid RoPE measurements", "", "GPU: NVIDIA GeForce RTX 4060 Laptop (8 GiB).",
                 "Seed 123; deterministic random block weights; no checkpoint or generated-video benchmark.",
                 "Every entry is a median of 12 shuffled rounds, two calls per sample, after three warmup calls.",
                 "Both original and patched compiled variants use default torch.compile with dynamic=False.", ""]
        for platform, report in reports.items():
            lines += [f"## {platform}", "", f"PyTorch {report['torch']}, CUDA {report['cuda']}. "
                      f"FlashAttention: {report.get('attention_version') or 'not used'}.", "", table(report), ""]
            errors = report.get("compiled_pair_error", {})
            if errors:
                lines += [f"Maximum absolute error, patched compiled vs original compiled: {max(errors.values()):.8g}.", ""]
        lines += ["The eager path retains the original implementation. The new real-valued path is selected only during compilation for uniform grids.",
                  "Heterogeneous batches retain the original per-sample path. Complex frequency tables, padding and FP32 outputs are preserved.", "",
                  "Both numerical variants use the same disabled host-metadata read so PyTorch 2.8 can compile the tensor work. The unmodified Tensor.tolist path failed with aten._local_scalar_dense and fell back to eager; those preliminary runs are not presented as compiled timings.", "",
                  "The block test instantiates the real upstream WanAttentionBlock: 1.3B dimensions for Wan2.1, 5B dimensions for Wan2.2. Grid (3,30,52) is a 3-frame latent chunk. Isolated RoPE also tests grid (21,30,52).",
                  "Windows block results explicitly substitute SDPA for FlashAttention in BOTH variants; the production attention source is unchanged.", "",
                  "FP32-output differences between eager and compiled blocks are recorded separately in raw JSON. They must not be confused with differences caused by this patch.",
                  "CUDA peaks in raw JSON include all live inputs, two blocks and compiler allocations, not just temporary RoPE workspace.",
                  "Single GPU only. No end-to-end video, pretrained-quality, distributed or older-PyTorch claim is made.", "",
                  "[Raw samples, errors, graph counts and memory](rope_benchmark_results.json)", "",
                  "```bash", "python -m pytest tests/test_rope.py -q",
                  "python tests/benchmark_rope.py --attention flash --output /tmp/rope_benchmark.json", "```", ""]
        (repo / "tests/rope_benchmark_results.md").write_text("\n".join(lines),encoding="utf-8")
        (repo / "tests/rope_benchmark_results.json").write_text(json.dumps(reports,indent=2),encoding="utf-8")
        primary = next(iter(reports.values()))
        block = [r for r in primary["rows"] if r["case"].startswith("upstream")]
        by = {r["variant"]:r for r in block}
        a, b = by["baseline_compiled"]["median_ms"], by["candidate_compiled"]["median_ms"]
        body = ("## Problem\n\nThe existing RoPE implementation reads tensor grid sizes with Tensor.tolist, then applies complex multiplication per sample. "
                "PyTorch 2.8 fails to lower the metadata read (aten._local_scalar_dense), and Inductor warns that complex operator code generation is unsupported on the tested versions. Frequency construction with torch.polar is precomputed and is not the timed operation here.\n\n"
                "## Change\n\nAdd a real-valued, vectorized path for uniform grids during torch.compile. "
                "Keep the existing eager and heterogeneous-batch arithmetic, frequency format, autocast policy, padding and FP32 output. Read grid metadata at an explicit Python boundary to avoid the PyTorch 2.8 scalar-lowering failure. "
                "No new runtime dependencies or changes to attention, checkpoints or model configuration.\n\n"
                "## Validation\n\nCheckpoint-free tests cover FP16/BF16/FP32, CPU/CUDA, uniform and mixed grids, padding, noncontiguous input, "
                "gradients, compiled uniform/heterogeneous paths, and changing grid values without freezing metadata. All 28 tests pass on WSL2 / PyTorch 2.8 / CUDA. "
                "The benchmark uses the real upstream block with identical deterministic random weights before/after.\n\n"
                "## Measurements\n\nBoth numerical variants use the same host-metadata boundary. This compares compiled tensor arithmetic, not a compiled candidate against an unnoticed eager fallback.\n\n" + table(primary) + "\n\n"
                f"On this GPU the compiled block median changes from {a:.3f} to {b:.3f} ms ({a/b:.3f}x). "
                "This is a block microbenchmark, not an end-to-end video speedup. "
                "Measurements are specific to the listed GPU/software/input shapes; this does not claim a universal 3.4x model speedup.\n\n"
                "See `tests/rope_benchmark_results.md` and `.json` for the environment, all samples, errors, graph counts, memory and Windows/SDPA cross-check.\n\n"
                "## Reproduce\n\n```bash\npython -m pytest tests/test_rope.py -q\n"
                "python tests/benchmark_rope.py --attention flash --output /tmp/rope_benchmark.json\npython tests/dump_rope_codegen.py --output /tmp/rope_codegen\n```\n\n"
                "Full-video generation and multi-GPU inference have not been tested. "
                "I would welcome feedback on whether this compile-only fast path fits the project's preferred approach.\n")
        (OUT / f"wan{version}_pr_body.md").write_text(body,encoding="utf-8")


if __name__ == "__main__":
    main()
