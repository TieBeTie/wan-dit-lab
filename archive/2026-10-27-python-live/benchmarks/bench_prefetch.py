"""Native prefetch sweep, late-transfer correctness and a profiled cost model.

Uses existing AOT packages. Does not export models or change generation defaults.
Run from a CUDA runtime environment: python benchmarks/bench_prefetch.py.
"""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ROOT / "artifacts/wan_aot"
OUT = ROOT / "artifacts/prefetch_check"


def simulate(pack, copy, compute, window):
    """Serial pack/copy/compute engines; pinned and VRAM slot reuse dependencies."""
    packed, copied, began, finished = [], [], [], []
    for i, (p, h, c) in enumerate(zip(pack, copy, compute)):
        previous_slot = i - window
        packed.append(max(packed[-1] if i else 0,
                          copied[previous_slot] if previous_slot >= 0 else 0,
                          began[previous_slot] if previous_slot >= 0 else 0) + p)
        copied.append(max(packed[-1], copied[-1] if i else 0,
                          finished[previous_slot] if previous_slot >= 0 else 0) + h)
        began.append(max(copied[-1], finished[-1] if i else 0))
        finished.append(began[-1] + c)
    return finished[-1]


def main():
    OUT.mkdir(exist_ok=True)
    cases = [dict(id=f"w{w}_r{r}", window=w, repeat=r, passes=7, copy_delay_ms=0, pack_delay_ms=0)
             for r in (1, 2) for w in (1, 2, 3, 4)]
    cases += [dict(id="late_copy", window=2, repeat=1, passes=4, copy_delay_ms=200, pack_delay_ms=0),
              dict(id="late_pack", window=2, repeat=1, passes=4, copy_delay_ms=0, pack_delay_ms=200)]
    protocol = dict(cases=cases, warmup_passes=2, exclude_final_cleanup_pass=True,
                    grid=[9, 30, 52], spatial_resolution=[480, 832],
                    transferred_weights="FP32 (existing native manifest); compiled BF16 arithmetic",
                    correctness=dict(relative_l2_limit=0.01, max_abs_limit=0.25),
                    caveat="CUDA event intervals include stream/host launch gaps; not SM active time. "
                           "Profile disables per-block memory snapshots. Single machine, sequential sweep.",
                    runner_sha256=hashlib.sha256((PACKAGES / "wan_aot_runner.exe").read_bytes()).hexdigest())
    path = OUT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == protocol, "Existing benchmark differs; use a new output directory"
    else:
        path.write_text(json.dumps(protocol, indent=2))
    expected = np.fromfile(PACKAGES / "expected.bin", dtype=np.float32).astype(np.float64)
    env = os.environ.copy()
    torch_lib = Path(os.__file__).parent / "site-packages/torch/lib"
    # The native executable also has its original adjacent LibTorch DLLs.
    env["PATH"] = str(ROOT / "artifacts/cuda_sdk/nvidia/cuda_runtime/bin") + os.pathsep + str(torch_lib) + os.pathsep + env["PATH"]
    results = []
    for case in cases:
        folder = OUT / case["id"]
        folder.mkdir(exist_ok=True)
        record_path = folder / "result.json"
        if record_path.exists():
            result = json.loads(record_path.read_text())
        else:
            command = [str(PACKAGES / "wan_aot_runner.exe"), str(PACKAGES), str(case["passes"]),
                       "--profile", "--window", str(case["window"]), "--output-dir", str(folder),
                       "--copy-delay-ms", str(case["copy_delay_ms"]),
                       "--pack-delay-ms", str(case["pack_delay_ms"])]
            (OUT / "status.json").write_text(json.dumps(dict(phase="running", current=case["id"])))
            print("START", case["id"], flush=True)
            with (folder / "native.log").open("w") as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            errors = []
            for name in ("native.bin", "native_second.bin", "native_last.bin"):
                actual = np.fromfile(folder / name, dtype=np.float32).astype(np.float64)
                assert actual.shape == expected.shape
                difference = actual - expected
                error = dict(file=name, relative_l2=float(np.linalg.norm(difference) / np.linalg.norm(expected)),
                             max_abs=float(np.max(np.abs(difference))))
                assert error["relative_l2"] <= 0.01 and error["max_abs"] <= 0.25, error
                errors.append(error)
            all_rows, walls = [], {}
            for line in (folder / "native.log").read_text().splitlines():
                if line.startswith("{"):
                    all_rows.append(json.loads(line))
                elif line.startswith("forward_seconds"):
                    fields = line.split()
                    walls[int(fields[1])] = float(fields[2])
            timings = [row for row in all_rows if row.get("timing")]
            assert len(timings) == case["passes"] * 30
            selected = [row for row in timings if 2 <= row["pass"] < case["passes"] - 1]
            memory = [row for row in all_rows if "stage" in row][-1]
            result = dict(case=case, errors=errors, timings=timings, steady={},
                          median_forward_s=float(np.median([walls[i] for i in range(2, case["passes"] - 1)])),
                          peak_allocated_gib=memory["peak_allocated_vram_gib"],
                          peak_reserved_gib=memory["peak_reserved_vram_gib"])
            for key in ("pack_ms", "h2d_ms", "compute_ms", "gpu_ready_gap_ms",
                        "gpu_dependency_wait_ms", "host_ready_wait_ms"):
                values = [row[key] for row in selected]
                result["steady"][key] = dict(mean=float(np.mean(values)),
                                             p50=float(np.quantile(values, .5)), p99=float(np.quantile(values, .99)))
            record_path.write_text(json.dumps(result, indent=2))
        results.append(result)
        print("DONE", case["id"], result["median_forward_s"], flush=True)
    normal = [r for r in results if not r["case"]["copy_delay_ms"] and not r["case"]["pack_delay_ms"]]
    calibration = [t for r in normal if r["case"]["window"] == 2 for t in r["timings"]
                   if 2 <= t["pass"] < r["case"]["passes"] - 1]
    costs = {key: [float(np.median([t[key] for t in calibration if t["block"] == i])) for i in range(30)]
             for key in ("pack_ms", "h2d_ms", "compute_ms")}
    predictions = {w: simulate(costs["pack_ms"], costs["h2d_ms"], costs["compute_ms"], w) / 1000
                   for w in (1, 2, 3, 4)}
    observed = {w: float(np.median([r["median_forward_s"] for r in normal if r["case"]["window"] == w]))
                for w in (1, 2, 3, 4)}
    output_hashes = {str(path.relative_to(OUT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for case in cases for path in sorted((OUT / case["id"]).glob("native*.bin"))}
    assert len(output_hashes) == 3 * len(cases)
    assert len(set(output_hashes.values())) == 1, "Prefetch changed the native output"
    (OUT / "output_hashes.json").write_text(json.dumps(output_hashes, indent=2))
    summary = dict(results=results, calibration_costs=costs, model_predicted_block_chain_s=predictions,
                   observed_whole_forward_s=observed,
                   bitwise_identical_outputs=len(output_hashes),
                   note="Model covers block chain only, ideal CPU submission and median stage costs. "
                        "Observed wall time also includes embeddings/output; compare rankings, not exact equality.")
    (OUT / "results.json").write_text(json.dumps(summary, indent=2))
    (OUT / "status.json").write_text(json.dumps(dict(phase="complete", cases=len(results))))
    print("ALL CHECKS PASSED", json.dumps(dict(predicted=predictions, observed=observed)), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        OUT.mkdir(exist_ok=True)
        (OUT / "status.json").write_text(json.dumps(dict(phase="failed", error=str(error))))
        raise
