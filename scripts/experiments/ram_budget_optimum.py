"""Adaptive whole-video RAM cache search: one prompt, 30s per candidate.

Single-run descriptive optimum, not a statistical equivalence test.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time

import psutil

from wan_dit_lab.json_files import read_json as read  # noqa: E402
from wan_dit_lab.json_files import write_json as write

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/experiments/2026-10-19/19_RAM_optimum_30s"
WARMUP = 10
TOLERANCE = .03
RESOLUTION_MIB = 64


def measured(record):
    from wan_dit_lab.harness import stream_summary

    result = stream_summary(record, WARMUP)
    if result.get("intervals", 0) < 20:
        raise RuntimeError("Too few steady intervals")
    return result


def cleanup(folder):
    for name in ("resource_backing", "latents"):
        target = (folder / name).resolve()
        if OUT.resolve() not in target.parents:
            raise RuntimeError("Cleanup outside experiment")
        if target.exists():
            for attempt in range(8):
                try:
                    shutil.rmtree(target)
                    break
                except PermissionError:
                    if attempt == 7:
                        raise
                    time.sleep(.5)
    for name in ("latents.json", "latents.pt"):
        (folder / name).unlink(missing_ok=True)


def report(rows, result=None):
    write(OUT / "comparison.json", rows)
    if not rows:
        return
    with (OUT / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# RAM cache search: 480x832, 30s, one prompt per candidate", "",
             "Fixed median profile. First 10 chunks excluded. Tolerance: median AND mean <= baseline +3%; resolution 64 MiB. One run per candidate: descriptive result, not statistical equivalence.", "",
             "| Cache MiB | Median chunk s | Mean chunk s | Peak RSS GiB | Read GiB | Write GiB | Within 3% |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | :--- |"]
    for row in rows:
        lines.append(f"| {row['cache_mib']} | {row['steady_median_s']:.3f} | {row['steady_mean_s']:.3f} | {row['rss_peak_gib']:.2f} | {row['disk_read_gib']:.2f} | {row['disk_write_gib']:.2f} | {row['eligible']} |")
    if result:
        lines += ["", json.dumps(result)]
    lines += ["", "Cache budget is not process RSS. File I/O includes OS file cache. All candidate latents and decoded frames must match the reference bitwise. Temporary KV and latent archives are deleted after each run."]
    (OUT / "Comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def workflow():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "status.json").exists():
        raise RuntimeError("Output exists; do not overwrite prior search")
    started = time.monotonic()
    status = dict(status="running", pid=os.getpid(), candidates=[], tolerance=TOLERANCE,
                  resolution_mib=RESOLUTION_MIB, warmup_chunks=WARMUP)
    config = read(ROOT / "configs/experiments/offloading_resources_pilot_480p.json")
    config["common"]["compile"] = "default"
    config["execution"]["require_compilation"] = True
    case = config["evaluation"][0] | dict(latent_frames=123, saved_video_frames=480)
    write(OUT / "config.json", config)
    base = read(ROOT / "configs/experiments/offloading_managed_resources_policy.json")
    base["resources"].update(vram_cache_gib=5.5, system_ram_reserve_gib=0.,
                             process_rss_limit_gib=None)
    write(OUT / "protocol.json", dict(prompt=case["prompt"], seed=case["seed"],
          common=config["common"], tolerance=TOLERANCE, warmup=WARMUP,
          resolution_mib=RESOLUTION_MIB, profile="median"))
    runtime = [*sorted((ROOT / "src/wan_dit_lab").rglob("*.py")), ROOT / "scripts/experiments/offloading_generate.py"]
    fingerprint = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in runtime}
    write(OUT / "sources.json", fingerprint)
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1", PYTHONPATH=str(ROOT / "src"),
               CUDA_HOME=str(ROOT / "artifacts/cuda_toolkit"))
    env["CUDA_PATH"] = env["CUDA_HOME"]
    env["PATH"] = str(Path(env["CUDA_HOME"]) / "bin") + os.pathsep + env["PATH"]
    rows, reference, baseline = [], None, None

    def save():
        status["elapsed_s"] = time.monotonic() - started
        write(OUT / "status.json", status)

    def trial(mib, calibrate=False):
        nonlocal reference, baseline
        if any(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != value
               for path, value in fingerprint.items()):
            raise RuntimeError("Runtime changed during search")
        if shutil.disk_usage(OUT).free < 10 * 2**30:
            raise OSError("Insufficient disk headroom before run")
        folder = OUT / ("calibration" if calibrate else f"ram_{mib:04d}mib")
        folder.mkdir()
        policy = json.loads(json.dumps(base))
        policy["resources"]["backing_ram_gib"] = mib / 1024
        if not calibrate:
            policy["resource_cost_model"] = dict(profile="median",
                calibration=str(OUT / "calibration/resource_calibration.json"))
        local_case = case | dict(calibration_probe=calibrate,
                                calibration_probe_node="decode_7")
        write(folder / "policy.json", policy)
        write(folder / "case.json", local_case)
        status.update(active_cache_mib=mib, active_folder=str(folder), status="running")
        save()
        worker = None
        try:
            with (folder / "run.log").open("w", encoding="utf-8") as log:
                worker = subprocess.Popen([sys.executable, "-u", "scripts/experiments/offloading_generate.py",
                    "--config", str(OUT / "config.json"), "--case", str(folder / "case.json"),
                    "--policy", str(folder / "policy.json"), "--output", str(folder)],
                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                while worker.poll() is None:
                    if shutil.disk_usage(OUT).free < 3 * 2**30:
                        raise OSError("Free disk below 3 GiB")
                    time.sleep(1)
                if worker.returncode:
                    raise RuntimeError(f"Candidate {mib} failed; see {folder / 'run.log'}")
            if not (folder / "complete.json").exists():
                raise RuntimeError("No completion marker")
            record = read(folder / "record.json")
            hashes = [(r["latent_hash"], r["decoded_hash"]) for r in record["chunks"]]
            if reference is None:
                reference = hashes
            elif hashes != reference:
                raise RuntimeError("Bitwise output mismatch")
            if calibrate:
                status["calibration_complete"] = True
                save()
                return True
            row = dict(cache_mib=mib, **measured(record))
            baseline = row if baseline is None else baseline
            row["eligible"] = (row["steady_median_s"] <= baseline["steady_median_s"] * (1 + TOLERANCE)
                               and row["steady_mean_s"] <= baseline["steady_mean_s"] * (1 + TOLERANCE))
            rows.append(row)
            status["candidates"] = rows
            report(rows)
            save()
            return row["eligible"]
        finally:
            if worker is not None and worker.poll() is None:
                root = psutil.Process(worker.pid)
                children = root.children(recursive=True)
                for process in [root, *children]:
                    try:
                        process.kill()
                    except psutil.NoSuchProcess:
                        pass
                psutil.wait_procs([root, *children], timeout=5)
            cleanup(folder)

    try:
        trial(2048, calibrate=True)
        trial(2048)
        low, high = 0, 2048
        candidate = 1024
        while candidate >= RESOLUTION_MIB:
            if trial(candidate):
                high = candidate
                candidate //= 2
            else:
                low = candidate
                break
        while high - low > RESOLUTION_MIB:
            candidate = ((low + high) // (2 * RESOLUTION_MIB)) * RESOLUTION_MIB
            if candidate <= low:
                break
            if trial(candidate):
                high = candidate
            else:
                low = candidate
        result = dict(selected_cache_mib=high, slower_or_untested_below_mib=low,
                      scope="single-prompt descriptive boundary; no statistical guarantee")
        report(rows, result)
        status.update(status="complete", result=result)
    except BaseException as error:
        status.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    workflow()
