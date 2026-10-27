"""Matched working-mode comparison: real torch.compile, 30s per RAM budget."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import os
import shutil
import subprocess
import time

import psutil

from scripts.experiments.ram_budget_optimum import ROOT, measured, read, write

OUT = ROOT / "outputs/experiments/2026-10-19/20_Offloading_compiled_30s"
PREVIOUS = ROOT / "outputs/experiments/2026-10-19/19_RAM_optimum_30s"


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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "status.json").exists():
        raise RuntimeError("Existing experiment; do not overwrite")
    config = read(PREVIOUS / "config.json")
    config["common"]["compile"] = "default"
    config["execution"]["require_compilation"] = True
    write(OUT / "config.json", config)
    case = read(PREVIOUS / "ram_2048mib/case.json")
    case["calibration_probe"] = False
    write(OUT / "case.json", case)
    calibration = read(PREVIOUS / "calibration/resource_calibration.json")
    write(OUT / "calibration.json", calibration)
    policy = read(PREVIOUS / "ram_2048mib/policy.json")
    policy["resource_cost_model"]["calibration"] = str(OUT / "calibration.json")
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1", PYTHONPATH=str(ROOT / "src"),
               CUDA_HOME=str(ROOT / "artifacts/cuda_toolkit"),
               TORCHINDUCTOR_COMPILE_THREADS="1")
    env["CUDA_PATH"] = env["CUDA_HOME"]
    env["PATH"] = str(ROOT / "artifacts/cuda_toolkit/bin") + os.pathsep + env["PATH"]
    status = dict(status="running", pid=os.getpid(), compile="default", completed=[])
    started = time.monotonic()
    records = []
    try:
        for mib in (2048, 832):
            folder = OUT / f"ram_{mib:04d}mib"
            folder.mkdir()
            policy["resources"]["backing_ram_gib"] = mib / 1024
            write(folder / "policy.json", policy)
            status.update(active_cache_mib=mib)
            write(OUT / "status.json", status)
            worker = None
            try:
                with (folder / "run.log").open("w", encoding="utf-8") as log:
                    worker = subprocess.Popen([sys.executable, "-u", "scripts/experiments/offloading_generate.py",
                        "--config", str(OUT / "config.json"), "--case", str(OUT / "case.json"),
                        "--policy", str(folder / "policy.json"), "--output", str(folder)],
                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    while worker.poll() is None:
                        if shutil.disk_usage(OUT).free < 3 * 2**30:
                            raise OSError("Less than 3 GiB free disk")
                        status["elapsed_s"] = time.monotonic() - started
                        if (folder / "progress.json").exists():
                            status["progress"] = read(folder / "progress.json")
                        write(OUT / "status.json", status)
                        time.sleep(2)
                    if worker.returncode or not (folder / "complete.json").exists():
                        raise RuntimeError(f"Compiled run failed: {folder / 'run.log'}")
                record = read(folder / "record.json")
                compilation = record["torch_compile"]
                if compilation["mode"] != "default" or compilation["unique_graphs"] < 1:
                    raise RuntimeError("Actual compilation not verified")
                if any("limit" in key for key in compilation["unimplemented"]):
                    raise RuntimeError("Dynamo compilation limit reached")
                hashes = [(r["latent_hash"], r["decoded_hash"]) for r in record["chunks"]]
                if records and hashes != records[0]["hashes"]:
                    raise RuntimeError("Compiled budgets changed outputs")
                rows = measured(record)
                steady = [r for r in record["chunks"] if r["index"] >= 7]
                rows["steady_fps_like_18_oct"] = sum(r["frames"] for r in steady) / sum(r["delivery_s"] for r in steady)
                rows.update(cache_mib=mib, compilation=compilation)
                records.append(dict(hashes=hashes, metrics=rows))
                status["completed"].append(rows)
                if mib == 2048:
                    # Refresh compute timings in the actual compiled mode.
                    # Existing measured transfer costs are retained explicitly.
                    for name in list(calibration["samples"]):
                        if name.startswith("dit/block/"):
                            del calibration["samples"][name]
                    for row in record["dit_trace"]:
                        if row["node"].startswith("denoise_") and int(row["node"].split("_")[-1]) >= 10:
                            calibration["samples"].setdefault(f"dit/block/{row['block']}", {}).setdefault(
                                "compute", []).append(row["compute_s"])
                    calibration["provenance"] = dict(
                        compute=str(folder / "record.json"), compile="default",
                        transfers=str(PREVIOUS / "calibration/resource_calibration.json"))
                    write(OUT / "calibration.json", calibration)
            finally:
                if worker is not None and worker.poll() is None:
                    process = psutil.Process(worker.pid)
                    children = process.children(recursive=True)
                    for p in [process, *children]:
                        try:
                            p.kill()
                        except psutil.NoSuchProcess:
                            pass
                    psutil.wait_procs([process, *children], timeout=5)
                cleanup(folder)
        rows = [r["metrics"] for r in records]
        write(OUT / "comparison.json", rows)
        lines = ["# Offloading: torch.compile(default), 480×832, 30 с", "",
                 "Один промпт/seed на вариант. FPS: та же формула и отсечение прогрева, что 18.10. Все латенты и декодированные кадры между бюджетами совпали побитно.", "",
                 "| RAM-кэш, MiB | FPS после прогрева | Среднее чанка, с | RSS пик, GiB | VRAM пик, GiB |",
                 "| ---: | ---: | ---: | ---: | ---: |"]
        for row in rows:
            lines.append(f"| {row['cache_mib']} | {row['steady_fps_like_18_oct']:.3f} | {row['steady_mean_s']:.3f} | {row['rss_peak_gib']:.2f} | {row['vram_peak_gib']:.2f} |")
        base, small = rows
        within = (small["steady_mean_s"] <= base["steady_mean_s"] * 1.03
                  and small["steady_median_s"] <= base["steady_median_s"] * 1.03)
        status.update(status="complete", within_3_percent=within, bitwise=True)
        lines += ["", f"832 MiB в допуске 3%: {within}. RAM-кэш не равен полной RAM процесса. Временные KV и латенты удалены."]
        (OUT / "Comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except BaseException as error:
        status.update(status="failed", error=repr(error))
        raise
    finally:
        status["elapsed_s"] = time.monotonic() - started
        write(OUT / "status.json", status)


if __name__ == "__main__":
    main()
