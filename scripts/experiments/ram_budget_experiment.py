"""Resource calibration followed by RAM-budget comparison, without a time limit."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import json
import os
import shutil
import subprocess
import time

import psutil

from wan_dit_lab.json_files import write_json as write  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/experiments/2026-10-19/18_RAM_budget_calibrated"
OUT.mkdir(parents=True, exist_ok=True)
if (OUT / "status.json").exists():
    raise RuntimeError("Existing run; use a new output folder")
start = time.monotonic()
deadline = float("inf")
status = dict(status="running", budget_s=None, phases=[], pid=os.getpid(), goal="Minimum RAM cache without measured slowdown; identical 480x832 workload; no wall-time limit")


def save():
    status["elapsed_s"] = time.monotonic() - start
    write(OUT / "status.json", status)

def execute(label, command):
    status["active_phase"] = label
    save()
    if time.monotonic() >= deadline:
        raise TimeoutError("Four-minute limit")
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1", PYTHONPATH=str(ROOT / "src"))
    env["CUDA_HOME"] = str(ROOT / "artifacts/cuda_toolkit")
    env["CUDA_PATH"] = env["CUDA_HOME"]
    env["PATH"] = str(Path(env["CUDA_HOME"]) / "bin") + os.pathsep + env["PATH"]
    with (OUT / (label + ".log")).open("w", encoding="utf-8") as log:
        worker = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            while worker.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Four-minute limit")
                if shutil.disk_usage(OUT).free < 3 * 2**30:
                    raise OSError("Less than 3 GiB free disk")
                time.sleep(.25)
        finally:
            if worker.poll() is None:
                process = psutil.Process(worker.pid)
                children = process.children(recursive=True)
                process.kill()
                for child in children:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                worker.wait()
                psutil.wait_procs(children, timeout=3)
    status["phases"].append(dict(name=label, exit_code=worker.returncode))
    save()
    if worker.returncode:
        raise RuntimeError(f"{label} failed; see log")

try:
    if shutil.disk_usage(OUT).free < 12 * 2**30:
        raise OSError("Need 12 GiB free before calibration")
    execute("tests", [sys.executable, "-m", "pytest", "tests/test_resource_cost.py", "tests/test_resource_cache.py", "-q"])
    config = json.loads((ROOT / "configs/experiments/offloading_resources_pilot_480p.json").read_text())
    config["common"]["compile"] = "default"
    config["execution"]["require_compilation"] = True
    write(OUT / "config.json", config)
    case = config["evaluation"][0] | dict(latent_frames=6, saved_video_frames=21,
                                         calibration_checkpoint=True)
    write(OUT / "case.json", case)
    policy = json.loads((ROOT / "configs/experiments/offloading_managed_resources_policy.json").read_text())
    policy["resources"].update(vram_cache_gib=5.5, backing_ram_gib=1.,
                               system_ram_reserve_gib=0., process_rss_limit_gib=None)
    write(OUT / "calibration_policy.json", policy)
    for label in ("calibration", "median_1g", "guarded_1g", "median_05g", "median_025g"):
        policy["resources"]["backing_ram_gib"] = {"calibration": .125, "median_1g": 1., "guarded_1g": 1., "median_05g": .5, "median_025g": .25}[label]
        case["calibration_probe"] = label == "calibration"
        write(OUT / "case.json", case)
        folder = OUT / label
        folder.mkdir(exist_ok=True)
        if label != "calibration":
            policy["resource_cost_model"] = dict(profile="guarded" if label.startswith("guarded") else "median",
                calibration=str(OUT / "calibration/resource_calibration.json"))
        write(folder / "policy.json", policy)
        execute(label, [sys.executable, "-u", "scripts/experiments/offloading_generate.py",
                        "--config", str(OUT / "config.json"), "--case", str(OUT / "case.json"),
                        "--policy", str(folder / "policy.json"), "--output", str(folder)])
        for name in ("resource_backing", "latents"):
            target = (folder / name).resolve()
            if OUT.resolve() not in target.parents:
                raise RuntimeError("Cleanup outside experiment")
            if target.exists():
                shutil.rmtree(target)
        for name in ("latents.json", "latents.pt"):
            (folder / name).unlink(missing_ok=True)
    status["status"] = "complete"
except Exception as error:
    status.update(status="time_limit" if isinstance(error, TimeoutError) else "failed", error=repr(error))
finally:
    removed = []
    for folder in (OUT / x for x in ("calibration", "median_1g", "guarded_1g", "median_05g", "median_025g")):
        for name in ("resource_backing", "latents"):
            target = (folder / name).resolve()
            if OUT.resolve() not in target.parents:
                raise RuntimeError("Cleanup outside experiment")
            if target.exists():
                for attempt in range(6):
                    try:
                        shutil.rmtree(target)
                        break
                    except PermissionError:
                        if attempt == 5:
                            raise
                        time.sleep(.5)
                removed.append(str(target.relative_to(OUT)))
        for name in ("latents.json", "latents.pt"):
            (folder / name).unlink(missing_ok=True)
    status["removed_temporary_data"] = removed
    save()
