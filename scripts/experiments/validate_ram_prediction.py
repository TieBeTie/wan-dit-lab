"""Validate offline-predicted RAM boundary with one full video at a time."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import os
import shutil
import subprocess
import time

import psutil

from scripts.experiments.ram_budget_optimum import OUT, ROOT, cleanup, measured, read, write


def main():
    if not read(OUT / "config.json")["common"].get("compile"):
        raise RuntimeError("Historical eager runs cannot be used for a working compiled benchmark")
    prediction = read(OUT / "budget_prediction.json")
    selected = prediction["selected_cache_bytes"] // 2**20
    reference = read(OUT / "ram_2048mib/record.json")
    expected = [(r["latent_hash"], r["decoded_hash"]) for r in reference["chunks"]]
    baseline = measured(reference)
    status = dict(status="running", pid=os.getpid(), mode="offline_prediction_validation",
                  predicted_cache_mib=selected, validations=[])
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1", PYTHONPATH=str(ROOT / "src"),
               CUDA_HOME=str(ROOT / "artifacts/cuda_toolkit"))
    env["CUDA_PATH"] = env["CUDA_HOME"]
    env["PATH"] = str(ROOT / "artifacts/cuda_toolkit/bin") + os.pathsep + env["PATH"]
    started = time.monotonic()
    try:
        for mib in range(selected, 2049, 64):
            folder = OUT / f"validation_{mib:04d}mib"
            folder.mkdir(exist_ok=False)
            case = read(OUT / "ram_2048mib/case.json")
            case["calibration_probe"] = False
            policy = read(OUT / "ram_2048mib/policy.json")
            policy["resources"]["backing_ram_gib"] = mib / 1024
            write(folder / "case.json", case)
            write(folder / "policy.json", policy)
            status.update(active_cache_mib=mib, active_folder=str(folder))
            write(OUT / "status.json", status)
            worker = None
            try:
                with (folder / "run.log").open("w", encoding="utf-8") as log:
                    worker = subprocess.Popen([sys.executable, "-u", "scripts/experiments/offloading_generate.py",
                        "--config", str(OUT / "config.json"), "--case", str(folder / "case.json"),
                        "--policy", str(folder / "policy.json"), "--output", str(folder)],
                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    while worker.poll() is None:
                        if shutil.disk_usage(OUT).free < 3 * 2**30:
                            raise OSError("Less than 3 GiB free disk")
                        status["elapsed_s"] = time.monotonic() - started
                        progress = folder / "progress.json"
                        if progress.exists():
                            status["progress"] = read(progress)
                        write(OUT / "status.json", status)
                        time.sleep(2)
                    if worker.returncode or not (folder / "complete.json").exists():
                        raise RuntimeError(f"Validation failed: {folder / 'run.log'}")
                record = read(folder / "record.json")
                if [(r["latent_hash"], r["decoded_hash"]) for r in record["chunks"]] != expected:
                    raise RuntimeError("Bitwise output mismatch")
                metrics = measured(record)
                passed = (metrics["steady_median_s"] <= 1.03 * baseline["steady_median_s"]
                          and metrics["steady_mean_s"] <= 1.03 * baseline["steady_mean_s"])
                status["validations"].append(dict(cache_mib=mib, **metrics, passed=passed, bitwise=True))
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
            if passed:
                status.update(status="complete", selected_cache_mib=mib,
                    scope="Predicted minimum on 64 MiB grid; selected budget validated on one prompt; lower neighbor predicted only")
                break
        else:
            raise RuntimeError("No validated budget within the search range")
        rows = [dict(cache_mib=2048, **baseline), *status["validations"]]
        write(OUT / "validation_results.json", dict(prediction=prediction["selected_cache_bytes"], rows=rows))
        lines = ["# RAM-кэш: расчёт и проверка", "",
                 "Один промпт/seed, 480×832, по 30 с. Первые 10 чанков исключены; 31 интервал в каждом замере.", "",
                 "| Кэш, MiB | Медиана чанка, с | Среднее, с | RSS пик, GiB | Read / Write, GiB |",
                 "| ---: | ---: | ---: | ---: | :--- |"]
        for row in rows:
            lines.append(f"| {row['cache_mib']} | {row['steady_median_s']:.3f} | {row['steady_mean_s']:.3f} | {row['rss_peak_gib']:.2f} | {row['disk_read_gib']:.2f} / {row['disk_write_gib']:.2f} |")
        lines += ["", f"Расчётный минимум: {selected} MiB. Проверенный кандидат: {status['selected_cache_mib']} MiB. Допуск 3% по среднему и медиане. Все латенты и кадры совпали побитно.", "",
                  "Это минимум по модели и один проверенный кандидат, не статистически установленный минимум для любых промптов. Модель фиксирует GPU-трассу и консервативно складывает I/O; файловый кэш ОС включён в измерения. Кэш RAM не равен полному RSS процесса. Временные KV и архивы латентов удалены."]
        (OUT / "Comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except BaseException as error:
        status.update(status="failed", error=repr(error))
        raise
    finally:
        status["elapsed_s"] = time.monotonic() - started
        write(OUT / "status.json", status)


if __name__ == "__main__":
    main()
