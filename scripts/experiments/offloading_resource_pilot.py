"""Small paired resource-migration pilot; no cost search or significance claims."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
import csv
import hashlib
import os
import shutil
import statistics
import subprocess
import time

import psutil

from scripts.experiments.offloading_experiment import fingerprint, matched, read, summarize, write

ROOT = Path(__file__).resolve().parents[2]


def check_disk_space(path, minimum_gib):
    free = shutil.disk_usage(path).free / 2**30
    if free < minimum_gib:
        raise RuntimeError(f"Low disk space: {free:.2f} GiB free, need {minimum_gib:g} GiB")


def stop_worker_tree(worker):
    try:
        root = psutil.Process(worker.pid)
        children = root.children(recursive=True)
        root.kill()
    except psutil.NoSuchProcess:
        return
    for child in children:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    worker.wait()


def metrics(record):
    result = summarize(record)
    for key in ("resource_deadline_misses", "resource_wait_total_s", "resource_wait_max_s",
                "ram_to_vram_gib"):
        result[f"weights_{key}"] = result.pop(key)
    events = record.get("tensor_resource_events", [])
    result.update(
        resource_save_count=sum(row["event"] == "save" for row in events),
        resource_restore_count=sum(row["event"] == "restore" for row in events),
        resource_save_restore_wall_s=sum(row["duration_s"] for row in events),
        backing_ram_peak_gib=record.get("backing_peak_bytes", {}).get("ram", 0) / 2**30,
        backing_disk_peak_gib=record.get("backing_peak_bytes", {}).get("disk", 0) / 2**30,
    )
    for event in ("save", "restore"):
        durations = [row["duration_s"] for row in events if row["event"] == event]
        result[f"resource_{event}_wall_total_s"] = sum(durations)
        result[f"resource_{event}_wall_max_s"] = max(durations, default=0)
        # Disk bytes include serialized payload/headers, not physical SSD traffic.
        result[f"resource_{event}_logical_gib"] = sum(
            row["bytes"] for row in events if row["event"] == event
        ) / 2**30
    result["latent_match"] = "pending paired comparison"
    return result


def aggregate(rows):
    result = {}
    for key in rows[0] if rows else ():
        values = [row[key] for row in rows if row[key] is not None]
        if not values:
            result[key] = None
        elif isinstance(values[0], (float, int)):
            reduce = max if "peak" in key or key.endswith("max_s") else (
                min if "min_free" in key or "min_available" in key else statistics.median
            )
            result[key] = reduce(values)
        else:
            result[key] = values[0]
    return result


def report(config, output):
    methods = list(config["policies"])
    groups = {method: [] for method in methods}
    runs, pairs = [], []
    for case in config["evaluation"]:
        records = {}
        for method in methods:
            folder = output / "evaluation" / case["id"] / method
            if (folder / "complete.json").exists():
                record = read(folder / "record.json")
                records[method] = record
        if len(records) == len(methods):
            matched(records[methods[0]], records[methods[1]])
            pairs.append(case["id"])
        for method, record in records.items():
            row = metrics(record)
            row["latent_match"] = "bitwise pair" if case["id"] in pairs else "pending pair"
            groups[method].append(row)
            runs.append(dict(case=case["id"], method=method, **row))
    summary = {method: aggregate(rows) for method, rows in groups.items()}
    for method, values in summary.items():
        if values:
            values["completed_runs"] = len(groups[method])
            values["latent_match"] = f"{len(pairs)}/{len(config['evaluation'])} bitwise pairs"
    write(output / "summary.json", dict(methods=summary, pairs=pairs, runs=runs,
                                         inference="descriptive pilot only"))
    keys = list(dict.fromkeys(key for values in summary.values() for key in values))
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", *methods])
        writer.writerows([key, *(summary[method].get(key) for method in methods)] for key in keys)
    lines = ["# Короткий тест offloading, 480×832", "",
             "2 промпта/seed, до 2 роликов на способ, по 30 с. До них 2 коротких прохода по 10 чанков.",
             "Это описательное сравнение явных размещений, без статистического вывода и без FlexGen-подбора.", "",
             "| Метрика | 2×2, KV в VRAM | Ресурсы в RAM, архив на диске |",
             "| :--- | ---: | ---: |"]
    for key in keys:
        values = []
        for method in methods:
            value = summary[method].get(key)
            values.append("·" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append(f"| {key} | {' | '.join(values)} |")
    lines += ["", "Квантили внутри запуска; сводка по запускам: медиана, пики максимум, свободная память минимум.",
              "weights_* учитывают только переносы весов. resource_* учитывают синхронные сохранения / восстановления.",
              "Логические байты файлов не равны физическому I/O SSD. Совпадение проверяется до MP4-кодирования.",
              "Время полной генерации включает загрузку и возможную компиляцию; steady-метрики отдельно. Профилирование не выбирает расписание."]
    (output / "Сравнение.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def workflow(config_path):
    config = read(config_path)
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=True)
    sources = fingerprint() | {
        "scripts/experiments/offloading_resource_pilot.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    }
    frozen = dict(config=config, sources=sources)
    if (output / "protocol.lock.json").exists() and read(output / "protocol.lock.json") != frozen:
        raise RuntimeError("Pilot protocol/source changed; use a new output directory")
    write(output / "protocol.lock.json", frozen)
    write(output / "protocol.json", config)
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1", TORCHINDUCTOR_COMPILE_THREADS="1")
    env["CUDA_HOME"] = str(ROOT / "artifacts/cuda_toolkit")
    env["CUDA_PATH"] = env["CUDA_HOME"]
    env["PATH"] = str(Path(env["CUDA_HOME"]) / "bin") + os.pathsep + env["PATH"]
    env["PYTHONPATH"] = str(ROOT / "src")

    def status(state, **extra):
        write(output / "status.json", dict(status=state, pid=os.getpid(), **extra))

    def launch(phase, case, method):
        if fingerprint() != {key: value for key, value in sources.items()
                             if key != "scripts/experiments/offloading_resource_pilot.py"}:
            raise RuntimeError("Runtime source changed during pilot")
        folder = output / phase / case["id"] / method
        if (folder / "complete.json").exists():
            return read(folder / "record.json")
        folder.mkdir(parents=True, exist_ok=True)
        check_disk_space(output, config["execution"].get("minimum_free_disk_gib", 12))
        status("running", phase=phase, case=case["id"], method=method)
        print(f"START {phase}/{case['id']}/{method}", flush=True)
        write(folder / "case.json", case | dict(process_start_perf=time.perf_counter()))
        write(folder / "policy.json", config["policies"][method])
        command = [sys.executable, "-u", str(ROOT / "scripts/experiments/offloading_generate.py"),
                   "--config", str(output / "protocol.json"),
                   "--case", str(folder / "case.json"), "--policy", str(folder / "policy.json"),
                   "--output", str(folder)]
        with (folder / "run.log").open("w", encoding="utf-8") as handle:
            worker = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            try:
                while worker.poll() is None:
                    check_disk_space(output, config["execution"].get("stop_below_free_disk_gib", 3))
                    try:
                        worker.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
                if worker.returncode:
                    raise subprocess.CalledProcessError(worker.returncode, command)
            except BaseException:
                stop_worker_tree(worker)
                raise
        if not (folder / "complete.json").exists():
            raise RuntimeError("Worker exited without a completed result")
        record = read(folder / "record.json")
        if max(row["rss"] for row in record["memory_samples"]) > config["execution"]["resource_budgets"]["ram_gib"] * 2**30:
            raise RuntimeError("Process RSS exceeded pilot RAM budget")
        print(f"DONE {phase}/{case['id']}/{method}: {record['full_run_s']:.2f}s", flush=True)
        return record

    try:
        case = config["evaluation"][0] | dict(latent_frames=30, saved_video_frames=117)
        warmups = [launch("preflight", case, method) for method in config["policies"]]
        matched(*warmups)
        write(output / "preflight.json", dict(status="complete", bitwise=True,
                                              metrics={key: metrics(value) for key, value in zip(config["policies"], warmups)}))
        for case in config["evaluation"]:
            for method in case["order"]:
                launch("evaluation", case, method)
                report(config, output)
        report(config, output)
        status("complete", pairs=len(config["evaluation"]), inference="descriptive pilot only")
        print("COMPLETE", flush=True)
    except BaseException as error:
        status("failed", error=repr(error))
        try:
            report(config, output)
        except Exception as report_error:
            print(f"Report incomplete: {report_error!r}", flush=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    workflow(parser.parse_args().config)
