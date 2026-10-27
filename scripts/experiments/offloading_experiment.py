"""Resumable gated workflow: smoke -> calibration -> freeze -> correctness -> trials.

No tests of significance are run on chunks. A distinct prompt/seed is one unit.
All subprocess failures stop the workflow; completed records are resumable only
under the same locked protocol and source fingerprint.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
import csv
import hashlib
import itertools
import math
import os
import random
import subprocess
import time

import numpy as np

from wan_dit_lab.json_files import read_json as read  # noqa: E402
from wan_dit_lab.json_files import write_json as write
from wan_dit_lab.offloading.offload_policy import select_policy
from wan_dit_lab.offloading.resource_schedule import sign_pvalue

ROOT = Path(__file__).resolve().parents[2]


def fingerprint():
    paths = sorted((ROOT / "src/wan_dit_lab").rglob("*.py")) + [
        ROOT / "scripts/experiments/offloading_generate.py",
        ROOT / "scripts/experiments/offloading_experiment.py",
    ]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def checkpoint_provenance():
    upstream = read(ROOT / "configs/upstream.json")
    files = [
        ROOT / upstream["local_dir"] / "models_t5_umt5-xxl-enc-bf16.pth",
        ROOT / "reference/INT8-DiT/wan_block/checkpoints/self_forcing_dmd.pt",
        ROOT / "reference/INT8-DiT/wan_block/checkpoints/taew2_1.pth",
    ]
    return {
        str(path.relative_to(ROOT)): dict(
            bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns
        )
        for path in files
    }


def prepare(config_path):
    config = read(config_path)
    source = read(ROOT / "configs/experiments/pc_fifo_480p.json")
    dataset = ROOT / source["prompt_source"].replace("\\", "/")
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() == source["prompt_source_sha256"]
    excluded = {p["prompt"] for p in config["calibration"]["prompts"]}
    population = list(
        dict.fromkeys(
            p.strip() for p in dataset.read_text(encoding="utf-8").splitlines() if p.strip()
        )
    )
    population = [p for p in population if p not in excluded]
    rng = random.Random(20261018)
    prompts = rng.sample(population, 24)
    orders = list(itertools.permutations(("fixed_2x2", "median", "guarded"))) * 4
    rng.shuffle(orders)
    config["evaluation"] = [
        dict(id=f"{i + 1:02}", prompt=prompt, seed=rng.randrange(2**31), order=list(orders[i]))
        for i, prompt in enumerate(prompts)
    ]
    config["statistics"] = dict(
        unit="distinct held-out prompt and assigned seed",
        pairs=24,
        primary="end-to-end speedup fixed_2x2 / candidate > 1.05",
        hypotheses=["median vs fixed_2x2", "guarded vs fixed_2x2"],
        test="exact two-sided paired sign test around speedup 1.05; Holm adjustment for two contrasts",
        alpha=0.05,
        ties="abs(speedup - 1.05) <= 1e-6, omitted from sign test; report count",
        power_assumption="independent units; probability of exceeding 1.05x speedup is .8, no ties",
        conservative_power_at_bonferroni_alpha_025=sum(
            math.comb(24, i) * 0.8**i * 0.2 ** (24 - i) for i in range(18, 25)
        ),
        reject_rule="all 24 triples complete; Holm-adjusted p<.05; more wins than losses",
        decision="reject or do not reject; no guarantee of significant results",
        secondary="guarded vs median, chunk quantiles, resource waits, memory, quality equality: descriptive",
    )
    config["execution"]["resource_budgets"] = dict(ram_gib=14.0, vram_gib=7.25)
    config["execution"]["calibration_tie_fraction"] = 0.01
    config["execution"]["readiness_wait_threshold_s"] = 0.0001
    config["execution"]["profile_fit"] = (
        "DiT compute timing uses chunks >=8, excluding compile/first-full-cache samples; retain all raw rows"
    )
    config["execution"]["calibration_estimator"] = (
        "phase/resource/node costs, median or max loading/min compute; same raw calibration dataset"
    )
    config["execution"]["candidate_scope"] = (
        "Independent pinned/GPU windows 1..4 for T5/DiT; other resource placements fixed by supported runtime. No arbitrary library/context/workspace swapping."
    )
    config["calibration"]["latent_frames_per_prompt"] = 30
    config["calibration"]["sample_plan"] = (
        "Eight disjoint prompts; 10 chunks each; steady fitted chunks8,9, five DiT passes/chunk,24 T5 blocks/prompt; retain per-block sample counts"
    )
    config["measurement"]["quantile_method"] = (
        "linear within each run, median of per-run metrics across prompts; memory peak worst across runs; raw runs retained"
    )
    config["measurement"]["end_to_end"] = (
        "parent process launch timestamp -> child writer close, includes import/loading/T5/generation/decode/writer; calibration separate"
    )
    config["status"] = "ready_for_tests"
    config["readiness"] = dict(
        full_pipeline_resource_plan=True,
        full_pipeline_cost_scheduler=True,
        telemetry_and_inventory_hooks=True,
        numerical_budgets_frozen=True,
        protocol_and_report_templates=True,
        calibration_profiles_frozen=False,
    )
    config["limits"] = (
        "24 distinct held-out prompt/seed triples, one run/condition, balanced condition order. Power assumes strong (.8 win probability) effect; no guarantee. Cost optimizer supports only legal staging windows; all other registered resources have fixed lifetimes/placements. Internal CUDA workspaces/library bytes remain untracked overhead."
    )
    write(config_path, config)


def matched(a, b):
    if a["context_hash"] != b["context_hash"] or a["written_frames"] != b["written_frames"]:
        raise RuntimeError("Text condition or output frame count changed")
    if [(x["latent_hash"], x["decoded_hash"]) for x in a["chunks"]] != [
        (x["latent_hash"], x["decoded_hash"]) for x in b["chunks"]
    ]:
        raise RuntimeError("Resource scheduling changed latent or decoded output")


def summarize(record):
    memory = record["memory_samples"]
    result = dict(full_run_s=record["full_run_s"], first_frame_s=record["first_frame_s"])
    chunks = [x for x in record["chunks"] if x["index"] >= 7]
    if chunks:
        for q in (0.5, 0.9, 0.99):
            result[f"chunk_delivery_p{int(q * 100)}_s"] = float(
                np.quantile([x["delivery_s"] for x in chunks], q)
            )
        result["steady_fps"] = sum(x["frames"] for x in chunks) / sum(
            x["delivery_s"] for x in chunks
        )
    for name, key in (("vram_live", "allocated"), ("ram_rss", "rss")):
        for q in (0.5, 0.9, 0.99):
            result[f"{name}_p{int(q * 100)}_gib"] = (
                float(np.quantile([s[key] for s in memory], q)) / 2**30
            )
    for name, key in [
        ("vram_peak_live", "peak_allocated"),
        ("vram_peak_reserved", "peak_reserved"),
        ("vram_device_peak_used", "device_used"),
        ("ram_rss_peak", "rss"),
        ("ram_private_peak", "private"),
        ("pinned_peak", "pinned_peak"),
    ]:
        values = [s[key] for s in memory if s[key] is not None]
        result[f"{name}_gib"] = max(values) / 2**30 if values else None
    for name, key in [
        ("vram_device_min_free", "device_free"),
        ("ram_system_min_available", "ram_available"),
    ]:
        result[f"{name}_gib"] = min(s[key] for s in memory) / 2**30
    for name, key in [
        ("vram_live_after", "allocated"),
        ("vram_device_free_after", "device_free"),
        ("ram_rss_after", "rss"),
        ("ram_system_available_after", "ram_available"),
    ]:
        result[f"{name}_gib"] = memory[-1][key] / 2**30
    waits = [r["dependency_wait_s"] for phase in ("t5", "dit") for r in record[f"{phase}_trace"]]
    result.update(
        resource_deadline_misses=sum(v > 0.0001 for v in waits),
        resource_wait_total_s=sum(waits),
        resource_wait_max_s=max(waits),
        latent_match="not compared yet",
        saved_frames=f"{record['written_frames']} / {record['written_frames'] / 16:g}s",
    )
    result["ram_to_vram_gib"] = (
        sum(
            record[f"{phase}_copy_count"] * record[f"{phase}_block_bytes"]
            for phase in ("t5", "dit")
        )
        / 2**30
    )
    return result


def holm(pvalues):
    adjusted, previous = {}, 0.0
    for rank, (name, value) in enumerate(sorted(pvalues.items(), key=lambda item: item[1])):
        previous = max(previous, min(1.0, value * (len(pvalues) - rank)))
        adjusted[name] = previous
    return adjusted


def report(config, output):
    groups = {name: [] for name in config["execution"]["order"]}
    inventories = {name: [] for name in groups}
    paired = []
    for pair in config["evaluation"]:
        records = {}
        for method in groups:
            folder = output / "evaluation" / pair["id"] / method
            if (folder / "complete.json").exists():
                records[method] = read(folder / "record.json")
        if len(records) == 3:
            for method in ("median", "guarded"):
                matched(records["fixed_2x2"], records[method])
            for method, record in records.items():
                groups[method].append(summarize(record))
                inventories[method].append(record)
            paired.append(
                dict(id=pair["id"], **{name: r["full_run_s"] for name, r in records.items()})
            )
    summary = {}
    for name, rows in groups.items():
        summary[name] = {}
        if not rows:
            continue
        for key in rows[0]:
            values = [row[key] for row in rows if row[key] is not None]
            if not values:
                summary[name][key] = None
            elif isinstance(values[0], (float, int)):
                reducer = (
                    max
                    if "peak" in key
                    else min
                    if "min_free" in key or "min_available" in key
                    else np.median
                )
                summary[name][key] = float(reducer(values))
            else:
                summary[name][key] = values[0]
    for values in summary.values():
        if values:
            values["latent_match"] = f"{len(paired)}/{len(config['evaluation'])} bitwise triples"
    if (output / "calibration.complete.json").exists():
        calibration_s = read(output / "calibration.complete.json").get("duration_s")
        for name in ("median", "guarded"):
            if summary[name]:
                summary[name]["calibration_s"] = calibration_s
    write(output / "summary.json", dict(methods=summary, pairs=paired))
    csv_path = output / "comparison.csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    display_rows = []
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "label", *groups])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: row[key] for key in ("metric", "label")}
                | {name: summary[name].get(row["metric"], "") for name in groups}
            )
            display_rows.append(
                [row["label"]] + [summary[name].get(row["metric"], "") for name in groups]
            )
    inventory_rows = []
    for category in config["inventory"]["categories"]:
        row = dict(category=category)
        for name in groups:
            counts, ram, gpu = [], [], []
            for record in inventories[name]:
                owners = {
                    event["resource"]
                    for event in record["resource_events"]
                    if event["event"] == "acquire"
                    and record.get("managed_resources", {}).get(event["resource"]) == category
                }
                counts.append(len(owners))
                for snap in record["inventory"]:
                    storages = [x for x in snap["storages"] if x["category"] == category]
                    ram.append(sum(x["bytes"] for x in storages if x["device"] == "cpu") / 2**30)
                    gpu.append(
                        sum(x["bytes"] for x in storages if x["device"].startswith("cuda")) / 2**30
                    )
            row[f"{name}_count"] = max(counts) if counts else ""
            row[f"{name}_peak_ram_gib"] = max(ram) if ram and category != "code" else ""
            row[f"{name}_peak_vram_gib"] = max(gpu) if gpu and category != "code" else ""
        inventory_rows.append(row)
    with (output / "resources.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory_rows[0]))
        writer.writeheader()
        writer.writerows(inventory_rows)

    def fmt(value):
        return f"{value:.4f}" if isinstance(value, (float, int)) else str(value or "·")

    text = [
        "# Offloading: 2×2 / медиана / запас",
        "",
        f"Завершено сопоставимых троек: {len(paired)}/{len(config['evaluation'])}.",
        "",
        "| Метрика | 2×2 | Медиана | Запас |",
        "| --- | ---: | ---: | ---: |",
    ]
    text.extend("| " + " | ".join(map(fmt, row)) + " |" for row in display_rows)
    text += [
        "",
        "Ресурсы: resources.csv. Объёмы категорий наблюдаются в выбранных снимках; внутренние workspace и код не атрибутированы. Неизвестные байты оставлены пустыми.",
        "",
    ]
    (output / "Сравнение.md").write_text("\n".join(text), encoding="utf-8")
    if len(paired) != len(config["evaluation"]):
        return
    contrasts = {}
    for name in ("median", "guarded"):
        speedups = [row["fixed_2x2"] / row[name] for row in paired]
        wins = sum(v > 1.05 + 1e-6 for v in speedups)
        losses = sum(v < 1.05 - 1e-6 for v in speedups)
        contrasts[name] = dict(
            wins=wins,
            losses=losses,
            ties=len(paired) - wins - losses,
            median_speedup=float(np.median(speedups)),
            p=sign_pvalue(wins, losses),
        )
    adjusted = holm({name: row["p"] for name, row in contrasts.items()})
    for name, row in contrasts.items():
        row.update(
            holm_p=adjusted[name],
            reject_for_benefit=adjusted[name] < 0.05 and row["wins"] > row["losses"],
            reject_null=adjusted[name] < 0.05,
        )
    write(output / "statistics.json", contrasts)


def workflow(config_path):
    config = read(config_path)
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=True)
    lock = dict(config=config, sources=fingerprint(), checkpoints=checkpoint_provenance())
    if (output / "protocol.lock.json").exists():
        if read(output / "protocol.lock.json") != lock:
            raise RuntimeError(
                "Protocol/source changed since run start; use a new experiment directory"
            )
    else:
        write(output / "protocol.lock.json", lock)
    frozen_config = output / "protocol.json"
    write(frozen_config, config)
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1", TORCHINDUCTOR_COMPILE_THREADS="1")
    env["CUDA_HOME"] = str(ROOT / "artifacts/cuda_toolkit")
    env["CUDA_PATH"] = env["CUDA_HOME"]
    env["PATH"] = str(Path(env["CUDA_HOME"]) / "bin") + os.pathsep + env["PATH"]
    env["PYTHONPATH"] = str(ROOT / "src")

    def status(phase, **extra):
        write(output / "status.json", dict(status="running", phase=phase, pid=os.getpid(), **extra))

    def launch(folder, case, policy):
        if fingerprint() != lock["sources"]:
            raise RuntimeError("Runtime source changed during experiment")
        if checkpoint_provenance() != lock["checkpoints"]:
            raise RuntimeError("Checkpoint files changed during experiment")
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "complete.json").exists():
            existing = read(folder / "record.json")
            if existing["policy"] != policy or any(
                existing["case"][k] != case[k] for k in ("prompt", "seed")
            ):
                raise RuntimeError("Completed record uses a different case or policy")
            if (
                max(s["rss"] for s in existing["memory_samples"])
                > config["execution"]["resource_budgets"]["ram_gib"] * 2**30
            ):
                raise RuntimeError("Completed run exceeds RAM budget")
            return existing
        case = case | dict(process_start_perf=time.perf_counter())
        write(folder / "case.json", case)
        write(folder / "policy.json", policy)
        command = [
            sys.executable,
            "-u",
            str(ROOT / "scripts/experiments/offloading_generate.py"),
            "--config",
            str(frozen_config.resolve()),
            "--case",
            str(folder / "case.json"),
            "--policy",
            str(folder / "policy.json"),
            "--output",
            str(folder),
        ]
        print("START", folder.relative_to(output), flush=True)
        with (folder / "run.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        if not (folder / "complete.json").exists():
            raise RuntimeError("Child exited without completed outputs")
        result = read(folder / "record.json")
        if (
            max(s["rss"] for s in result["memory_samples"])
            > config["execution"]["resource_budgets"]["ram_gib"] * 2**30
        ):
            raise RuntimeError("Observed process RSS exceeds frozen RAM budget")
        print("DONE", folder.relative_to(output), result["full_run_s"], flush=True)
        return result

    fixed = dict(profile="fixed_2x2", t5=dict(ram=2, gpu=2), dit=dict(ram=2, gpu=2))
    calibration_cases = config["calibration"]["prompts"]
    try:
        status("smoke")
        smoke_case = calibration_cases[0] | dict(latent_frames=3, saved_video_frames=9)
        launch(output / "smoke/fixed_2x2", smoke_case, fixed)
        status("calibration")
        calibration_started = time.perf_counter()
        records = []
        for i, case in enumerate(calibration_cases):
            status("calibration", completed=i, total=len(calibration_cases), case=case["id"])
            records.append(
                launch(
                    output / "calibration" / case["id"],
                    case | dict(latent_frames=30, saved_video_frames=117),
                    fixed,
                )
            )
        budget = config["execution"]["resource_budgets"]
        policies = dict(fixed_2x2=fixed)
        for name in ("median", "guarded"):
            policies[name] = select_policy(
                records, name, budget["vram_gib"] * 2**30, budget["ram_gib"] * 2**30
            )
        write(output / "policies.json", policies)
        status("correctness_gate")
        for name in ("median", "guarded"):
            matched(
                records[0],
                launch(
                    output / "gates" / name,
                    calibration_cases[0] | dict(latent_frames=30, saved_video_frames=117),
                    policies[name],
                ),
            )
        write(
            output / "calibration.complete.json",
            dict(
                status="complete",
                prompts=len(records),
                policies=policies,
                duration_s=time.perf_counter() - calibration_started,
            ),
        )
        for i, pair in enumerate(config["evaluation"]):
            for name in pair["order"]:
                status(
                    "evaluation",
                    pair=pair["id"],
                    method=name,
                    completed_pairs=i,
                    total_pairs=len(config["evaluation"]),
                )
                launch(output / "evaluation" / pair["id"] / name, pair, policies[name])
                report(config, output)
        report(config, output)
        write(
            output / "status.json",
            dict(
                status="complete",
                completed_pairs=len(config["evaluation"]),
                runs=3 * len(config["evaluation"]),
            ),
        )
    except Exception as error:
        write(output / "failure.json", dict(status="failed", error=str(error), time=time.time()))
        status("failed", error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run", "report"])
    parser.add_argument("--config", default="configs/experiments/offloading_full_30s.json")
    args = parser.parse_args()
    path = Path(args.config)
    if args.action == "prepare":
        prepare(path)
    elif args.action == "run":
        workflow(path)
    else:
        config = read(path)
        report(config, ROOT / config["output"])


if __name__ == "__main__":
    main()
