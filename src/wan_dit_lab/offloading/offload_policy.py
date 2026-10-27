"""Derive bounded staging policies from raw, held-out calibration traces."""

import math
import statistics

from wan_dit_lab.offloading.resource_schedule import simulate


def select_policy(records, profile, gpu_budget, ram_budget, generated_chunks=41):
    if profile not in ("median", "guarded") or not records:
        raise ValueError("Unknown profile or empty calibration")
    result = dict(
        profile=profile,
        fixed_placements={
            "context": "CPU source; GPU consumer copy",
            "KV_and_decoder_state": "VRAM",
            "activations": "VRAM while used",
            "latents_archive": "RAM",
            "decoder": "VRAM first decode to last decode",
            "code": "runtime-managed",
        },
        candidate_scope="Independent RAM/VRAM staging windows 1..4 in T5 and DiT; other resources have fixed legal placement",
        gpu_budget_gib=gpu_budget / 2**30,
        ram_budget_gib=ram_budget / 2**30,
    )
    for phase, blocks in (("t5", 24), ("dit", 30)):
        rows = [
            r
            for record in records
            for r in record[f"{phase}_trace"]
            if phase == "t5"
            or (r["node"].startswith("denoise_") and int(r["node"].split("_")[-1]) >= 8)
        ]
        if not rows:
            raise ValueError(f"No compute samples for {phase}")
        reduce_load = max if profile == "guarded" else statistics.median
        reduce_compute = min if profile == "guarded" else statistics.median
        costs = {
            key: [
                reduce_compute([r[key] for r in rows if r["block"] == block])
                if key == "compute_s"
                else reduce_load([r[key] for r in rows if r["block"] == block])
                for block in range(blocks)
            ]
            for key in ("pack_s", "copy_s", "compute_s")
        }
        # Both first-use and warmed peaks matter, not just tensor parameter size.
        phase_samples = [
            s
            for record in records
            for s in record["memory_samples"]
            if (
                s["stage"] in ("text", "acquire_t5")
                if phase == "t5"
                else s["stage"].startswith(
                    ("denoise_", "decode_", "acquire_dit", "acquire_stream", "acquire_decoder")
                )
            )
        ]
        if not phase_samples:
            raise ValueError("Missing phase memory snapshots")
        block_bytes = max(record[f"{phase}_block_bytes"] for record in records)
        base_gpu = max(s["peak_allocated"] for s in phase_samples) - 2 * block_bytes
        base_ram = max(s["rss"] for s in phase_samples) - 2 * block_bytes
        # Preserve measured transient-workspace margin as well as resident resources.
        candidates = []
        for ram in range(1, 5):
            for gpu in range(1, 5):
                peak_gpu = max(0, base_gpu) + gpu * block_bytes
                peak_ram = max(0, base_ram) + ram * block_bytes
                if peak_gpu > gpu_budget or peak_ram > ram_budget:
                    continue
                time_s = simulate(costs["pack_s"], costs["copy_s"], costs["compute_s"], ram, gpu)
                # Stalls are measured against a chain with all weights ready.
                first_load = costs["pack_s"][0] + costs["copy_s"][0]
                excess = max(0.0, time_s - first_load - sum(costs["compute_s"]))
                candidates.append(
                    dict(
                        ram=ram,
                        gpu=gpu,
                        predicted_chain_s=time_s,
                        predicted_excess_s=excess,
                        estimated_peak_gpu=peak_gpu,
                        estimated_peak_ram=peak_ram,
                    )
                )
        if not candidates:
            raise RuntimeError(f"No {phase} policy fits frozen budgets")
        best_time = min(row["predicted_chain_s"] for row in candidates)
        # A 1% predicted tie buys no additional buffers. Frozen before evaluation.
        near = [row for row in candidates if row["predicted_chain_s"] <= best_time * 1.01]
        if profile == "guarded":
            best_excess = min(row["predicted_excess_s"] for row in candidates)
            near = [row for row in candidates if row["predicted_excess_s"] <= best_excess + 0.0001]
        chosen = min(
            near,
            key=lambda row: (
                row["estimated_peak_gpu"] + row["estimated_peak_ram"],
                row["gpu"],
                row["ram"],
            ),
        )
        result[phase] = chosen | dict(
            costs=costs,
            candidates=candidates,
            samples=len(rows),
            predicted_zero_excess=chosen["predicted_excess_s"] < 0.0001,
        )
    result["predicted_block_work_s"] = (
        result["t5"]["predicted_chain_s"]
        + result["dit"]["predicted_chain_s"] * generated_chunks * 5
    )
    reduce_fixed = max if profile == "guarded" else statistics.median
    fixed = [
        sum(row["duration_s"] for row in record["resource_events"] if row["event"] == "acquire")
        + statistics.median(
            [row["duration_s"] for row in record["node_times"] if row["node"].startswith("decode_")]
        )
        * generated_chunks
        for record in records
    ]
    result["predicted_fixed_stage_s"] = reduce_fixed(fixed)
    result["predicted_total_s"] = (
        result["predicted_block_work_s"] + result["predicted_fixed_stage_s"]
    )
    result["prediction_limit"] = (
        "Coarse serial/overlap model. Fixed resource loads and decoder costs included; first-use compilation excluded from steady block estimates. Stage estimates may overlap/double-count startup; validate against real timings, no accuracy guarantee."
    )
    if not math.isfinite(result["predicted_block_work_s"]):
        raise ValueError("Nonfinite costs")
    return result
