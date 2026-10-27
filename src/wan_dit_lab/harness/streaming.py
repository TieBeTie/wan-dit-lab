"""Summaries of already collected records, selected by harness config."""

import statistics

from .config import load_config
from .memory import summarize_memory
from .statistics import summarize


def memory_metrics(record, warmup):
    memory = record.get("memory_samples", [])
    backing = record.get("backing_peak_bytes", {}).get("ram")
    payload = record.get("pinned_staging_peak_bytes", record.get("weight_staging_peak_bytes"))
    result = dict(ram_cache_peak_gib=backing / 2**30 if backing is not None else None,
                  pinned_payload_peak_gib=payload / 2**30 if payload is not None else None,
                  memory=summarize_memory(memory))
    pinned = [r["pinned_peak"] for r in memory if r.get("pinned_peak") is not None]
    result["pinned_peak_gib"] = max(pinned) / 2**30 if pinned else None
    for name, key in (("rss_peak_gib", "rss"), ("private_commit_peak_gib", "private"),
                      ("vram_peak_gib", "peak_allocated"),
                      ("vram_reserved_peak_gib", "peak_reserved"),
                      ("vram_device_peak_used_gib", "device_used")):
        values = [r[key] for r in memory if r.get(key) is not None]
        result[name] = max(values) / 2**30 if values else None
    free = [r["device_free"] for r in memory if r.get("device_free") is not None]
    result["vram_device_min_free_gib"] = min(free) / 2**30 if free else None
    steady_memory = [r for r in memory if r.get("stage", "").startswith(("denoise_", "decode_"))
                     and int(r["stage"].rsplit("_", 1)[-1]) >= warmup]
    for name, key, reduce in (("steady_rss_peak_gib", "rss", max),
                              ("steady_ram_min_available_gib", "ram_available", min)):
        values = [r[key] for r in steady_memory if r.get(key) is not None]
        result[name] = reduce(values) / 2**30 if values else None
    return result


def io_metrics(record):
    events, result = record.get("tensor_resource_events", []), {}
    for operation in ("read", "write"):
        rows = [r for r in events if r["event"] == operation]
        measured = "tensor_resource_events" in record
        result[f"disk_{operation}_gib"] = sum(r["bytes"] for r in rows) / 2**30 if measured else None
        result[f"disk_{operation}_wall_s"] = sum(r["duration_s"] for r in rows) if measured else None
    return result


def readiness_metrics(record, warmup):
    events, result = record.get("tensor_resource_events", []), {}
    consumers = [r for r in events if r["event"] == "consumer_ready"]
    result["resource_readiness"] = dict(
        scope="all measured consumers including warmup; wall spans may overlap GPU work",
        consumers=len(consumers), misses=sum(not r["ready_at_request"] for r in consumers),
        acquisition_wall_s=sum(r["duration_s"] for r in consumers),
        late_acquisition_wall_s=sum(r["duration_s"] for r in consumers if not r["ready_at_request"]),
        prefetch_wait_wall_s=sum(r["duration_s"] for r in events if r["event"] == "prefetch_wait"),
    ) if consumers else None  # missing telemetry is not zero waits
    if consumers:
        steady = [r for r in consumers if r.get("node")
                  and r["node"].startswith(("denoise_", "decode_"))
                  and int(r["node"].rsplit("_", 1)[-1]) >= warmup]
        result["resource_readiness"]["steady"] = dict(consumers=len(steady),
            misses=sum(not r["ready_at_request"] for r in steady),
            late_acquisition_wall_s=sum(r["duration_s"] for r in steady if not r["ready_at_request"]))
        operations = {}
        for row in steady:
            key = row["resource"].split("/")[0] + "/" + row.get("consumer", "unknown")
            summary = operations.setdefault(key, dict(consumers=0, misses=0, late_wall_s=0.))
            summary["consumers"] += 1
            if not row["ready_at_request"]:
                summary["misses"] += 1
                summary["late_wall_s"] += row["duration_s"]
        result["resource_readiness"]["steady_operations"] = operations
    traces = record.get("dit_trace", []) + record.get("t5_trace", [])
    measured_weights = [r for r in traces if "pack_wait_s" in r]
    result["weight_readiness"] = dict(
        consumers=len(measured_weights), pack_queue_wall_s=sum(r["pack_wait_s"] for r in measured_weights),
        copy_misses=sum(not r["copy_ready_at_request"] for r in measured_weights),
        gpu_dependency_wait_s=sum(r["dependency_wait_s"] for r in measured_weights),
        compute_stream_gap_s=sum(r.get("compute_stream_gap_s") or 0 for r in measured_weights),
    ) if measured_weights else None
    if measured_weights:
        phases = {}
        for row in measured_weights:
            node = row.get("node", "initialization")
            if node.startswith("denoise_"):
                phase = "steady_dit" if int(node.rsplit("_", 1)[-1]) >= warmup else "warmup_dit"
            else:
                phase = "initialization"
            summary = phases.setdefault(phase, dict(consumers=0, compute_s=0.,
                compute_stream_gap_s=0., gpu_dependency_wait_s=0.))
            summary["consumers"] += 1
            summary["compute_s"] += row["compute_s"]
            summary["compute_stream_gap_s"] += row.get("compute_stream_gap_s") or 0.
            summary["gpu_dependency_wait_s"] += row["dependency_wait_s"]
        result["weight_readiness"]["phases"] = phases
    return result


def stream_summary(record, warmup=None, *, config=None):
    cfg = load_config(config)
    warmup = cfg.warmup_chunks if warmup is None else warmup
    chunks = [r for r in record["chunks"] if r["index"] >= warmup and r["delivery_s"] is not None]
    if not chunks:
        return {}
    times = [r["delivery_s"] for r in chunks]
    result = dict(intervals=len(times), warmup_chunks=warmup, full_run_s=record.get("full_run_s"))
    if cfg.metrics.latency:
        q = summarize([value * 1000 for value in times])
        result.update(steady_median_s=statistics.median(times), steady_mean_s=statistics.mean(times),
                      **{f"steady_{name}_s": q[f"{name}_ms"] / 1000 for name in ("p90", "p95", "p99")})
    if cfg.metrics.fps:
        result["steady_fps"] = sum(r["frames"] for r in chunks) / sum(times)
    for enabled, collect in ((cfg.metrics.memory, lambda: memory_metrics(record, warmup)),
                             (cfg.metrics.disk_io, lambda: io_metrics(record)),
                             (cfg.metrics.readiness, lambda: readiness_metrics(record, warmup))):
        if enabled:
            result.update(collect())
    return result
