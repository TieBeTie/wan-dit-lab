import statistics


def summarize(ts):
    if not ts:
        raise ValueError("No measured durations")
    q = statistics.quantiles(ts, n=100, method="inclusive") if len(ts) > 1 else ts * 99
    return {"median_ms": statistics.median(ts), "p50_ms": q[49], "p90_ms": q[89],
            "p95_ms": q[94], "p99_ms": q[98]}
