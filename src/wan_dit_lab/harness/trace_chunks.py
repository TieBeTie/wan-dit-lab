"""Chunks, passes and idle time of a wan.exe run read from an Nsight Systems trace (sqlite export). Measurement only.

    .venv\\Scripts\\python.exe -m wan_dit_lab.harness.trace_chunks <trace.sqlite> <out.json> [blocks] [passes]

A block call holds exactly one self-attention Sage kernel (the longest launch of the Sage kernel, the cross-attention
launch of the same kernel is short). Its start anchors the call. The anchors of a run come in the order chunk, pass,
block, so `blocks * passes` anchors make a chunk and `blocks` anchors a pass.
"""

import json
import sqlite3
import sys

SAGE = "qk_int_sv_f8_attn_kernel"


def busy(intervals):
    """Length of the union of [(start, end)]."""
    total, edge = 0, None
    for start, end in sorted(intervals):
        if edge is None or start > edge:
            total += end - start
            edge = end
        elif end > edge:
            total += end - edge
            edge = end
    return total


def idle(intervals, start, end):
    """Time in [start, end] covered by none of the intervals."""
    clipped = [(max(a, start), min(b, end)) for a, b in intervals if b > start and a < end]
    return (end - start) - busy(clipped)


def split(anchors, blocks, passes):
    """[chunk][pass] -> the anchors of that pass; the number of anchors must be a whole number of chunks."""
    per = blocks * passes
    if len(anchors) % per:
        raise ValueError(f"{len(anchors)} anchors do not make whole chunks of {blocks} blocks x {passes} passes")
    return [[anchors[c * per + p * blocks:c * per + (p + 1) * blocks] for p in range(passes)]
            for c in range(len(anchors) // per)]


def durations(parts, end):
    """(chunk durations, [[pass durations] per chunk]) from the first anchor of the next unit, the last one up to end."""
    starts = [[p[0] for p in chunk] for chunk in parts]
    flat = [s for chunk in starts for s in chunk] + [end]
    chunk_starts = [chunk[0] for chunk in starts] + [end]
    chunk_ns = [chunk_starts[i + 1] - chunk_starts[i] for i in range(len(parts))]
    k = len(starts[0])
    pass_ns = [[flat[c * k + p + 1] - flat[c * k + p] for p in range(k)] for c in range(len(parts))]
    return chunk_ns, pass_ns


def by_name(kernels, start, stop):
    """{name: (launches, total ns)} of the kernels that start in [start, stop)."""
    out = {}
    for s, e, n in kernels:
        if start <= s < stop:
            count, total = out.get(n, (0, 0))
            out[n] = (count + 1, total + e - s)
    return out


def read(path):
    """Kernel rows (start, end, name) and copy rows (start, end, bytes, kind) of the trace, in start order."""
    db = sqlite3.connect(path)
    kernels = db.execute("select k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k "
                         "join StringIds s on s.id = k.shortName order by k.start").fetchall()
    copies = db.execute("select start, end, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY order by start").fetchall()
    return kernels, copies


def analyse(kernels, copies, blocks, passes, min_sage_ns=2_000_000):
    anchors = [s for s, e, n in kernels if n == SAGE and e - s >= min_sage_ns]
    parts = split(anchors, blocks, passes)
    end = max(e for s, e, n in kernels)
    chunk_ns, pass_ns = durations(parts, end)
    spans = [(s, e) for s, e, n in kernels]
    first = anchors[0]
    out = []
    for c, chunk in enumerate(parts):
        start = chunk[0][0]
        stop = start + chunk_ns[c]
        row = {"chunk": c, "chunk_s": chunk_ns[c] / 1e9, "passes_s": [x / 1e9 for x in pass_ns[c]],
               "gpu_idle_s": idle(spans, start, stop) / 1e9,
               "sage_self_ms_per_block": sum(e - s for s, e, n in kernels if n == SAGE and e - s >= min_sage_ns
                                             and start <= s < stop) / (blocks * passes) / 1e6,
               "copy_s": busy([(a, b) for a, b, size, kind in copies if b > start and a < stop]) / 1e9}
        out.append(row)
    return {"first_anchor_ns": first, "chunks": out}


COPY, STREAM_SYNC, EVENT_SYNC = "cuMemcpyDtoHAsync_v2", "cuStreamSynchronize", "cuEventSynchronize"


def ready_moments(calls, copies):
    """(moments, flush) at which wan.exe calls Probe::ready, from driver API calls [(start, end, name)] in start order.

    Generator::advance ends with start_decode (one cuMemcpyDtoHAsync per latent frame, `copies` of them), then
    metrics.chunk, whose debug "transfers" synchronizes the compute stream, then ready. A moment is the end of that stream
    sync, or of the last copy call when there is none before the next decode. The flush is the last cuEventSynchronize
    (finish_decode of Generator::flush) after the last decode."""
    moments, run, i = [], [], 0
    while i < len(calls):
        start, end, name = calls[i]
        if name == COPY:
            run.append(end)
            i += 1
            continue
        if len(run) >= copies:
            moment = run[-1]
            if name == STREAM_SYNC:
                moment = end
            moments.append(moment)
        run = []
        i += 1
    if len(run) >= copies:
        moments.append(run[-1])
    last = moments[-1] if moments else None
    flush = max((e for s, e, n in calls if n == EVENT_SYNC and last is not None and s >= last), default=None)
    return moments, flush


def _mean(frames, seconds):
    return {"frames": frames, "seconds": seconds, "fps": frames / seconds if seconds else None}


def metric_samples(moments, flush, frames, full_from=None):
    """The samples of Totals (crates/wan/src/telemetry.rs) for ready at `moments` and at `flush`, `frames[k]` the frames
    of chunk k. ready at moments[k] emits chunk k - 1 (advance queues chunk k before it waits for k - 1), the first
    one is the warmup, the flush emits the last chunk over the gap after the last advance.

    metric: what wan.exe reports. computed: the frames of the chunks computed in the counted time (2 to n - 1).
    full_window: the same from chunk `full_from` on. last_chunk: the last interval with the last chunk's frames."""
    n = len(moments)
    samples = [(frames[k - 1], moments[k] - moments[k - 1]) for k in range(2, n)] + [(frames[n - 1], flush - moments[n - 1])]
    out = {"warmup": {"frames": frames[0], "seconds": None}, "samples": samples,
           "metric": _mean(sum(f for f, _ in samples), sum(s for _, s in samples)),
           "computed": _mean(sum(frames[2:]), moments[n - 1] - moments[1]),
           "last_chunk": _mean(frames[n - 1], moments[n - 1] - moments[n - 2])}
    if full_from is not None:
        out["full_window"] = _mean(sum(frames[full_from:]), moments[n - 1] - moments[full_from - 1])
    return out


def read_calls(path):
    """Driver API calls (start, end, name) of copies and syncs, in start order."""
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    return db.execute("select r.start, r.end, s.value from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s "
                      "on s.id = r.nameId where s.value in (?, ?, ?) order by r.start",
                      (COPY, STREAM_SYNC, EVENT_SYNC)).fetchall()


def main():
    path, dest = sys.argv[1], sys.argv[2]
    blocks = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    passes = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    kernels, copies = read(path)
    result = analyse(kernels, copies, blocks, passes)
    with open(dest, "w") as f:
        json.dump(result, f, indent=1)
    for row in result["chunks"]:
        print(row)


if __name__ == "__main__":
    main()
