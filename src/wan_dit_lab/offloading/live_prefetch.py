"""Warmup-only timing for Live's fixed two-slot prefetch, not a FlexGen policy.

No synchronization is added per block. Resolve at the existing chunk fence.
GPU timestamps, rather than an early CPU query, identify late transfers.
"""

import hashlib
import json

import torch

from wan_dit_lab.harness import summarize
from wan_dit_lab.offloading.offload import StreamedRows


def calibration_key(sources, workload, runtime):
    value = json.dumps(dict(sources=sources, workload=workload, runtime=runtime), sort_keys=True)
    return hashlib.sha256(value.encode()).hexdigest()


def summarize_prefetch(rows):
    def summary(samples):
        return dict(consumers=len(samples), late_consumers=sum(r["late_ms"] > 0 for r in samples),
                    late_ms_total=sum(max(r["late_ms"], 0.) for r in samples),
                    **{name: dict(**summarize([r[field] for r in samples]),
                                  max_ms=max(r[field] for r in samples))
                       for name, field in (("copy", "copy_ms"), ("compute", "compute_ms"),
                                           ("dependency_wait", "dependency_wait_ms"))})

    if not rows:
        raise ValueError("No calibration samples")
    return summary(rows) | dict(per_block={str(i): summary([r for r in rows if r["block"] == i])
                                          for i in sorted({r["block"] for r in rows})})


class TracedStreamedRows(StreamedRows):
    """Same two buffers and dependency events, with optional warmup timestamps."""

    instances = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracing, self.samples, self.transfers = True, [], {}
        self.current_sample = None
        self.instances.append(self)

    def event(self, stream=None):
        event = torch.cuda.Event(enable_timing=True)
        event.record(stream or torch.cuda.current_stream(self.device))
        return event

    def copy(self, i, wait=True):
        if not self.tracing:
            return super().copy(i, wait)
        slot = self.slot_of[i]
        with torch.cuda.stream(self.copy_stream):
            if wait:
                self.copy_stream.wait_event(self.freed[slot])
            row = dict(block=i, copy_begin=self.event(self.copy_stream))
            self.slot_rows[slot].copy_(self.host[i], non_blocking=True)
            row["copy_end"] = self.event(self.copy_stream)
            self.copied[slot].record(self.copy_stream)
            self.transfers[i] = row

    def get_next(self):
        i = self.i
        # a copy issued while tracing was off has no timestamps: tracing restarted mid-stream skips it
        if self.tracing and i in self.slot_of and i in self.transfers:
            row = self.transfers.pop(i)
            row["demand"] = self.event()
            block = super().get_next()
            row["begin"] = self.event()
            self.current_sample = row
            return block
        return super().get_next()

    def release(self, block):
        if self.tracing and self.current_sample is not None:
            self.current_sample["end"] = self.event()
            self.samples.append(self.current_sample)
            self.current_sample = None
        return super().release(block)

    def timings(self):
        return [dict(block=r["block"], copy_ms=r["copy_begin"].elapsed_time(r["copy_end"]),
                     compute_ms=r["begin"].elapsed_time(r["end"]),
                     dependency_wait_ms=r["demand"].elapsed_time(r["begin"]),
                     late_ms=r["demand"].elapsed_time(r["copy_end"])) for r in self.samples]

    def drain(self):
        """Timings of the blocks run since the last drain. Call only after completed chunk D2H."""
        rows = self.timings()
        self.samples.clear()
        return rows

    def finish_calibration(self):
        """Call only after completed chunk D2H, which already fences its compute."""
        self.tracing = False
        rows = self.timings()
        self.transfers.clear()
        return summarize_prefetch(rows) | dict(samples=rows,
            policy="fixed two GPU slots, weights in pinned RAM", scope="warmup only")
