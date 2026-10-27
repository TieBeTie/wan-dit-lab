"""Optional CUDA event profiling; resolve only at existing chunk/phase fences."""

import time

import torch


class OffloadTrace:
    def __init__(self, label):
        self.label = label
        self.rows = []
        self.copies = []
        self.current = None
        self.node = "initialization"

    @staticmethod
    def event(stream=None):
        event = torch.cuda.Event(enable_timing=True)
        event.record(stream)
        return event

    def begin(self, row):
        row["node"] = self.node
        stream = torch.cuda.current_stream()
        row["copy_ready_at_request"] = row["copied"].query()
        row["demand"] = self.event(stream)
        stream.wait_event(row["copied"])
        row["begin"] = self.event(stream)
        self.current = row

    def end(self):
        row = self.current
        row["end"] = self.event()
        self.rows.append(row)
        self.current = None

    def resolve(self, start=0):
        result = []
        for index, row in enumerate(self.rows[start:], start=start):
            previous = self.rows[index - 1] if index else None
            result.append(
                dict(
                    label=self.label,
                    node=row["node"],
                    block=row["block"],
                    pack_s=row["pack_s"],
                    pack_wait_s=row.get("pack_wait_s", 0.),
                    pack_ready_at_request=row.get("pack_ready_at_request"),
                    copy_ready_at_request=row.get("copy_ready_at_request"),
                    copy_s=row["copy_begin"].elapsed_time(row["copied"]) / 1000,
                    compute_s=row["begin"].elapsed_time(row["end"]) / 1000,
                    dependency_wait_s=row["demand"].elapsed_time(row["begin"]) / 1000,
                    compute_stream_gap_s=(previous["end"].elapsed_time(row["demand"]) / 1000
                                          if previous and previous["node"] == row["node"] else None),
                )
            )
        return result


def clock():
    return time.perf_counter()
