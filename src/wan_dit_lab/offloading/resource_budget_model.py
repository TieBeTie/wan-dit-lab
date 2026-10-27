"""CPU-only replay of retained backing and measured I/O costs.

GPU residency/use sequence is held fixed from a completed reference trace.
Reads are conservatively serialized; this does not simulate asynchronous overlap.
The result is a budget prediction to validate, not a latency guarantee.
"""
import math
from collections import deque
from functools import lru_cache

from wan_dit_lab.offloading.resource_cost import ResourceCosts, resource_key


class BackingReplay:
    def __init__(self, events, costs, recent_limit=8):
        self.events, self.costs, self.recent_limit = events, costs, recent_limit
        self.cost_table = {name: {op: costs.seconds(name, op) for op in ops}
                           for name, ops in costs.samples.items()}
        cross = list(dict.fromkeys(e["resource"] for e in events
                     if e["event"] == "save" and e["resource"].startswith("cross_kv/")))
        self.cycles = ({f"self_kv/{i}": i for i in range(30)},
                       {name: i for i, name in enumerate(cross)})

    def seconds(self, name, operation):
        try:
            return self.cost_table[resource_key(name)][operation]
        except KeyError:
            raise ValueError(f"Missing calibration: {resource_key(name)} {operation}") from None

    @lru_cache(maxsize=8192)
    def _solve(self, values, required):
        rows = [(i, size, cost) for i, (size, cost) in enumerate(values)]
        return tuple(self.costs.victims(rows, required))

    def run(self, budget, warmup=10):
        entries, recent, cursor = {}, deque(), [-1, -1]
        used = peak = chunk = 0
        read_bytes = write_bytes = 0
        read_s = write_s = 0.
        per_chunk = {}

        def touch(name):
            for i, positions in enumerate(self.cycles):
                if name in positions:
                    cursor[i] = positions[name]

        def distance(name):
            for i, positions in enumerate(self.cycles):
                if name in positions and cursor[i] >= 0:
                    return (positions[name] - cursor[i]) % len(positions) or len(positions)
            return 1

        def io(name, operation, size):
            nonlocal read_bytes, write_bytes, read_s, write_s
            seconds = self.seconds(name, operation)
            per_chunk[chunk] = per_chunk.get(chunk, 0.) + seconds
            if operation == "disk_write":
                write_bytes += size
                write_s += seconds
            else:
                read_bytes += size
                read_s += seconds

        def spill(name):
            nonlocal used
            entry = entries[name]
            if not entry["ram"]:
                return
            if not entry["file"]:
                io(name, "disk_write", entry["size"])
                entry["file"] = True
            entry["ram"] = False
            used -= entry["size"]

        def delete(name):
            nonlocal used
            entry = entries.pop(name, None)
            if entry and entry["ram"]:
                used -= entry["size"]

        for event in self.events:
            kind, name = event["event"], event.get("resource", "")
            if kind == "save":
                touch(name)
                size = event["bytes"]
                old = entries.get(name)
                credit = old["size"] if old and old["ram"] else 0
                admit = size <= budget
                if admit:
                    required = used + max(0, size - credit) - budget
                    if required > 0:
                        candidates = [n for n, e in entries.items() if e["ram"] and n != name]
                        values = tuple((entries[n]["size"], (self.seconds(n, "disk_restore") + (0 if entries[n]["file"] else
                             self.seconds(n, "disk_write"))) / distance(n)) for n in candidates)
                        selected = self._solve(values, required)
                        for i in selected:
                            spill(candidates[i])
                delete(name)
                entries[name] = dict(size=size, ram=admit, file=not admit)
                if admit:
                    used += size
                else:
                    io(name, "disk_write", size)
                peak = max(peak, used)
                if used > budget:
                    raise AssertionError("Replay exceeded cache budget")
                if name.startswith("history/recent/"):
                    recent.append(name)
                    if len(recent) > self.recent_limit:
                        delete(recent.popleft())
                if name.startswith("latents/"):
                    chunk += 1
            elif kind == "restore":
                if name not in entries:
                    raise ValueError(f"Missing lifetime in trace: {name}")
                if not entries[name]["ram"]:
                    io(name, "disk_restore", entries[name]["size"])
                touch(name)
            elif kind == "prefetch":
                # Replay only observed prefetch opportunities. A lower-budget
                # run may expose new ones, hence use the serialized upper estimate.
                entry = entries.get(name)
                if entry and not entry["ram"] and used + entry["size"] <= budget:
                    io(name, "disk_restore", entry["size"])
                    entry["ram"] = True
                    used += entry["size"]
                    peak = max(peak, used)
        steady = sum(seconds for i, seconds in per_chunk.items() if warmup <= i < chunk)
        return dict(cache_bytes=budget, peak_bytes=peak, read_bytes=read_bytes,
                    write_bytes=write_bytes, serialized_read_s=read_s,
                    serialized_write_s=write_s, steady_io_s=steady,
                    io_by_chunk=per_chunk, solver_cache=self._solve.cache_info()._asdict())


def predict_budgets(record, calibration, budgets, warmup=10, tolerance=.03):
    costs = ResourceCosts(calibration["samples"], "median")
    replay = BackingReplay(record["tensor_resource_events"], costs,
                          record.get("protocol_recent_limit", 8))
    reference_budget = round(record["resource_placements"]["backing_ram_gib"] * 2**30)
    reference = replay.run(reference_budget, warmup)
    deliveries = [r["delivery_s"] for r in record["chunks"][warmup:] if r["delivery_s"] is not None]
    observed_s = sum(deliveries)
    if not deliveries or not math.isfinite(observed_s):
        raise ValueError("No measured steady reference")
    rows = []
    for budget in sorted(set(budgets)):
        result = replay.run(budget, warmup)
        predicted_s = observed_s + result["steady_io_s"] - reference["steady_io_s"]
        rows.append(result | dict(predicted_steady_s=predicted_s,
                                  predicted_mean_chunk_s=predicted_s / len(deliveries),
                                  predicted_slowdown=predicted_s / observed_s - 1,
                                  eligible=predicted_s <= observed_s * (1 + tolerance)))
    eligible = [r for r in rows if r["eligible"]]
    return dict(selected_cache_bytes=eligible[0]["cache_bytes"] if eligible else None,
                reference_replay=reference, observed_reference_steady_s=observed_s,
                warmup_chunks=warmup, intervals=len(deliveries), tolerance=tolerance,
                scope="Fixed GPU access trace; CPU cache replay; serialized I/O estimate; validate with video",
                rows=rows)
