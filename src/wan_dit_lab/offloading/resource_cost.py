"""Measured three-tier retention optimizer inspired by FlexGen's cost model.

Binary whole-resource placements, rather than FlexGen's tensor fractions.
Profiles are frozen before evaluation; maxima are empirical, not guarantees.
"""
import math
import re
import statistics


def resource_key(name):
    return re.sub(r"/[0-9a-f]{32}$", "/item", name)


class ResourceCosts:
    def __init__(self, samples, profile="median"):
        if profile not in ("median", "guarded"):
            raise ValueError("Unknown cost profile")
        self.profile = profile
        self.samples = samples
        for operations in samples.values():
            for values in operations.values():
                if not values or any(not math.isfinite(x) or x < 0 for x in values):
                    raise ValueError("Costs must be finite nonnegative samples")

    def seconds(self, name, operation):
        values = self.samples.get(resource_key(name), {}).get(operation)
        if values is None:
            raise ValueError(f"Missing calibration: {resource_key(name)} {operation}")
        reduce = min if operation == "compute" and self.profile == "guarded" else (
            max if self.profile == "guarded" else statistics.median)
        return reduce(values)

    def eviction(self, name, tier, dirty, distance):
        # Cost of evicting now and restoring at the next use. Writes count separately.
        cost = self.seconds(name, "disk_restore" if tier == "ram" else "gpu_restore")
        if dirty:
            cost += self.seconds(name, "disk_write" if tier == "ram" else "host_save")
        return cost / max(1, distance)

    def victims(self, rows, required):
        """Minimize estimated lost reuse benefit while freeing enough bytes."""
        from scipy.optimize import Bounds, LinearConstraint, milp
        if required <= 0:
            return []
        if sum(row[1] for row in rows) < required:
            raise MemoryError("No legal placement fits the memory budget")
        costs = [row[2] for row in rows]
        sizes = [row[1] for row in rows]
        result = milp(costs, integrality=[1] * len(rows),
                      bounds=Bounds(0, 1),
                      constraints=LinearConstraint([sizes], required, math.inf),
                      options={"time_limit": 5})
        if not result.success:
            raise RuntimeError(f"Resource placement optimization failed: {result.message}")
        return [row[0] for row, selected in zip(rows, result.x) if selected > .5]


def calibration_from_events(events):
    samples = {}
    for row in events:
        operation = {"write": "disk_write", "save": "host_save",
                     "restore": "gpu_restore", "device_prefetch": "gpu_restore"}.get(row["event"])
        if row["event"] in ("restore", "device_prefetch") and row.get("tier") == "disk":
            # End-to-end disk restore includes mapping faults and memory transfer.
            samples.setdefault(resource_key(row["resource"]), {}).setdefault(
                "disk_restore", []).append(row["duration_s"])
        if operation:
            samples.setdefault(resource_key(row["resource"]), {}).setdefault(
                operation, []).append(row["duration_s"])
    return samples
