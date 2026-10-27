"""Explicit resource lifetimes and a small, testable three-engine cost model.

This is not an FX compiler pass or the FlexGen optimizer. Unsupported resources
keep a fixed placement; candidate schedules change only legal staging windows.
"""

import math
import sys
from dataclasses import dataclass
from typing import Callable


@dataclass
class Resource:
    name: str
    acquire: Callable
    release: Callable


@dataclass
class Node:
    name: str
    uses: tuple[str, ...]
    run: Callable


class ResourcePlan:
    def __init__(self, resources, nodes, before=None, after=None):
        self.resources = {r.name: r for r in resources}
        if len(self.resources) != len(resources):
            raise ValueError("Duplicate resource")
        self.nodes, self.values = nodes, {}
        self.before, self.after = before, after
        self.last = {}
        for i, node in enumerate(nodes):
            if len(set(node.uses)) != len(node.uses):
                raise ValueError("Duplicate use")
            for name in node.uses:
                if name not in self.resources:
                    raise ValueError(f"Unknown resource {name}")
                self.last[name] = i

    def __getitem__(self, name):
        return self.values[name]

    def _release(self, name):
        if name in self.values:
            value = self.values.pop(name)
            self.resources[name].release(value)

    def run(self):
        try:
            for i, node in enumerate(self.nodes):
                for name in node.uses:
                    if name not in self.values:
                        self.values[name] = self.resources[name].acquire(self)
                if self.before:
                    self.before(node, self)
                node.run(self)
                if self.after:
                    self.after(node, self)
                for name in reversed(node.uses):
                    if self.last[name] == i:
                        self._release(name)
        finally:
            # Acquire must clean its own partially created object on exception.
            active_error, cleanup_error = sys.exc_info()[0] is not None, None
            for name in list(self.values)[::-1]:
                try:
                    self._release(name)
                except Exception as error:
                    cleanup_error = cleanup_error or error
            if cleanup_error is not None and not active_error:
                raise cleanup_error


def simulate(pack, copy, compute, ram_window, gpu_window):
    """Seconds, serialized pack/copy/compute queues with independent safe slots."""
    if ram_window < 1 or gpu_window < 1 or not (len(pack) == len(copy) == len(compute)):
        raise ValueError("Invalid windows or costs")
    packed, copied, begun, finished = [], [], [], []
    for i, (p, h, c) in enumerate(zip(pack, copy, compute)):
        if min(p, h, c) < 0 or not all(math.isfinite(x) for x in (p, h, c)):
            raise ValueError("Negative duration")
        pin_free = copied[i - ram_window] if i >= ram_window else 0
        # Runtime requests a slot refill after submitting its previous user.
        requested = begun[i - gpu_window] if i >= gpu_window else 0
        packed.append(max(packed[-1] if i else 0, pin_free, requested) + p)
        gpu_free = finished[i - gpu_window] if i >= gpu_window else 0
        copied.append(max(packed[-1], copied[-1] if i else 0, gpu_free) + h)
        begun.append(max(copied[-1], finished[-1] if i else 0))
        finished.append(begun[-1] + c)
    return finished[-1] if finished else 0


def sign_pvalue(wins, losses):
    """Exact two-sided paired sign test, ties omitted."""
    import math

    n = wins + losses
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2**n)
