"""Record one stateful generation, without replaying it for probes."""

import copy

import torch

from .config import load_config, workload_config
from .provenance import metadata
from .streaming import stream_summary


class WorkloadRun:
    """Streaming adapter of the common harness, without repeating stateful work."""

    def __init__(self, label, config, case, policy, device=0):
        self.config = workload_config(config)
        self.harness = load_config(config.get("harness"))
        self.before = dict(torch._dynamo.utils.counters["stats"])
        self.header = dict(label=label, function="streaming_video", number=1, repeat=1,
                           config=self.harness.model_dump(),
                           settings=self.config["common"] | case, policy=copy.deepcopy(policy),
                           **(metadata(device) if self.harness.metrics.metadata and
                              self.config["execution"].get("record_metadata", True) else {}))

    def finish(self, record):
        stats = torch._dynamo.utils.counters["stats"]
        record["torch_compile"] = dict(
            mode=self.config["common"]["compile"],
            unique_graphs=stats["unique_graphs"] - self.before.get("unique_graphs", 0),
            calls_captured=stats["calls_captured"] - self.before.get("calls_captured", 0),
            unimplemented=dict(torch._dynamo.utils.counters["unimplemented"]),
        )
        warmup = self.config["execution"].get("warmup_chunks", 10)
        record["harness"] = self.header | dict(warmup_chunks=warmup,
            summary=stream_summary(record, warmup, config=self.harness), compilation=record["torch_compile"])

    def validate(self, record):
        if self.config["execution"]["require_compilation"] and record["torch_compile"]["unique_graphs"] < 1:
            raise RuntimeError("Compilation required, but no compiled graphs were captured")
