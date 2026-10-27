"""Typed harness defaults plus an experiment's small override."""

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = Path.home() / "Documents/Experiments/DIT/outputs"  # out of the Obsidian vault: it indexes nothing here


def located(path):
    """A path from a config or a script: outputs/... lives in OUTPUTS, anything else relative is in the project."""
    path = Path(path)
    if path.is_absolute():
        return path
    return OUTPUTS.joinpath(*path.parts[1:]) if path.parts[:1] == ("outputs",) else ROOT / path


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Metrics(Settings):
    latency: bool = True
    fps: bool = True
    wall: bool = True
    memory: bool = True
    metadata: bool = True
    kernels: bool = True
    disk_io: bool = True
    readiness: bool = True
    readiness_steady: bool = False  # Live: GPU timestamps of streamed blocks in the measured chunks too, not only warmup
    display: bool = False  # Live: duration of each window tick and the depth of the frame queue
    stages: bool = False  # Live: DiT and decoder GPU time of a chunk, T5 time of a prompt
    compilation: bool = True


class Timing(Settings):
    mode: Literal["cuda_graph", "cuda_events"] = "cuda_graph"
    settle_ms: float = Field(200., ge=0)


class Checks(Settings):
    tests: list[str] = Field(default_factory=list)
    profiler_attempts: int = Field(2, gt=0)
    require_compilation: bool = True
    max_latency_regression: float = Field(.05, ge=0)
    max_block_relative_error_increase: float = Field(.05, ge=0)


class Memory(Settings):
    sample_interval_s: float = Field(.1, gt=0)
    vram_fraction: float | None = Field(None, gt=0, le=1)


class Output(Settings):
    path: str | None = None


class HarnessConfig(Settings):
    warmup: int = Field(12, ge=0)
    number: int = Field(30, gt=0)
    repeat: int = Field(3, gt=0)
    seed: int = 0
    warmup_chunks: int = Field(6, gt=0)
    measured_chunks: int = Field(24, gt=0)
    timing: Timing = Field(default_factory=Timing)
    metrics: Metrics = Field(default_factory=Metrics)
    checks: Checks = Field(default_factory=Checks)
    memory: Memory = Field(default_factory=Memory)
    output: Output = Field(default_factory=Output)


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(
            result.get(key), dict) else copy.deepcopy(value)
    return result


@lru_cache(maxsize=1)
def defaults():
    return json.loads((ROOT / "configs/harness.json").read_text(encoding="utf-8"))


def load_config(value=None):
    if isinstance(value, HarnessConfig):
        return value.model_copy(deep=True)
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_text(encoding="utf-8-sig"))
    return HarnessConfig.model_validate(merge(defaults(), value or {}))


def workload_config(config):
    """Legacy video configs keep their existing compilation contract."""
    config = copy.deepcopy(config)
    common, execution = config.setdefault("common", {}), config.setdefault("execution", {})
    cfg = load_config(config.get("harness"))
    common.setdefault("compile", "default")
    execution.setdefault("require_compilation", cfg.checks.require_compilation)
    if execution["require_compilation"] and common["compile"] not in (
        "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
    ):
        raise ValueError("Working benchmark requires an explicit torch.compile mode")
    return config
