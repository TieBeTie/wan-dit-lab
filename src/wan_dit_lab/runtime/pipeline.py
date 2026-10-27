"""Ordered model adapters followed by compilation. No custom compiler framework."""

import copy
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, model_validator

from wan_dit_lab.harness.config import ROOT, Settings, load_config, merge


class Rope(Settings):
    name: Literal["rope"]
    backend: Literal["original", "triton"] = "original"


class CacheStore(Settings):
    name: Literal["kv_store"]
    backend: Literal["original", "triton"] = "original"


class Compile(Settings):
    name: Literal["compile"]
    mode: Literal["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"] = "default"
    options: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def compatible_options(self):
        if self.options and self.mode != "default":
            raise ValueError("Choose a compile mode or explicit options")
        return self

    def kwargs(self):
        return dict(options=self.options) if self.options else dict(mode=self.mode)


class Run(Settings):
    mode: Literal["live", "benchmark"] = "benchmark"
    keep_open: bool = True  # live: the window stays after the measured chunks; false closes it, for scripted runs


class Debug(Settings):
    dump_graph: bool = False
    recompiles: bool = False


Pass = Annotated[Rope | CacheStore | Compile, Field(discriminator="name")]
adapter = TypeAdapter(list[Pass])


def validate(pipeline):
    steps = adapter.validate_python(pipeline)
    names = [step.name for step in steps]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate pipeline step")
    if "compile" in names and names[-1] != "compile":
        raise ValueError("Model replacements must precede compile")
    return steps


def compile_settings(pipeline):
    return next((step.kwargs() for step in validate(pipeline) if step.name == "compile"), {})


def resolve_experiment(raw, seen=()):
    config = copy.deepcopy(raw)
    if base := config.pop("base", None):
        path = (ROOT / Path(base)).resolve()
        if path in seen:
            raise ValueError(f"Configuration inheritance cycle: {path}")
        parent = resolve_experiment(json.loads(path.read_text(encoding="utf-8-sig")), (*seen, path))
        config = merge(parent, config)
    config["run"] = Run.model_validate(config.get("run", {})).model_dump()
    config["debug"] = Debug.model_validate(config.get("debug", {})).model_dump()
    config["harness"] = load_config(config.get("harness")).model_dump()
    config["pipeline"] = [step.model_dump() for step in validate(config.get("pipeline", [{"name": "compile"}]))]
    for variant in config.get("variants", []):
        variant["pipeline"] = [step.model_dump() for step in validate(variant["pipeline"])]
    return config


def live_arguments(config):
    """One workload description shared by the Live and benchmark entry points."""
    return ["--model", config["model"], "--height", str(config["height"]),
            "--width", str(config["width"]), "--seed", str(config["seed"]),
            "--stream-blocks", str(config["stream_blocks"]), "--prompt", config["prompt"],
            "--rope-interpolate" if config["rope_interpolate"] else "--no-rope-interpolate"]


@contextmanager
def install(pipeline):
    """Only this process is changed. Return compile kwargs for model setup."""
    from wan_dit_lab.generation.causal import KVCache
    from wan_dit_lab.models import multi_head_attention as attention
    from wan_dit_lab.models.rope import rope_apply as original_rope
    from wan_dit_lab.quantization import w4a4

    steps = validate(pipeline)
    saved = attention.rope_apply, w4a4.rope_apply, KVCache.store
    compilation = {}
    try:
        for step in steps:
            if step.name == "rope":
                selected = original_rope
                if step.backend == "triton":
                    from wan_dit_lab.kernels.rope_triton import rope_apply

                    selected = rope_apply
                attention.rope_apply = w4a4.rope_apply = selected
            elif step.name == "kv_store":
                KVCache.store = saved[2]
                if step.backend == "triton":
                    from wan_dit_lab.kernels.cache_fp8 import install_cache_store

                    install_cache_store()
            else:
                compilation = step.kwargs()
        yield compilation
    finally:
        attention.rope_apply, w4a4.rope_apply, KVCache.store = saved
