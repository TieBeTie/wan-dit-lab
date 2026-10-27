"""Single entry point for video experiments: one JSON config in; calibration if missing, videos, summary out.

    .venv\\Scripts\\python.exe scripts\\run_experiment.py --config configs\\experiments\\longlive2_cuts_384p.json

As in the RAG runner of the thesis (Master Thesis/experiments/runners/run_rag.py): a top-level section present in the
config turns its stage on, an absent one skips it; the output goes to outputs/experiments/{date}/{time}_{config stem}/
unless "output" names it, with a copy of the config. The config is checked by pydantic before any model is loaded.

    {
      "common":  {...},           # model, resolution, schedule, cache: Common below; extra keys are kept as notes
      "execution": {...},         # resource budgets, compilation, warmup chunks
      "policy": "configs/experiments/<profile>.json" or {...},
      "policy_overrides": {...},  # top-level fields replaced in that policy, e.g. its resource_cost_model
      "calibration": {...},       # optional settings; the calibration itself runs whenever its file is missing
      "runs": [{"id": ..., "prompts": [...], "switch": "shot", "switch_every": 8, "latent_frames": 192}, ...],
      "compare": {...}            # present: the runs side by side in one video
    }

Calibration: when the policy names a cost model ("resource_cost_model": {"calibration": path}) whose file does not
exist (or "calibration": {"force": true}), a short video of this model and resolution runs first with the cost model
off and read-only probes of every live resource (outputs/experiments/2026-10-19/22_Calibrated_prefetch_5s), and its
traces are merged by scripts/calibration/calibrate_resource_costs.py into that path. Each video runs scripts/experiments/offloading_generate.py
in a process of its own; summary.json gets speed, waits for resources, RAM and VRAM of each.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


import argparse
import os
import subprocess
import time
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wan_dit_lab.harness.config import OUTPUTS, located
from wan_dit_lab.json_files import read_json, write_json  # noqa: E402

# the paths are Cyrillic: a console in cp1252 cannot print them (as in run_rag.py)
if (getattr(sys.stdout, "encoding", "") or "").lower().replace("-", "") != "utf8" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
SWITCHES = ("swap", "recache", "blend", "decay", "reanchor", "recache_ring", "shot")  # causal.SWITCHES


class Common(BaseModel):
    """What the model and the stream are: the "common" of offloading_generate.py."""
    # notes such as "weights" or "decoder" go through as they are; "model_config" is the JSON name of model_config_path
    model_config = ConfigDict(extra="allow", populate_by_name=True, protected_namespaces=())

    height: int
    width: int
    fps: int = 16
    latent_frames: int = Field(description="of each run unless the run says otherwise")
    chunk: int = 3
    vae_stride: int = 8
    context_frames: int = 21
    sink_frames: int = 0
    positions: Literal["video", "window"] = "video"
    pc_recent: int | None = None
    cross_cache: bool = True
    context_noise: int = 0
    attention: Literal["cudnn", "efficient", "sage"] = "cudnn"
    compile: str | None = "default"
    shift: float = 5.0
    sampler: Literal["dmd", "unipc"] = "dmd"
    steps: int | None = None  # UniPC steps; None: the 4 of the trained schedule
    kv_fp8: bool = False  # self-attention cache in FP8 (half the bytes; at 704x1280 BF16 needs 10 GB)
    forcing_kv: str | None = None  # Forcing-KV: path of the static heads (forcing_kv_heads.py), Sage over FP8 only
    write_pass: bool = True  # False: no pass at t = 0 to write a chunk to the cache, it keeps the keys and values of
    #   the last step (of x at its noise level): a third less work with 2 steps, outside what the model was trained on
    collect_act_amax: str | None = None  # path: save the largest |x| per input channel of every linear layer
    linear_fp8: Literal["tensor", "row"] | None = None  # linear layers of the blocks in FP8 W8A8, scale per
    #   tensor or per row (token of x, output row of W); None: bfloat16
    multi_shot_rope_offset: int = 8
    rope_dtype: Literal["float32", "float64"] = "float64"  # of the cos and sin tables: float64 as the reference Wan,
    #   float32 our speed-up (E23)
    # off the resolution the model is trained on, its height and width in pixels (None: as trained): Q K^T scaled by
    # sqrt(log_T N / d), T the keys of the same attention there (Jin et al., NeurIPS 2023), and RoPE positions of
    # height and width spread over the trained grid
    attention_entropy_from: tuple[int, int] | None = None
    rope_interpolate_from: tuple[int, int] | None = None
    rope_interpolate_strength: float = 1.0  # of the stretch of rope_interpolate_from, geometric: (trained / now) ** s
    model_config_path: str | None = Field(None, alias="model_config")
    taehv: str | None = None

    @model_validator(mode="after")
    def check(self):
        cell = 2 * self.vae_stride  # a token is a 2x2 patch of latent cells
        assert self.height % cell == 0 and self.width % cell == 0, f"height and width must be multiples of {cell}"
        assert self.latent_frames % self.chunk == 0, f"latent_frames must be a multiple of the chunk ({self.chunk})"
        window = self.context_frames
        assert self.sink_frames == window - self.chunk or 0 <= self.sink_frames <= window - 2 * self.chunk, \
            "the ring takes a whole chunk at a time: sink == window - chunk or sink <= window - 2 chunk"
        for path in (self.model_config_path, self.taehv):
            assert path is None or (ROOT / path).exists(), f"missing file {path}"
        return self

    def dump(self):
        out = self.model_dump(by_alias=True, exclude_none=False)
        return {k: v for k, v in out.items() if v is not None or k in ("pc_recent", "compile")}


class Execution(BaseModel):
    model_config = ConfigDict(extra="allow")
    resource_budgets: dict = {"ram_gib": 14.0, "vram_gib": 7.25}
    require_compilation: bool = True
    warmup_chunks: int = 4


class Run(BaseModel):
    """One video: its prompts, how they change, and what it changes in Common."""
    model_config = ConfigDict(extra="forbid")
    id: str
    prompts: list[str] = []
    prompt_refs: list[str] = []  # "<prompt set JSON>#<id>" (configs/prompts): prompts of a set instead of their text
    seed: int = 0
    switch: str = "swap"
    switch_every: int | None = None  # chunks; None: the prompts share the video equally
    blend_chunks: int = 4
    decay_mu: tuple[float, float] = (0.1, 0.7)
    latent_frames: int | None = None
    common: dict = {}  # overrides of Common for this run only
    policy: dict = {}  # top-level fields of the policy replaced for this run only (e.g. its checkpoint)

    @model_validator(mode="after")
    def check(self):
        assert set(self.switch.split("+")) <= set(SWITCHES), f"unknown switch {self.switch}"
        for ref in self.prompt_refs:
            path, _, key = ref.partition("#")
            texts = {p["id"]: p["text"] for p in read_json(ROOT / path)["prompts"]}
            assert key in texts, f"no prompt {key} in {path}"
            self.prompts.append(texts[key])
        assert self.prompts, "a run needs prompts or prompt_refs"
        return self


class Calibration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    latent_frames: int | None = None  # default: the window full and one chunk more
    probe_repeats: int = 3
    force: bool = False


class Compare(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "comparison"
    runs: list[str] | None = None  # default: all


class Experiment(BaseModel):
    model_config = ConfigDict(extra="allow")  # "question" and other notes
    id: str | None = None
    output: str | None = None
    common: Common
    harness: dict = Field(default_factory=dict)
    execution: Execution = Execution()
    policy: str | dict
    policy_overrides: dict = {}
    calibration: Calibration | None = None
    runs: list[Run]
    compare: Compare | None = None

    @model_validator(mode="after")
    def check(self):
        ids = [r.id for r in self.runs]
        assert len(ids) == len(set(ids)) and "calibration" not in ids, "run ids must be unique and not 'calibration'"
        return self




def merged(under, over):
    """over laid on under: dicts merged key by key, any other value of over replaces that of under."""
    out = dict(under)
    for key, value in over.items():
        out[key] = merged(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def with_base(raw, seen=()):
    """A config with the fields of its "base" config (configs/base) under its own, the base's base first. The base
    is the model as it is, and an experiment writes only what it changes (its speed-ups, its factors)."""
    raw = dict(raw)
    base = raw.pop("base", None)
    if base is None:
        return raw
    assert base not in seen, f"a cycle of base configs at {base}"
    return merged(with_base(read_json(ROOT / base), (*seen, base)), raw)


def video_frames(latent):
    return latent * 4 - 3


def generate(name, out, common, execution, policy, run, label, harness=None):
    """One video by offloading_generate.py in its own process; its record."""
    common = Common(**{**common.dump(), **run.common})
    latent = run.latent_frames or common.latent_frames
    assert latent % common.chunk == 0, f"{name}: latent_frames must be a multiple of {common.chunk}"
    frames = video_frames(latent)
    config = dict(id=f"{label}_{name}", output=str(out), execution=execution.model_dump(), harness=harness or {},
                  common=dict(common.dump(), latent_frames=latent, prompt=run.prompts[0], seed=run.seed,
                              generated_video_frames=frames, saved_video_frames=frames,
                              saved_video_seconds=frames / common.fps))
    case = dict(id=name, prompt=run.prompts[0], prompts=run.prompts, seed=run.seed, switch=run.switch,
                switch_every=run.switch_every, blend_chunks=run.blend_chunks, decay_mu=list(run.decay_mu),
                latent_frames=latent, saved_video_frames=frames, calibration_probe=False)
    case.update(getattr(run, "probe", {}))
    inputs = out / "_inputs"
    policy = {**policy, **run.policy}
    for kind, value in (("config", config), ("case", case), ("policy", policy)):
        write_json(inputs / f"{kind}_{name}.json", value)
    started = time.perf_counter()
    print(f"[{name}] {common.height}x{common.width}, {latent} latent frames, {run.switch} ...", flush=True)
    with open(out / f"{name}.log", "w", encoding="utf-8") as log, open(out / f"{name}.err.log", "w", encoding="utf-8") as err:
        result = subprocess.run(
            [str(PYTHON), str(ROOT / "scripts" / "experiments" / "offloading_generate.py"),
             *(f for kind in ("config", "case", "policy") for f in (f"--{kind}", str(inputs / f"{kind}_{name}.json"))),
             "--output", str(out / name)], cwd=ROOT, stdout=log, stderr=err,
            # without it the cache, the streamed weights and the activations fragment the allocator: at 704x1280
            # 420 MB stayed reserved and unused and the first chunk ran out of memory
            env={**os.environ, "PYTORCH_ALLOC_CONF": os.environ.get("PYTORCH_ALLOC_CONF", "expandable_segments:True")})
    if result.returncode:
        raise SystemExit(f"[{name}] failed ({result.returncode}): see {out / f'{name}.err.log'}")
    print(f"[{name}] done in {time.perf_counter() - started:.0f} s", flush=True)
    return read_json(out / name / "record.json")


class ProbeRun(Run):
    probe: dict = {}


def calibrate(path, out, experiment, policy, label):
    """The policy's calibration file from a short run of this model and resolution, probing every live resource."""
    common, settings = experiment.common, experiment.calibration or Calibration()
    latent = settings.latent_frames or (-(-common.context_frames // common.chunk) + 1) * common.chunk
    count = latent // common.chunk
    probe = {k: v for k, v in policy.items() if k != "resource_cost_model"}
    probe["profile"] = f"{policy.get('profile', 'policy')}_calibration"
    first = experiment.runs[0]
    run = ProbeRun(id="calibration", prompts=first.prompts[:1], seed=first.seed, latent_frames=latent,
                   common=first.common,
                   # the stream and its KV are released after the last denoise node: probe one decode before it
                   probe=dict(calibration_probe=True, calibration_probe_node=f"decode_{count - 2}",
                              calibration_probe_repeats=settings.probe_repeats))
    print(f"calibration {'forced' if settings.force else 'missing'}: {path}; {latent} latent frames", flush=True)
    record = generate("calibration", out, common, experiment.execution, probe, run, label, harness=experiment.harness)
    sys.path.insert(0, str(ROOT / "scripts"))
    from scripts.calibration.calibrate_resource_costs import merge
    merged = merge([record])
    merged["source"] = str((out / "calibration" / "record.json").relative_to(OUTPUTS.parent)).replace("\\", "/")
    merged["model"] = dict(model_config=common.model_config_path, height=common.height, width=common.width,
                           chunk=common.chunk, context_frames=common.context_frames)
    write_json(path, merged)
    print(f"calibration written: {path}", flush=True)


def summary(name, record):
    s = record["harness"]["summary"]
    ready = (s.get("resource_readiness") or {}).get("steady") or {}
    weights = s.get("weight_readiness") or {}
    return dict(run=name, steady_fps=s.get("steady_fps"), chunk_p90_s=s.get("steady_p90_s"),
                resource_waits=ready.get("misses"), late_wait_s=ready.get("late_acquisition_wall_s"),
                weight_copy_waits=weights.get("copy_misses"), rss_peak_gib=s.get("rss_peak_gib"),
                vram_tensors_peak_gib=s.get("vram_peak_gib"), vram_device_peak_gib=s.get("vram_device_peak_used_gib"),
                disk_write_gib=s.get("disk_write_gib"), switch_s=[round(x["duration_s"], 2) for x in record.get("switches", [])],
                video=f"{name}/video.mp4")


def compare(out, settings, runs, fps):
    """The runs side by side, each labelled, with the time and the prompt shown."""
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    chosen = [r for r in runs if settings.runs is None or r.id in settings.runs]
    font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 24)
    readers = [imageio.get_reader(out / r.id / "video.mp4") for r in chosen]
    writer = imageio.get_writer(out / f"{settings.name}.mp4", fps=fps, quality=8, macro_block_size=8)
    for i, frames in enumerate(zip(*readers)):
        tiles = []
        for frame, run in zip(frames, chosen):
            image = Image.fromarray(frame)
            ImageDraw.Draw(image).text((12, 8), f"{run.id}  {i / fps:.1f} s", font=font, fill="white",
                                       stroke_width=3, stroke_fill="black")
            tiles.append(np.asarray(image))
        writer.append_data(np.hstack(tiles))
    writer.close()
    for reader in readers:
        reader.close()
    print(f"comparison: {out / f'{settings.name}.mp4'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--only", nargs="*", help="only these run ids")
    parser.add_argument("--check", action="store_true", help="validate the config and stop")
    args = parser.parse_args()
    raw = with_base(read_json(args.config))  # written whole to the output, the base included
    experiment = Experiment(**raw)
    label = experiment.id or args.config.stem
    now = datetime.now()
    out = located(experiment.output or f"outputs/experiments/{now:%Y-%m-%d}/{now:%H%M%S}_{args.config.stem}")
    policy = experiment.policy
    policy = read_json(ROOT / policy) if isinstance(policy, str) else dict(policy)
    policy.update(experiment.policy_overrides)
    cost = (policy.get("resource_cost_model") or {}).get("calibration")
    runs = [r for r in experiment.runs if not args.only or r.id in args.only]
    calibrating = bool(cost) and (not located(cost).exists() or (experiment.calibration and experiment.calibration.force))
    print(f"{label}: {len(runs)} runs -> {out}; calibration {'will run' if calibrating else 'present' if cost else 'not used'}")
    if args.check:
        return
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "experiment.json", raw)
    if calibrating:
        calibrate(located(cost), out, experiment, policy, label)
    rows = []
    for run in runs:
        rows.append(summary(run.id, generate(run.id, out, experiment.common, experiment.execution, policy, run, label,
                                             harness=experiment.harness)))
        write_json(out / "summary.json", rows)
    if experiment.compare and len(runs) > 1:
        compare(out, experiment.compare, runs, experiment.common.fps)
    for row in rows:
        print(f"{row['run']}: {row['steady_fps']:.2f} fps, waits {row['resource_waits']}, "
              f"RAM {row['rss_peak_gib']} GiB, VRAM {row['vram_tensors_peak_gib']} GiB", flush=True)


if __name__ == "__main__":
    main()
