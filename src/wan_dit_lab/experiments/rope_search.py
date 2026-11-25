"""The pieces of the gradient-free search over the RoPE profile at a lower resolution (scripts/experiments/rope_cma.py).

A point is what generate.rope_sweep of wan.exe takes ({height, width, ntk_pi_mix}): wan.exe runs every chain of a
config at every point of its sweep in one process and writes <run>/<tag>/<chain>.mp4. The loss of a point is the
first chunk's LPIPS after alignment plus 1 minus DINO against the native run of the same seed and chains
(scripts/reports/evaluate_videos.py), averaged over the chains named. Meaningful only with kernels.lora_sum "ordered":
then a point repeats its video bit for bit, and two points differ by the point alone.
"""

import copy
import json
import os
from pathlib import Path


def point(x, space):
    """The sweep point of vector x: each coordinate clipped to its bounds in `space` ([{name, low, high}]) and rounded
    to the thousandths the folder tag of wan.exe keeps."""
    if len(x) != len(space):
        raise ValueError(f"a point of this space takes {len(space)} numbers, got {len(x)}")
    return {s["name"]: round(min(max(float(v), s["low"]), s["high"]), 3) for v, s in zip(x, space)}


def tag(p):
    """The folder wan.exe writes a sweep point into (crates/wan/src/rope.rs point_tag)."""
    base = f"h{round(p['height'] * 1000):04d}-w{round(p['width'] * 1000):04d}"
    return base if p.get("ntk_pi_mix") is None else f"{base}-mix{round(p['ntk_pi_mix'] * 1000):03d}"


def generation_config(base, points, seed, directory):
    """A copy of the pipeline config `base` that sweeps `points` with `seed` into `directory`."""
    config = copy.deepcopy(base)
    config[1]["rope_sweep"] = [dict(p) for p in points]
    config[1]["seed"] = seed
    config[1]["output"]["directory"] = str(directory)
    return config


def first_chunk_loss(row):
    return row["lpips_aligned_first"] + 1.0 - row["dino_reference_first"]


def point_losses(rows, chains):
    """{tag: mean first-chunk loss over `chains`} from the rows of evaluate_videos (run ids <chain>__<tag>)."""
    by_run = {r["run"]: r for r in rows}
    tags = sorted({r["run"].split("__", 1)[1] for r in rows if "__" in r["run"]})
    losses = {}
    for t in tags:
        values = []
        for chain in chains:
            run = f"{chain}__{t}"
            if run not in by_run:
                raise KeyError(f"no evaluation of {run}")
            values.append(first_chunk_loss(by_run[run]))
        losses[t] = sum(values) / len(values)
    return losses


def _link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:  # another drive
        target.write_bytes(Path(source).read_bytes())


def evaluation_layout(run, reference, chains, tags, destination):
    """The folders evaluate_videos.py reads: an experiment of runs <chain>__<tag> (the sweep's videos) and a reference
    of runs <chain> (the native run), each <id>/video.mp4 with an experiment.json; videos are hard links."""
    destination = Path(destination)
    experiment, ref = destination / "runs", destination / "reference"
    runs = []
    for t in tags:
        for chain in chains:
            _link(Path(run) / t / f"{chain}.mp4", experiment / f"{chain}__{t}" / "video.mp4")
            runs.append({"id": f"{chain}__{t}", "prompts": [chain]})
    (experiment / "experiment.json").write_text(json.dumps({"runs": runs}), encoding="utf-8")
    for chain in chains:
        _link(Path(reference) / f"{chain}.mp4", ref / chain / "video.mp4")
    (ref / "experiment.json").write_text(json.dumps({"runs": [{"id": c, "prompts": [c]} for c in chains]}),
                                         encoding="utf-8")
    return experiment, ref


def append_generation(log, generation, vectors, losses, extra):
    """One line of the search's log: what was told to the optimizer, so a stopped search resumes (replay)."""
    record = {"generation": generation, "vectors": [list(map(float, v)) for v in vectors],
              "losses": [float(l) for l in losses], **extra}
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def replay(log):
    """[(vectors, losses)] of every generation in the log, in order; [] without a log."""
    if not Path(log).exists():
        return []
    records = [json.loads(line) for line in Path(log).read_text(encoding="utf-8").splitlines() if line.strip()]
    return [(r["vectors"], r["losses"]) for r in records]
