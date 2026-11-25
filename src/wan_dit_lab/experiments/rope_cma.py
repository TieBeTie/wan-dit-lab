"""CMA-ES (pycma) over the RoPE profile of a lower resolution, scored by the first chunk against the native size.

    cd src
    ..\\.venv\\Scripts\\python.exe -m wan_dit_lab.experiments.rope_cma <spec.json>

A module, not a script under scripts\\experiments\\: the GPU queue of wan.exe (crates/wan/src/queue.rs gpu_processes)
counts every process run from scripts\\experiments\\ as a GPU job, so a wan.exe started by such a driver waits for its
own parent forever (2026-11-11: the first search stood three hours at its first point).

The spec (no defaults, every key named; paths relative to the repository or absolute):
    base_config      pipeline config at the lower size, kernels.lora_sum "ordered", one chunk per chain
    references       {seed: run folder of the native config with the same chains and seed}
    train_seeds      seeds of the search; held_out_seeds: of the final check
    train_chains     chains scored in the search; held_out_chains: only in the final check
    space            [{name: height | width | ntk_pi_mix, low, high}]
    x0, sigma0, popsize, max_points, cma_seed
    baselines        sweep points checked with the best one at the end
    work             folder (not on the system SSD) for runs, evaluations and log.jsonl
    eval_device      --device of evaluate_videos.py ("cuda": the card is free between two wan.exe runs)
Every generation is appended to <work>/log.jsonl, a second start replays it and goes on. wan.exe runs through its
queue (bin\\wan.exe --config), one process per seed and generation.
"""
import json
import subprocess
import sys
from pathlib import Path

import cma

from wan_dit_lab.experiments import rope_search as rs

ROOT = Path(__file__).resolve().parents[3]
KEYS = {"base_config", "references", "train_seeds", "held_out_seeds", "train_chains", "held_out_chains", "space", "x0",
        "sigma0", "popsize", "max_points", "cma_seed", "baselines", "work", "eval_device"}


def run_sweep(base, points, seed, chains, folder):
    """One wan.exe process: `points` for `seed` over `chains`, returns its run folder (<folder>/<timestamp>)."""
    config = rs.generation_config(base, points, seed, folder.as_posix())
    config[1]["prompts"]["chains"] = list(chains)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "config.json"
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(folder / "wan.log", "w", encoding="utf-8") as log:
        code = subprocess.call([str(ROOT / "bin" / "wan.exe"), "--config", str(path)], cwd=ROOT, stdout=log,
                               stderr=subprocess.STDOUT)
    if code:
        raise RuntimeError(f"wan.exe failed ({code}), see {folder / 'wan.log'}")
    return sorted(p for p in folder.iterdir() if p.is_dir() and p.name != "eval")[-1]


def evaluate(run, reference, chains, tags, folder, name, device):
    """evaluate_videos.py in a process of its own: its models on the card only while it runs. Kept in this process
    (evaluate_videos.Evaluator) they would hold a CUDA context of some 0.3 GB beside wan.exe, which at 608x1120 peaks
    at 7.8 of 8 GB."""
    experiment, ref = rs.evaluation_layout(run, reference, chains, tags, folder)
    with open(folder / "evaluate.log", "w", encoding="utf-8") as log:
        code = subprocess.call([sys.executable, str(ROOT / "scripts" / "reports" / "evaluate_videos.py"),
                                "--experiment", str(experiment), "--reference", str(ref), "--name", name,
                                "--device", device], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if code:
        raise RuntimeError(f"evaluate_videos failed ({code}), see {folder / 'evaluate.log'}")
    rows = json.loads((experiment / f"evaluation_{name}.json").read_text(encoding="utf-8"))
    return [r for r in rows if isinstance(r, dict) and "lpips_aligned_first" in r]


def score_groups(spec, base, points, seeds, groups, folder):
    """{group: ({tag: mean first-chunk loss over seeds and the group's chains}, per seed)}. One wan.exe run per seed
    over the chains of every group in the order given, which must be the order of the references' runs: the noise of
    a seed runs on through the chains, so a chain scored without the chains before it is another video."""
    chains = [c for group in groups.values() for c in group]
    tags = sorted({rs.tag(p) for p in points})
    unique = [next(p for p in points if rs.tag(p) == t) for t in tags]
    rows_by_seed = {}
    for seed in seeds:
        run = run_sweep(base, unique, seed, chains, folder / f"s{seed}")
        rows_by_seed[seed] = evaluate(run, spec["references"][str(seed)], chains, tags, folder / f"s{seed}" / "eval",
                                      f"s{seed}", spec["eval_device"])
    out = {}
    for name, group in groups.items():
        per_seed = {seed: rs.point_losses(rows, group) for seed, rows in rows_by_seed.items()}
        out[name] = ({t: sum(per_seed[s][t] for s in seeds) / len(seeds) for t in tags}, per_seed)
    return out


def score(spec, base, points, seeds, chains, folder):
    """{tag: mean first-chunk loss over seeds and chains}, and per seed."""
    return score_groups(spec, base, points, seeds, {"all": chains}, folder)["all"]


def final_check(spec, base, candidates, folder):
    """The candidates on every seed, training and held-out chains scored apart from one run of all of them."""
    seeds = sorted(set(spec["train_seeds"]) | set(spec["held_out_seeds"]))
    groups = {"train": spec["train_chains"], "held_out": spec["held_out_chains"]}
    return seeds, {name: {"mean": mean, "per_seed": per_seed}
                   for name, (mean, per_seed) in score_groups(spec, base, candidates, seeds, groups, folder).items()}


def main(spec_path):
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    if set(spec) != KEYS:
        raise SystemExit(f"spec: missing {sorted(KEYS - set(spec))}, unknown {sorted(set(spec) - KEYS)}")
    base = json.loads((ROOT / spec["base_config"]).read_text(encoding="utf-8"))
    if base[0]["kernels"]["lora_sum"] != "ordered":
        raise SystemExit("base_config: kernels.lora_sum must be ordered, or a point does not repeat its video")
    work = Path(spec["work"])
    work.mkdir(parents=True, exist_ok=True)
    log = work / "log.jsonl"
    space = spec["space"]
    es = cma.CMAEvolutionStrategy(spec["x0"], spec["sigma0"], {
        "bounds": [[s["low"] for s in space], [s["high"] for s in space]], "popsize": spec["popsize"],
        "seed": spec["cma_seed"], "verbose": -9})
    told = 0
    for vectors, losses in rs.replay(log):
        es.ask()
        es.tell(vectors, losses)
        told += len(vectors)
    generation = len(rs.replay(log))
    while told < spec["max_points"] and not es.stop():
        vectors = es.ask()
        points = [rs.point(x, space) for x in vectors]
        losses, per_seed = score(spec, base, points, spec["train_seeds"], spec["train_chains"], work / f"g{generation:03d}")
        values = [losses[rs.tag(p)] for p in points]
        es.tell(vectors, values)
        rs.append_generation(log, generation, vectors, values, {"points": points, "per_seed": per_seed})
        told += len(vectors)
        best = min(zip(values, points), key=lambda v: v[0])
        print(f"generation {generation}: best {best[0]:.4f} at {rs.tag(best[1])}, {told} points", flush=True)
        generation += 1
    # the final check: the best point told and the baselines, on every seed, training and held-out chains apart
    history = [(l, v) for vectors, losses in rs.replay(log) for v, l in zip(vectors, losses)]
    best = rs.point(min(history)[1], space)
    candidates = [best] + [dict(b) for b in spec["baselines"]]
    seeds, final = final_check(spec, base, candidates, work / "final")
    report = {"best": best, "best_tag": rs.tag(best), "baselines": [rs.tag(b) for b in spec["baselines"]],
              "points_told": told, "seeds": seeds, "final": final}
    (work / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["final"], indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
