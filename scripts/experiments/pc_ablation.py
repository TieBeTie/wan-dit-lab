"""Fixed paired PC/FIFO experiment. No changes to the production sampler."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
import functools
import gc
import hashlib
import json
import os
import random
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "artifacts/pc_eval_deps")]
from wan_dit_lab.json_files import write_json  # noqa: E402

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def prepare(config_path):
    source = ROOT / "reference/DeepForcing/prompts/MovieGenVideoBench.txt"
    prompts = [s.strip() for s in source.read_text(encoding="utf-8").splitlines() if s.strip()]
    rng = random.Random(20261017)
    selected = rng.sample(range(len(prompts)), 16)
    config = {
        "id": "PC_FIFO_BF16_256p_16prompts",
        "output": "outputs/experiments/2026-10-17/10_PC_FIFO_BF16_256p_16prompts",
        "selection_seed": 20261017,
        "prompt_source": str(source.relative_to(ROOT)),
        "prompt_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "prompt_population_size": len(prompts),
        "width": 448,
        "height": 256,
        "latent_frames": 81,
        "fps": 16,
        "weights": "self_forcing_dmd.pt / generator_ema",
        "precision": "BF16 weights and KV",
        "shift": 2.6794565082283346,
        "context_frames": 21,
        "sink_frames": 10,
        "positions": "window",
        "decoder": "TAEHV FP16",
        "compile": "default",
        "attention": "cudnn",
        "cross_cache": True,
        "context_noise": 0,
        "conditions": {"FIFO": {"pc_recent": None}, "PC": {"pc_recent": 1}},
        "primary": {
            "metric": "MUSIQ KonIQ10k raw score, higher is better",
            "frame_indices": [240, 256, 272, 288, 304],
            "definition": "mean MUSIQ of five frames, 15..19 seconds at 1 Hz",
            "pair_difference": "PC minus FIFO",
            "unit": "one independently selected distinct benchmark prompt with one assigned seed",
            "test": "exact two-sided binomial sign test, ties |difference| <= 1e-6 excluded",
            "null": "P(PC score > FIFO score | not tied) = 0.5",
            "alpha": 0.05,
            "all_pairs_required": 16,
            "minimum_wins_for_significance_without_ties": 13,
            "decision": "reject H0 for benefit if p<.05 and wins>losses; reject for harm if reversed; otherwise do not reject H0",
        },
        "secondary": {
            "early_frame_indices": [0, 16, 32, 48, 64],
            "metrics": [
                "early MUSIQ",
                "early minus late MUSIQ (quality drop)",
                "adjacent decoded frame mean absolute RGB difference (descriptive motion proxy)",
                "steady generation seconds per video second",
                "peak allocated GPU memory",
            ],
            "inferential_tests": "none; descriptive only",
        },
        "limitations": [
            "20 s at 256p, one seed per prompt",
            "MUSIQ is frame quality, not prompt adherence or temporal coherence",
            "Not a full VBench evaluation; benchmark prompts only",
            "No early stopping or choosing a different primary metric after seeing results",
            "No automatic change to default inference settings",
        ],
        "pairs": [],
    }
    for i, index in enumerate(selected):
        order = ["FIFO", "PC"]
        rng.shuffle(order)
        config["pairs"].append(
            {
                "id": f"{i + 1:02d}",
                "source_line": index + 1,
                "prompt": prompts[index],
                "seed": rng.randrange(2**31),
                "order": order,
            }
        )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    assert not config_path.exists(), "Protocol already exists; do not overwrite preregistration."
    write_json(config_path, config)
    print(f"Protocol saved before generation: {config_path}", flush=True)


def metric_model():
    import torch
    from pyiqa.archs.musiq_arch import MUSIQ

    torch.manual_seed(0)
    return MUSIQ(pretrained="koniq10k").eval().to("cuda")


def smoke_metric():
    import torch

    model = metric_model()
    with torch.inference_mode():
        value = model(torch.full((1, 3, 256, 448), 0.5, device="cuda"))
    assert torch.isfinite(value).all()
    print("MUSIQ ready", value.item(), flush=True)


def generate(config_path):
    import imageio.v2 as imageio
    import numpy as np
    import torch

    from wan_dit_lab.generation.causal import Stream
    from wan_dit_lab.models.checkpoint import model_config, self_forcing_weights
    from wan_dit_lab.models.model import WanModel
    from wan_dit_lab.models.rope import rope
    from wan_dit_lab.models.text_encoder import TextEncoder
    from wan_dit_lab.models.vae import TAEHVStream
    from wan_dit_lab.offloading.offload import BlockOffloader, OnDevice

    config = json.loads(config_path.read_text(encoding="utf-8"))
    out = ROOT / config["output"]
    meta = out / "_data"
    meta.mkdir(parents=True, exist_ok=True)
    protocol_copy = meta / "protocol.json"
    if protocol_copy.exists():
        assert json.loads(protocol_copy.read_text(encoding="utf-8")) == config
    else:
        write_json(protocol_copy, config)
    with zipfile.ZipFile(meta / "source_snapshot.zip", "w") as archive:
        for path in sorted((ROOT / "src/wan_dit_lab").rglob("*.py")):
            archive.write(path, str(path.relative_to(ROOT)))
        archive.write(Path(__file__), "scripts/experiments/pc_ablation.py")
    provenance = {
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "src/wan_dit_lab").rglob("*.py")
        },
    }
    if (meta / "provenance.json").exists():
        old = json.loads((meta / "provenance.json").read_text())
        assert old["source_sha256"] == provenance["source_sha256"], (
            "Runtime source changed across runs"
        )
    write_json(meta / "provenance.json", provenance)
    contexts_path = meta / "contexts.pt"
    if not contexts_path.exists():
        encoder = TextEncoder(torch.device("cuda"))
        contexts = []
        for pair in config["pairs"]:
            print("Encoding prompt", pair["id"], flush=True)
            contexts.append(encoder([pair["prompt"]]))
        torch.save(torch.cat(contexts), contexts_path)
        del encoder, contexts
        gc.collect()
        torch.cuda.empty_cache()
    contexts = torch.load(contexts_path, weights_only=True)
    model_cfg = model_config()
    keys = [
        "dim",
        "eps",
        "ffn_dim",
        "freq_dim",
        "in_dim",
        "num_heads",
        "num_layers",
        "out_dim",
        "text_len",
    ]
    with torch.device("meta"):
        model = WanModel(**{k: model_cfg[k] for k in keys})
    model.load_state_dict(self_forcing_weights(), assign=True)
    model.eval()
    device = torch.device("cuda")
    placement = config.get("offloading", {})
    if placement.get("enabled", False):
        model.blocks.to(dtype=torch.bfloat16)
        model.block_offloader = BlockOffloader(
            model.blocks, placement["window"], device, ram_window=placement["ram_window"]
        )
    else:
        model.block_offloader = OnDevice(model.blocks, device)
    model.to(device)
    forward = torch.compile(type(model.blocks[0]).forward, mode="default")
    for block in model.blocks:
        block.forward = functools.partial(forward, block)
    d = model_cfg["dim"] // model_cfg["num_heads"]
    freqs = torch.cat(
        [rope(1024, d - 4 * (d // 6)), rope(1024, 2 * (d // 6)), rope(1024, 2 * (d // 6))], dim=-1
    ).to(device)
    decoder = TAEHVStream(device)
    state = {"phase": "generation", "completed": [], "total": len(config["pairs"]) * 2}
    started = time.time()
    for pair_index, pair in enumerate(config["pairs"]):
        directory = out / pair["id"]
        directory.mkdir(exist_ok=True)
        for condition in pair["order"]:
            record_path = directory / (condition + ".json")
            if record_path.exists():
                state["completed"].append(pair["id"] + "/" + condition)
                continue
            state.update(current=pair["id"] + "/" + condition, chunk=0,
                         chunks_total=config["latent_frames"] // 3,
                         elapsed_s=round(time.time() - started))
            write_json(meta / "status.json", state)
            print("START", state["current"], flush=True)
            decoder.last.clear()
            decoder.trim = 3
            generator = torch.Generator(device).manual_seed(pair["seed"])
            stream = Stream(
                model,
                contexts[pair_index : pair_index + 1].to(device),
                freqs,
                config["height"],
                config["width"],
                generator,
                max_frames=config["context_frames"],
                sink_frames=config["sink_frames"],
                attention=config["attention"],
                positions=config["positions"],
                context_noise=0,
                pc_recent=config["conditions"][condition]["pc_recent"],
                cross_cache=True,
                shift=config["shift"],
            )
            writer = imageio.get_writer(directory / (condition + ".mp4"), fps=16, quality=8)
            chunk_times = []
            sample_frames = {}
            prefix = []
            counter = 0
            motion_sum = 0.0
            motion_n = 0
            last_frame = None
            sampled = set(
                config["primary"]["frame_indices"] + config["secondary"]["early_frame_indices"]
            )
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                for chunk_index in range(config["latent_frames"] // 3):
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    latent = stream.next_chunk()
                    if chunk_index < 7:
                        prefix.append(latent.cpu())
                    frames = decoder(latent)
                    frames = (
                        ((frames.permute(1, 2, 3, 0) + 1) * 127.5)
                        .round()
                        .to(torch.uint8)
                        .cpu()
                        .numpy()
                    )
                    for frame in frames:
                        if counter in sampled:
                            sample_frames[counter] = frame.copy()
                        if last_frame is not None:
                            motion_sum += float(
                                np.abs(
                                    frame.astype(np.float32) - last_frame.astype(np.float32)
                                ).mean()
                                / 255
                            )
                            motion_n += 1
                        last_frame = frame
                        writer.append_data(frame)
                        counter += 1
                    torch.cuda.synchronize()
                    chunk_times.append(time.perf_counter() - tick)
                    state.update(chunk=chunk_index + 1, elapsed_s=round(time.time() - started),
                                 last_chunk_s=round(chunk_times[-1], 3),
                                 allocated_gib=torch.cuda.memory_allocated() / 2**30,
                                 peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
                    write_json(meta / "status.json", state)
                    if chunk_index % 9 == 8:
                        print(state["current"], f"chunk {chunk_index + 1}/27", flush=True)
            writer.close()
            assert counter == 321 and set(sample_frames) == sampled
            prefix = torch.cat(prefix, dim=1).float().numpy()
            prefix_hash = hashlib.sha256(prefix.tobytes()).hexdigest()
            np.savez_compressed(
                directory / (condition + ".samples.npz"),
                **{str(k): v for k, v in sample_frames.items()},
            )
            record = {
                "condition": condition,
                "pair": pair,
                "video_frames": counter,
                "video_seconds": counter / 16,
                "common_prefix_latent_frames": 21,
                "common_prefix_sha256": prefix_hash,
                "chunk_s": chunk_times,
                "steady_s_per_video_s": float(np.mean(chunk_times[8:]) / 0.75),
                "motion_rgb_mae": motion_sum / motion_n,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "height": config["height"], "width": config["width"],
                "offloading": placement,
            }
            other = directory / ("FIFO.json" if condition == "PC" else "PC.json")
            if other.exists():
                previous = json.loads(other.read_text(encoding="utf-8"))
                # The methods must agree before the first differently retained cache affects generation.
                assert prefix_hash == previous["common_prefix_sha256"], (
                    "Unmatched pre-treatment trajectories"
                )
            write_json(record_path, record)
            del stream, latent, frames, prefix
            gc.collect()
            torch.cuda.empty_cache()
            state["completed"].append(state["current"])
            write_json(meta / "status.json", state)
            print("DONE", state["current"], f"{sum(chunk_times):.1f}s", flush=True)
    state.update(phase="generated", current=None, elapsed_s=round(time.time() - started))
    decoder.close()
    model.block_offloader.close()
    write_json(meta / "status.json", state)
    print("ALL VIDEOS GENERATED", flush=True)


def evaluate(config_path):
    import importlib.metadata

    import numpy as np
    import torch
    from scipy.stats import binomtest

    config = json.loads(config_path.read_text(encoding="utf-8"))
    out = ROOT / config["output"]
    model = metric_model()
    checkpoint = Path.home() / ".cache/torch/hub/pyiqa/musiq_koniq_ckpt-e95806b9.pth"
    write_json(
        out / "_data/evaluator.json",
        {
            "pyiqa": importlib.metadata.version("pyiqa"),
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "input": ("uncompressed decoded RGB in [0,1], before MP4 encoding, native "
                      f"{config['height']}x{config['width']}"),
        },
    )
    results = []
    with torch.inference_mode():
        for pair in config["pairs"]:
            item = {
                "id": pair["id"],
                "source_line": pair["source_line"],
                "seed": pair["seed"],
                "prompt": pair["prompt"],
            }
            for condition in ["FIFO", "PC"]:
                directory = out / pair["id"]
                record = json.loads((directory / (condition + ".json")).read_text(encoding="utf-8"))
                images = np.load(directory / (condition + ".samples.npz"))
                scores = {}
                for key in images.files:
                    tensor = (
                        torch.from_numpy(images[key].copy())
                        .permute(2, 0, 1)[None]
                        .float()
                        .to("cuda")
                        / 255
                    )
                    scores[int(key)] = float(model(tensor).item())
                    assert np.isfinite(scores[int(key)]), "Non-finite MUSIQ score"
                early = float(
                    np.mean([scores[i] for i in config["secondary"]["early_frame_indices"]])
                )
                late = float(np.mean([scores[i] for i in config["primary"]["frame_indices"]]))
                item[condition] = {
                    "early_musiq": early,
                    "late_musiq": late,
                    "quality_drop": early - late,
                    "frame_scores": scores,
                    "motion_rgb_mae": record["motion_rgb_mae"],
                    "steady_s_per_video_s": record["steady_s_per_video_s"],
                    "peak_allocated_gib": record["peak_allocated_gib"],
                }
            item["delta"] = item["PC"]["late_musiq"] - item["FIFO"]["late_musiq"]
            results.append(item)
            write_json(out / "_data/metrics_progress.json", results)
            print("SCORED", pair["id"], f"delta={item['delta']:.4f}", flush=True)
    differences = np.array([r["delta"] for r in results])
    assert np.isfinite(differences).all()
    assert len(results) == config["primary"]["all_pairs_required"]
    wins = int((differences > 1e-6).sum())
    losses = int((differences < -1e-6).sum())
    test = binomtest(wins, wins + losses, p=0.5, alternative="two-sided") if wins + losses else None
    rng = np.random.default_rng(20261017)
    boot = rng.choice(differences, size=(20000, len(differences)), replace=True).mean(axis=1)
    pvalue = float(test.pvalue) if test else 1.0
    decision = "do_not_reject_H0"
    if pvalue < 0.05:
        decision = "reject_H0_PC_higher_MUSIQ" if wins > losses else "reject_H0_PC_lower_MUSIQ"
    summary = {
        "primary": "tail MUSIQ, PC - FIFO",
        "n_pairs": len(results),
        "wins": wins,
        "losses": losses,
        "ties": len(results) - wins - losses,
        "exact_sign_test_p_two_sided": pvalue,
        "mean_delta": float(differences.mean()),
        "median_delta": float(np.median(differences)),
        "bootstrap_mean_delta_95ci": np.quantile(boot, [0.025, 0.975]).tolist(),
        "decision": decision,
        "conditions": {},
        "pairs": results,
        "limitations": config["limitations"],
    }
    for condition in ["FIFO", "PC"]:
        summary["conditions"][condition] = {
            k: float(np.mean([r[condition][k] for r in results]))
            for k in [
                "early_musiq",
                "late_musiq",
                "quality_drop",
                "motion_rgb_mae",
                "steady_s_per_video_s",
                "peak_allocated_gib",
            ]
        }
    write_json(out / "_data/results.json", summary)
    write_json(
        out / "_data/status.json",
        {"phase": "complete", "completed_pairs": len(results), "decision": decision},
    )
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("pairs", "limitations")}, indent=2
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "smoke-metric", "generate", "evaluate"])
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/experiments/pc_fifo_256p.json"
    )
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.config)
    elif args.action == "smoke-metric":
        smoke_metric()
    elif args.action == "generate":
        generate(args.config)
    else:
        evaluate(args.config)


if __name__ == "__main__":
    main()
