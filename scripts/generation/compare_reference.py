"""My WanModel against the reference one on the same inputs: the same text embeddings, the same noise, the same
sampler (Euler, shift, steps, guidance). Both models go through the same block offloader, one model at a time.

Takes the settings of a run of generate.py from outputs/<run>.json and writes next to it:
<run>.compare.json (velocity at the first step, final latents, PSNR of the decoded frames),
<run>.compare.mp4 and <run>.compare.png (mine above, the reference below).

    PYTHONPATH=src python scripts/generation/compare_reference.py --run iceland-33f-320p-25s
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
import gc
import json

import imageio.v2 as imageio
import numpy as np
import torch

from wan_dit_lab.generation.sampling import sample
from wan_dit_lab.harness import git_commit, gpu_state
from wan_dit_lab.harness.config import OUTPUTS
from wan_dit_lab.models.checkpoint import (
    REFERENCE,
    load_weights,
    mapped_weights,
    model_config,
)
from wan_dit_lab.models.model import WanModel
from wan_dit_lab.models.rope import rope
from wan_dit_lab.models.text_encoder import encode
from wan_dit_lab.models.vae import decode
from wan_dit_lab.offloading.offload import BlockOffloader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REFERENCE))
from wan.modules.model import WanModel as ReferenceWanModel  # noqa: E402

from scripts.generation.generate import FPS, KEYS, phase  # noqa: E402


class OffloadedBlock(torch.nn.Module):
    """Stands in for one block of a model that loops over its blocks by itself: runs the next block the offloader
    has put on the GPU. The real blocks stay outside the model, so that moving the model leaves them on the CPU."""

    def __init__(self, offloader):
        super().__init__()
        self.offloader = offloader

    def forward(self, *args, **kwargs):
        block = self.offloader.get_next()
        out = block(*args, **kwargs)
        self.offloader.release(block)
        return out


class Reference:
    """The reference model behind the call of mine: model(X [B, C, F, H, W], t [B], context [B, L, D], grid_sizes,
    freqs). It takes lists and the sequence length instead, and computes its own RoPE frequencies."""

    def __init__(self, model, seq_len):
        self.model, self.seq_len = model, seq_len

    def __call__(self, X, t, context, grid_sizes, freqs):
        return self.model(list(X), t, list(context), self.seq_len)


def build(which, config, device, window, ram_window):
    """One of the two models with its blocks in RAM and `window` of them on the GPU; returns it and its offloader."""
    if which == "mine":
        with torch.device("meta"):
            model = WanModel(**{key: config[key] for key in KEYS})
        model.load_state_dict(mapped_weights(), assign=True)
        model.eval()
        model.block_offloader = BlockOffloader(model.blocks, window, device, ram_window=ram_window)
        model.to(device)
        return model, model.block_offloader
    model = ReferenceWanModel(**{key: config[key] for key in KEYS}, model_type="t2v")
    load_weights(model)
    model.eval()
    blocks = list(model.blocks)
    offloader = BlockOffloader(blocks, window, device, ram_window=ram_window)
    model.blocks = torch.nn.ModuleList([OffloadedBlock(offloader) for _ in blocks])
    model.to(device)
    offloader.start()
    return model, offloader


def psnr(a, b):
    """Peak signal-to-noise ratio of two videos with values in [-1, 1], in dB."""
    mse = ((a - b) ** 2).mean().item()
    return float("inf") if mse == 0 else 10 * np.log10(4 / mse)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="name of a run of generate.py in outputs/")
    args = parser.parse_args()
    out = OUTPUTS
    run = json.loads((out / f"{args.run}.json").read_text(encoding="utf-8"))
    device = torch.device("cuda")
    record = {"run": args.run, "commit": git_commit(), "gpu": gpu_state()}

    with phase(record, "text_encoder"):
        context, context_null = encode([run["prompt"], run["negative"]], device).chunk(2)
    config = model_config()
    d = config["dim"] // config["num_heads"]
    freqs = torch.cat([rope(1024, d - 4 * (d // 6)), rope(1024, 2 * (d // 6)), rope(1024, 2 * (d // 6))],
                      dim=-1).to(device)
    latent = (16, (run["frames"] - 1) // 4 + 1, run["height"] // 8, run["width"] // 8)
    grid = (latent[1], latent[2] // 2, latent[3] // 2)
    noise = torch.randn(latent, device=device, generator=torch.Generator(device).manual_seed(run["seed"]))
    contexts = torch.cat([context, context_null]).to(device)
    sizes = torch.tensor([grid, grid])
    t0 = torch.tensor([1000.0, 1000.0], device=device)  # the first step of the sampler with sigma 1

    velocity, latents = {}, {}
    for which in ("mine", "reference"):
        with phase(record, which):
            model, offloader = build(which, config, device, run["window"], run["ram_window"])
            call = model if which == "mine" else Reference(model, grid[0] * grid[1] * grid[2])
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
                velocity[which] = call(torch.stack([noise, noise]), t0, contexts, sizes, freqs).float().cpu()
            latents[which] = sample(call, noise, context.to(device), context_null.to(device), torch.tensor([grid]),
                                    freqs, run["steps"], run["shift"], run["guide"]).cpu()
            offloader.stop()
            del model, call, offloader
            gc.collect()

    def compare(a, b):
        return {"rel_l2": round(((a - b).norm() / b.norm()).item(), 5),
                "cos": round(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(), 6),
                "max_abs": round((a - b).abs().max().item(), 5)}

    record["first_step_velocity"] = compare(velocity["mine"], velocity["reference"])
    record["final_latents"] = compare(latents["mine"], latents["reference"])
    with phase(record, "vae"):
        videos = {which: decode(latents[which].to(device), device) for which in latents}
    record["video_psnr_db"] = round(psnr(videos["mine"], videos["reference"]), 2)

    frames = {k: ((v.permute(1, 2, 3, 0) + 1) * 127.5).round().to(torch.uint8).numpy() for k, v in videos.items()}
    stacked = np.concatenate([frames["mine"], frames["reference"]], axis=1)  # mine above, the reference below
    imageio.mimwrite(out / f"{args.run}.compare.mp4", stacked, fps=FPS, quality=8)
    imageio.imwrite(out / f"{args.run}.compare.png",
                    np.concatenate([stacked[0], stacked[len(stacked) // 2], stacked[-1]], axis=1))
    (out / f"{args.run}.compare.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("first_step_velocity", "final_latents", "video_psnr_db")}), flush=True)


if __name__ == "__main__":
    main()
