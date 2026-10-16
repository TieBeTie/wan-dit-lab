"""Text to video with my WanModel on an 8 GB GPU: T5, then the DiT, then the VAE, one model at a time.

Writes to outputs/: <name>.mp4, <name>.png (first, middle and last frame side by side), <name>.latents.pt and
<name>.json (prompt, seed, frames, size, steps, guidance, commit, time and peak VRAM of every phase).

    PYTHONPATH=src python scripts/generate.py --prompt "A tree sways in the wind" --frames 33
"""

import argparse
import gc
import json
import time

import imageio.v2 as imageio
import numpy as np
import torch

from wan_dit_lab.checkpoint import ROOT, mapped_weights, model_config
from wan_dit_lab.harness import git_commit, gpu_state
from wan_dit_lab.model import WanModel
from wan_dit_lab.offload import BlockOffloader
from wan_dit_lab.rope import rope
from wan_dit_lab.sampling import sample
from wan_dit_lab.text_encoder import NEGATIVE_PROMPT, encode
from wan_dit_lab.vae import decode

KEYS = ["dim", "eps", "ffn_dim", "freq_dim", "in_dim", "num_heads", "num_layers", "out_dim", "text_len"]
FPS = 16  # sample_fps of the upstream shared config


def phase(record, name):
    """Context of one phase: frees the GPU after it and records its time and peak VRAM."""

    class Phase:
        def __enter__(self):
            torch.cuda.reset_peak_memory_stats()
            self.start = time.perf_counter()

        def __exit__(self, *exc):
            torch.cuda.synchronize()
            record[f"{name}_s"] = round(time.perf_counter() - self.start, 1)
            record[f"{name}_peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            gc.collect()
            torch.cuda.empty_cache()
            print(f"{name}: {record[f'{name}_s']} s, peak VRAM {record[f'{name}_peak_vram_gib']} GiB", flush=True)

    return Phase()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative", default=NEGATIVE_PROMPT)
    parser.add_argument("--frames", type=int, default=33, help="4n + 1")
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--shift", type=float, default=8.0, help="README of the checkpoint: 8 to 12 for 1.3B")
    parser.add_argument("--guide", type=float, default=6.0, help="README of the checkpoint: 6 for 1.3B")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--ram-window", type=int, default=2)
    parser.add_argument("--name", default=None)
    parser.add_argument("--compile", action="store_true", help="torch.compile (inductor) of the two GPU slots of the blocks")
    args = parser.parse_args()
    if (args.frames - 1) % 4 or args.height % 16 or args.width % 16:
        parser.error("frames must be 4n + 1, height and width multiples of 16")

    device = torch.device("cuda")
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    name = args.name or time.strftime("%Y%m%d-%H%M%S")
    record = vars(args) | {"commit": git_commit(), "gpu": gpu_state()}

    with phase(record, "text_encoder"):
        context, context_null = encode([args.prompt, args.negative], device).chunk(2)

    with phase(record, "denoiser"):
        config = model_config()
        with torch.device("meta"):  # the weights stay in the checkpoint file: RAM holds only the pinned window
            model = WanModel(**{key: config[key] for key in KEYS})
        model.load_state_dict(mapped_weights(), assign=True)
        model.eval()
        model.block_offloader = BlockOffloader(model.blocks, args.window, device, ram_window=args.ram_window)
        model.to(device)
        if args.compile:  # the slots are the modules that compute; their parameters are views of the GPU rows
            for slot in model.block_offloader.slots:
                slot.compile()
        d = config["dim"] // config["num_heads"]
        freqs = torch.cat([rope(1024, d - 4 * (d // 6)), rope(1024, 2 * (d // 6)), rope(1024, 2 * (d // 6))],
                          dim=-1).to(device)
        latent = (16, (args.frames - 1) // 4 + 1, args.height // 8, args.width // 8)  # VAE stride (4, 8, 8)
        grid = (latent[1], latent[2] // 2, latent[3] // 2)  # patch (1, 2, 2)
        noise = torch.randn(latent, device=device, generator=torch.Generator(device).manual_seed(args.seed))
        start = time.perf_counter()

        def progress(i):
            print(f"  step {i + 1}/{args.steps}, {time.perf_counter() - start:.0f} s", flush=True)

        latents = sample(model, noise, context.to(device), context_null.to(device), torch.tensor([grid]), freqs,
                         args.steps, args.shift, args.guide, on_step=progress)
        model.block_offloader.stop()
        del model
    torch.save(latents.cpu(), out / f"{name}.latents.pt")

    with phase(record, "vae"):
        video = decode(latents, device)
    frames = ((video.permute(1, 2, 3, 0) + 1) * 127.5).round().to(torch.uint8).numpy()
    imageio.mimwrite(out / f"{name}.mp4", frames, fps=FPS, quality=8)
    imageio.imwrite(out / f"{name}.png", np.concatenate([frames[0], frames[len(frames) // 2], frames[-1]], axis=1))
    record["tokens"] = grid[0] * grid[1] * grid[2]
    (out / f"{name}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out / name}.mp4", flush=True)


if __name__ == "__main__":
    main()
