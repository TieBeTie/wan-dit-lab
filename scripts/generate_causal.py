"""Text to video with Self-Forcing on my WanModel: four steps per chunk of 3 latent frames, no guidance, the keys
and values of the finished frames in a cache. T5, then the DiT, then the VAE, one model at a time.

Writes to outputs/: <name>.mp4, <name>.png (first, middle and last frame), <name>.latents.pt and <name>.json
(settings, commit, time and peak VRAM of every phase, time of every chunk).

    PYTHONPATH=src python scripts/generate_causal.py --prompt "A tree sways in the wind" --latent-frames 9
"""

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from wan_dit_lab import causal
from wan_dit_lab.checkpoint import ROOT, model_config, self_forcing_weights
from wan_dit_lab.harness import git_commit, gpu_state
from wan_dit_lab.model import WanModel
from wan_dit_lab.fp8 import to_fp8
from wan_dit_lab.offload import BlockOffloader, OnDevice
from wan_dit_lab.rope import rope
from wan_dit_lab.text_encoder import encode
from wan_dit_lab.vae import TAEHVStream, decode

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import FPS, KEYS, phase  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--latent-frames", type=int, default=9, help="a multiple of 3; video frames 4F - 3")
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--ram-window", type=int, default=2)
    parser.add_argument("--on-gpu", action="store_true", help="all blocks on the GPU in bfloat16, no offloading")
    parser.add_argument("--compile", nargs="?", const="default", default=None,
                        help="torch.compile (inductor) of the modules that compute; a mode may follow: reduce-overhead")
    parser.add_argument("--context-frames", type=int, default=causal.CONTEXT,
                        help="latent frames attention sees, the chunk included; trained with 21")
    parser.add_argument("--cuda-graph", action="store_true",
                        help="replay each pass from a CUDA graph once the cache is full")
    parser.add_argument("--decoder", choices=["wan", "taehv"], default="wan",
                        help="taehv: every chunk decoded as soon as it is generated, as a live stream would")
    parser.add_argument("--fp8", action="store_true", help="linear layers of the blocks in FP8 (with --on-gpu)")
    parser.add_argument("--attention", choices=["cudnn", "efficient", "sage"], default="cudnn",
                        help="the SDPA kernel of self-attention over the cache")
    parser.add_argument("--sink-frames", type=int, default=0, help="latent frames at the start kept in a rolling cache")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    if args.latent_frames % causal.CHUNK or args.height % 16 or args.width % 16:
        parser.error(f"latent frames must be a multiple of {causal.CHUNK}, height and width multiples of 16")

    device = torch.device("cuda")
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    name = args.name or time.strftime("causal-%Y%m%d-%H%M%S")
    record = vars(args) | {"commit": git_commit(), "gpu": gpu_state(), "steps": list(causal.STEPS),
                           "shift": causal.SHIFT}

    with phase(record, "text_encoder"):
        context = encode([args.prompt], device)

    with phase(record, "denoiser"):
        config = model_config()
        with torch.device("meta"):
            model = WanModel(**{key: config[key] for key in KEYS})
        model.load_state_dict(self_forcing_weights(), assign=True)
        model.eval()
        if args.on_gpu:
            model.block_offloader = OnDevice(model.blocks, device)
        else:
            model.block_offloader = BlockOffloader(model.blocks, args.window, device, ram_window=args.ram_window)
        model.to(device)
        if args.fp8:
            for block in model.blocks:
                to_fp8(block)
        if args.compile:  # the blocks themselves, or the two slots that stand for them when offloading
            blocks = model.blocks if args.on_gpu else model.block_offloader.slots
            # one compiled forward for all of them; with CUDA graphs (reduce-overhead) its output is copied: the next
            # block replays the same graph into the same output memory
            forward = torch.compile(type(blocks[0]).forward, mode=args.compile)
            if args.compile == "reduce-overhead":
                forward = lambda *a, f=forward, **k: f(*a, **k).clone()  # noqa: E731
            for block in blocks:
                block.forward = functools.partial(forward, block)
        d = config["dim"] // config["num_heads"]
        freqs = torch.cat([rope(1024, d - 4 * (d // 6)), rope(1024, 2 * (d // 6)), rope(1024, 2 * (d // 6))],
                          dim=-1).to(device)
        latent = (16, args.latent_frames, args.height // 8, args.width // 8)
        noise = torch.randn(latent, device=device, generator=torch.Generator(device).manual_seed(args.seed))
        fresh = torch.Generator(device).manual_seed(args.seed + 1)  # the noise between the steps
        stream = TAEHVStream(device) if args.decoder == "taehv" else None
        chunk_s, decode_s, pieces, last = [], [], [], [time.perf_counter()]

        def progress(index, x0):
            torch.cuda.synchronize()
            now = time.perf_counter()
            chunk_s.append(round(now - last[0], 2))
            line = f"  chunk {index + 1}/{args.latent_frames // causal.CHUNK}, {chunk_s[-1]} s"
            if stream is not None:
                pieces.append(stream(x0))
                torch.cuda.synchronize()
                decode_s.append(round(time.perf_counter() - now, 3))
                line += f", decoded in {decode_s[-1]} s"
            last[0] = time.perf_counter()
            print(line, flush=True)

        latents = causal.generate(model, noise, context.to(device), freqs, generator=fresh, on_chunk=progress,
                                 attention=args.attention, sink_frames=args.sink_frames, max_frames=args.context_frames,
                                 cuda_graph=args.cuda_graph)
        model.block_offloader.stop()
        del model
        record["chunk_s"] = chunk_s
        filled = args.context_frames // causal.CHUNK - 1  # chunks until the cache holds the whole window
        full = len(chunk_s) > filled
        record["chunk_median_s"] = round(float(np.median(chunk_s)), 2)
        if full:
            record["chunk_median_full_cache_s"] = round(float(np.median(chunk_s[filled + 1:])), 2)
        if stream is not None:
            record["decode_s"] = decode_s
            record["decode_median_s"] = round(float(np.median(decode_s)), 3)
        print({k: v for k, v in record.items() if k.endswith("median_s") or k.endswith("cache_s")}, flush=True)
    torch.save(latents.cpu(), out / f"{name}.latents.pt")

    if stream is not None:
        video = torch.cat(pieces, dim=1).cpu()
    else:
        with phase(record, "vae"):
            video = decode(latents, device)
    frames = ((video.permute(1, 2, 3, 0) + 1) * 127.5).round().to(torch.uint8).numpy()
    imageio.mimwrite(out / f"{name}.mp4", frames, fps=FPS, quality=8)
    imageio.imwrite(out / f"{name}.png", np.concatenate([frames[0], frames[len(frames) // 2], frames[-1]], axis=1))
    (out / f"{name}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out / name}.mp4", flush=True)


if __name__ == "__main__":
    main()
