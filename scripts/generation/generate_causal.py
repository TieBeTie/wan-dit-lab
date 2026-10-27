"""Text to video with Self-Forcing on my WanModel: four steps per chunk of 3 latent frames, no guidance, the keys
and values of the finished frames in a cache. T5, then the DiT, then the VAE, one model at a time.

Writes to outputs/: <name>.mp4, <name>.png (first, middle and last frame), <name>.latents.pt and <name>.json
(settings, commit, time and peak VRAM of every phase, time of every chunk).

    PYTHONPATH=src python scripts/generation/generate_causal.py --prompt "A tree sways in the wind" --latent-frames 9

Several --prompt: the video goes from one to the next every --switch-every chunks, changed the --switch way
(causal.SWITCHES, joined by "+").
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
import functools
import json
import math
import time
from contextlib import ExitStack

import imageio.v2 as imageio
import numpy as np
import torch

from wan_dit_lab.generation import causal
from wan_dit_lab.harness import git_commit, gpu_state
from wan_dit_lab.harness.config import OUTPUTS
from wan_dit_lab.models.checkpoint import model_config, self_forcing_weights
from wan_dit_lab.models.model import WanModel
from wan_dit_lab.models.rope import rope
from wan_dit_lab.models.text_encoder import encode
from wan_dit_lab.models.vae import TAEHVStream, decode
from wan_dit_lab.offloading.offload import BlockOffloader, OnDevice
from wan_dit_lab.quantization.fp8 import to_fp8

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.generation.generate import FPS, KEYS, phase  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", required=True, action="append", help="one or more, in the order of the video")
    parser.add_argument("--switch-every", type=int, default=None,
                        help="chunks per prompt (by default the video is shared evenly among the prompts)")
    parser.add_argument("--switch", default="swap", help="how the prompt changes: " + ", ".join(causal.SWITCHES))
    parser.add_argument("--blend-chunks", type=int, default=4, help="chunks of a blend switch")
    parser.add_argument("--decay-mu", type=float, nargs=2, default=(0.1, 0.7), help="mu_min mu_max of a decay switch")
    parser.add_argument("--latent-frames", type=int, default=9, help="a multiple of 3; video frames 4F - 3")
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shift", type=float, default=causal.SHIFT, help="positive flow timestep shift")
    parser.add_argument("--resolution-shift", action="store_true",
                        help="multiply shift by sqrt(H*W/(480*832)); experimental SD3 Eq. 23 transfer")
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--ram-window", type=int, default=2)
    parser.add_argument("--on-gpu", action="store_true", help="all blocks on the GPU in bfloat16, no offloading")
    parser.add_argument("--offload-bf16", action="store_true",
                        help="keep offloaded block weights in BF16, matching --on-gpu block storage")
    parser.add_argument("--compile", nargs="?", const="default", default=None,
                        help="torch.compile (inductor) of the modules that compute; a mode may follow: reduce-overhead")
    parser.add_argument("--context-frames", type=int, default=causal.CONTEXT,
                        help="latent frames attention sees, the chunk included; trained with 21")
    parser.add_argument("--cuda-graph", action="store_true",
                        help="replay each pass from a CUDA graph once the cache is full")
    parser.add_argument("--decoder", choices=["wan", "taehv", "none"], default="wan",
                        help="taehv: every chunk decoded as soon as it is generated, as a live stream would; "
                             "none: the latents alone (scripts/generation/decode_taehv.py decodes them later)")
    parser.add_argument("--fp8", action="store_true", help="linear layers of the blocks in FP8 (with --on-gpu)")
    parser.add_argument("--attention", choices=["cudnn", "efficient", "sage"], default="cudnn",
                        help="the SDPA kernel of self-attention over the cache")
    parser.add_argument("--sink-frames", type=int, default=0, help="latent frames at the start kept in a rolling cache")
    parser.add_argument("--positions", choices=["video", "window"], default="video",
                        help="RoPE time position of a frame: in the video, or its place in the attention window")
    parser.add_argument("--context-noise", type=int, default=0,
                        help="timestep a finished chunk is written to the cache at (0 as trained, 1000 pure noise)")
    parser.add_argument("--kv-fp8", action="store_true", help="keys and values of the cache in FP8, half the memory")
    parser.add_argument("--pc-recent", type=int, default=None,
                        help="Participative Compression: evict the ring tokens the chunk scores lowest, never from the "
                             "newest PC_RECENT frames (by default the oldest frames are evicted)")
    parser.add_argument("--cross-cache", action="store_true",
                        help="cross-attention keys and values computed once per prompt (needed by --switch blend)")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    if args.latent_frames % causal.CHUNK or args.height % 16 or args.width % 16:
        parser.error(f"latent frames must be a multiple of {causal.CHUNK}, height and width multiples of 16")
    if not math.isfinite(args.shift) or args.shift <= 0:
        parser.error("shift must be finite and positive")
    effective_shift = args.shift * (math.sqrt(args.height * args.width / (480 * 832))
                                    if args.resolution_shift else 1.0)

    device = torch.device("cuda")
    out = OUTPUTS
    out.mkdir(exist_ok=True)
    name = args.name or time.strftime("causal-%Y%m%d-%H%M%S")
    record = vars(args) | {"commit": git_commit(), "gpu": gpu_state(), "steps": list(causal.STEPS),
                           "shift": effective_shift, "sigmas": causal.sigmas(shift=effective_shift)}
    print(f"shift={effective_shift:.6f}, sigmas={record['sigmas']}", flush=True)

    with phase(record, "text_encoder"):
        contexts = encode(args.prompt, device)
    chunk_count = args.latent_frames // causal.CHUNK
    every = args.switch_every or -(-chunk_count // len(args.prompt))
    record["prompt_at_chunk"] = {i * every: p for i, p in enumerate(args.prompt) if i * every < chunk_count}

    with phase(record, "denoiser"):
        latents, previews, streamed = denoise(
            args, contexts, device, out, name, record, effective_shift, chunk_count, every)
    torch.save(latents.cpu(), out / f"{name}.latents.pt")
    if args.decoder == "none":
        (out / f"{name}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {out / name}.latents.pt", flush=True)
        return

    if streamed:
        imageio.imwrite(out / f"{name}.png", np.concatenate(previews, axis=1))
    else:
        with phase(record, "vae"):
            video = decode(latents, device)
        frames = ((video.permute(1, 2, 3, 0) + 1) * 127.5).round().to(torch.uint8).numpy()
        imageio.mimwrite(out / f"{name}.mp4", frames, fps=FPS, quality=8)
        imageio.imwrite(out / f"{name}.png", np.concatenate([frames[0], frames[len(frames) // 2], frames[-1]], axis=1))
    (out / f"{name}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out / name}.mp4", flush=True)


def denoise(args, contexts, device, out, name, record, effective_shift, chunk_count, every):
    """Own all denoiser/streaming-decoder state until its last chunk, including on failure."""
    with ExitStack() as resources:
        config = model_config()
        with torch.device("meta"):
            model = WanModel(**{key: config[key] for key in KEYS})
        model.load_state_dict(self_forcing_weights(), assign=True)
        model.eval()
        if args.on_gpu:
            model.block_offloader = OnDevice(model.blocks, device)
        else:
            if args.offload_bf16:
                model.blocks.to(dtype=torch.bfloat16)
            model.block_offloader = BlockOffloader(model.blocks, args.window, device, ram_window=args.ram_window)
        resources.callback(model.block_offloader.close)
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
        # one generator drawn chunk by chunk, as live_video.py draws it: the same seed gives the same video there
        generator = torch.Generator(device).manual_seed(args.seed)
        streaming = args.decoder == "taehv"
        stream = None
        chunk_s, decode_s, last = [], [], [time.perf_counter()]
        # Write decoded chunks immediately: a minute of float32 RGB must not accumulate on the GPU or in RAM.
        writer = imageio.get_writer(out / f"{name}.mp4", fps=FPS, quality=8) if streaming else None
        if writer is not None:
            resources.callback(writer.close)
        previews, frame_count = [], [0]

        def progress(index, x0):
            nonlocal stream
            torch.cuda.synchronize()
            now = time.perf_counter()
            chunk_s.append(round(now - last[0], 2))
            line = f"  chunk {index + 1}/{args.latent_frames // causal.CHUNK}, {chunk_s[-1]} s"
            if streaming:
                if stream is None:
                    stream = TAEHVStream(device)
                    resources.callback(stream.close)
                decoded = stream(x0)
                frames = ((decoded.permute(1, 2, 3, 0) + 1) * 127.5).round().to(torch.uint8).cpu().numpy()
                if index in (0, chunk_count // 2, chunk_count - 1):
                    previews.append(frames[-1 if index else 0].copy())
                for frame in frames:
                    writer.append_data(frame)
                frame_count[0] += len(frames)
                torch.cuda.synchronize()
                decode_s.append(round(time.perf_counter() - now, 3))
                line += f", decoded in {decode_s[-1]} s"
            last[0] = time.perf_counter()
            print(line, flush=True)

        video = causal.Stream(model, contexts[:1].to(device), freqs, args.height, args.width, generator,
                              max_frames=args.context_frames, sink_frames=args.sink_frames, attention=args.attention,
                              cuda_graph=args.cuda_graph, positions=args.positions,
                              context_noise=args.context_noise, kv_fp8=args.kv_fp8, pc_recent=args.pc_recent,
                              cross_cache=args.cross_cache, shift=effective_shift)
        chunks, switch_s = [], []
        for index in range(chunk_count):
            if index and index % every == 0 and index // every < len(args.prompt):
                torch.cuda.synchronize()
                start = time.perf_counter()
                video.set_prompt(contexts[index // every][None].to(device), args.switch, args.blend_chunks,
                                 tuple(args.decay_mu))
                torch.cuda.synchronize()
                switch_s.append(round(time.perf_counter() - start, 2))
                last[0] += switch_s[-1]  # not counted in the time of the chunk
                print(f"  prompt {index // every + 1} ({args.switch}) in {switch_s[-1]} s", flush=True)
            chunk = video.next_chunk()
            progress(index, chunk)
            chunks.append(chunk.cpu())
        if writer is not None:
            record["video_frames"] = frame_count[0]
            record["video_seconds"] = frame_count[0] / FPS
        record["switch_s"] = switch_s
        latents = torch.cat(chunks, dim=1)
        chunks.clear()
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
        return latents, previews, streaming


if __name__ == "__main__":
    main()
