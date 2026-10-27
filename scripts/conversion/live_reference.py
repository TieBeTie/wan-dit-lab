"""The reference of the Rust host (native/live): Live's old Python path, the preset longlive2 as it was built,
over a script of prompts and cuts, with the noise of every chunk kept, so that the host can be run on the same
noise and compared chunk by chunk.

    python scripts/conversion/live_reference.py --out artifacts/live/reference.safetensors [--eager]

Script: prompt A for 4 chunks, a shot (/scene) to B for 2, a new video (/new) of C for 1. Saved: the T5 embeddings
of A, B, C [1, 512, 4096]; per chunk the noise and the clean latents x0 [48, 8, 30, 54]; the frames (uint8
[T, 480, 864, 3]) of chunks 0, 3, 4 and 6. --eager runs the blocks without torch.compile: the difference of the two
runs is the tolerance of the host, which runs Inductor's code too, compiled ahead of time."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import argparse
import json

import torch
from safetensors.torch import save_file

from wan_dit_lab.models.text_encoder import FastTextEncoder
from wan_dit_lab.models.vae import TAEHVStream
from wan_dit_lab.runtime.live_presets import MODELS, SCENE_PREFIX, build_stream

PROMPTS = {"A": "A lone oak tree on a grassy hill sways in the wind",
           "B": SCENE_PREFIX + "Snow starts to fall on the oak tree",
           "C": "A red fox walks through a snowy forest"}
SCRIPT = [("A", None), ("A", None), ("A", None), ("A", None), ("B", "shot"), ("B", None), ("C", "new")]
FRAMES = (0, 3, 4, 6)
# the videos of the quality comparison (live_quality.py): one prompt each, a new video each
RENDER_PROMPTS = {"oak": "A lone oak tree on a grassy hill sways in the wind",
                  "fox": "A red fox walks through a snowy forest",
                  "market": "A busy night market in Taipei, people walking under neon signs"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "live" / "reference.safetensors")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bench", type=int, default=0, help="time this many chunks of prompt A instead, as live bench")
    parser.add_argument("--render", type=Path, default=None,
                        help="videos of RENDER_PROMPTS x --seeds, --chunks each, into this directory (live_quality.py)")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    parser.add_argument("--chunks", type=int, default=12)
    args = parser.parse_args()
    device = torch.device("cuda")
    spec = MODELS["longlive2"]
    encoder = FastTextEncoder(device, weights_file=ROOT / "artifacts" / "t5_int8_fast.bin")
    prompts = RENDER_PROMPTS if args.render else PROMPTS
    embeddings = {name: encoder([text]) for name, text in prompts.items()}
    del encoder
    torch.cuda.empty_cache()
    stream = build_stream(spec, next(iter(embeddings.values())), device, compiled=not args.eager)
    if args.render:
        return render(stream, spec, embeddings, device, args)
    decoder = TAEHVStream(device, checkpoint=spec["taehv"])
    g = torch.Generator(device).manual_seed(args.seed)
    if args.bench:
        return bench(stream, decoder, g, args.bench)
    saved = {f"embedding_{name}": e.float().contiguous() for name, e in embeddings.items()}
    prompt = "A"
    for i, (name, cut) in enumerate(SCRIPT):
        if name != prompt:
            stream.set_prompt(embeddings[name].to(device), cut or "swap")
            prompt = name
        noise = torch.randn(stream.shape, generator=g, device=device)
        x0 = stream.next_chunk(noise=noise)
        video = decoder(x0)
        pixels = ((video.permute(1, 2, 3, 0) + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
        saved[f"noise_{i}"], saved[f"x0_{i}"] = noise.cpu(), x0.float().cpu()
        if i in FRAMES:
            saved[f"frames_{i}"] = pixels.cpu().contiguous()
        print(f"chunk {i} ({name}{', ' + cut if cut else ''}): |x0| {float(x0.float().abs().mean()):.4f}, "
              f"{pixels.shape[0]} frames", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    script = json.dumps([[n, c] for n, c in SCRIPT])
    save_file(saved, str(args.out), metadata=dict(script=script, prompts=json.dumps(PROMPTS), eager=str(args.eager)))
    print(args.out)


def render(stream, spec, embeddings, device, args):
    """For each prompt and seed a new video of args.chunks chunks: video_<prompt>_<seed>.safetensors with the noise of
    every chunk (the Rust host renders the same noise: live render) and all the frames."""
    args.render.mkdir(parents=True, exist_ok=True)
    for name, embedding in embeddings.items():
        for seed in args.seeds:
            stream.set_prompt(embedding.to(device), "new")
            decoder = TAEHVStream(device, checkpoint=spec["taehv"])
            g = torch.Generator(device).manual_seed(seed)
            saved, frames = {}, []
            for i in range(args.chunks):
                noise = torch.randn(stream.shape, generator=g, device=device)
                video = decoder(stream.next_chunk(noise=noise))
                frames.append(((video.permute(1, 2, 3, 0) + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu())
                saved[f"noise_{i}"] = noise.cpu()
            saved["frames"] = torch.cat(frames).contiguous()
            path = args.render / f"video_{name}_{seed}.safetensors"
            save_file(saved, str(path), metadata=dict(prompt=RENDER_PROMPTS[name], seed=str(seed), chunks=str(args.chunks)))
            print(path, saved["frames"].shape[0], "frames", flush=True)


def memory():
    """Private committed and working set of this process, and the GPU in use, in GiB (as live bench reports)."""
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (n, ctypes.c_size_t) for n in ("peak_ws", "ws", "qpp", "qp", "qpnp", "qnp", "pf", "peak_pf", "private")]
    c = Counters(cb=ctypes.sizeof(Counters))
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    free, total = torch.cuda.mem_get_info()
    gib = 1 << 30
    return f"RAM private {c.private / gib:.2f} GiB, working set {c.ws / gib:.2f} GiB | GPU used {(total - free) / gib:.2f} GiB"


def bench(stream, decoder, g, chunks):
    """Chunks as the old Python Live made them in its generating thread: DiT, TAEHV, pixels to the CPU."""
    import time
    print(memory(), flush=True)
    times = []
    for i in range(chunks):
        start = time.perf_counter()
        video = decoder(stream.next_chunk(noise=torch.randn(stream.shape, generator=g, device=g.device)))
        pixels = ((video.permute(1, 2, 3, 0) + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        seconds = time.perf_counter() - start
        times.append(seconds)
        print(f"chunk {i}: {seconds:.3f} s, {len(pixels)} frames, {len(pixels) / seconds:.1f} fps", flush=True)
    steady = sum(times[4:]) / len(times[4:])
    print(f"steady chunk: {steady:.3f} s, {32 / steady:.2f} fps")
    print(memory(), flush=True)


if __name__ == "__main__":
    main()
