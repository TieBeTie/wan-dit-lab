"""Profile of one chunk of Self-Forcing: a Chrome trace for Perfetto (ui.perfetto.dev) and a table of the kernels.

The model is on the GPU in bfloat16; the context is random, T5 does not matter for the time of the DiT. The first
chunks warm up (and compile with --compile); the chunk after them is recorded, with `--warmup` chunks in the cache.

    PYTHONPATH=src python scripts/generation/profile_causal.py --height 320 --width 576
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse

import torch
from torch.profiler import ProfilerActivity, profile

from wan_dit_lab.generation import causal
from wan_dit_lab.harness.config import OUTPUTS
from wan_dit_lab.models.checkpoint import model_config, self_forcing_weights
from wan_dit_lab.models.model import WanModel
from wan_dit_lab.models.rope import rope
from wan_dit_lab.offloading.offload import OnDevice
from wan_dit_lab.quantization.fp8 import to_fp8

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.generation.generate import KEYS  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--warmup", type=int, default=7,
                        help="chunks before the recorded one (6 fill a window of 21, then one more after it rolls)")
    parser.add_argument("--compile", nargs="?", const="default", default=None)
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--attention", choices=["cudnn", "efficient", "sage"], default="cudnn")
    parser.add_argument("--context-frames", type=int, default=causal.CONTEXT, help="window of attention, chunk included")
    parser.add_argument("--sink-frames", type=int, default=0)
    parser.add_argument("--positions", choices=["video", "window"], default="video")
    parser.add_argument("--pc-recent", type=int, default=None)
    parser.add_argument("--kv-fp8", action="store_true")
    parser.add_argument("--cross-cache", action="store_true")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    device = torch.device("cuda")
    name = args.name or f"profile-causal-{args.height}p" + (f"-{args.compile}" if args.compile else "")

    config = model_config()
    with torch.device("meta"):
        model = WanModel(**{key: config[key] for key in KEYS})
    model.load_state_dict(self_forcing_weights(), assign=True)
    model.eval()
    model.block_offloader = OnDevice(model.blocks, device)
    model.to(device)
    if args.fp8:
        for block in model.blocks:
            to_fp8(block)
    if args.compile:
        forward = torch.compile(type(model.blocks[0]).forward, mode=args.compile)
        for block in model.blocks:
            block.forward = lambda *a, block=block, **k: forward(block, *a, **k)
    d = config["dim"] // config["num_heads"]
    freqs = torch.cat([rope(1024, d - 4 * (d // 6)), rope(1024, 2 * (d // 6)), rope(1024, 2 * (d // 6))],
                      dim=-1).to(device)
    frames = causal.CHUNK * (args.warmup + 1)
    noise = torch.randn(16, frames, args.height // 8, args.width // 8, device=device)
    context = torch.randn(1, 512, 4096, device=device)
    profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])

    def on_chunk(index, x0):
        torch.cuda.synchronize()
        if index == args.warmup - 1:
            profiler.start()

    causal.generate(model, noise, context, freqs, max_frames=args.context_frames, on_chunk=on_chunk,
                    attention=args.attention, sink_frames=args.sink_frames, positions=args.positions,
                    pc_recent=args.pc_recent, kv_fp8=args.kv_fp8, cross_cache=args.cross_cache)
    torch.cuda.synchronize()
    profiler.stop()
    out = OUTPUTS
    profiler.export_chrome_trace(str(out / f"{name}.json"))
    table = profiler.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=60)
    (out / f"{name}.txt").write_text(table, encoding="utf-8")
    print(table)
    # GPU time of the recorded chunk by kind of kernel, from their names
    kinds = {"attention (SDPA kernels)": ("fmha", "cudnn", "flash", "attention"),
             "matmul (linear layers)": ("gemm", "cutlass", "xmma", "scaled_mm", "cublas"),
             "copies": ("memcpy", "copy_", "Memcpy")}
    total, spent = 0.0, {kind: 0.0 for kind in [*kinds, "fused elementwise and norms (triton, other)"]}
    for event in profiler.key_averages():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        time = event.self_device_time_total
        total += time
        kind = next((k for k, keys in kinds.items() if any(key in event.key for key in keys)),
                    "fused elementwise and norms (triton, other)")
        spent[kind] += time
    for kind, time in spent.items():
        print(f"{kind:45s} {time / 1e3:9.1f} ms  {100 * time / max(total, 1):5.1f} %")
    print(f"{'all GPU kernels':45s} {total / 1e3:9.1f} ms")
    print(f"wrote {out / name}.json")


if __name__ == "__main__":
    main()
