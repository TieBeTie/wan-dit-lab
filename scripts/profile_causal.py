"""Profile of one chunk of Self-Forcing: a Chrome trace for Perfetto (ui.perfetto.dev) and a table of the kernels.

The model is on the GPU in bfloat16; the context is random, T5 does not matter for the time of the DiT. The first
chunks warm up (and compile with --compile); the chunk after them is recorded, with `--warmup` chunks in the cache.

    PYTHONPATH=src python scripts/profile_causal.py --height 320 --width 576
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from wan_dit_lab import causal
from wan_dit_lab.checkpoint import ROOT, model_config, self_forcing_weights
from wan_dit_lab.model import WanModel
from wan_dit_lab.fp8 import to_fp8
from wan_dit_lab.offload import OnDevice
from wan_dit_lab.rope import rope

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import KEYS  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--warmup", type=int, default=2, help="chunks before the recorded one")
    parser.add_argument("--compile", nargs="?", const="default", default=None)
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--attention", choices=["cudnn", "efficient", "sage"], default="cudnn")
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

    causal.generate(model, noise, context, freqs, on_chunk=on_chunk, attention=args.attention)
    torch.cuda.synchronize()
    profiler.stop()
    out = ROOT / "outputs"
    profiler.export_chrome_trace(str(out / f"{name}.json"))
    table = profiler.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=60)
    (out / f"{name}.txt").write_text(table, encoding="utf-8")
    print(table)
    print(f"wrote {out / name}.json")


if __name__ == "__main__":
    main()
