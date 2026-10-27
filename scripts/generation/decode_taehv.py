"""Decodes saved latents with TAEHV chunk by chunk, as the live pipeline does: outputs/<run>.latents.pt to
outputs/<run>-taehv.mp4.

    PYTHONPATH=src python scripts/generation/decode_taehv.py <run>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


import imageio.v2 as imageio
import torch

from wan_dit_lab.generation import causal
from wan_dit_lab.harness.config import OUTPUTS
from wan_dit_lab.models.vae import TAEHVStream


def main():
    run = sys.argv[1]
    device = torch.device("cuda")
    latents = torch.load(OUTPUTS / f"{run}.latents.pt").to(device)
    stream = TAEHVStream(device)
    video = torch.cat([stream(latents[:, i:i + causal.CHUNK]) for i in range(0, latents.shape[1], causal.CHUNK)], dim=1)
    frames = ((video.permute(1, 2, 3, 0) + 1) * 127.5).round().to(torch.uint8).cpu().numpy()
    imageio.mimwrite(OUTPUTS / f"{run}-taehv.mp4", frames, fps=16, quality=8)
    print(f"wrote outputs/{run}-taehv.mp4: {len(frames)} frames")


if __name__ == "__main__":
    main()
