"""Decoders of the latents, taken ready from the reference code and run in float16.

decode: the Wan VAE, the whole video at once; under autocast it does not fit 8 GB.
TAEHVStream: TAEHV (madebyollin), a small decoder of the same latents for previews, chunk by chunk.
"""

import sys

import torch

from wan_dit_lab.checkpoint import REFERENCE, checkpoint_dir

if str(REFERENCE) not in sys.path:
    sys.path.insert(0, str(REFERENCE))
from taehv import TAEHV, MemBlock  # noqa: E402
from wan.modules.vae import WanVAE  # noqa: E402


def decode(latents, device):
    """Latents [16, F, H, W] to a video [3, 4(F - 1) + 1, 8H, 8W] with values in [-1, 1], float32 on the CPU."""
    vae = WanVAE(vae_pth=str(checkpoint_dir() / "Wan2.1_VAE.pth"), dtype=torch.float16, device=device)
    vae.model.half()
    with torch.no_grad():
        video = vae.model.decode(latents.to(device, torch.float16)[None], vae.scale)
    return video[0].float().clamp(-1, 1).cpu()


class TAEHVStream:
    """TAEHV decoding a video chunk by chunk as the chunks come, the same frames as decoding it whole.

    Inside a chunk the frames go through the decoder together. A MemBlock sees, besides each frame, the frame
    before it; for the first frame of a chunk that is the last frame of the chunk before, kept here (zeros before the
    first chunk, as TAEHV pads). The decoder upsamples time 4 times; the first 3 frames of the video are cut, as
    TAEHV does, so that F latent frames give 4F - 3."""

    def __init__(self, device, checkpoint=REFERENCE / "checkpoints" / "taew2_1.pth"):
        self.decoder = TAEHV(str(checkpoint)).to(device, torch.float16).decoder
        self.last = {}  # index of a MemBlock -> its input for the last frame of the previous chunk, [N, 1, C, H, W]
        self.trim = 3

    @torch.no_grad()
    def __call__(self, latents):
        """Latents of a chunk [16, F, H, W] to frames [3, 4F, 8H, 8W] (fewer for the first chunk), values in [-1, 1]."""
        x = latents.permute(1, 0, 2, 3).to(torch.float16)  # [T, C, H, W]: one video, its frames as the batch
        for i, block in enumerate(self.decoder):
            if isinstance(block, MemBlock):
                past = self.last.get(i, torch.zeros_like(x[:1]))
                self.last[i] = x[-1:]
                x = block(x, torch.cat([past, x[:-1]]))
            else:
                x = block(x)
        frames = x.permute(1, 0, 2, 3)[:, self.trim:]  # [3, T', H, W]
        self.trim = 0
        return frames.float().mul(2).sub(1).clamp(-1, 1)
