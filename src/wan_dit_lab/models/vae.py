"""Decoders of the latents, taken ready from the reference code and run in float16.

decode: the Wan VAE, the whole video at once; under autocast it does not fit 8 GB.
TAEHVStream: TAEHV (madebyollin), a small decoder of the same latents for previews, chunk by chunk.
"""

import sys
from contextlib import ExitStack

import torch

from wan_dit_lab.models.checkpoint import REFERENCE, checkpoint_dir
from wan_dit_lab.offloading.resource_store import StateBinding, bind_module

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

    def __init__(self, device, checkpoint=REFERENCE / "checkpoints" / "taew2_1.pth", frames_at_once=1):
        taehv, self.memblock = TAEHV, MemBlock
        if "taew2_2" in str(checkpoint):
            # the Wan2.2 latents (48 channels, 16x16 pixels a cell) need the newer taehv.py: it ends with a pixel
            # shuffle of 2 (reference/taehv, madebyollin/taehv 011dfc2)
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "taehv_wan22", REFERENCE.parents[2] / "reference" / "taehv" / "taehv.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            taehv, self.memblock = module.TAEHV, module.MemBlock
        model = taehv(str(checkpoint))
        self.patch_size = getattr(model, "patch_size", 1)
        self.decoder = model.to(device, torch.float16).decoder
        self.last = {}  # index of a MemBlock -> its input for the last frame of the previous chunk, [N, 1, C, H, W]
        self.trim = 3
        # latent frames decoded together: at 480x832 a whole chunk of 3 needs 0.6 GB per activation, more than is
        # left beside the model and its cache; one at a time gives the same frames, the MemBlocks carry over
        self.frames_at_once = frames_at_once
        self._offload_bindings = []

    def configure_offloading(self, store, device, weights_tier="vram", state_tier="vram"):
        if self._offload_bindings:
            raise RuntimeError("Decoder offloading is already configured")
        if any(tier not in ("vram", "ram", "disk", "auto") for tier in (weights_tier, state_tier)):
            raise ValueError("Unknown decoder placement")
        if weights_tier != "vram":
            self._offload_bindings.append(
                bind_module(self.decoder, store, "vae/weights", device, weights_tier)
            )
        if state_tier != "vram":
            self._offload_bindings.append(StateBinding(
                store, "vae/state", lambda: self.last,
                lambda values: setattr(self, "last", values), self.last_clear,
                device, state_tier,
            ))
        for binding in self._offload_bindings:
            binding.evict()

    def last_clear(self):
        self.last = {}

    def close(self):
        """End a video and release decoder weights and recurrent frame state."""
        for binding in self._offload_bindings:
            binding.close()
        self._offload_bindings.clear()
        self.last.clear()
        self.decoder = None

    @torch.no_grad()
    def __call__(self, latents):
        """Latents of a chunk [16, F, H, W] to frames [3, 4F, 8H, 8W] (fewer for the first chunk), values in [-1, 1]."""
        with ExitStack() as stack:
            for binding in self._offload_bindings:
                stack.enter_context(binding.resident())
            return self._decode(latents)

    def _decode(self, latents):
        if latents.shape[1] > self.frames_at_once:
            n = self.frames_at_once
            return torch.cat([self(latents[:, i:i + n]) for i in range(0, latents.shape[1], n)], dim=1)
        x = latents.permute(1, 0, 2, 3).to(torch.float16)  # [T, C, H, W]: one video, its frames as the batch
        for i, block in enumerate(self.decoder):
            if isinstance(block, self.memblock):
                past = self.last.get(i, torch.zeros_like(x[:1]))
                self.last[i] = x[-1:]
                x = block(x, torch.cat([past, x[:-1]]))
            else:
                x = block(x)
        if self.patch_size > 1:
            x = torch.nn.functional.pixel_shuffle(x, self.patch_size)
        frames = x.permute(1, 0, 2, 3)[:, self.trim:]  # [3, T', H, W]
        self.trim = 0
        return frames.float().mul(2).sub(1).clamp(-1, 1)
