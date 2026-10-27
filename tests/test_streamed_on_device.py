"""offload.StreamedOnDevice: blocks copied from pinned RAM into two GPU slots give the same video as all on the GPU."""

import pytest
import torch

from wan_dit_lab.generation.causal import Stream
from wan_dit_lab.quantization.fp8 import to_fp8
from wan_dit_lab.models.model import WanModel
from wan_dit_lab.offloading.offload import OnDevice, StreamedOnDevice
from wan_dit_lab.models.rope import rope


def latents(streamed, fp8=False, chunks=4):
    device = torch.device("cuda")
    torch.manual_seed(37)
    model = WanModel(dim=128, eps=1e-6, ffn_dim=256, freq_dim=256, in_dim=16,
                     num_heads=2, num_layers=6, out_dim=16, text_len=8, text_dim=32)
    model.requires_grad_(False).eval()
    for block in model.blocks:
        block.to(device)
        if fp8:
            to_fp8(block.to(torch.bfloat16))
    model.block_offloader = (StreamedOnDevice(model.blocks, device, streamed) if streamed
                             else OnDevice(model.blocks, device, dtype=None))
    model.to(device)
    d = 64
    freqs = torch.cat([rope(1024, d - 4 * (d // 6)), rope(1024, 2 * (d // 6)), rope(1024, 2 * (d // 6))], dim=-1).to(device)
    stream = Stream(model, torch.randn(1, 8, 32, device=device), freqs, 32, 32, torch.Generator(device).manual_seed(5),
                    dtype=torch.bfloat16 if fp8 else torch.float32, max_frames=12, sink_frames=3, attention="efficient")
    return [stream.next_chunk() for _ in range(chunks)]


@pytest.mark.gpu
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("streamed", [(1, 4), (0, 2, 3, 5)])
def test_streamed_blocks_give_the_same_video(streamed, fp8):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    for a, b in zip(latents(None, fp8), latents(streamed, fp8)):
        assert torch.equal(a, b)
