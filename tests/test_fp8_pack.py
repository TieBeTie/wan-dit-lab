"""A block_fp8_v1 pack streamed by BlockOffloader gives the video of the same blocks turned FP8 on the GPU."""

import copy
import functools

import pytest
import torch

from wan_dit_lab.generation.causal import Stream
from wan_dit_lab.quantization.fp8 import fp8_slot, to_fp8
from wan_dit_lab.models.model import WanModel
from wan_dit_lab.offloading.offload import BlockOffloader, OnDevice
from wan_dit_lab.models.rope import rope
from wan_dit_lab.quantization.weight_pack import build_fp8, mapped_fp8


def tiny():
    torch.manual_seed(37)
    model = WanModel(dim=128, eps=1e-6, ffn_dim=256, freq_dim=256, in_dim=16,
                     num_heads=2, num_layers=4, out_dim=16, text_len=8, text_dim=32)
    return model.requires_grad_(False).eval()


def video(model, offloader, device):
    model.block_offloader = offloader
    model.to(device)
    d = 64
    freqs = torch.cat([rope(1024, d - 4 * (d // 6)), rope(1024, 2 * (d // 6)), rope(1024, 2 * (d // 6))], dim=-1).to(device)
    torch.manual_seed(3)
    stream = Stream(model, torch.randn(1, 8, 32, device=device), freqs, 32, 32, torch.Generator(device).manual_seed(5),
                    dtype=torch.bfloat16, max_frames=12, sink_frames=3, attention="efficient")
    return [stream.next_chunk() for _ in range(4)]


@pytest.mark.gpu
def test_fp8_pack_matches_fp8_on_the_gpu(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    source = tmp_path / "source.pt"
    source.write_bytes(b"checkpoint identity")
    model = tiny()
    for block in model.blocks:
        block.to(torch.bfloat16)
    reference = copy.deepcopy(model)
    for block in reference.blocks:
        for m in block.modules():  # the pack quantizes from float32, rounding once; so does the reference
            if isinstance(m, torch.nn.Linear):
                m.weight.data = m.weight.data.float()
        to_fp8(block.to(device))
    expected = video(reference, OnDevice(reference.blocks, device, dtype=None), device)

    build_fp8(model.blocks, tmp_path / "blocks.fp8", source)
    packed, layout = mapped_fp8(model.blocks, tmp_path / "blocks.fp8", source)
    offloader = BlockOffloader(model.blocks, 2, device, ram_window=2, dtype=torch.uint8, packed=packed,
                               slot=functools.partial(fp8_slot, like=model.blocks[0], layout=layout))
    actual = video(model, offloader, device)
    offloader.close()
    for a, b in zip(actual, expected):
        assert torch.equal(a, b)
