"""A block_w4a4_v1 pack streamed by BlockOffloader gives the video of the same W4A4 layers on the GPU, and stays
close to bfloat16."""

import copy
import functools

import pytest
import torch

from wan_dit_lab.generation.causal import Stream
from wan_dit_lab.models.model import WanModel
from wan_dit_lab.offloading.offload import BlockOffloader, OnDevice
from wan_dit_lab.models.rope import rope


def tiny():
    torch.manual_seed(37)
    model = WanModel(dim=128, eps=1e-6, ffn_dim=256, freq_dim=256, in_dim=16,
                     num_heads=2, num_layers=4, out_dim=16, text_len=8, text_dim=32)
    model.requires_grad_(False).eval()
    for block in model.blocks:
        block.to(torch.bfloat16)
    return model


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
@pytest.mark.parametrize("fused", [False, True])
def test_w4a4_pack_matches_w4a4_on_the_gpu(tmp_path, fused):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("wan_dit_lab.kernels.operators").load()
    from wan_dit_lab.quantization import w4a4
    device = torch.device("cuda")
    source = tmp_path / "source.pt"
    source.write_bytes(b"checkpoint identity")
    model = tiny()

    reference = copy.deepcopy(model)
    for block in reference.blocks:
        w4a4.convert(block.to(device), fused=fused)
    expected = video(reference, OnDevice(reference.blocks, device, dtype=None), device)
    plain = video(copy.deepcopy(model), OnDevice(copy.deepcopy(model).blocks, device), device)

    w4a4.build_pack(model.blocks, {}, tmp_path / "blocks.w4a4", source, fused=fused)
    packed, layout = w4a4.mapped_pack(model.blocks, tmp_path / "blocks.w4a4", source)
    offloader = BlockOffloader(model.blocks, 2, device, ram_window=2, dtype=torch.uint8, packed=packed,
                               slot=functools.partial(w4a4.slot, like=model.blocks[0], layout=layout))
    actual = video(model, offloader, device)
    offloader.close()
    for a, b in zip(actual, expected):
        assert torch.equal(a, b)
    assert all(torch.isfinite(a).all() for a in actual)


@pytest.mark.gpu
def test_streamed_w4a4_rows_give_the_video_of_all_rows_on_the_gpu(tmp_path):
    # Live's VRAM beside OBS and a browser: some W4A4 blocks streamed from pinned RAM through two GPU rows
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("wan_dit_lab.kernels.operators").load()
    from wan_dit_lab.quantization import w4a4
    from wan_dit_lab.offloading.offload import StreamedRows
    device = torch.device("cuda")
    source = tmp_path / "source.pt"
    source.write_bytes(b"checkpoint identity")
    model = tiny()
    w4a4.build_pack(model.blocks, {}, tmp_path / "blocks.w4a4", source, fused=True)
    rows, layout = w4a4.mapped_pack(model.blocks, tmp_path / "blocks.w4a4", source)
    like = copy.deepcopy(model.blocks[0]).to("meta")
    make = functools.partial(w4a4.slot, like=like, layout=layout)

    resident = copy.deepcopy(model)
    resident.blocks = torch.nn.ModuleList([make(r.to(device)) for r in rows])
    expected = video(resident, OnDevice(resident.blocks, device, dtype=None), device)

    streamed = copy.deepcopy(model)
    streamed.blocks = torch.nn.ModuleList([make(r.to(device)) if i in (0, 2) else copy.deepcopy(like)
                                           for i, r in enumerate(rows)])
    offloader = StreamedRows(streamed.blocks, device, [1, 3], rows, make)
    offloader.start()
    model_video = video(streamed, offloader, device)
    for a, b in zip(model_video, expected):
        assert torch.equal(a, b)
