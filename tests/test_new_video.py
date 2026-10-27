"""Stream.set_prompt(..., "new"): after it the next chunk is the first chunk of a new video under that prompt."""

import pytest
import torch

from wan_dit_lab.generation.causal import Stream
from wan_dit_lab.models.model import WanModel
from wan_dit_lab.offloading.offload import OnDevice
from wan_dit_lab.models.rope import rope


@pytest.mark.gpu
def test_new_video_starts_from_an_empty_cache():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    dev = torch.device("cuda")
    torch.manual_seed(37)
    model = WanModel(dim=128, eps=1e-6, ffn_dim=256, freq_dim=256, in_dim=16, num_heads=2, num_layers=2, out_dim=16,
                     text_len=8, text_dim=32).requires_grad_(False).eval()
    model.block_offloader = OnDevice(model.blocks, dev, dtype=None)
    model.block_offloader.start()
    model.to(dev)
    d = 64
    freqs = torch.cat([rope(1024, d - 4 * (d // 6)), rope(1024, 2 * (d // 6)), rope(1024, 2 * (d // 6))], dim=-1).to(dev)
    first, second = torch.randn(1, 8, 32, device=dev), torch.randn(1, 8, 32, device=dev)
    make = lambda context: Stream(model, context, freqs, 32, 32, torch.Generator(dev).manual_seed(5),
                                  dtype=torch.bfloat16, max_frames=12, sink_frames=3, attention="efficient")
    old = make(first)
    for _ in range(4):
        old.next_chunk()
    old.set_prompt(second, "new")
    old.generator = torch.Generator(dev).manual_seed(5)  # the sampler draws fresh noise at every level from it
    fresh = make(second)
    noise = torch.randn(old.shape, generator=torch.Generator(dev).manual_seed(9), device=dev)
    assert old.start == 0 and all(c.saved == 0 for c in old.caches)
    assert torch.equal(old.next_chunk(noise=noise.clone()), fresh.next_chunk(noise=noise.clone()))


@pytest.mark.gpu
def test_cuda_graph_gives_the_same_chunks_also_across_a_new_video():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    dev = torch.device("cuda")
    torch.manual_seed(37)
    model = WanModel(dim=128, eps=1e-6, ffn_dim=256, freq_dim=256, in_dim=16, num_heads=2, num_layers=2, out_dim=16,
                     text_len=8, text_dim=32).requires_grad_(False).eval()
    model.block_offloader = OnDevice(model.blocks, dev, dtype=None)
    model.block_offloader.start()
    model.to(dev)
    d = 64
    freqs = torch.cat([rope(1024, d - 4 * (d // 6)), rope(1024, 2 * (d // 6)), rope(1024, 2 * (d // 6))], dim=-1).to(dev)
    first, second = torch.randn(1, 8, 32, device=dev), torch.randn(1, 8, 32, device=dev)

    def video(graph):
        s = Stream(model, first, freqs, 32, 32, torch.Generator(dev).manual_seed(5), dtype=torch.bfloat16,
                   max_frames=12, sink_frames=3, attention="efficient", cuda_graph=graph)
        out = [s.next_chunk() for _ in range(5)]
        s.set_prompt(second, "new")
        out += [s.next_chunk() for _ in range(5)]
        s.set_prompt(first, "swap")
        out += [s.next_chunk() for _ in range(2)]
        return out

    for a, b in zip(video(False), video(True)):
        assert torch.equal(a, b)


@pytest.mark.gpu
def test_without_the_write_pass_the_last_step_fills_the_cache_also_under_a_cuda_graph():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    dev = torch.device("cuda")
    torch.manual_seed(37)
    model = WanModel(dim=128, eps=1e-6, ffn_dim=256, freq_dim=256, in_dim=16, num_heads=2, num_layers=2, out_dim=16,
                     text_len=8, text_dim=32).requires_grad_(False).eval()
    model.block_offloader = OnDevice(model.blocks, dev, dtype=None)
    model.block_offloader.start()
    model.to(dev)
    d = 64
    freqs = torch.cat([rope(1024, d - 4 * (d // 6)), rope(1024, 2 * (d // 6)), rope(1024, 2 * (d // 6))], dim=-1).to(dev)
    context = torch.randn(1, 8, 32, device=dev)

    def video(graph, write_pass):  # Live's sampler: 2 UniPC steps
        s = Stream(model, context, freqs, 32, 32, torch.Generator(dev).manual_seed(5), dtype=torch.bfloat16,
                   max_frames=12, sink_frames=3, attention="efficient", cuda_graph=graph, write_pass=write_pass,
                   sampler="unipc", steps=2)
        return [s.next_chunk() for _ in range(6)]

    written, kept = video(False, True), video(False, False)
    assert torch.equal(written[0], kept[0])  # the first chunk sees no cache
    assert not torch.equal(written[1], kept[1])  # later ones see the keys of the last step instead of the clean chunk
    for a, b in zip(kept, video(True, False)):
        assert torch.equal(a, b)
