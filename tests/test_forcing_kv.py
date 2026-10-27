"""Forcing-KV on Sage over the FP8 cache: static heads see the sink, the newest finished frame and the chunk; the
others the whole window; the same under a CUDA graph."""

import pytest
import torch

from wan_dit_lab.generation.causal import KVCache, Stream
from wan_dit_lab.models.model import WanModel
from wan_dit_lab.offloading.offload import OnDevice
from wan_dit_lab.models.rope import rope


@pytest.fixture
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("sageattention")
    return torch.device("cuda")


@pytest.mark.gpu
def test_static_heads_attend_to_the_sink_the_newest_frame_and_the_chunk(cuda):
    from wan_dit_lab.kernels import sage_kv
    torch.manual_seed(0)
    per, chunk, H, D = 32, 2, 4, 128
    cache = KVCache(max_tokens=6 * per, sink_tokens=2 * per, sage=True, fp8=True, chunk=chunk)
    cache.set_heads([0, 2], H)
    new = lambda: [torch.randn(1, chunk * per, H, D, device=cuda, dtype=torch.bfloat16) for _ in range(3)]
    for _ in range(5):  # sink, then the ring, then one chunk evicting the oldest ring frames
        Q, K, V = new()
        cache.attend(Q, K, V)
        cache.commit()
    assert cache.full()
    Q, K, V = new()
    out = cache.attend(Q, K, V)
    order = cache.frame_of * cache.per_frame + cache.place_in_frame
    ring = order.argsort(descending=True)[:per] + cache.sink_tokens  # the newest frame, newest token first
    assert bool((cache.frame_of[ring - cache.sink_tokens] == cache.frame_of.max()).all())
    tokens = torch.cat([torch.arange(cache.sink_tokens, device=cuda), ring])
    for heads, keep in (([0, 2], tokens), ([1, 3], torch.arange(cache.max_tokens, device=cuda))):
        pick = lambda x: x[:, keep][:, :, heads].contiguous()
        want = sage_kv.attend(Q[:, :, heads].contiguous(), pick(cache.k), pick(cache.ks), pick(cache.v), pick(cache.vs),
                              len(keep), K[:, :, heads].contiguous(), V[:, :, heads].contiguous())
        assert torch.equal(out[:, :, heads], want)


@pytest.mark.gpu
def test_forcing_kv_gives_the_same_chunks_under_a_cuda_graph(cuda):
    torch.manual_seed(37)
    model = WanModel(dim=256, eps=1e-6, ffn_dim=256, freq_dim=256, in_dim=16, num_heads=2, num_layers=2, out_dim=16,
                     text_len=8, text_dim=32).requires_grad_(False).eval()
    model.block_offloader = OnDevice(model.blocks, cuda, dtype=None)
    model.block_offloader.start()
    model.to(cuda)
    d = 128
    freqs = torch.cat([rope(1024, d - 4 * (d // 6)), rope(1024, 2 * (d // 6)), rope(1024, 2 * (d // 6))], dim=-1).to(cuda)
    context = torch.randn(1, 8, 32, device=cuda)

    def video(graph, static):
        s = Stream(model, context, freqs, 32, 32, torch.Generator(cuda).manual_seed(5), dtype=torch.bfloat16,
                   max_frames=12, sink_frames=3, attention="sage", kv_fp8=True, cuda_graph=graph,
                   static_heads=static)
        return [s.next_chunk() for _ in range(6)]

    full, pruned = video(False, None), video(False, {0: [1], 1: [0]})
    assert torch.equal(full[0], pruned[0]) and not torch.equal(full[-1], pruned[-1])
    for a, b in zip(pruned, video(True, {0: [1], 1: [0]})):
        assert torch.equal(a, b)
