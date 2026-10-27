"""Stream.set_prompt "recache_ring" (LongLive's recache with a global sink) on a tiny model, on the GPU
(the SDPA kernels of causal.ATTENTION have no CPU version)."""

import pytest
import torch

from wan_dit_lab.generation.causal import Stream
from wan_dit_lab.models.model import WanModel
from wan_dit_lab.offloading.offload import OnDevice
from wan_dit_lab.models.rope import rope


def stream(seed=12):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    torch.manual_seed(37)
    model = WanModel(dim=128, eps=1e-6, ffn_dim=256, freq_dim=256, in_dim=16,
                     num_heads=2, num_layers=2, out_dim=16, text_len=8, text_dim=32)
    model.requires_grad_(False).eval()
    model.block_offloader = OnDevice(model.blocks, device, dtype=torch.float32)
    model.block_offloader.start()
    model.to(device)
    d = 128 // 2
    freqs = torch.cat([rope(1024, d - 4 * (d // 6)), rope(1024, 2 * (d // 6)), rope(1024, 2 * (d // 6))], dim=-1).to(device)
    context = torch.randn(1, 8, 32, device=device)
    return Stream(model, context, freqs, 32, 32, torch.Generator(device).manual_seed(seed), dtype=torch.float32,
                  max_frames=12, sink_frames=3, positions="video", attention="efficient")


@pytest.mark.gpu
def test_same_prompt_before_eviction_rebuilds_the_same_cache():
    """Nothing evicted yet: each ring chunk was written seeing the sink and the chunks before it, as now."""
    s = stream()
    for _ in range(3):  # the sink, then two ring chunks: 9 frames, the ring of 6 is full, nothing evicted
        s.next_chunk()
    before = [(c.k[:, :c.saved].clone(), c.v[:, :c.saved].clone(), c.saved, c.frames) for c in s.caches]
    s.set_prompt(s.context.clone(), "recache_ring")
    for c, (k, v, saved, frames) in zip(s.caches, before):
        assert (c.saved, c.frames) == (saved, frames)
        assert torch.allclose(c.k[:, :saved], k, atol=1e-5) and torch.allclose(c.v[:, :saved], v, atol=1e-5)


@pytest.mark.gpu
def test_new_prompt_keeps_the_sink_and_changes_the_ring():
    s = stream()
    for _ in range(5):  # ring evicted twice
        s.next_chunk()
    sink = s.caches[0].sink_tokens
    before = [(c.k[:, :c.saved].clone(), c.saved) for c in s.caches]
    s.set_prompt(torch.randn_like(s.context), "recache_ring")
    for index, (c, (k, saved)) in enumerate(zip(s.caches, before)):
        assert c.saved == saved
        assert torch.equal(c.k[:, :sink], k[:, :sink])  # the sink untouched
        # the keys of the first block come from the latents alone (the prompt enters after self-attention, in
        # cross-attention); from the second block on they carry the new prompt
        assert torch.allclose(c.k[:, sink:saved], k[:, sink:], atol=1e-5) == (index == 0)
    s.next_chunk()  # and the video goes on


@pytest.mark.gpu
def test_shot_pins_its_first_chunk_and_moves_positions():
    """LongLive 2.0's scene cut: the first chunk of the shot stays in the ring however many chunks follow, and the
    shot is placed shot_rope_offset frames later in time."""
    s = stream()
    for _ in range(4):
        s.next_chunk()
    s.set_prompt(torch.randn_like(s.context), "shot")
    assert s.rope_offset == s.shot_rope_offset
    s.next_chunk()  # the first chunk of the shot, frames 12-14 of the video
    cache = s.caches[1]
    sink = cache.sink_tokens
    pinned = cache.pinned.nonzero().flatten()
    assert len(pinned) == cache.per_frame * 3 and set(cache.frame_of[pinned].tolist()) == {12, 13, 14}
    keys = cache.k[:, sink + pinned].clone()
    for _ in range(4):  # the ring of 6 frames turns over twice; the pinned 3 stay
        s.next_chunk()
    assert set(cache.frame_of[pinned].tolist()) == {12, 13, 14}
    assert torch.equal(cache.k[:, sink + pinned], keys)
    s.set_prompt(torch.randn_like(s.context), "shot")
    s.next_chunk()  # the next shot releases the old pin and pins its own first chunk
    assert set(cache.frame_of[cache.pinned].tolist()) == {27, 28, 29}
