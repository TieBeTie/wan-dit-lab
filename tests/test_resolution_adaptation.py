"""Off the trained resolution: attention scaled as Jin et al. (KVCache.entropy_tokens) and RoPE positions spread over
the trained grid (offloading_generate.rope_at); at the trained sizes both are the model as trained."""

import math
import sys
from pathlib import Path

import torch

from wan_dit_lab.generation.causal import KVCache
from wan_dit_lab.models.rope import rope

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_scale_is_the_trained_one_at_the_trained_resolution_and_softer_below():
    assert math.isclose(KVCache(64, entropy_ratio=1.0).scale(1000, 128), 128 ** -0.5, rel_tol=1e-12)
    smaller, larger = KVCache(64, entropy_ratio=2.256), KVCache(64, entropy_ratio=0.5)  # fewer, more tokens than trained
    assert smaller.scale(12480, 128) < 128 ** -0.5 < larger.scale(12480, 128)
    assert math.isclose(smaller.scale(12480, 128) * 128 ** 0.5, math.sqrt(math.log(12480) / math.log(12480 * 2.256)))
    assert KVCache(64).scale(400, 128) is None


def test_attention_at_the_trained_number_of_keys_is_unchanged():
    torch.manual_seed(0)
    n, heads, d = 32, 2, 64
    Q, K, V = (torch.randn(1, n, heads, d) for _ in range(3))
    plain = KVCache(4 * n, chunk=1).attend(Q, K, V)
    scaled = KVCache(4 * n, chunk=1, entropy_ratio=1.0).attend(Q, K, V)
    softer = KVCache(4 * n, chunk=1, entropy_ratio=4.0).attend(Q, K, V)
    assert torch.allclose(plain, scaled, atol=1e-6)
    assert not torch.allclose(plain, softer, atol=1e-3)


def test_rope_at_scale_one_is_rope():
    from wan_dit_lab.models.positions import rope_at
    assert torch.equal(rope_at(1024, 42, 1.0), rope(1024, 42))
    assert torch.allclose(rope_at(64, 42, 1.5)[:, 2], rope(64, 42)[:, 3])  # position 2 * 1.5 is position 3
