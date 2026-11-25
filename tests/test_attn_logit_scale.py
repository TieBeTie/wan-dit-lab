"""The attention logit multiplier lambda, carried by the RoPE table: q and k both pass through the table, so a table
scaled by sqrt(lambda) multiplies every self-attention logit by lambda and adds no operator to the compiled graph."""
import math

import pytest
import torch

from wan_dit_lab.runtime import live_presets

BASE = dict(height=512, width=928, native=(704, 1280), vae_stride=16, rope_dtype="float64",
            rope_interpolate=True, rope_stretch=0.3, rope_ntk_aware=True)


def rotate(x, table):
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a * table[0] - b * table[1], b * table[0] + a * table[1]), -1).flatten(-2)


@pytest.fixture(autouse=True)
def small_table(monkeypatch):
    monkeypatch.setattr(live_presets, "POSITIONS", 8)


def test_scale_one_leaves_the_table_unchanged():
    plain = live_presets.rope_table(BASE, 128, "cpu")
    assert torch.equal(live_presets.rope_table(dict(BASE, attn_logit_scale=1.0), 128, "cpu"), plain)


@pytest.mark.parametrize("scale", [0.93, 0.965, 1.035])
def test_every_logit_is_multiplied_by_the_scale(scale):
    plain = live_presets.rope_table(BASE, 128, "cpu")
    scaled = live_presets.rope_table(dict(BASE, attn_logit_scale=scale), 128, "cpu")
    q, k = torch.randn(2, 8, 128, dtype=torch.float64)
    logits = rotate(q, plain[..., :8, :]) @ rotate(k, plain[..., :8, :]).T
    shrunk = rotate(q, scaled[..., :8, :]) @ rotate(k, scaled[..., :8, :]).T
    torch.testing.assert_close(shrunk, scale * logits)


def test_scale_applies_without_interpolation_too():
    base = dict(BASE, rope_interpolate=False)
    plain = live_presets.rope_table(base, 128, "cpu")
    scaled = live_presets.rope_table(dict(base, attn_logit_scale=0.81), 128, "cpu")
    torch.testing.assert_close(scaled, plain * math.sqrt(0.81))


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_scale_is_refused(bad):
    with pytest.raises(ValueError):
        live_presets.rope_table(dict(BASE, attn_logit_scale=bad), 128, "cpu")


def test_scale_changes_the_export_identity(tmp_path):
    from wan_dit_lab.runtime.live_pipeline import export_identity
    export = {"model": {}, "rope": {"interpolate": True, "stretch": .3, "ntk_aware": True}}
    before = export_identity(export, tmp_path, sources=[], versions={})
    export["rope"]["attn_logit_scale"] = 0.93
    assert export_identity(export, tmp_path, sources=[], versions={}) != before
