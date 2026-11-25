"""Frequency-aware phases, axis isolation and an unchanged compiled consumer graph."""
import math

import pytest
import torch

from wan_dit_lab.models.positions import rope_at, spatial_rope


@pytest.mark.parametrize("scale", [0.9, 1.1, 1.2])
def test_frequency_anchor_is_one_rotation_on_the_trained_axis(scale):
    # Choose L so that pair 4 has exactly one rotation across the trained axis.
    trained = 2 * math.pi * 10000 ** (8 / 42)
    table = spatial_rope(2, 42, scale, trained, frequency_aware=True)
    angles = torch.atan2(table[1, 1], table[0, 1])
    theta = 10000 ** (-2 * torch.arange(21, dtype=torch.float64) / 42)
    torch.testing.assert_close(angles[0], theta[0])
    torch.testing.assert_close(angles[2], theta[2] * math.sqrt(scale))
    torch.testing.assert_close(angles[4:], theta[4:] * scale)


@pytest.mark.parametrize("scale", [0.9, 1.1])
@pytest.mark.parametrize("trained", [22, 40])
def test_profile_matches_base_change_with_directional_pi_clipping(scale, trained):
    target = 42 / 2 * math.log(trained / (2 * math.pi), 10000)
    base = 10000 / scale ** (42 / (2 * target))
    pair = torch.arange(21, dtype=torch.float64)
    theta = 10000 ** (-2 * pair / 42)
    changed = base ** (-2 * pair / 42)
    # The shrinking-grid extension reverses clipping, keeping the change bounded by M.
    expected = torch.minimum(changed, theta * scale) if scale > 1 else torch.maximum(changed, theta * scale)
    actual = spatial_rope(2, 42, scale, trained, frequency_aware=True)
    torch.testing.assert_close(torch.atan2(actual[1, 1], actual[0, 1]), expected)


def test_frequency_profile_preserves_time_and_identity(monkeypatch):
    from wan_dit_lab.runtime import live_presets
    monkeypatch.setattr(live_presets, "POSITIONS", 8)
    base = dict(height=512, width=928, native=(704, 1280), vae_stride=16,
                rope_dtype="float64", rope_interpolate=False)
    plain = live_presets.rope_table(base, 128, "cpu")
    identity = live_presets.rope_table(dict(base, rope_interpolate=True, rope_stretch=0,
                                           rope_frequency_aware=True), 128, "cpu")
    torch.testing.assert_close(identity, plain, rtol=1e-12, atol=1e-12)
    changed = live_presets.rope_table(dict(base, rope_interpolate=True, rope_stretch=.3,
                                          rope_frequency_aware=True), 128, "cpu")
    assert torch.equal(changed[..., :22], plain[..., :22])
    assert not torch.equal(changed[..., 22:], plain[..., 22:])


@pytest.mark.parametrize("other", [{"ntk_aware": True}, {"by_parts": {"alpha": .25, "beta": 2}}])
def test_frequency_profile_cannot_be_combined_with_another_profile(other):
    with pytest.raises(ValueError, match="one RoPE frequency profile"):
        spatial_rope(2, 42, 1.1, 22, frequency_aware=True, **other)


def test_short_axis_without_a_high_frequency_band_uses_uniform_scale():
    torch.testing.assert_close(spatial_rope(8, 42, 1.1, 4, frequency_aware=True), rope_at(8, 42, 1.1))


@pytest.mark.parametrize("trained", [0, -1, math.inf, math.nan])
def test_invalid_trained_length_is_rejected(trained):
    with pytest.raises(ValueError, match="positive finite trained axis"):
        spatial_rope(2, 42, 1.1, trained, frequency_aware=True)


@pytest.mark.parametrize("profile", [{}, {"ntk_aware": True}, {"by_parts": {"alpha": .25, "beta": 2}}])
def test_disabled_frequency_profile_leaves_legacy_tables_identical(profile):
    assert torch.equal(spatial_rope(8, 42, 1.1, 22, **profile),
                       spatial_rope(8, 42, 1.1, 22, frequency_aware=False, **profile))


def test_frequency_profile_is_no_export_input(tmp_path):
    # the height and width RoPE table is an input of the packages, wan.exe computes it (rope.rs)
    from wan_dit_lab.runtime.live_pipeline import export_identity
    export = {"model": {}, "rope": {"interpolate": True, "stretch": .3, "ntk_aware": False}}
    before = export_identity(export, tmp_path, sources=[], versions={})
    export["rope"]["frequency_aware"] = True
    assert export_identity(export, tmp_path, sources=[], versions={}) == before


def test_exported_rotation_has_the_same_operators_for_both_profiles():
    class Rotate(torch.nn.Module):
        def __init__(self, table):
            super().__init__()
            self.register_buffer("cs", table.float())

        def forward(self, x):
            a, b = x[..., 0::2], x[..., 1::2]
            return torch.stack((a * self.cs[0] - b * self.cs[1],
                                b * self.cs[0] + a * self.cs[1]), -1).flatten(-2)

    x = torch.zeros(8, 42)
    graphs = []
    for profile in ({"ntk_aware": True}, {"frequency_aware": True}):
        module = Rotate(spatial_rope(8, 42, 1.1, 22, **profile))
        graph = torch.export.export(module, (x,)).graph
        graphs.append([(n.op, str(n.target)) for n in graph.nodes])
    assert graphs[0] == graphs[1]
