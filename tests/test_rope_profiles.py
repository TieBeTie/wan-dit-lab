import math
import torch
from wan_dit_lab.models.positions import spatial_rope


def test_profiles_preserve_time_and_zero_strength():
    from wan_dit_lab.runtime.live_presets import rope_table
    base = dict(height=512, width=928, native=(704,1280), vae_stride=16, rope_dtype="float64", rope_interpolate=False)
    original = rope_table(base, 128, "cpu")
    for profile in ({"rope_ntk_aware": True}, {"rope_by_parts": {"alpha": .25, "beta": 2}}):
        actual = rope_table(dict(base, rope_interpolate=True, rope_stretch=.3, **profile), 128, "cpu")
        assert torch.equal(actual[..., :22], original[..., :22])
        zero = rope_table(dict(base, rope_interpolate=True, rope_stretch=0., **profile), 128, "cpu")
        torch.testing.assert_close(zero, original, rtol=1e-12, atol=1e-12)


def test_ntk_preserves_fastest_and_scales_slowest():
    table = spatial_rope(2, 42, 1.1, 22, ntk_aware=True)
    angles = torch.atan2(table[1, 1], table[0, 1])
    theta = 10000 ** (-2 * torch.arange(21, dtype=torch.float64) / 42)
    torch.testing.assert_close(angles[0], theta[0])
    torch.testing.assert_close(angles[-1], theta[-1] * 1.1)


def test_by_parts_preserves_fast_pair_and_scales_slow_pair():
    table = spatial_rope(2, 42, 1.1, 22, by_parts={"alpha": .25, "beta": 2})
    angles = torch.atan2(table[1, 1], table[0, 1])
    theta = 10000 ** (-2 * torch.arange(21, dtype=torch.float64) / 42)
    torch.testing.assert_close(angles[0], theta[0])
    torch.testing.assert_close(angles[-1], theta[-1] * 1.1)


def _hw(spec):
    from wan_dit_lab.runtime.live_presets import rope_table
    base = dict(height=608, width=1120, native=(704, 1280), vae_stride=16, rope_dtype="float64", rope_interpolate=True)
    return rope_table(dict(base, **spec), 128, "cpu")[:, :40, 22:]


def test_m_per_axis_is_the_scale_of_its_axis():
    # rope_m: M of height and width given directly; equal to the stretch that gives the same scale per side
    s = math.log(1.2) / math.log(22 / 19)
    by_stretch = _hw({"rope_stretch": s, "rope_ntk_aware": True})
    by_m = _hw({"rope_m": {"height": 1.2, "width": (40 / 35) ** s}, "rope_ntk_aware": True})
    torch.testing.assert_close(by_m, by_stretch, rtol=1e-12, atol=1e-12)
    # width alone: the height columns stay those of M 1.2
    other = _hw({"rope_m": {"height": 1.2, "width": 1.4}, "rope_ntk_aware": True})
    torch.testing.assert_close(other[..., :21], by_stretch[..., :21], rtol=1e-12, atol=1e-12)
    assert not torch.allclose(other[..., 21:], by_stretch[..., 21:])


def test_ntk_pi_mix_goes_from_ntk_to_pi():
    m = {"height": 1.2, "width": 1.2}
    ntk = _hw({"rope_m": m, "rope_ntk_aware": True})
    pi = _hw({"rope_m": m})
    torch.testing.assert_close(_hw({"rope_m": m, "rope_ntk_aware": True, "rope_ntk_pi_mix": 0.0}), ntk, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(_hw({"rope_m": m, "rope_ntk_aware": True, "rope_ntk_pi_mix": 1.0}), pi, rtol=1e-12, atol=1e-12)
    # half way: the fastest pair is scaled by sqrt(M), the slowest by M
    half = spatial_rope(2, 42, 1.2, 22, ntk_aware=True, ntk_pi_mix=0.5)
    angles = torch.atan2(half[1, 1], half[0, 1])
    theta = 10000 ** (-2 * torch.arange(21, dtype=torch.float64) / 42)
    torch.testing.assert_close(angles[0], theta[0] * 1.2 ** 0.5)
    torch.testing.assert_close(angles[-1], theta[-1] * 1.2)
