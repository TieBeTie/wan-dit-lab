import math

import pytest

from wan_dit_lab.generation.causal import sigmas


def test_resolution_shift_composes_with_checkpoint_schedule():
    # SD3 Eq. 23 maps the already shifted checkpoint levels to the new resolution.
    ratio = math.sqrt((256 * 448) / (480 * 832))
    original = sigmas()
    mapped = [ratio * t / (1 + (ratio - 1) * t) for t in original]
    assert sigmas(shift=5 * ratio) == pytest.approx(mapped)
    assert original == pytest.approx([1, 0.9375, 5 / 6, 0.625])
    assert mapped[0] == 1
    assert all(a > b for a, b in zip(mapped, mapped[1:]))
