import pytest
import torch

from wan_dit_lab.kernels.rope_compatible import rope, rope_apply


def complex_reference(x, grids, frequencies, start_frame):
    pairs = x.shape[-1] // 2
    time, height, width = torch.complex(*frequencies).split(
        [pairs - 2 * (pairs // 3), pairs // 3, pairs // 3], dim=-1)
    output = []
    for sample, (f, h, w) in zip(x, grids):
        length = f * h * w
        phase = torch.cat((
            time[start_frame:start_frame + f, None, None].expand(f, h, w, -1),
            height[None, :h, None].expand(f, h, w, -1),
            width[None, None, :w].expand(f, h, w, -1)), dim=-1).reshape(length, 1, pairs)
        values = torch.view_as_complex(sample[:length].double().reshape(length, x.shape[2], pairs, 2))
        rotated = torch.view_as_real(values * phase).flatten(-2).to(x.dtype)
        output.append(torch.cat((rotated, sample[length:])))
    return torch.stack(output)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("grids,length", [
    (((2, 2, 3), (2, 2, 3)), 12),
    (((2, 2, 3), (1, 3, 2)), 15),
    (((2, 2, 3), (1, 3, 4)), 12),
])
@pytest.mark.parametrize("start_frame", [0, 3])
def test_batch_matches_complex_reference_and_preserves_padding(dtype, grids, length, start_frame):
    torch.manual_seed(7)
    frequencies = torch.cat([rope(16, 4)] * 3, dim=-1)
    x = torch.randn(2, length, 2, 12).to(dtype).requires_grad_()
    actual = rope_apply(x, torch.tensor(grids), frequencies, start_frame)
    expected = complex_reference(x, grids, frequencies, start_frame)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    probe = torch.randn_like(actual)
    gradient = torch.autograd.grad(actual, x, probe)[0]
    expected_gradient = torch.autograd.grad(expected, x, probe)[0]
    torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)
    for i, (f, h, w) in enumerate(grids):
        torch.testing.assert_close(actual[i, f*h*w:], x[i, f*h*w:], rtol=0, atol=0)


def test_invalid_grid_does_not_silently_broadcast():
    frequencies = torch.cat([rope(8, 4)] * 3, dim=-1)
    with pytest.raises(ValueError, match="per batch"):
        rope_apply(torch.randn(2, 12, 1, 12), ((2, 2, 3),), frequencies)
    with pytest.raises(ValueError, match="exceeds"):
        rope_apply(torch.randn(1, 4, 1, 12), ((2, 2, 3),), frequencies)
