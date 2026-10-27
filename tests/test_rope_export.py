import torch

from wan_dit_lab.models.rope import rope, rope_apply


class StaticGridRoPE(torch.nn.Module):
    def forward(self, x, frequencies):
        return rope_apply(x, ((2, 2, 3),), frequencies)


def test_static_grid_exports_and_matches_tensor_grid():
    torch.manual_seed(0)
    frequencies = torch.cat([rope(8, 2)] * 3, dim=-1)
    x = torch.randn(1, 12, 2, 6)
    model = StaticGridRoPE()
    exported = torch.export.export(model, (x, frequencies)).module()
    for sample in (x, torch.randn_like(x)):
        expected = rope_apply(sample, torch.tensor([[2, 2, 3]]), frequencies)
        torch.testing.assert_close(model(sample, frequencies), expected, rtol=0, atol=0)
        torch.testing.assert_close(exported(sample, frequencies), expected, rtol=0, atol=0)
