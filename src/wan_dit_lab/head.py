import torch

from wan_dit_lab.layer_norm import LayerNorm


class Head(torch.nn.Module):
    def __init__(self, dim, out_dim, eps, linear=torch.nn.Linear, norm=LayerNorm):
        super().__init__()
        self.norm = norm(dim, eps=eps, elementwise_affine=False)
        self.head = linear(dim, 4 * out_dim)
        self.modulation = torch.nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, X, t):
        # X: [B, N, dim], t: [B, dim] (выход time embedding), modulation: [1, 2, dim]
        # (modulation + t) -> [B, 2, dim] -> shift, scale = ?
        # out = head(norm(X) * (1 + scale) + shift)
        shift, scale = (self.modulation + t.unsqueeze(1)).chunk(2, dim=1)
        return self.head(self.norm(X) * (1 + scale) + shift)
