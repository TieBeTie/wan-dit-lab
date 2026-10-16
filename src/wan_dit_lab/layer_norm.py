import torch


class LayerNorm(torch.nn.LayerNorm):
    def forward(self, X):
        # autocast computes layer norm in float32; the result goes back to the input type, as in the reference
        return super().forward(X).type_as(X)
