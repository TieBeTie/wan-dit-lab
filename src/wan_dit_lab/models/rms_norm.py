import torch


class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, X):
        # statistics in float32 (autocast does not do it for plain arithmetic), result back to the input type, then the weight
        X32 = X.float()
        return (X32 * torch.rsqrt(X32.pow(2).mean(dim=-1, keepdim=True) + self.eps)).type_as(X) * self.weight
