"""Linear layers in FP8 (E4M3) on the tensor cores of Ada: half the bytes of bfloat16 and twice its peak rate.

W8A8 with one scale per tensor: the weight is quantized once, the activation on every call by its own absolute
maximum. torch._scaled_mm multiplies the two FP8 matrices, rescales and adds the bias in one kernel. A scale per
row is not supported by _scaled_mm on this GPU (docs/roadmap.md).
"""

import torch

FP8 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max  # 448


def quantize(X):
    """X in FP8 and the scale it was divided by: X = scale * X_fp8, the largest |X| goes to 448."""
    scale = X.abs().amax().float().clamp(min=1e-12) / FP8_MAX
    return (X / scale).to(FP8), scale


class Fp8Linear(torch.nn.Module):
    """nn.Linear with the weight in FP8: y = x W^T + b, x quantized on the fly; the output in bfloat16."""

    def __init__(self, linear):
        super().__init__()
        W, self.scale = quantize(linear.weight.detach())
        self.register_buffer("weight_t", W.t())  # [in, out], column-major: the layout _scaled_mm wants for b
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach().to(torch.bfloat16))
        self.out_features = linear.out_features

    def forward(self, X):
        x, scale = quantize(X.reshape(-1, X.shape[-1]))
        y = torch._scaled_mm(x, self.weight_t, scale_a=scale, scale_b=self.scale, bias=self.bias,
                             out_dtype=torch.bfloat16)
        return y.reshape(*X.shape[:-1], self.out_features)


def to_fp8(module):
    """Replaces every nn.Linear inside `module` by an Fp8Linear, in place; the weights must be on the GPU."""
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            setattr(module, name, Fp8Linear(child))
        else:
            to_fp8(child)
    return module
