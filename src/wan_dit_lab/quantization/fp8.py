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


def quantize_rows(X):
    """X [..., d] in FP8 with one scale per row: X = scale * X_fp8, the largest |X| of each row goes to 448. In the KV
    cache a scale per token and head, so that a few large keys do not flatten the small ones."""
    scale = X.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-12) / FP8_MAX
    return (X / scale).to(FP8), scale


class DynamicFp8Linear(torch.nn.Module):
    """W8A8 over an nn.Linear whose weight changes in place (a slot of BlockOffloader): the weight is quantized on
    every call, so it stays whatever block was copied in. rows False: one scale per tensor, as Fp8Linear. rows True:
    one scale per token of x and one per output row of W; _scaled_mm has no row-wise scales on Ada (sm_89), so it
    multiplies with unit scales and the scales are applied to its output, y_ij = s_i t_j sum_k x_ik w_jk."""

    def __init__(self, linear, rows=False):
        super().__init__()
        self.linear, self.rows = linear, rows
        self.register_buffer("one", torch.ones((), device=linear.weight.device), persistent=False)

    def forward(self, X):
        W, b = self.linear.weight, self.linear.bias
        x = X.reshape(-1, X.shape[-1])
        if self.rows:
            (xq, xs), (wq, ws) = quantize_rows(x), quantize_rows(W)
            y = torch._scaled_mm(xq, wq.t(), scale_a=self.one, scale_b=self.one, out_dtype=torch.bfloat16)
            y = y * (xs * ws.t())
            y = y if b is None else y + b
        else:
            (xq, xs), (wq, ws) = quantize(x), quantize(W)
            y = torch._scaled_mm(xq, wq.t(), scale_a=xs, scale_b=ws, out_dtype=torch.bfloat16,
                                 bias=None if b is None else b.to(torch.bfloat16))
        return y.to(torch.bfloat16).reshape(*X.shape[:-1], W.shape[0])


def to_dynamic_fp8(module, rows=False):
    """Every nn.Linear inside `module` wrapped by a DynamicFp8Linear, in place."""
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            setattr(module, name, DynamicFp8Linear(child, rows))
        else:
            to_dynamic_fp8(child, rows)
    return module


class PackedFp8Linear(torch.nn.Module):
    """A linear layer whose FP8 weight, its scale and its bias are views of a row of bytes (a block of a block_fp8_v1
    pack in a GPU slot of BlockOffloader): nothing is quantized but the activation, and a new block copied into the
    row is the new layer."""

    def __init__(self, weight, scale, bias):
        super().__init__()
        self.weight, self.scale, self.bias = weight, scale, bias  # [out, in] FP8, a 0-d float32, [out] bfloat16
        self.out_features = weight.shape[0]

    def forward(self, X):
        x, scale = quantize(X.reshape(-1, X.shape[-1]))
        y = torch._scaled_mm(x, self.weight.t(), scale_a=scale, scale_b=self.scale, bias=self.bias,
                             out_dtype=torch.bfloat16)
        return y.reshape(*X.shape[:-1], self.out_features)


def fp8_slot(row, like, layout):
    """A copy of the block `like` (structure only) over `row`, a uint8 row of a block_fp8_v1 pack: each nn.Linear a
    PackedFp8Linear on the views its layout entries name, every other parameter a bfloat16 view."""
    import copy

    def view(entry):
        start, end = entry["offset"], entry["offset"] + entry["bytes"]
        dtype = {"fp8": FP8, "bf16": torch.bfloat16, "f32": torch.float32}[entry["dtype"]]
        return row[start:end].view(dtype).view(entry["shape"])

    entries = {e["name"]: e for e in layout}
    module = copy.deepcopy(like).to("meta").requires_grad_(False)
    for name, child in list(module.named_modules()):
        if isinstance(child, torch.nn.Linear):
            parent, _, attr = name.rpartition(".")
            bias = entries.get(f"{name}.bias")
            setattr(module.get_submodule(parent) if parent else module, attr, PackedFp8Linear(
                view(entries[f"{name}.weight"]), view(entries[f"{name}.scale"]), None if bias is None else view(bias)))
    for name, _ in list(module.named_parameters()):
        parent, _, attr = name.rpartition(".")
        owner = module.get_submodule(parent) if parent else module
        setattr(owner, attr, torch.nn.Parameter(view(entries[name]), requires_grad=False))
    return module


def to_fp8(module):
    """Replaces every nn.Linear inside `module` by an Fp8Linear, in place; the weights must be on the GPU."""
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            setattr(module, name, Fp8Linear(child))
        else:
            to_fp8(child)
    return module
