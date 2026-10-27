"""Linear layers in 4 bits, weights and activations (W4A4), as SVDQuant (Li et al., "SVDQuant: Absorbing Outliers by
Low-Rank Components for 4-Bit Diffusion Models"), on the INT4 tensor cores of Ada through the kernels of Nunchaku.

y = x W^T + b is computed as

    y = (x / s) (W diag(s))^T + b,      W diag(s) = U V + R,      y = x (V diag(1/s))^T U^T + Q(x / s) Q(R)^T + b

s smooths the outliers of the activations into the weights (per input channel), U V (rank r, 16 bits) takes the
largest singular directions of the smoothed weight, and the residual R and the smoothed activation go through INT4
with one scale per group of 64 input channels. Found by tests against x W^T (tests/test_w4a4.py): the kernel divides
x by s (given packed, as the scales) before quantizing it, and its low-rank branch reads x itself, not x / s, so the
down projection carries the 1/s. The kernels (live::w4a4_linear and live::w4a4_gelu_mlp of
native/ops, built from reference/nunchaku, Apache 2.0, commit 302e0e9) fuse the smoothing, the activation quantization and the down projection x V^T into one
pass, and the INT4 product, the up projection and the bias into another.

The packing of weights, scales and low-rank factors into the order of the MMA tiles is Nunchaku's own
(nunchaku/lora/flux/packer.py), executed from the reference checkout with its two helpers given here.
"""

from pathlib import Path

import torch

from wan_dit_lab.kernels import operators
from wan_dit_lab.models.multi_head_attention import MultiHeadAttention
from wan_dit_lab.models.rope import rope_apply

ROOT = Path(__file__).resolve().parents[3]
GROUP = 64  # input channels per INT4 scale
FORMATS = ("block_w4a4_v1", "block_w4a4_v2")  # a layer per nn.Linear; v2 with q, k, v as one and the fused GELU MLP


def ceil_divide(x, divisor):
    return (x + divisor - 1) // divisor


def pad(tensor, divisor, dim, fill_value=0):
    """nunchaku/lora/flux/utils.py: the tensor padded so that the dimensions `dim` are multiples of `divisor`."""
    if isinstance(divisor, int):
        if divisor <= 1:
            return tensor
    elif all(d <= 1 for d in divisor):
        return tensor
    if tensor is None:
        return None
    shape = list(tensor.shape)
    if isinstance(dim, int):
        shape[dim] = ceil_divide(shape[dim], divisor) * divisor
    else:
        divisor = [divisor] * len(dim) if isinstance(divisor, int) else divisor
        for d, div in zip(dim, divisor, strict=True):
            shape[d] = ceil_divide(shape[d], div) * div
    result = torch.full(shape, fill_value, dtype=tensor.dtype, device=tensor.device)
    result[tuple(slice(0, extent) for extent in tensor.shape)] = tensor
    return result


def _packer_class():
    source = ROOT / "reference" / "nunchaku" / "nunchaku" / "lora" / "flux" / "packer.py"
    code = source.read_text(encoding="utf-8")
    code = code.replace("from ...utils import ceil_divide\n", "").replace("from .utils import pad\n", "")
    namespace = {"ceil_divide": ceil_divide, "pad": pad, "__name__": "nunchaku_packer"}
    exec(compile(code, str(source), "exec"), namespace)
    return namespace["NunchakuWeightPacker"]


_PACKER = None


def packer():
    global _PACKER
    if _PACKER is None:
        _PACKER = _packer_class()(bits=4)
    return _PACKER


MSE_STEPS = torch.linspace(0.5, 1.0, 26).tolist()  # the scales tried, as fractions of max|x| / 7


@torch.no_grad()
def int4_groups(R, rule="max"):
    """R [N, K] in symmetric INT4 groups of GROUP input channels: the integers [N, K] and the scales [N, K / GROUP].
    rule "max": the scale puts the largest |x| of a group on 7. "mse": of the scales MSE_STEPS * that one, the one
    with the least squared error over the group, the rule of LongLive 2.0's NVFP4 (weight_scale_rule: mse); a larger
    value is clipped to the end of the grid so that the rest of the group gets a finer one."""
    N, K = R.shape
    groups = R.float().view(N, K // GROUP, GROUP)
    top = groups.abs().amax(dim=-1).clamp(min=1e-8) / 7
    scale = top
    if rule == "mse":
        best = torch.full_like(top, float("inf"))
        for c in MSE_STEPS:
            s = top * c
            err = ((groups - torch.round(groups / s[..., None]).clamp(-8, 7) * s[..., None]) ** 2).sum(-1)
            better = err < best
            best, scale = torch.where(better, err, best), torch.where(better, s, scale)
    elif rule != "max":
        raise ValueError(rule)
    q = torch.round(groups / scale[..., None]).clamp(-8, 7).to(torch.int32).view(N, K)
    return q, scale


@torch.no_grad()
def gptq(R, H, damp=0.01, block=128):
    """GPTQ (Frantar et al., ICLR 2023) of R [N, K] into INT4 groups of GROUP input channels, symmetric, one scale per
    row and group taken from the columns as GPTQ has updated them: minimises ||X R^T - X Q(R)^T|| for inputs with
    H = X^T X [K, K]. Columns are rounded in order; the error of each is spread over the columns not yet rounded
    through the Cholesky factor of H^-1. Returns the integers [N, K] and the scales [N, K / GROUP]."""
    W = R.clone().float()
    N, K = W.shape
    H = H.float().clone()
    dead = torch.diag(H) == 0  # channels that never carry anything: any value there costs nothing
    H[dead, dead] = 1
    W[:, dead] = 0
    H += damp * torch.diag(H).mean() * torch.eye(K, device=H.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    del H
    q = torch.zeros(N, K, dtype=torch.int32, device=W.device)
    scale = torch.zeros(N, K // GROUP, device=W.device)
    for i1 in range(0, K, block):
        i2 = min(i1 + block, K)
        W1, Err = W[:, i1:i2].clone(), torch.zeros(N, i2 - i1, device=W.device)
        Hb = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if col % GROUP == 0:  # the scale of a group from its columns as they are now
                g = torch.cat([W1[:, i:], W[:, i2:]], dim=1)[:, :GROUP]
                scale[:, col // GROUP] = g.abs().amax(dim=1).clamp(min=1e-8) / 7
            sc = scale[:, col // GROUP]
            w = W1[:, i]
            qi = torch.round(w / sc).clamp(-8, 7)
            q[:, col] = qi.to(torch.int32)
            err = (w - qi * sc) / Hb[i, i]
            W1[:, i:] -= err[:, None] * Hb[i, i:][None, :]
            Err[:, i] = err
        W[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    return q, scale


@torch.no_grad()
def quantize(weight, bias=None, smooth=None, rank=32, shift=0.0, hessian=None, rule="max"):
    """The tensors of a W4A4 layer for y = x W^T + b. weight [N, K]; smooth [K] (ones when None); returns
    dict(qweight [N, K/2] int8, wscales, proj_down [K, r], proj_up [N, r], smooth [K], bias [N]) in Nunchaku's packed
    layouts, bfloat16 where not INT4. shift: the kernel quantizes x + shift, unsigned (the second layer of gelu_mlp);
    the low-rank branch reads x itself, so the bias takes off shift (R / s) 1, what the shift adds through INT4.
    hessian [K, K]: X^T X of the inputs the INT4 branch multiplies by R (x + shift for the second layer of gelu_mlp,
    before smoothing): R is rounded by GPTQ in the smoothed space instead of to the nearest. rule: how the scale of
    an INT4 group is chosen (int4_groups)."""
    p = packer()
    W = weight.float()
    N, K = W.shape
    s = torch.ones(K, device=W.device) if smooth is None else smooth.float().to(W.device)
    Ws = W * s[None, :]  # W diag(s): the activation is divided by s
    U, S, Vh = torch.linalg.svd(Ws, full_matrices=False)
    up = U[:, :rank] * S[:rank]  # [N, r]
    down = Vh[:rank]  # [r, K]
    R = Ws - up @ down
    if hessian is None:  # to the nearest
        q, scale = int4_groups(R, rule)
    else:  # the INT4 branch sees x / s: the Hessian of x / s is diag(1/s) H diag(1/s)
        # a wide layer (the feed forward's second, K = 14336: H is 0.8 GB, GPTQ holds several of it) on the CPU,
        # beside a model on the GPU
        where = torch.device("cpu") if K > 8192 else W.device
        Hs = hessian.float().to(where) / s.to(where)[:, None] / s.to(where)[None, :]
        q, scale = gptq(R.to(where), Hs)
        q, scale = q.to(W.device), scale.to(W.device)
    b = torch.zeros(N, device=W.device) if bias is None else bias.float().to(W.device)
    if shift:
        Rq = (q.view(N, K // GROUP, GROUP).float() * scale[..., None]).view(N, K)  # R as INT4 holds it
        b = b - shift * (Rq / s[None, :]).sum(dim=1)
    vector = lambda v: p.pack_scale(p.pad_scale(v.to(torch.bfloat16), group_size=-1), group_size=-1)  # packed [N]
    out = dict(
        qweight=p.pack_weight(p.pad_weight(q)),
        wscales=p.pack_scale(p.pad_scale(scale.to(torch.bfloat16), group_size=GROUP), group_size=GROUP),  # [N, K/64]
        proj_down=p.pack_lowrank_weight((down / s[None, :]).to(torch.bfloat16), down=True),  # the branch reads x
        proj_up=p.pack_lowrank_weight(up.to(torch.bfloat16), down=False),
        smooth=vector(s),  # the kernel divides x by it
        bias=vector(b),
    )
    return out


def smoothing(act_amax, weight, alpha=0.5):
    """SmoothQuant's s_j = max|x_j|^alpha / max|W_:j|^(1 - alpha), from the largest |x| of each input channel."""
    w = weight.float().abs().amax(dim=0).clamp(min=1e-5)
    return (act_amax.float().to(w.device).clamp(min=1e-5) ** alpha / w ** (1 - alpha)).clamp(min=1e-2, max=1e2)


def linear(x, t):
    """y = x W^T + b for x [M, K] bfloat16 and t the tensors of quantize: live::w4a4_linear (native/ops), the
    activation quantized to INT4 with its low-rank branch, then Nunchaku's W4A4 GEMM."""
    return w4a4_linear(x, *(t[part] for part in PARTS))


GELU_SHIFT = 0.171875  # the kernel's SHIFT_GELU: GELU(x) > -0.17, so GELU(x) + GELU_SHIFT is quantized unsigned


def gelu_mlp(x, t1, t2):
    """y = GELU(x W1^T + b1) W2^T + b2, GELU in tanh form as Wan's feed forward, for x [M, K] bfloat16; t1, t2 the
    tensors of quantize, t2 made with shift=GELU_SHIFT. Nunchaku's fused_gelu_mlp (nunchaku/ops/fused.py): the first
    GEMM applies in its epilogue GELU, the down projection of the second layer's low-rank branch (on GELU's output)
    and the INT4 quantization of GELU's output + shift for the second GEMM, so the hidden layer never goes to memory
    in 16 bits (live::w4a4_gelu_mlp)."""
    return w4a4_gelu_mlp(x, *(t1[part] for part in PARTS), *(t2[part] for part in PARTS))


PARTS = ("qweight", "wscales", "proj_down", "proj_up", "smooth", "bias")  # of a W4A4 layer, as quantize gives them


# The kernels as one operator each (C++, native/ops), so that torch.compile keeps the rest of a block in its graph
# around them and AOTInductor calls them without Python.
def w4a4_linear(x, qweight, wscales, proj_down, proj_up, smooth, bias):
    return operators.load().w4a4_linear(x, qweight, wscales, proj_down, proj_up, smooth, bias)


def w4a4_gelu_mlp(*args):
    return operators.load().w4a4_gelu_mlp(*args)


class PackedW4A4Linear(torch.nn.Module):
    """A linear layer in W4A4 whose tensors are views of a row of bytes (a block of a block_w4a4_v1 pack in a slot of
    BlockOffloader), as fp8.PackedFp8Linear is for FP8."""

    def __init__(self, **parts):
        super().__init__()
        self.parts = [parts[name] for name in PARTS]
        self.out_features = parts["proj_up"].shape[0]

    def forward(self, X):
        y = w4a4_linear(X.reshape(-1, X.shape[-1]).to(torch.bfloat16).contiguous(), *self.parts)
        return y.reshape(*X.shape[:-1], self.out_features)


class PackedW4A4GeluMLP(torch.nn.Module):
    """Wan's feed forward (Linear, GELU in tanh form, Linear) in W4A4 as one gelu_mlp; its tensors are views as in
    PackedW4A4Linear."""

    def __init__(self, fc1, fc2):
        super().__init__()
        self.parts = [fc1[name] for name in PARTS] + [fc2[name] for name in PARTS]
        self.out_features = fc2["proj_up"].shape[0]

    def forward(self, X):
        y = w4a4_gelu_mlp(X.reshape(-1, X.shape[-1]).to(torch.bfloat16).contiguous(), *self.parts)
        return y.reshape(*X.shape[:-1], self.out_features)


class FusedQKVAttention(MultiHeadAttention):
    """Self-attention whose q, k and v are one W4A4 layer qkv [3 dim, dim]: X is quantized once and goes through one
    GEMM instead of three. The rest as MultiHeadAttention.forward."""

    def forward(self, X, grid_sizes=None, freq=None, context=None, start_frame=0, cache=None, context_kv=None):
        assert context is None and context_kv is None, "q, k and v of one input: self-attention only"
        B, L = X.shape[0], X.shape[-2]
        q, k, v = self.qkv(X).split(self.dim, dim=-1)
        Q = rope_apply(self.norm_q(q).reshape(B, L, self.num_heads, self.head_dim), grid_sizes, freq, start_frame)
        K = rope_apply(self.norm_k(k).reshape(B, L, self.num_heads, self.head_dim), grid_sizes, freq, start_frame)
        V = v.reshape(B, L, self.num_heads, self.head_dim)
        if cache is not None:
            return self.o(cache.attend(Q, K, V).reshape(B, L, self.dim))
        A = torch.nn.functional.scaled_dot_product_attention(Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2))
        return self.o(A.transpose(1, 2).reshape(B, L, self.dim))


QKV = ("self_attn.q", "self_attn.k", "self_attn.v")  # one input: one layer self_attn.qkv in a fused pack
FFN = ("ffn.0", "ffn.2")  # Linear, GELU, Linear: one gelu_mlp in a fused pack


def layers(block, fused=False, keep=()):
    """The W4A4 layers of a block: {name: [(name, nn.Linear) it is made of, stacked by output]}; fused: q, k and v of
    self-attention as one layer self_attn.qkv; keep: names of nn.Linear left in bfloat16."""
    out = {}
    for name, m in block.named_modules():
        if isinstance(m, torch.nn.Linear) and name not in keep:
            out.setdefault("self_attn.qkv" if fused and name in QKV else name, []).append((name, m))
    return out


def pack_layout(block, fused=False, rank=32, keep=()):
    """Where each tensor of a block lies in its row of a pack (block_w4a4_v1, or v2 when fused): the six parts of
    every layer of `layers` (their shapes from quantize on a zero weight), every other parameter in bfloat16; each
    entry 16-byte aligned."""
    groups = layers(block, fused, keep)
    group_of = {name: g for g, members in groups.items() for name, _ in members}
    layout, offset = [], 0

    def add(name, dtype, shape, size, **extra):
        nonlocal offset
        layout.append(dict(name=name, dtype=dtype, shape=list(shape), offset=offset, bytes=size, **extra))
        offset += -(-size // 16) * 16

    def shape(g):
        members = [m for _, m in groups[g]]
        return sum(m.out_features for m in members), members[0].in_features

    shapes = {}
    for g in groups:
        key = shape(g)
        if key not in shapes:
            parts = quantize(torch.zeros(key, device="cuda"), None, rank=rank)
            shapes[key] = {k: (str(v.dtype).removeprefix("torch."), tuple(v.shape), v.numel() * v.element_size())
                           for k, v in parts.items()}
    for name, p in block.named_parameters():
        owner = name.rsplit(".", 1)[0]
        if owner in group_of:
            g = group_of[owner]
            if name.endswith(".weight") and groups[g][0][0] == owner:
                for part, (dtype, size_shape, size) in shapes[shape(g)].items():
                    # made for gelu_mlp: a slot must install it so even without self_attn.qkv beside it
                    add(f"{g}.{part}", dtype, size_shape, size, **({"gelu_mlp": True} if fused and g == FFN[1] else {}))
            continue  # the bias is one of the parts
        add(name, "bfloat16", p.shape, p.numel() * 2)
    return layout, offset


@torch.no_grad()
def layer_parts(block, index, amax, alpha=0.5, rank=32, fused=False, keep=(), hessians=None, rule="max"):
    """The tensors of every W4A4 layer of block number `index` (quantize, on the GPU): {layer: tensors}. The
    smoothing from amax {"<block>.<linear>": largest |x| of each input channel}; in a fused pack the second layer of
    the feed forward is made for gelu_mlp, whose kernel quantizes GELU's output + GELU_SHIFT."""
    out = {}
    for g, members in layers(block, fused, keep).items():
        W = torch.cat([m.weight.detach().to("cuda", torch.float32) for _, m in members])
        b = None
        if any(m.bias is not None for _, m in members):
            b = torch.cat([torch.zeros(m.out_features, device="cuda") if m.bias is None else
                           m.bias.detach().to("cuda", torch.float32) for _, m in members])
        shift = GELU_SHIFT if fused and g == FFN[1] else 0.0
        a = amax.get(f"{index}.{members[0][0]}")  # the members share their input
        s = None if a is None else smoothing(a.to("cuda") + shift, W, alpha)
        H = None if hessians is None else hessians.get(g)  # X^T X of the layer's input (members share it)
        out[g] = quantize(W, b, smooth=s, rank=rank, shift=shift, hessian=H, rule=rule)
    return out


def install(block, parts, gelu_mlp=None):
    """block with its W4A4 layers from parts {layer of `layers`: tensors of quantize}; self_attn.qkv brings
    FusedQKVAttention, gelu_mlp (by default: a fused set, one with self_attn.qkv) PackedW4A4GeluMLP."""
    if gelu_mlp is None:
        gelu_mlp = "self_attn.qkv" in parts
    if "self_attn.qkv" in parts:
        attention = block.self_attn
        del attention.q, attention.k, attention.v
        attention.__class__ = FusedQKVAttention
        attention.qkv = PackedW4A4Linear(**parts["self_attn.qkv"])
    if gelu_mlp and FFN[1] in parts:
        block.ffn = PackedW4A4GeluMLP(parts[FFN[0]], parts[FFN[1]])
    for name, child in list(block.named_modules()):
        if isinstance(child, torch.nn.Linear) and name in parts:  # the others stay nn.Linear in bfloat16
            parent, _, attr = name.rpartition(".")
            setattr(block.get_submodule(parent) if parent else block, attr, PackedW4A4Linear(**parts[name]))
    return block


def convert(block, index=0, amax=None, alpha=0.5, rank=32, fused=False, keep=()):
    """block, on the GPU, in W4A4 in place: what a slot of its pack holds, with the tensors in memory of their own."""
    return install(block, layer_parts(block, index, amax or {}, alpha, rank, fused, keep), gelu_mlp=fused)


@torch.no_grad()
def build_pack(blocks, amax, path, source, alpha=0.5, rank=32, fused=False, keep=(), hessians=None, rule="max"):
    """The blocks as a pack: per block one uint8 row (pack_layout). amax: {"<block>.<linear>": the largest |x| of
    each input channel of that linear layer} from a calibration run (ActStats). fused: block_w4a4_v2, with q, k, v
    of self-attention as one layer and the feed forward as one gelu_mlp; else block_w4a4_v1, a layer per nn.Linear."""
    import hashlib
    import json
    import os

    from wan_dit_lab.quantization.weight_pack import source_identity
    path = Path(path)
    manifest, temporary = path.with_suffix(path.suffix + ".json"), path.with_suffix(path.suffix + ".tmp")
    if any(p.exists() for p in (path, manifest, temporary)):
        raise FileExistsError(path)
    layout, size = pack_layout(blocks[0], fused, rank, keep)
    where = {e["name"]: e for e in layout}
    digest, identity = hashlib.sha256(), source_identity(source)
    try:
        with temporary.open("xb") as stream:
            for index, block in enumerate(blocks):
                row = torch.zeros(size, dtype=torch.uint8)
                params = dict(block.named_parameters())
                # hessians(index): {layer: X^T X} of that block, measured on demand (GPTQ), or None (to the nearest)
                tensors = layer_parts(block, index, amax, alpha, rank, fused, keep,
                                      None if hessians is None else hessians(index), rule)
                for g, parts in tensors.items():
                    for part, value in parts.items():
                        e = where[f"{g}.{part}"]
                        row[e["offset"]:e["offset"] + e["bytes"]] = value.contiguous().cpu().reshape(-1).view(torch.uint8)
                linear_parts = {f"{g}.{part}" for g in tensors for part in PARTS}
                for e in layout:  # the rest in bfloat16; a linear bias is a packed part, not this raw parameter
                    if e["name"] in params and e["name"] not in linear_parts:
                        data = params[e["name"]].detach().to(torch.bfloat16).cpu().reshape(-1).view(torch.uint8)
                        row[e["offset"]:e["offset"] + e["bytes"]] = data
                payload = row.numpy().tobytes()
                digest.update(payload)
                stream.write(payload)
                print(f"block {index} packed", flush=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        metadata = dict(format="block_w4a4_v2" if fused else "block_w4a4_v1", blocks=len(blocks), layout=layout,
                        bytes_per_block=size, bytes=size * len(blocks), sha256=digest.hexdigest(), source=identity,
                        alpha=alpha, rank=rank, calibrated=sorted(amax), bfloat16_layers=sorted(keep),
                        rounding="GPTQ" if hessians is not None else "nearest", scale_rule=rule,
                        quantization="SVDQuant W4A4: INT4 groups of 64, low-rank branch of rank r in bfloat16, "
                        "SmoothQuant smoothing from calibrated activation maxima"
                        + ("; q, k, v of self-attention as one layer, feed forward as Nunchaku's fused GELU MLP "
                           "(unsigned INT4 hidden activation shifted by 0.171875)" if fused else ""))
        manifest.write_text(json.dumps(metadata, indent=1), encoding="utf-8")
        return {k: v for k, v in metadata.items() if k not in ("layout", "calibrated")}
    except BaseException:
        temporary.unlink(missing_ok=True)
        path.unlink(missing_ok=True)
        manifest.unlink(missing_ok=True)
        raise


def mapped_pack(blocks, path, source):
    """A block_w4a4_v1 or v2 pack mapped read-only: uint8 [blocks, bytes per block], and its layout."""
    import json
    import warnings

    import numpy as np

    from wan_dit_lab.quantization.weight_pack import source_identity
    path = Path(path)
    metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
    if metadata["format"] not in FORMATS or metadata["source"] != source_identity(source):
        raise ValueError("Weight pack has a different checkpoint or format")
    size = metadata["bytes_per_block"]
    if metadata["blocks"] != len(blocks) or path.stat().st_size != size * len(blocks):
        raise ValueError("W4A4 weight pack is incomplete")
    data = np.memmap(path, dtype=np.uint8, mode="r")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return torch.from_numpy(data).reshape(len(blocks), size), metadata["layout"]


def slot(row, like, layout):
    """A copy of the block `like` (structure only) over `row`, a uint8 row of a block_w4a4_v1 or v2 pack."""
    import copy

    def view(entry):
        dtype = getattr(torch, entry["dtype"])
        start = entry["offset"]
        return row[start:start + entry["bytes"]].view(dtype).view(entry["shape"])

    entries = {e["name"]: e for e in layout}
    module = copy.deepcopy(like).to("meta").requires_grad_(False)
    names = [e["name"].removesuffix(".qweight") for e in layout if e["name"].endswith(".qweight")]
    gelu_mlp = any(e.get("gelu_mlp") or e["name"].startswith("self_attn.qkv.") for e in layout)
    install(module, {g: {part: view(entries[f"{g}.{part}"]) for part in PARTS} for g in names}, gelu_mlp)
    for name, _ in list(module.named_parameters()):
        parent, _, attr = name.rpartition(".")
        owner = module.get_submodule(parent) if parent else module
        setattr(owner, attr, torch.nn.Parameter(view(entries[name]), requires_grad=False))
    return module


class ActStats:
    """The largest |x| of each input channel of every linear layer of every block, gathered while a model runs
    through an offloader that gives its blocks in order (get_next): the calibration of the smoothing."""

    def __init__(self, offloader, modules, blocks):
        self.amax, self.calls, self.blocks, self.block = {}, 0, blocks, 0
        get_next = offloader.get_next

        def counted():
            self.block = self.calls % self.blocks
            self.calls += 1
            return get_next()

        offloader.get_next = counted
        for module in modules:
            for name, child in module.named_modules():
                if child is not module and hasattr(child, "out_features") and not list(child.children()):
                    child.register_forward_pre_hook(lambda m, args, name=name: self.record(name, args[0]))

    @torch.no_grad()
    def record(self, name, x):
        key = f"{self.block}.{name}"
        a = x.reshape(-1, x.shape[-1]).abs().amax(dim=0).float()
        self.amax[key] = a if key not in self.amax else torch.maximum(self.amax[key], a)
