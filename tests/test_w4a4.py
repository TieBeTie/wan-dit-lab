"""w4a4: the conventions of Nunchaku's W4A4 kernels against x W^T + b (smoothing, low-rank branch, bias)."""

import pytest
import torch


@pytest.mark.gpu
def test_low_rank_weight_is_exact_and_smoothing_helps():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    w4a4 = pytest.importorskip("wan_dit_lab.quantization.w4a4")
    pytest.importorskip("wan_dit_lab.kernels.operators").load()
    torch.manual_seed(0)
    M, K, N = 512, 1024, 768
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    x[:, :8] *= 20  # outlier channels
    b = torch.randn(N, device="cuda") * 0.1

    def err(W, t):
        ref = x.float() @ W.t() + b
        return float((w4a4.linear(x, t).float() - ref).norm() / ref.norm())

    low = (torch.randn(N, 32, device="cuda") @ torch.randn(32, K, device="cuda")) * 0.01
    s = torch.rand(K, device="cuda") + 0.5
    assert err(low, w4a4.quantize(low, b, smooth=s)) < 0.01  # all of it in the 16-bit branch, whatever s is
    W = torch.randn(N, K, device="cuda") * 0.02
    plain = err(W, w4a4.quantize(W, b))
    smooth = err(W, w4a4.quantize(W, b, smooth=w4a4.smoothing(x.abs().amax(dim=0), W)))
    assert smooth < plain < 0.25


@pytest.mark.gpu
def test_gelu_mlp_is_the_feed_forward_and_needs_the_shift_in_the_bias():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    w4a4 = pytest.importorskip("wan_dit_lab.quantization.w4a4")
    pytest.importorskip("wan_dit_lab.kernels.operators").load()
    torch.manual_seed(0)
    M, K, H, N = 300, 256, 1024, 256  # M not a multiple of the padding
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    x[:, :4] *= 10
    W1, b1 = torch.randn(H, K, device="cuda") * 0.08, torch.randn(H, device="cuda") * 0.1
    W2, b2 = torch.randn(N, H, device="cuda") * 0.05, torch.randn(N, device="cuda") * 0.1
    gelu = torch.nn.GELU(approximate="tanh")
    h = gelu(x.float() @ W1.t() + b1)
    ref = h @ W2.t() + b2
    err = lambda y: float((y.float() - ref).norm() / ref.norm())
    offset = lambda y: float((y.float() - ref).mean(dim=0).norm() / ref.mean(dim=0).norm())  # systematic, per column
    s1 = w4a4.smoothing(x.abs().amax(dim=0), W1)
    t1 = w4a4.quantize(W1, b1, smooth=s1)
    s2 = w4a4.smoothing(h.abs().amax(dim=0) + w4a4.GELU_SHIFT, W2)
    unfused = w4a4.linear(gelu(w4a4.linear(x, t1)), w4a4.quantize(W2, b2, smooth=s2))
    fused = w4a4.gelu_mlp(x, t1, w4a4.quantize(W2, b2, smooth=s2, shift=w4a4.GELU_SHIFT))
    uncorrected = w4a4.gelu_mlp(x, t1, w4a4.quantize(W2, b2, smooth=s2))
    print(f"error unfused {err(unfused):.4f} fused {err(fused):.4f} without the bias correction {err(uncorrected):.4f}"
          f"; offset {offset(unfused):.4f} {offset(fused):.4f} {offset(uncorrected):.4f}")
    assert err(fused) < 0.2 and err(fused) < err(unfused)  # the hidden layer quantized unsigned: twice the levels
    # GELU over quantization noise leaves an offset in both paths; the shift, uncorrected, doubles it
    assert offset(fused) < 1.1 * offset(unfused) and offset(uncorrected) > 1.5 * offset(fused)


@pytest.mark.gpu
def test_fused_block_stays_as_close_to_bfloat16_as_a_layer_per_linear():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    w4a4 = pytest.importorskip("wan_dit_lab.quantization.w4a4")
    pytest.importorskip("wan_dit_lab.kernels.operators").load()
    import copy
    from wan_dit_lab.models.block import WanBlock
    from wan_dit_lab.models.rope import rope
    torch.manual_seed(1)
    block = WanBlock(256, 1024, 2, 1e-6).cuda().eval()
    d = 128
    freqs = torch.cat([rope(1024, d - 4 * (d // 6)), rope(1024, 2 * (d // 6)), rope(1024, 2 * (d // 6))], dim=-1).cuda()
    X = torch.randn(1, 2 * 8 * 16, 256, device="cuda")
    e = torch.randn(1, 6, 256, device="cuda") * 0.1
    context = torch.randn(1, 16, 256, device="cuda")
    kwargs = dict(grid_sizes=((2, 8, 16),), freqs=freqs, context=context)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref = block(X, e, **kwargs).float() - X
        err = {}
        for fused in (False, True):
            q = w4a4.convert(copy.deepcopy(block), fused=fused)
            assert isinstance(q.self_attn, w4a4.FusedQKVAttention) == fused
            err[fused] = float(((q(X, e, **kwargs).float() - X) - ref).norm() / ref.norm())
    print(err)
    assert err[True] < 0.2 and err[True] < 1.3 * err[False]


@pytest.mark.gpu
@pytest.mark.parametrize("keep", [("cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o"),
                                  ("self_attn.q", "self_attn.k", "self_attn.v"), ("self_attn.o",), ("ffn.0", "ffn.2")])
def test_layers_kept_in_bfloat16_through_a_pack(tmp_path, keep):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    w4a4 = pytest.importorskip("wan_dit_lab.quantization.w4a4")
    pytest.importorskip("wan_dit_lab.kernels.operators").load()
    import copy
    from wan_dit_lab.models.block import WanBlock
    from wan_dit_lab.models.rope import rope
    torch.manual_seed(1)
    block = WanBlock(256, 1024, 2, 1e-6).cuda().eval().to(torch.bfloat16)
    d = 128
    freqs = torch.cat([rope(1024, d - 4 * (d // 6)), rope(1024, 2 * (d // 6)), rope(1024, 2 * (d // 6))], dim=-1).cuda()
    X = torch.randn(1, 2 * 8 * 16, 256, device="cuda")
    e = torch.randn(1, 6, 256, device="cuda") * 0.1
    kwargs = dict(grid_sizes=((2, 8, 16),), freqs=freqs, context=torch.randn(1, 16, 256, device="cuda"))
    source = tmp_path / "source.pt"
    source.write_bytes(b"0")
    path = tmp_path / "blocks.w4a4"
    w4a4.build_pack([block], {}, path, source, fused=True, keep=keep)
    rows, layout = w4a4.mapped_pack([block], path, source)
    packed = w4a4.slot(rows[0].cuda(), block, layout)
    assert isinstance(packed.get_submodule(keep[0]), torch.nn.Linear)
    assert isinstance(packed.self_attn, w4a4.FusedQKVAttention) == ("self_attn.q" not in keep)
    assert isinstance(packed.ffn, w4a4.PackedW4A4GeluMLP) == ("ffn.0" not in keep)
    all_w4a4 = w4a4.convert(copy.deepcopy(block), fused=True)  # outside autocast, as build_pack quantizes
    kept = w4a4.convert(copy.deepcopy(block), fused=True, keep=keep)
    # Serialization must preserve every tensor exactly. Nunchaku's fused MLP
    # reduces its low-rank activations with floating-point atomics, so comparing
    # separately allocated executions bitwise tests their reduction order too.
    expected_modules = dict(kept.named_modules())
    assert dict(packed.named_modules()).keys() == expected_modules.keys()
    for name, module in packed.named_modules():
        expected = expected_modules[name]
        assert type(module) is type(expected), name
        if hasattr(module, "parts"):
            assert len(module.parts) == len(expected.parts)
            for actual_part, expected_part in zip(module.parts, expected.parts):
                torch.testing.assert_close(actual_part, expected_part, rtol=0, atol=0)
    expected_parameters = dict(kept.named_parameters())
    assert dict(packed.named_parameters()).keys() == expected_parameters.keys()
    for name, parameter in packed.named_parameters():
        torch.testing.assert_close(parameter, expected_parameters[name], rtol=0, atol=0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref = block(X, e, **kwargs).float() - X
        all_w4a4 = all_w4a4(X, e, **kwargs).float() - X
        out = kept(X, e, **kwargs).float() - X
        packed_out = packed(X, e, **kwargs).float() - X
    assert (out - ref).norm() < 1.01 * (all_w4a4 - ref).norm()  # one small layer may change little on random weights
    assert (packed_out - ref).norm() < 1.01 * (all_w4a4 - ref).norm()


def test_gptq_keeps_the_layer_output_closer_than_rounding_to_the_nearest():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    w4a4 = pytest.importorskip("wan_dit_lab.quantization.w4a4")
    torch.manual_seed(0)
    K, N = 256, 192
    mix = torch.randn(K, K, device="cuda") / K ** 0.5
    X = torch.randn(4096, K, device="cuda") @ mix + 0.3 * torch.randn(4096, K, device="cuda")  # correlated inputs
    R = torch.randn(N, K, device="cuda")
    groups = R.view(N, K // w4a4.GROUP, w4a4.GROUP)
    near_scale = groups.abs().amax(-1) / 7
    near = (torch.round(groups / near_scale[..., None]).clamp(-8, 7) * near_scale[..., None]).view(N, K)
    q, scale = w4a4.gptq(R, X.T @ X)
    assert q.min() >= -8 and q.max() <= 7 and scale.shape == (N, K // w4a4.GROUP)
    fitted = (q.view(N, K // w4a4.GROUP, w4a4.GROUP).float() * scale[..., None]).view(N, K)
    err = lambda W: float((X @ (R - W).T).norm())
    assert err(fitted) < 0.8 * err(near)


def test_mse_scales_fit_heavy_tailed_weights_better_than_the_largest_value():
    # LongLive 2.0 was distilled with NVFP4 whose scales are chosen by MSE: for INT4 groups of 64 the same rule
    # gives up the largest value of a group for the others, and reconstructs the group better on the whole
    w4a4 = pytest.importorskip("wan_dit_lab.quantization.w4a4")
    torch.manual_seed(0)
    R = torch.distributions.StudentT(3.0).sample((128, 512))  # heavy tails, as the residuals of real layers
    q_max, s_max = w4a4.int4_groups(R, rule="max")
    q_mse, s_mse = w4a4.int4_groups(R, rule="mse")
    back = lambda q, s: (q.view(128, -1, w4a4.GROUP).float() * s[..., None]).view(128, 512)
    assert q_mse.min() >= -8 and q_mse.max() <= 7
    err = lambda q, s: float(((R - back(q, s)) ** 2).sum())
    assert err(q_mse, s_mse) < 0.9 * err(q_max, s_max)
    assert torch.equal(q_max, torch.round(R.view(128, -1, w4a4.GROUP) / s_max[..., None]).clamp(-8, 7)
                       .to(torch.int32).view(128, 512))  # "max" is the rounding the packs have used so far


@pytest.mark.gpu
def test_a_layer_with_mse_scales_stays_close_to_bfloat16():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    w4a4 = pytest.importorskip("wan_dit_lab.quantization.w4a4")
    pytest.importorskip("wan_dit_lab.kernels.operators").load()
    torch.manual_seed(0)
    W = torch.distributions.StudentT(3.0).sample((512, 1024)).cuda() * 0.02
    x = torch.randn(256, 1024, device="cuda", dtype=torch.bfloat16)
    ref = x.float() @ W.T
    err = {rule: float((w4a4.linear(x, w4a4.quantize(W, rule=rule)).float() - ref).norm() / ref.norm())
           for rule in ("max", "mse")}
    assert err["mse"] < err["max"] < 0.2, err
