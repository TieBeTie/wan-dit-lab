"""Contract first: CUDA tests for the optional fused RoPE, without model weights."""

import pytest
import torch

pytest.importorskip("triton")
from wan_dit_lab.models.rope import rope
from wan_dit_lab.kernels.rope_triton import rope_apply, rope_apply_wan

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required")]


def frequencies(dim=128, positions=64):
    pairs = dim // 2
    spatial = pairs // 3
    return torch.cat([rope(positions, 2 * n) for n in
                      (pairs - 2 * spatial, spatial, spatial)], dim=-1).cuda()


def oracle(x, grids, freq, start=0, *, complex_multiply=False):
    """FP64 oracle. Complex multiplication is a separate rounding cross-check."""
    pairs = x.shape[-1] // 2
    spatial = pairs // 3
    time, height, width = torch.complex(*freq).split(
        (pairs - 2 * spatial, spatial, spatial), dim=-1)
    outputs = []
    for sample, (f, h, w) in zip(x, grids):
        length = f * h * w
        phase = torch.cat((
            time[start:start + f, None, None].expand(f, h, w, -1),
            height[None, :h, None].expand(f, h, w, -1),
            width[None, None, :w].expand(f, h, w, -1)), dim=-1)
        values = sample[:length].double().reshape(length, x.shape[2], pairs, 2)
        phase = phase.reshape(length, 1, pairs)
        if complex_multiply:
            rotated = torch.view_as_real(
                torch.view_as_complex(values.contiguous()) * phase).flatten(-2).to(x.dtype)
        else:
            even, odd = values[..., 0], values[..., 1]
            rotated = torch.stack((even * phase.real - odd * phase.imag,
                                   odd * phase.real + even * phase.imag), dim=-1)
            rotated = rotated.flatten(-2).to(x.dtype)
        outputs.append(torch.cat((rotated, sample[length:]), dim=0))
    return torch.stack(outputs)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_forward_backward_same_shape_batches(dtype, batch):
    torch.manual_seed(12)
    grids = ((2, 3, 5),) * batch
    x = torch.randn(batch, 30, 3, 128, device="cuda", dtype=dtype, requires_grad=True)
    freq = frequencies()
    actual, expected = rope_apply(x, grids, freq, 7), oracle(x, grids, freq, 7)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # A strided upstream gradient is a legitimate autograd input.
    probe = torch.randn(batch, 30, 3, 256, device="cuda", dtype=dtype)[..., ::2]
    grad = torch.autograd.grad(actual, x, probe)[0]
    grad_ref = torch.autograd.grad(expected, x, probe)[0]
    torch.testing.assert_close(grad, grad_ref, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_strided_input_heterogeneous_grids_padding_and_no_mutation(dtype):
    grids = ((2, 3, 5), (1, 2, 4))
    x = torch.randn(2, 33, 2, 256, device="cuda", dtype=dtype)[..., ::2]
    before = x.clone()
    actual = rope_apply(x, torch.tensor(grids), frequencies(), 3)
    torch.testing.assert_close(actual, oracle(x, grids, frequencies(), 3), rtol=0, atol=0)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    for b, (f, h, w) in enumerate(grids):
        torch.testing.assert_close(actual[b, f*h*w:], x[b, f*h*w:], rtol=0, atol=0)
    assert actual.dtype == x.dtype and actual.shape == x.shape
    assert actual.data_ptr() != x.data_ptr()


@pytest.mark.parametrize("dim", [2, 6, 12, 64, 128])
def test_dimension_splits_tail_and_noncontiguous_frequencies(dim):
    freq = frequencies(dim)
    storage = torch.zeros(2, 64, dim, device="cuda", dtype=torch.float64)
    storage[..., ::2] = freq
    freq = storage[..., ::2]
    x = torch.randn(1, 15, 3, dim, device="cuda")
    torch.testing.assert_close(rope_apply(x, ((1, 3, 5),), freq, 11),
                               oracle(x, ((1, 3, 5),), freq, 11), rtol=0, atol=0)


def test_rotation_norm_relative_position_and_chunk_offset():
    freq = frequencies(12)
    x = torch.randn(1, 12, 2, 12, device="cuda", dtype=torch.float64)
    grids = ((3, 2, 2),)
    rotated = rope_apply(x, grids, freq, 5)
    torch.testing.assert_close(rotated.square().sum(-1), x.square().sum(-1),
                               rtol=2e-14, atol=2e-14)
    chunks = torch.cat([rope_apply(x[:, i*4:(i+1)*4], ((1, 2, 2),), freq, 5+i)
                        for i in range(3)], dim=1)
    torch.testing.assert_close(rotated, chunks, rtol=0, atol=0)
    q = torch.randn_like(x)
    def dot(a, b):
        return (a * b).sum(-1)
    before = dot(rope_apply(x, grids, freq, 2), rope_apply(q, grids, freq, 7))
    after = dot(rope_apply(x, grids, freq, 9), rope_apply(q, grids, freq, 14))
    torch.testing.assert_close(before, after, rtol=2e-13, atol=2e-13)


def test_gradcheck_and_gradgradcheck():
    x = torch.randn(1, 2, 1, 6, device="cuda", dtype=torch.float64, requires_grad=True)
    def fn(value):
        return rope_apply(value, ((1, 1, 2),), frequencies(6), 2)
    assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-7, rtol=1e-5)
    assert torch.autograd.gradgradcheck(fn, (x,), eps=1e-6, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_complex_oracle_rounding_crosscheck(dtype):
    torch.manual_seed(12)
    x = torch.randn(4, 30, 3, 128, device="cuda", dtype=dtype, requires_grad=True)
    grids, freq = ((2, 3, 5),) * 4, frequencies()
    actual = rope_apply(x, grids, freq, 7)
    expected = oracle(x, grids, freq, 7, complex_multiply=True)
    # Complex CUDA multiply can fuse a product and sum. Require at most one
    # representable step for low precision, roundoff-scale error for FP64.
    tolerance = dict(rtol=2e-13, atol=1e-15) if dtype == torch.float64 else dict(
        rtol=torch.finfo(dtype).eps, atol=torch.finfo(dtype).tiny)
    torch.testing.assert_close(actual, expected, **tolerance)
    probe = torch.randn_like(x)
    torch.testing.assert_close(torch.autograd.grad(actual, x, probe)[0],
                               torch.autograd.grad(expected, x, probe)[0], **tolerance)


def test_compile_fullgraph_and_compiled_backward():
    compiled = torch.compile(rope_apply, fullgraph=True, dynamic=False)
    for batch, grids in [(1, ((2, 3, 5),)), (2, ((1, 3, 5), (2, 3, 5)))]:
        x = torch.randn(batch, 30, 2, 12, device="cuda", requires_grad=True)
        freq = frequencies(12)
        actual, expected = compiled(x, grids, freq, 3), oracle(x, grids, freq, 3)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        probe = torch.randn_like(actual)
        torch.testing.assert_close(torch.autograd.grad(actual, x, probe)[0],
                                   torch.autograd.grad(expected, x, probe)[0], rtol=0, atol=0)


def test_upstream_complex_frequency_fp32_output_contract():
    freq = frequencies(12)
    complex_freq = torch.complex(*freq)
    x = torch.randn(2, 33, 2, 12, device="cuda", dtype=torch.bfloat16)
    grids = ((2, 3, 5), (1, 2, 4))
    actual = rope_apply_wan(x, grids, complex_freq)
    expected = oracle(x.float(), grids, freq)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_operator_registration_and_empty_tensor():
    from wan_dit_lab.kernels.rope_triton import _rotate
    x = torch.randn(1, 6, 1, 12, device="cuda", requires_grad=True)
    torch.library.opcheck(_rotate, (x, frequencies(12), "1,2,3", 0, False, x.dtype))
    empty = torch.empty(1, 0, 2, 12, device="cuda")
    assert rope_apply(empty, ((0, 2, 3),), frequencies(12)).shape == empty.shape


def test_current_stream_autocast_and_actual_single_kernel():
    from wan_dit_lab.harness.profiler import profile_kernels

    x = torch.randn(1, 30, 2, 128, device="cuda", dtype=torch.bfloat16)
    freq, grids = frequencies(), ((2, 3, 5),)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.autocast("cuda", dtype=torch.bfloat16):
        actual = rope_apply(x, grids, freq, 3)
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(actual, oracle(x, grids, freq, 3), rtol=0, atol=0)
    compiled = torch.compile(lambda: rope_apply(x, grids, freq, 3), fullgraph=True, dynamic=False)
    compiled()
    assert profile_kernels(compiled)["count"] == 1


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_live_fp32_tables_preserve_original_arithmetic(dtype):
    from wan_dit_lab.models.rope import rope_apply as original

    torch.manual_seed(19)
    freq = frequencies().float()
    grids = ((2, 3, 5),)
    x = torch.randn(1, 30, 3, 128, device="cuda", dtype=dtype)
    actual = rope_apply(x, grids, freq, 9)
    expected = original(x, grids, freq, 9)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_complete_block_integration_does_not_patch_original_module():
    import copy

    import wan_dit_lab.models.multi_head_attention as original_attention
    from wan_dit_lab.models.rope import rope_apply as original_apply
    from wan_dit_lab.harness.checks import block_accuracy_gate
    from wan_dit_lab.experiments.rope_validation import make_block_pair

    torch.manual_seed(8)
    with torch.inference_mode():
        blocks = make_block_pair(128, 256, 2, torch.bfloat16)
        x = torch.randn(1, 30, 128, device="cuda", dtype=torch.bfloat16)
        time = torch.randn(1, 6, 128, device="cuda", dtype=torch.bfloat16)
        context = torch.randn(1, 5, 128, device="cuda", dtype=torch.bfloat16)
        freq = frequencies(64)
        outputs = {}
        for label, block in blocks.items():
            fn = torch.compile(block, fullgraph=True, dynamic=False)
            outputs[label] = fn(x, time, grid_sizes=((2, 3, 5),), freqs=freq,
                                context=context, start_frame=7)
        eager = [block(x, time, grid_sizes=((2, 3, 5),), freqs=freq,
                       context=context, start_frame=7) for block in blocks.values()]
        torch.testing.assert_close(eager[0], eager[1], rtol=0, atol=0)
        float_block = copy.deepcopy(blocks["original"]).float()
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            reference = float_block(x.float(), time.float(), grid_sizes=((2, 3, 5),),
                                    freqs=freq, context=context.float(), start_frame=7)
        gate = block_accuracy_gate(outputs["triton"], outputs["original"], reference, .05)
        assert gate["passed"], gate
    assert original_attention.rope_apply is original_apply


@pytest.mark.parametrize("problem", ["odd_dim", "grid_count", "grid_length", "negative_start",
                                    "frequency_bounds", "frequency_grad"])
def test_invalid_contract_is_rejected(problem):
    x = torch.randn(1, 6, 1, 12, device="cuda")
    grids, start, freq = ((1, 2, 3),), 0, frequencies(12)
    if problem == "odd_dim":
        x = x[..., :11]
    elif problem == "grid_count":
        grids = grids * 2
    elif problem == "grid_length":
        grids = ((2, 2, 3),)
    elif problem == "negative_start":
        start = -1
    elif problem == "frequency_bounds":
        start = 64
    else:
        freq.requires_grad_()
    with pytest.raises((ValueError, TypeError)):
        rope_apply(x, grids, freq, start)
