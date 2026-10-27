"""TDD contract for fused FP8 quantization directly into persistent cache slices."""

import pytest
import torch

pytest.importorskip("triton")
from wan_dit_lab.kernels.cache_fp8 import store_kv_fp8
from wan_dit_lab.generation.causal import KVCache, quantize_rows

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def buffers(batch, tokens, heads, dim):
    shape = (batch, tokens, heads, dim)
    return (torch.full(shape, 2., device="cuda").to(torch.float8_e4m3fn),
            torch.full((batch, tokens, heads, 1), 7., device="cuda"),
            torch.full(shape, -2., device="cuda").to(torch.float8_e4m3fn),
            torch.full((batch, tokens, heads, 1), 9., device="cuda"))


def compare_bits(actual, expected):
    if actual.dtype == torch.float8_e4m3fn:
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    else:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("batch,heads,dim", [(1, 2, 64), (2, 3, 128), (1, 1, 160)])
def test_bitwise_values_scales_and_untouched_cache(dtype, batch, heads, dim):
    torch.manual_seed(23)
    k = torch.randn(batch, 7, heads, dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    actual = buffers(batch, 17, heads, dim)
    expected = tuple(t.clone() for t in actual)
    for x, cache, scales in ((k, expected[0], expected[1]), (v, expected[2], expected[3])):
        cache[:, 5:12], scales[:, 5:12] = quantize_rows(x)
    before_k, before_v = k.clone(), v.clone()
    store_kv_fp8(k, v, *actual, 5)
    for left, right in zip(actual, expected):
        compare_bits(left, right)
    compare_bits(k, before_k)
    compare_bits(v, before_v)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_zero_tiny_large_and_fp8_rounding_boundaries(dtype):
    values = torch.tensor([0., -0., 1e-15, -1e-15, .5, -.5, 1., -1., 3., -3., 448., -448.],
                          device="cuda").to(dtype)
    k = values.repeat(8).reshape(1, 1, 1, 96)
    v = torch.zeros_like(k)
    actual = buffers(1, 3, 1, 96)
    store_kv_fp8(k, v, *actual, 1)
    for x, cache, scales in ((k, actual[0], actual[1]), (v, actual[2], actual[3])):
        q, scale = quantize_rows(x)
        compare_bits(cache[:, 1:2], q)
        compare_bits(scales[:, 1:2], scale)


def test_strided_inputs_mixed_dtypes_and_nondefault_stream():
    k = torch.randn(2, 5, 3, 256, device="cuda", dtype=torch.float32)[..., ::2]
    v = torch.randn(2, 10, 3, 128, device="cuda", dtype=torch.bfloat16)[:, ::2]
    actual = buffers(2, 11, 3, 128)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        store_kv_fp8(k, v, *actual, 6)
    torch.cuda.current_stream().wait_stream(stream)
    for x, cache, scales in ((k, actual[0], actual[1]), (v, actual[2], actual[3])):
        q, scale = quantize_rows(x)
        compare_bits(cache[:, 6:], q)
        compare_bits(scales[:, 6:], scale)


def test_compilation_capture_and_exactly_one_kernel():
    from wan_dit_lab.harness.profiler import profile_kernels

    k = torch.randn(1, 7, 2, 128, device="cuda", dtype=torch.bfloat16)
    v, actual = torch.randn_like(k), buffers(1, 12, 2, 128)
    def run():
        store_kv_fp8(k, v, *actual, 4)
    compiled = torch.compile(run, fullgraph=True, dynamic=False)
    compiled()
    assert profile_kernels(compiled)["count"] == 1
    for _ in range(3):
        compiled()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    graph.replay()
    for x, cache, scales in ((k, actual[0], actual[1]), (v, actual[2], actual[3])):
        q, scale = quantize_rows(x)
        compare_bits(cache[:, 4:11], q)
        compare_bits(scales[:, 4:11], scale)


def test_cache_integration_and_repeated_offsets():
    k = torch.randn(1, 6, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    baseline, candidate = KVCache(12, fp8=True), KVCache(12, fp8=True)
    baseline.allocate(v)
    candidate.allocate(v)
    for offset in (0, 6, 12, 6):
        baseline.store(offset, k, v)
        store_kv_fp8(k, v, candidate.k, candidate.ks, candidate.v, candidate.vs, offset)
        for name in ("k", "ks", "v", "vs"):
            compare_bits(getattr(candidate, name)[:, offset:offset+6],
                         getattr(baseline, name)[:, offset:offset+6])


@pytest.mark.parametrize("problem", ["negative", "overflow", "shape", "cache_dtype", "scale_dtype", "grad"])
def test_invalid_contract_rejected_before_write(problem):
    k = torch.randn(1, 5, 2, 128, device="cuda")
    v, actual, offset = k.clone(), list(buffers(1, 8, 2, 128)), 1
    if problem == "negative":
        offset = -1
    elif problem == "overflow":
        offset = 4
    elif problem == "shape":
        v = v[:, :3]
    elif problem == "cache_dtype":
        actual[0] = actual[0].float()
    elif problem == "scale_dtype":
        actual[1] = actual[1].half()
    else:
        k.requires_grad_()
    with pytest.raises((ValueError, TypeError)):
        store_kv_fp8(k, v, *actual, offset)


def test_optional_install_preserves_non_fp8_path_and_can_be_restored():
    from wan_dit_lab.kernels.cache_fp8 import install_cache_store

    original = KVCache.store
    try:
        assert install_cache_store() is original
        k = torch.randn(1, 6, 2, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        for fp8 in (False, True):
            cache = KVCache(12, fp8=fp8)
            cache.allocate(v)
            cache.store(3, k, v)
            expected = quantize_rows(k)[0] if fp8 else k
            compare_bits(cache.k[:, 3:9], expected)
    finally:
        KVCache.store = original


def test_fp32_values_around_fp8_midpoints_do_not_round_through_fp16():
    # 84.021 -> 84 in FP16 -> 80 in FP8, but direct FP32 -> FP8 is 88.
    k = torch.tensor([448., 84.021, -84.021, 1.06251, -1.06251, 1.06249, -1.06249, 0.],
                     device="cuda").reshape(1, 1, 1, 8)
    actual = buffers(1, 3, 1, 8)
    store_kv_fp8(k, k, *actual, 1)
    q, scale = quantize_rows(k)
    compare_bits(actual[0][:, 1:2], q)
    compare_bits(actual[1][:, 1:2], scale)
