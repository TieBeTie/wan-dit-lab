"""sage_kv: Sage's quantization read straight from the FP8 cache gives Sage's layouts and Sage's attention."""

import pytest
import torch


def setup(at=200, n=150, nq=150, H=3, D=128, seed=0):
    """A cache of `at` tokens in FP8 with scale 1 (so its values are exact in bfloat16) and a chunk of n tokens."""
    torch.manual_seed(seed)
    dev = "cuda"
    total = at + n + 64  # the buffer is longer than what is used, as in KVCache
    k = (torch.randn(1, total, H, D, device=dev) + 0.5).to(torch.float8_e4m3fn)
    v = (torch.randn(1, total, H, D, device=dev) * 0.7).to(torch.float8_e4m3fn)
    ks = torch.ones(1, total, H, 1, device=dev)
    vs = torch.ones(1, total, H, 1, device=dev)
    K = (torch.randn(1, n, H, D, device=dev) + 0.5).to(torch.bfloat16)
    V = torch.randn(1, n, H, D, device=dev).to(torch.bfloat16)
    Q = torch.randn(1, nq, H, D, device=dev).to(torch.bfloat16)
    K_full = torch.cat([k[:, :at].to(torch.bfloat16), K], dim=1)
    V_full = torch.cat([v[:, :at].to(torch.bfloat16), V], dim=1)
    return Q, k, ks, v, vs, at, K, V, K_full, V_full


@pytest.fixture
def sage():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("sageattention")
    from wan_dit_lab.kernels import sage_kv
    return sage_kv


@pytest.mark.gpu
def test_values_come_out_in_sage_layout(sage):
    from sageattention.quant import per_channel_fp8
    Q, k, ks, v, vs, at, K, V, K_full, V_full = setup()
    expected, expected_scale, _ = per_channel_fp8(V_full, tensor_layout="NHD", scale_max=sage.V_MAX, smooth_v=False)
    captured = {}
    original = sage.torch.empty

    def keep(*shape, **kw):
        t = original(*shape, **kw)
        if kw.get("dtype") == torch.float8_e4m3fn:
            captured["v"] = t
        return t

    sage.torch.empty = keep
    try:
        sage.attend(Q, k, ks, v, vs, at, K, V)
    finally:
        sage.torch.empty = original
    v_fp8 = captured["v"]
    assert v_fp8.shape == expected.shape
    a, b = v_fp8.view(torch.uint8), expected.view(torch.uint8)
    assert (a == b).float().mean() > 0.999, (a != b).nonzero()[:8]


@pytest.mark.gpu
def test_keys_come_out_as_sage_quantizes_them(sage):
    from sageattention.triton.quant_per_thread import per_thread_int8
    Q, k, ks, v, vs, at, K, V, K_full, V_full = setup()
    km = K_full.float().mean(dim=1, keepdim=True)
    _, _, k_ref, k_scale_ref = per_thread_int8(Q, (K_full.float() - km).to(torch.bfloat16), None, tensor_layout="NHD")
    captured = {}
    original = sage.torch.empty

    def keep(*shape, **kw):
        t = original(*shape, **kw)
        if kw.get("dtype") == torch.int8 and "k" not in captured:
            captured["k"] = t
        elif kw.get("dtype") == torch.float32 and "k_scale" not in captured and "k" in captured:
            captured["k_scale"] = t
        return t

    sage.torch.empty = keep
    try:
        sage.attend(Q, k, ks, v, vs, at, K, V)
    finally:
        sage.torch.empty = original
    assert captured["k"].shape == k_ref.shape and captured["k_scale"].shape == k_scale_ref.shape
    assert (captured["k"].int() - k_ref.int()).abs().max() <= 1  # K - mean rounded to bfloat16 for the reference
    assert torch.allclose(captured["k_scale"], k_scale_ref, rtol=2e-2)


@pytest.mark.gpu
@pytest.mark.parametrize("at, n", [(200, 150), (0, 7040), (21120, 7040), (3120 * 3, 3120)])
def test_attention_is_sageattn_over_the_unpacked_cache(sage, at, n):
    from sageattention import sageattn
    Q, k, ks, v, vs, at, K, V, K_full, V_full = setup(at=at, n=n, nq=n, H=2)
    exact = torch.nn.functional.scaled_dot_product_attention(
        Q.transpose(1, 2).float(), K_full.transpose(1, 2).float(), V_full.transpose(1, 2).float()).transpose(1, 2)
    reference = sageattn(Q, K_full, V_full, tensor_layout="NHD", is_causal=False)
    mine = sage.attend(Q, k, ks, v, vs, at, K, V)
    err = lambda x: float((x.float() - exact).norm() / exact.norm())
    print(f"at {at} n {n}: sageattn {err(reference):.4f}, from the FP8 cache {err(mine):.4f}, "
          f"between {float((mine.float() - reference.float()).norm() / reference.float().norm()):.4f}")
    assert err(mine) < 1.2 * err(reference) + 1e-3


@pytest.mark.gpu
def test_kv_cache_attends_through_sage_kv_as_through_the_unpacked_copy(sage):
    from wan_dit_lab.generation.causal import KVCache
    torch.manual_seed(3)
    per, chunk, H, D = 39, 2, 2, 128  # tokens of a frame not a multiple of 64
    cache = KVCache(4 * chunk * per, chunk * per, sage=True, fp8=True, chunk=chunk)
    for step in range(5):  # filling, then full with eviction
        n = chunk * per
        Q = torch.randn(1, n, H, D, device="cuda")
        K = torch.randn(1, n, H, D, device="cuda") + 0.3
        V = torch.randn(1, n, H, D, device="cuda", dtype=torch.bfloat16)
        at = cache.saved
        new = cache.attend(Q, K, V)
        old = cache.attend_unpacked(Q, K, V, at, n, False, False)
        diff = float((new.float() - old.float()).norm() / old.float().norm())
        assert new.shape == old.shape and diff < 0.06, (step, diff)
        cache.commit()
    assert cache.full()


@pytest.mark.gpu
def test_chosen_heads_and_tokens_read_from_the_whole_cache(sage):
    # heads and tokens maps give what the same call gives over a cache that holds only those heads and tokens
    Q, k, ks, v, vs, at, K, V, K_full, V_full = setup(H=4)
    heads = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    tokens = torch.tensor(list(range(40)) + list(range(120, 184)), device="cuda", dtype=torch.int32)
    picked = lambda x: x[:, tokens.long()][:, :, heads.long()].contiguous()
    expected = sage.attend(Q[:, :, heads.long()].contiguous(), picked(k), picked(ks), picked(v), picked(vs),
                           len(tokens), K[:, :, heads.long()].contiguous(), V[:, :, heads.long()].contiguous())
    out = sage.attend(Q[:, :, heads.long()].contiguous(), k, ks, v, vs, len(tokens), K[:, :, heads.long()].contiguous(),
                      V[:, :, heads.long()].contiguous(), heads=heads, tokens=tokens)
    assert torch.equal(out, expected)
