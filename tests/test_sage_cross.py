"""Text K/V are prepared once; cross-attention only prepares each new query."""
import pytest
import torch


def test_prepared_text_is_reused_and_cross_attention_has_no_rope(monkeypatch):
    from wan_dit_lab.kernels import sage_cross
    from wan_dit_lab.models.multi_head_attention import MultiHeadAttention
    attention = MultiHeadAttention(12, 2)
    attention.sage_cross_tokens = 7
    calls = []
    key, value = torch.empty(1, dtype=torch.uint8), torch.empty(2, dtype=torch.uint8)
    def prepare(k, v, weight, eps):
        calls.append(("kv", k.shape))
        return key, value
    def attend(q, weight, eps, k, v, tokens):
        assert k is key and v is value and tokens == 7
        calls.append(("q", q.shape))
        return q
    monkeypatch.setattr(sage_cross, "prepare_kv", prepare)
    monkeypatch.setattr(sage_cross, "attend", attend)
    def forbidden(*args):
        pytest.fail("Sage cross owns query/key normalization; no separate RoPE")
    attention.norm_q.forward = forbidden
    attention.norm_k.forward = forbidden
    attention.rope_kernel = forbidden
    context = attention.keys_values(torch.randn(1, 7, 12))
    for _ in range(3):
        assert attention(torch.randn(1, 5, 12), context_kv=context).shape == (1, 5, 12)
    assert calls == [("kv", torch.Size([1, 7, 2, 6]))] + [("q", torch.Size([1, 5, 2, 6]))] * 3


@pytest.mark.gpu
@pytest.mark.parametrize("batch,nk,nq,heads", [(1, 79, 67, 3), (2, 128, 79, 2), (1, 512, 3712, 24)])
def test_sage_cross_accuracy_and_opaque_buffers(batch, nk, nq, heads):
    from wan_dit_lab.kernels import sage_cross
    from wan_dit_lab.models.rms_norm import RMSNorm
    torch.manual_seed(37)
    q = torch.randn(batch, nq, heads, 128, device="cuda", dtype=torch.bfloat16)
    k = (torch.randn(batch, nk, heads, 128, device="cuda") + .5).bfloat16()
    v = torch.randn_like(k)
    wq, wk = [torch.randn(heads * 128, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    def norm(x, w):
        x = x.float().flatten(-2)
        return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * w.float()).view(*x.shape[:2], heads, 128)
    packed = sage_cross.prepare_kv(k, v, wk, 1e-6)
    assert len(packed) == 2 and all(t.dtype == torch.uint8 and t.is_contiguous() for t in packed)
    previous = [t.clone() for t in packed]
    actual = sage_cross.attend(q, wq, 1e-6, *packed, nk)
    qr, kr = norm(q, wq), norm(k, wk)
    reference = torch.nn.functional.scaled_dot_product_attention(qr.transpose(1, 2), kr.transpose(1, 2),
                                                                 v.float().transpose(1, 2)).transpose(1, 2)
    error = (actual.float() - reference).norm() / reference.norm()
    from sageattention import sageattn_qk_int8_pv_fp8_cuda
    official = sageattn_qk_int8_pv_fp8_cuda(qr.bfloat16(), kr.bfloat16(), v, tensor_layout="NHD",
                                           qk_quant_gran="per_thread", pv_accum_dtype="fp32+fp16")
    official_error = (official.float() - reference).norm() / reference.norm()
    assert error <= official_error * 1.05 + 1e-7, (float(error), float(official_error))
    assert all(torch.equal(a, b) for a, b in zip(packed, previous)), "query must not rewrite text K/V"
    restored_k, restored_v = sage_cross.dequantize(*packed, batch, nk, heads, 128)
    centered = kr - kr.mean(1, keepdim=True)
    # Verify the actual per-thread quantizer, rather than assuming an arbitrary
    # percentage for the inherent INT8 error on random outlier-heavy weights.
    tiles = (nk + 63) // 64
    grouped = torch.nn.functional.pad(centered, (0, 0, 0, 0, 0, tiles*64-nk)).view(batch, tiles, 8, 4, 2, heads, 128)
    scale = grouped.abs().amax((2, 4, 6), keepdim=True) / 127 + 1e-7
    y = grouped / scale
    expected = (y + torch.where(y >= 0, .5, -.5)).trunc().to(torch.int8).view(batch, tiles*64, heads, 128)[:, :nk]
    k8, _, ks, _ = sage_cross.unpack(*packed, batch, nk, heads, 128)
    torch.testing.assert_close(ks, scale[:, :, 0, :, 0, :, 0].permute(0, 3, 1, 2).reshape(batch, heads, tiles*4),
                               rtol=3e-6, atol=1e-8)
    assert (k8.int() - expected.int()).abs().max() <= 1
    assert (k8 == expected).float().mean() > .999
    assert (restored_v - v.float()).norm() / v.float().norm() < .035


@pytest.mark.gpu
def test_sage_cross_aot_contains_preparation_and_reuses_opaque_text_buffers(tmp_path):
    import zipfile
    from wan_dit_lab.kernels import sage_cross, operators
    operators.load()
    class Query(torch.nn.Module):
        def forward(self, q, weight, key, value):
            return sage_cross.attend(q, weight, 1e-6, key, value, 128)
    q = torch.randn(1, 79, 2, 128, device="cuda", dtype=torch.bfloat16)
    k, v = [torch.randn(1, 128, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    weight = torch.ones(256, device="cuda", dtype=torch.bfloat16)
    packed = sage_cross.prepare_kv(k, v, weight, 1e-6)
    inputs = (q, weight, *packed)
    with torch.no_grad():
        eager = Query()(*inputs)
        program = torch.export.export(Query(), inputs)
        path = torch._inductor.aoti_compile_and_package(program, package_path=str(tmp_path / "query.pt2"))
        loaded = torch._inductor.aoti_load_package(path)
        torch.testing.assert_close(loaded(*inputs), eager, rtol=0, atol=0)
    with zipfile.ZipFile(path) as archive:
        source = b"\n".join(archive.read(name) for name in archive.namelist() if name.endswith(".cpp"))
    assert b"_cross_query_int8" in source
    assert b"_cross_key_int8" not in source, "text preparation belongs to the prompt package"


@pytest.mark.gpu
def test_prompt_preparation_is_aot_code_and_preserves_both_packed_buffers(tmp_path):
    from wan_dit_lab.kernels import sage_cross
    class Text(torch.nn.Module):
        def forward(self, k, v, weight):
            return sage_cross.prepare_kv(k, v, weight, 1e-6)
    k, v = [torch.randn(1, 512, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    weight = torch.ones(256, device="cuda", dtype=torch.bfloat16)
    inputs = (k, v, weight)
    with torch.no_grad():
        expected = Text()(*inputs)
        exported = torch.export.export(Text(), inputs)
        path = torch._inductor.aoti_compile_and_package(exported, package_path=str(tmp_path / "prompt.pt2"))
        compiled = torch._inductor.aoti_load_package(path)
        assert all(torch.equal(a, b) for a, b in zip(compiled(*inputs), expected))


@pytest.mark.gpu
def test_a_constant_text_value_is_preserved_despite_quantized_attention_weights():
    from wan_dit_lab.kernels import sage_cross
    torch.manual_seed(91)
    q = torch.randn(1, 79, 2, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 128, 2, 128, device="cuda", dtype=torch.bfloat16)
    constant = torch.randn(1, 1, 2, 128, device="cuda", dtype=torch.bfloat16) * 3
    v = constant.expand_as(k).contiguous()
    weight = torch.ones(256, device="cuda", dtype=torch.bfloat16)
    prepared = sage_cross.prepare_kv(k, v, weight, 1e-6)
    output = sage_cross.attend(q, weight, 1e-6, *prepared, 128)
    torch.testing.assert_close(output, constant.expand_as(output), rtol=0, atol=0)
