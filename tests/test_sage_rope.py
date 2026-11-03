"""The fused preparation must preserve Sage's scales, positions and FP8 cache contract."""
import pytest
import torch


@pytest.mark.gpu
def test_fused_rotation_preserves_the_live_fma_rounding_order():
    import triton
    import triton.language as tl
    from wan_dit_lab.kernels.sage_rope import _rotate_raw
    @triton.jit
    def probe(X, R, W, F, OUTPUT):
        t, d = tl.arange(0, 4), tl.arange(0, 128)
        x = _rotate_raw(X,R,W,F,0,0,t,t<4,d,4,128,(512,128,128,1),(256,64,1),1,4,0,False)
        tl.store(OUTPUT+t[:,None]*128+d[None,:],x)
    torch.manual_seed(67)
    x=torch.randn(1,4,1,128,device="cuda",dtype=torch.bfloat16)
    weight=torch.randn(128,device="cuda",dtype=torch.bfloat16)
    r=torch.ones(4,device="cuda")
    freq=torch.randn(2,4,64,device="cuda")
    actual=torch.empty(x.shape,device="cuda")
    probe[(1,)](x,r,weight,freq,actual)
    pairs=torch.arange(64,device="cuda")
    positions=torch.where(pairs[None,:]<43,0,torch.arange(4,device="cuda")[:,None])
    # Temporal pairs and the height axis use row 0, the width axis uses token.
    c,s=freq[0,positions,pairs],freq[1,positions,pairs]
    e=x[0,:,0,::2].float()*weight[::2].float()
    o=x[0,:,0,1::2].float()*weight[1::2].float()
    # FP64 here emulates correctly rounded FMA on bounded FP32 operands.
    expected_e=(e.double()*c.double()-(o*s).double()).float()
    expected_o=(o.double()*c.double()+(e*s).double()).float()
    expected=torch.stack((expected_e,expected_o),-1).view_as(actual)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("at,grid,start,batch", [(0, (2, 3, 13), 0, 1), (117, (2, 3, 13), 41, 2),
                                                (234, (2, 3, 13), 65520, 1)])
def test_fused_preparation_matches_independent_fp32_math_and_cache(at, grid, start, batch):
    from wan_dit_lab.kernels.sage_rope import prepare
    from wan_dit_lab.kernels.sage_kv import quant_q, BLKK
    from wan_dit_lab.kernels.cache_fp8 import store_kv_fp8
    from wan_dit_lab.models.rope import rope, rope_apply

    torch.manual_seed(139)
    n, h, d = grid[0] * grid[1] * grid[2], 3, 128
    raw = torch.randn(batch, n, 3, h, d, device="cuda", dtype=torch.bfloat16)
    q, k, v = raw.unbind(2)  # the production GEMM returns strided Q/K/V views
    wq, wk = [torch.randn(h * d, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    table = torch.cat([rope(65536, 44), rope(65536, 42), rope(65536, 42)], -1).float().cuda()
    def normalize(x, weight):
        x = x.float().flatten(-2)
        return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * weight.float()).view(batch, n, h, d)
    basis = torch.zeros(q.shape, device="cuda")
    basis[..., ::2] = 1
    cs = rope_apply(basis, (grid,), table, start)
    c, s = cs[..., ::2], cs[..., 1::2]
    def rotate(x):
        e, o = x[..., ::2], x[..., 1::2]
        # Emulate the live FMA contract independently with FP64 products.
        even = (e.double()*c.double() - (o*s).double()).float()
        odd = (o.double()*c.double() + (e*s).double()).float()
        return torch.stack((even, odd), -1).flatten(-2)
    qr = rotate(normalize(q, wq))
    kr = rotate(normalize(k, wk)).bfloat16()
    capacity = at + n + 17
    ck, cv = [torch.randn(batch, capacity, h, d, device="cuda").to(torch.float8_e4m3fn) for _ in range(2)]
    ks, vs = [torch.rand(batch, capacity, h, 1, device="cuda") * .03 for _ in range(2)]
    buffers = (ck, ks, cv, vs)
    saved = [x.clone() for x in buffers]
    q8, k8, v8, qs, kscale, vscale = prepare(q, k, v, wq, wk, table, grid, start, *buffers, at, 1e-6, 1e-6)
    expected_q, expected_qs = quant_q(qr)
    torch.testing.assert_close(qs, expected_qs, rtol=2e-6, atol=1e-8)
    assert (q8.int() - expected_q.int()).abs().max() <= 1
    assert (q8 == expected_q).float().mean() > .999
    full_k = torch.cat([saved[0][:, :at].float() * saved[1][:, :at], kr.float()], 1)
    centered = full_k - full_k.mean(1, keepdim=True)
    for block in range((at + n + BLKK - 1) // BLKK):
        for group in range(4):
            ids = [block * 64 + i * 8 + group * 2 + j for i in range(8) for j in range(2)]
            ids = [i for i in ids if i < at + n]
            if not ids:
                continue
            x = centered[:, ids]
            scale = x.abs().amax((1, 3)) / 127 + 1e-7
            torch.testing.assert_close(kscale[:, :, block * 4 + group], scale, rtol=3e-6, atol=1e-8)
            y = x / scale[:, None, :, None]
            expected = (y + torch.where(y >= 0, .5, -.5)).trunc().to(torch.int8)
            assert (k8[:, ids].int() - expected.int()).abs().max() <= 1
    store_kv_fp8(kr, v, *saved, at)
    for actual, expected in zip(buffers, saved):
        if actual.dtype == torch.float8_e4m3fn:
            # Only FP8 boundary ties may differ after the independent RMS reduction.
            assert (actual.view(torch.uint8) == expected.view(torch.uint8)).float().mean() > .999
        else:
            torch.testing.assert_close(actual, expected, rtol=3e-6, atol=1e-9)
    assert v8.dtype == torch.float8_e4m3fn and vscale.shape == (batch, h, d)


@pytest.mark.parametrize("fused", [False, True])
def test_joint_attention_receives_raw_projections_and_owns_norm_and_rope(fused):
    from wan_dit_lab.models.multi_head_attention import MultiHeadAttention
    from wan_dit_lab.quantization.w4a4 import FusedQKVAttention
    attention = (FusedQKVAttention if fused else MultiHeadAttention)(12, 2)
    if fused:
        attention.qkv = torch.nn.Linear(12, 36)
    attention.sage_rope = True
    def forbidden(*args):
        pytest.fail("normalization/RoPE must be owned by the joint preparation")
    attention.norm_q.forward = forbidden
    attention.norm_k.forward = forbidden
    attention.rope_kernel = forbidden
    seen = []
    class Cache:
        def attend_rope(self, q, k, v, norm_q, norm_k, grid, freq, start):
            seen.append((q.shape, norm_q, norm_k, grid, start))
            return v
    x = torch.randn(1, 4, 12)
    assert attention(x, ((1, 2, 2),), None, start_frame=17, cache=Cache()).shape == x.shape
    assert seen == [(torch.Size([1, 4, 2, 6]), attention.norm_q, attention.norm_k, ((1, 2, 2),), 17)]


@pytest.mark.gpu
@pytest.mark.parametrize("at,start", [(0, 0), (78, 1024), (234, 65520)])
def test_joint_attention_retains_accuracy_against_fp64_rope(at, start):
    from wan_dit_lab.kernels import sage_rope, sage_kv
    from wan_dit_lab.models.rope import rope, rope_apply
    from wan_dit_lab.models.rms_norm import RMSNorm
    from wan_dit_lab.harness.checks import block_accuracy_gate
    torch.manual_seed(17)
    b, n, h, d = 1, 78, 3, 128
    raw = torch.randn(b, n, 3, h, d, device="cuda", dtype=torch.bfloat16)
    q, k, v = raw.unbind(2)
    norms = [RMSNorm(h*d).cuda() for _ in range(2)]
    freq64 = torch.cat([rope(65536, 44), rope(65536, 42), rope(65536, 42)], -1).cuda()
    cache = [torch.randn(b, at+n+32, h, d, device="cuda").to(torch.float8_e4m3fn),
             torch.rand(b, at+n+32, h, 1, device="cuda")*.03]
    cache += [cache[0].clone(), cache[1].clone()]
    def rotate(x, norm, dtype, table):
        x = x.to(dtype).flatten(-2)
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True)+norm.eps) * norm.weight.to(dtype)
        return rope_apply(x.view(b,n,h,d), ((2,3,13),), table, start)
    qr, kr = [rotate(x, norm, torch.float64, freq64).float() for x, norm in zip((q,k), norms)]
    full_k = torch.cat([cache[0][:,:at].float()*cache[1][:,:at], kr], 1)
    full_v = torch.cat([cache[2][:,:at].float()*cache[3][:,:at], v.float()], 1)
    reference = torch.nn.functional.scaled_dot_product_attention(qr.transpose(1,2), full_k.transpose(1,2),
                                                                 full_v.transpose(1,2)).transpose(1,2)
    qb, kb = [rotate(x, norm, torch.float32, freq64.float()) for x,norm in zip((q,k),norms)]
    baseline = sage_kv.attend(qb, *cache, at, kb.bfloat16(), v)
    candidate = sage_rope.attend(q,k,v,*norms,(2,3,13),freq64.float(),start,*cache,at)
    gate = block_accuracy_gate(candidate, baseline, reference, .05)
    assert gate["passed"], gate


@pytest.mark.gpu
def test_joint_sage_rope_is_packaged_as_native_gpu_code(tmp_path):
    import zipfile
    from wan_dit_lab.kernels import sage_rope, operators
    from wan_dit_lab.models.rope import rope
    operators.load()
    class Joint(torch.nn.Module):
        def forward(self, q, k, v, wq, wk, freq, ck, ks, cv, vs):
            q8, k8, v8, qs, kscale, vscale = sage_rope.prepare(q,k,v,wq,wk,freq,(2,3,13),17,
                                                             ck,ks,cv,vs,64,1e-6,1e-6)
            return torch.ops.live.sage_sm89(q8,k8,v8,qs,kscale,vscale,128**-.5)
    b,n,h,d=1,78,2,128
    q,k,v = torch.randn(b,n,3,h,d,device="cuda",dtype=torch.bfloat16).unbind(2)
    wq,wk = [torch.ones(h*d,device="cuda",dtype=torch.bfloat16) for _ in range(2)]
    freq=torch.cat([rope(64,44),rope(64,42),rope(64,42)],-1).float().cuda()
    cache=[torch.randn(b,160,h,d,device="cuda").to(torch.float8_e4m3fn),torch.ones(b,160,h,1,device="cuda")*.02]
    cache += [cache[0].clone(),cache[1].clone()]
    inputs=(q,k,v,wq,wk,freq,*cache)
    with torch.no_grad():
        expected=Joint()(*inputs)
        exported=torch.export.export(Joint(),inputs)
        package=torch._inductor.aoti_compile_and_package(exported,package_path=str(tmp_path/"joint.pt2"))
        compiled=torch._inductor.aoti_load_package(package)
        torch.testing.assert_close(compiled(*inputs),expected,rtol=0,atol=0)
    with zipfile.ZipFile(package) as archive:
        contents=b"".join(archive.read(name) for name in archive.namelist() if name.endswith(".cpp"))
        assert b"_q_rope_int8" in contents and b"_k_rope_int8" in contents
