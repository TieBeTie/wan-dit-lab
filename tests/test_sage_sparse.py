"""live::sage_sm89_sparse runs the dense Sage kernel over a list of key blocks per query block.

Its contract, checked bit for bit: the full list is live::sage_sm89, and any list is live::sage_sm89 over the keys
and values of those blocks alone, gathered in the same order (the online softmax visits the same blocks in the
same order, so not even the rounding may differ)."""
import pytest
import torch

CTA_Q, CTA_K = 128, 64


def operands(n, L, h, seed):
    """Random operands in Sage's sm89 layouts: q, k INT8 [1, tokens, h, 128], v FP8 [1, 128, h, padded tokens],
    per-thread scales of q [1, h, query blocks * 32] and k [1, h, key blocks * 4], the channel scales of v."""
    g = torch.Generator().manual_seed(seed)
    padded = -(-L // CTA_K) * CTA_K
    tensors = (torch.randint(-127, 128, (1, n, h, 128), generator=g, dtype=torch.int8),
               torch.randint(-127, 128, (1, L, h, 128), generator=g, dtype=torch.int8),
               (torch.rand(1, 128, h, padded, generator=g) * 4 - 2).to(torch.float8_e4m3fn),
               torch.rand(1, h, -(-n // CTA_Q) * 32, generator=g) * 1e-2,
               torch.rand(1, h, padded // CTA_K * 4, generator=g) * 1e-2,
               torch.rand(1, h, 128, generator=g) * 1e-2)
    return [x.cuda() for x in tensors]


def table(keep):
    from wan_dit_lab.kernels.sparse_blocks import table
    return table(keep)


def gathered(q, k, v, qs, ks, vs, blocks, L):
    """The operands of the dense kernel over the key blocks `blocks` of k, v alone, in that order."""
    tokens = torch.cat([torch.arange(b * CTA_K, (b + 1) * CTA_K) for b in blocks]).cuda()
    length = (len(blocks) - 1) * CTA_K + min(CTA_K, L - blocks[-1] * CTA_K)
    scale = torch.cat([torch.arange(b * 4, b * 4 + 4) for b in blocks]).cuda()
    padded_k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, v.shape[-1] - k.shape[1]))
    return (q, padded_k[:, tokens][:, :length].contiguous(), v[..., tokens].contiguous(), qs,
            ks[..., scale].contiguous(), vs)


@pytest.mark.gpu
@pytest.mark.parametrize("n,L,h", [(256, 256, 2), (880, 2640, 3), (928, 3776, 2)])
def test_the_full_list_is_the_dense_kernel_bit_for_bit(n, L, h):
    from wan_dit_lab.kernels import operators
    ops = operators.load()
    q, k, v, qs, ks, vs = operands(n, L, h, seed=n)
    keep = torch.ones(1, h, -(-n // CTA_Q), -(-L // CTA_K), dtype=torch.bool, device="cuda")
    lut, valid = table(keep)
    dense = ops.sage_sm89(q, k, v, qs, ks, vs, 128 ** -.5)
    assert torch.equal(ops.sage_sm89_sparse(q, k, v, qs, ks, vs, lut, valid, 128 ** -.5), dense)


@pytest.mark.gpu
@pytest.mark.parametrize("n,L,h", [(256, 1000, 2), (880, 2640, 2)])
def test_a_list_is_the_dense_kernel_over_those_blocks_alone(n, L, h):
    from wan_dit_lab.kernels import operators
    ops = operators.load()
    q, k, v, qs, ks, vs = operands(n, L, h, seed=L)
    nq, nk = -(-n // CTA_Q), -(-L // CTA_K)
    g = torch.Generator().manual_seed(11)
    keep = torch.rand(1, h, nq, nk, generator=g) < .4
    keep[..., 0] = False  # the first block may be skipped too: the pointers start past it
    keep[..., -1] = True  # the contract: the last block is always run
    keep = keep.cuda()
    lut, valid = table(keep)
    sparse = ops.sage_sm89_sparse(q, k, v, qs, ks, vs, lut, valid, 128 ** -.5)
    for head in range(h):
        for block in range(nq):
            rows = slice(block * CTA_Q, min(n, (block + 1) * CTA_Q))
            kept = keep[0, head, block].nonzero().flatten().tolist()
            one = [x[:, :, head:head + 1].contiguous() for x in (q[:, rows], k, v)]
            scales = (qs[:, head:head + 1, block * 32:(block + 1) * 32].contiguous(), ks[:, head:head + 1].contiguous(),
                      vs[:, head:head + 1].contiguous())
            expected = ops.sage_sm89(*gathered(one[0], one[1], one[2], scales[0], scales[1], scales[2], kept, L),
                                     128 ** -.5)
            assert torch.equal(sparse[:, rows, head:head + 1], expected), (head, block, kept)


def test_the_table_lists_the_kept_blocks_ascending():
    from wan_dit_lab.kernels.sparse_blocks import table
    kept = torch.tensor([[[[False, True, False, True, True], [True, False, False, False, True]]]])
    lut, valid = table(kept)
    assert lut.dtype == valid.dtype == torch.int32
    assert valid.tolist() == [[[3, 2]]]
    assert lut[0, 0, 0, :3].tolist() == [1, 3, 4] and lut[0, 0, 1, :2].tolist() == [0, 4]


def shares(*probabilities):
    """Block means whose estimated shares of the attention are `probabilities` (one query block, one head)."""
    logits = torch.tensor(probabilities).log()
    q = torch.zeros(1, 1, 1, len(probabilities) + 1)
    q[..., 0] = 1
    k = torch.zeros(1, 1, len(probabilities), len(probabilities) + 1)
    k[0, 0, :, 0] = logits
    return q, k, torch.ones(len(probabilities))


def test_skip_mass_zero_keeps_every_allowed_block_and_the_forced_ones():
    from wan_dit_lab.kernels.sparse_blocks import keep
    q, k, tokens = shares(.5, .3, .15, .05)
    allowed = torch.tensor([True, False, True, True])
    forced = torch.tensor([False, True, False, False])
    assert keep(q, k, tokens, allowed, forced, 0.0, 1.0).flatten().tolist() == [True, True, True, True]
    assert keep(q, k, tokens, allowed, ~forced & forced, 0.0, 1.0).flatten().tolist() == [True, False, True, True]


@pytest.mark.parametrize("skip_mass,expected", [(.04, [True, True, True, True]), (.1, [True, True, True, False]),
                                                (.25, [True, True, False, False]), (.55, [True, False, False, False])])
def test_the_weakest_blocks_are_skipped_while_their_shares_add_up_to_less_than_skip_mass(skip_mass, expected):
    from wan_dit_lab.kernels.sparse_blocks import keep
    q, k, tokens = shares(.5, .3, .15, .05)
    everything, nothing = torch.ones(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool)
    assert keep(q, k, tokens, everything, nothing, skip_mass, 1.0).flatten().tolist() == expected


def test_forced_blocks_are_never_skipped_and_disallowed_ones_take_no_share():
    from wan_dit_lab.kernels.sparse_blocks import keep
    q, k, tokens = shares(.5, .3, .15, .05)
    forced = torch.tensor([False, False, False, True])
    everything = torch.ones(4, dtype=torch.bool)
    assert keep(q, k, tokens, everything, forced, .25, 1.0).flatten().tolist() == [True, True, False, True]
    # without block 0 the shares are .6, .3, .1: skip_mass .15 skips block 3 alone
    allowed = torch.tensor([False, True, True, True])
    assert keep(q, k, tokens, allowed, ~everything, .15, 1.0).flatten().tolist() == [False, True, True, False]


def test_a_block_of_fewer_tokens_has_a_proportionally_smaller_share():
    from wan_dit_lab.kernels.sparse_blocks import keep
    q, k, _ = shares(.25, .25, .25, .25)
    tokens = torch.tensor([64., 64., 64., 8.])  # 8 of 200 tokens: share .04
    everything, nothing = torch.ones(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool)
    assert keep(q, k, tokens, everything, nothing, .05, 1.0).flatten().tolist() == [True, True, True, False]


def joint_inputs(at, grid, h=3, seed=3):
    from wan_dit_lab.models.rope import rope
    torch.manual_seed(seed)
    n = grid[0] * grid[1] * grid[2]
    q, k, v = torch.randn(1, n, 3, h, 128, device="cuda", dtype=torch.bfloat16).unbind(2)
    wq, wk = [torch.randn(h * 128, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    table = torch.cat([rope(4096, 44), rope(4096, 42), rope(4096, 42)], -1).float().cuda()
    cache = [torch.randn(1, at + n + 40, h, 128, device="cuda").to(torch.float8_e4m3fn),
             torch.rand(1, at + n + 40, h, 1, device="cuda") * .03]
    return q, k, v, wq, wk, table, cache + [cache[0].clone(), cache[1].clone()]


def same(a, b):
    bits = lambda x: x.view(torch.uint8) if x.dtype == torch.float8_e4m3fn else x
    return torch.equal(bits(a), bits(b))


@pytest.mark.gpu
@pytest.mark.parametrize("at,grid", [(0, (2, 3, 13)), (128, (2, 4, 29)), (3712, (8, 16, 29))])
def test_pooling_leaves_the_preparation_bit_for_bit_and_sums_the_rotated_queries_and_keys(at, grid):
    from wan_dit_lab.kernels import sage_rope
    q, k, v, wq, wk, freq, cache = joint_inputs(at, grid)
    n, h = q.shape[1], q.shape[2]
    first = [c.clone() for c in cache]
    dense = sage_rope.prepare(q, k, v, wq, wk, freq, grid, 5, *first, at, 1e-6, 1e-6)
    second = [c.clone() for c in cache]
    *pooled, q_sums, k_sums = sage_rope.prepare(q, k, v, wq, wk, freq, grid, 5, *second, at, 1e-6, 1e-6, pool=True)
    assert all(same(a, b) for a, b in zip(list(dense) + first, pooled + second))
    # the rotated queries, recovered from the dense path's own INT8 Q and its per-thread scales
    q8, _, _, qs = dense[:4]
    token = torch.arange(n, device="cuda")
    rotated_q = q8.float() * qs[:, :, (token // 32) * 8 + token % 8].permute(0, 2, 1)[..., None]
    expected_q = torch.nn.functional.pad(rotated_q, (0, 0, 0, 0, 0, -n % CTA_Q)).view(1, -1, CTA_Q, h, 128).sum(2)
    torch.testing.assert_close(q_sums, expected_q.permute(0, 2, 1, 3), rtol=0, atol=CTA_Q * qs.max().item())
    # the rotated keys: the cache holds them all in FP8 now, the chunk's too
    keys = first[0][:, :at + n].float() * first[1][:, :at + n]
    expected_k = torch.nn.functional.pad(keys, (0, 0, 0, 0, 0, -(at + n) % CTA_K)).view(1, -1, CTA_K, h, 128).sum(2)
    torch.testing.assert_close(k_sums, expected_k.permute(0, 2, 1, 3), rtol=.02, atol=.05 * CTA_K)


@pytest.mark.gpu
@pytest.mark.parametrize("at,grid", [(0, (2, 3, 13)), (3712, (8, 16, 29))])
def test_sparse_attention_with_skip_mass_zero_is_the_dense_one_bit_for_bit(at, grid):
    from wan_dit_lab.kernels import sage_rope
    from wan_dit_lab.models.rms_norm import RMSNorm
    q, k, v, _, _, freq, cache = joint_inputs(at, grid)
    norms = [RMSNorm(3 * 128).cuda() for _ in range(2)]
    first, second = [c.clone() for c in cache], [c.clone() for c in cache]
    dense = sage_rope.attend(q, k, v, *norms, grid, freq, 5, *first, at)
    allowed = torch.ones(-(-(at + q.shape[1]) // CTA_K), device="cuda", dtype=torch.int32)
    sparse = sage_rope.attend(q, k, v, *norms, grid, freq, 5, *second, at, skip_mass=0.0, key_keep=allowed)
    assert torch.equal(dense, sparse)
    assert all(same(a, b) for a, b in zip(first, second))


@pytest.mark.gpu
def test_disallowed_frames_are_skipped():
    # with every cached block disallowed the chunk attends to itself alone, as over an empty cache; only the K mean
    # (the centering over all keys) and the V channel scales still see the cached tokens
    from wan_dit_lab.kernels import sage_rope
    from wan_dit_lab.models.rms_norm import RMSNorm
    grid, at = (2, 4, 16), 512
    q, k, v, _, _, freq, cache = joint_inputs(at, grid)
    norms = [RMSNorm(3 * 128).cuda() for _ in range(2)]
    n = q.shape[1]
    alone = sage_rope.attend(q, k, v, *norms, grid, freq, 5, *[c.clone() for c in cache], 0)
    disallowed = torch.zeros((at + n) // CTA_K, device="cuda", dtype=torch.int32)
    sparse = sage_rope.attend(q, k, v, *norms, grid, freq, 5, *[c.clone() for c in cache], at, skip_mass=0.0,
                              key_keep=disallowed)
    dense = sage_rope.attend(q, k, v, *norms, grid, freq, 5, *[c.clone() for c in cache], at)
    error = lambda x: (x.float() - alone.float()).abs().mean().item()
    assert error(sparse) < .1 * error(dense)


@pytest.mark.gpu
@pytest.mark.parametrize("skip_mass", [None, 0.0, .1])
def test_the_package_holds_the_sparse_code_only_when_it_is_on(tmp_path, skip_mass):
    import zipfile
    from wan_dit_lab.kernels import sage_rope, operators
    from wan_dit_lab.models.rms_norm import RMSNorm
    operators.load()
    grid, at = (2, 4, 16), 256
    q, k, v, _, _, freq, cache = joint_inputs(at, grid, h=2)
    norms = [RMSNorm(2 * 128).cuda() for _ in range(2)]

    class Joint(torch.nn.Module):
        def forward(self, q, k, v, ck, ks, cv, vs, key_keep):
            return sage_rope.attend(q, k, v, *norms, grid, freq, 5, ck, ks, cv, vs, at, skip_mass=skip_mass,
                                    key_keep=None if skip_mass is None else key_keep)
    key_keep = torch.ones((at + q.shape[1]) // CTA_K, device="cuda", dtype=torch.int32)
    key_keep[1] = 0
    with torch.no_grad():
        expected = Joint()(q, k, v, *[c.clone() for c in cache], key_keep)
        inputs = (q, k, v, *[c.clone() for c in cache], key_keep)
        package = torch._inductor.aoti_compile_and_package(torch.export.export(Joint(), inputs),
                                                           package_path=str(tmp_path / "joint.pt2"))
        actual = torch._inductor.aoti_load_package(package)(q, k, v, *[c.clone() for c in cache], key_keep)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with zipfile.ZipFile(package) as archive:
        code = b"".join(archive.read(name) for name in archive.namelist() if name.endswith((".cpp", ".py", ".json")))
    # off, the package is the dense one: no sparse operator, no pooled preparation
    assert (b"sage_sm89_sparse" in code) == (skip_mass is not None)
    assert (b"sage_rope_prepare_pooled" in code) == (skip_mass is not None)
