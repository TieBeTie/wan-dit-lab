"""The kernels requested by a run must describe what its AOT packages actually contain."""

import copy
import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from wan_dit_lab.runtime.live_kernels import read_kernels, validate_kernels, validate_partial_export

ROOT = Path(__file__).resolve().parents[1]
KERNELS = {
    "attention": "sage_sm89",
    "cross_attention": ["cudnn", "efficient"],
    "linear": "nunchaku_w4a4",
    "kv_cache": "fp8_e4m3fn",
    "kv_store": "triton",
    "rope": "torch_inductor",
    "rope_dtype": "float32",
}


def test_fp8_original_recipe_is_accepted():
    kernels = {**KERNELS, "linear": "torch_fp8", "attention": "torch_sdpa",
               "cross_attention": ["math"], "kv_cache": "bfloat16", "kv_store": "torch",
               "rope": "longlive_triton"}
    assert validate_kernels(kernels) == kernels


def test_dense_sdpa_accepts_memory_efficient_backend():
    # MATH materializes the scores: at 704x1280 a compiled block grew by 1.3 GiB per chunk of context (5.3 GiB at a
    # full window) and spilled out of 8 GiB; the original LongLive computes this attention with FlashAttention.
    kernels = {**KERNELS, "linear": "torch_fp8", "attention": "torch_sdpa", "cross_attention": ["efficient"],
               "rope": "longlive_triton"}
    assert validate_kernels(kernels) == kernels


def test_point_a_accepts_fp8_storage_with_sdpa():
    from wan_dit_lab.runtime.live_pipeline import load_pipeline
    export, _ = load_pipeline(ROOT / "configs/experiments/director-original/point-a-fp8-kv.proposed.json")
    assert export["kernels"]["attention"] == "torch_sdpa"
    assert export["kernels"]["kv_cache"] == "fp8_e4m3fn"


@pytest.mark.gpu
def test_fp8_sdpa_reads_past_and_keeps_current_chunk_unquantized(exporter, monkeypatch, tmp_path):
    import torch
    from wan_dit_lab.kernels.cache_fp8 import store_kv_fp8
    torch.manual_seed(7)
    q, k, v = [torch.randn(1, 5, 2, 16, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    ck, cv = [torch.zeros(1, 11, 2, 16, device="cuda", dtype=torch.float8_e4m3fn) for _ in range(2)]
    ks, vs = [torch.ones(1, 11, 2, 1, device="cuda") for _ in range(2)]
    old_k, old_v = [torch.randn(1, 3, 2, 16, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    store_kv_fp8(old_k, old_v, ck, ks, cv, vs, 0)
    expected_k = torch.cat([(ck[:, :3].float() * ks[:, :3]).bfloat16(), k], 1)
    expected_v = torch.cat([(cv[:, :3].float() * vs[:, :3]).bfloat16(), v], 1)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        expected = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), expected_k.transpose(1, 2), expected_v.transpose(1, 2)).transpose(1, 2)
        monkeypatch.setattr(exporter.sage_kv, "attend", lambda *a: pytest.fail("FP8 SDPA called Sage"))
        got = exporter.Cache(3, ck, ks, cv, vs, attention="torch_sdpa").attend(q, k, v)
    torch.testing.assert_close(got, expected, rtol=.005, atol=.005)
    assert ck[:, 3:8].float().count_nonzero() > 0
    assert ck[:, 8:].float().count_nonzero() == 0
    class Attention(torch.nn.Module):
        def forward(self, q, k, v, ck, ks, cv, vs):
            return exporter.Cache(3, ck, ks, cv, vs, attention="torch_sdpa").attend(q, k, v)
    export = exporter.Exporter(tmp_path, None, {**KERNELS, "cross_attention": ["math"]})
    export("fp8_sdpa_probe", Attention(), (q, k, v, ck, ks, cv, vs))


@pytest.mark.gpu
def test_fp8_cache_attention_exports_with_memory_efficient_sdpa(exporter, tmp_path):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from wan_dit_lab.kernels.cache_fp8 import store_kv_fp8
    torch.manual_seed(7)
    q, k, v = [torch.randn(1, 160, 2, 48, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    ck, cv = [torch.zeros(1, 320, 2, 48, device="cuda", dtype=torch.float8_e4m3fn) for _ in range(2)]
    ks, vs = [torch.ones(1, 320, 2, 1, device="cuda") for _ in range(2)]
    old_k, old_v = [torch.randn(1, 160, 2, 48, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    store_kv_fp8(old_k, old_v, ck, ks, cv, vs, 0)
    class Attention(torch.nn.Module):
        def forward(self, q, k, v, ck, ks, cv, vs):
            return exporter.Cache(160, ck, ks, cv, vs, attention="torch_sdpa").attend(q, k, v)
    inputs = (q, k, v, ck, ks, cv, vs)
    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
        math = Attention()(*inputs)
    export = exporter.Exporter(tmp_path, None, {**KERNELS, "attention": "torch_sdpa", "cross_attention": ["efficient"]})
    efficient = export("fp8_sdpa_efficient_probe", Attention(), inputs)
    assert exporter.relative(efficient, math) < 0.01


def test_fp8_experiment_accepts_automatic_memory_policy():
    from wan_dit_lab.runtime.live_pipeline import load_pipeline
    _, run = load_pipeline(ROOT / "configs/experiments/director-original/s2-fp8-native.proposed.json")
    assert run["offloading"]["t5"] == {"activations": "auto", "resident": "auto"}
    assert run["debug"] == "fps"


def test_fp8_bound_rebinds_packed_weights_on_each_row(exporter):
    import torch
    from wan_dit_lab.quantization.weight_pack import fp8_layout
    block = torch.nn.Module()
    block.self_attn = torch.nn.Identity()
    block.proj = torch.nn.Linear(16, 16).bfloat16()
    layout, size = fp8_layout(block)
    bound = exporter.Bound(block, layout, size, linear="torch_fp8", rope="longlive_triton",
                           attention="torch_sdpa", cross_attention=["math"])
    row = torch.zeros(size, dtype=torch.uint8)
    bound.bind(row)
    first = bound.block.proj.weight.data_ptr()
    other = row.clone()
    bound.bind(other)
    assert bound.block.proj.weight.data_ptr() != first
    assert bound.block.proj.weight.untyped_storage().data_ptr() == other.data_ptr()


@pytest.mark.gpu
def test_fp8_row_executes_in_compiled_package_and_rebinds(exporter, tmp_path, monkeypatch):
    import torch
    from wan_dit_lab.quantization.weight_pack import build_fp8, mapped_fp8
    block = torch.nn.Module()
    block.self_attn = torch.nn.Identity()
    block.proj = torch.nn.Linear(32, 32).bfloat16().eval()
    other = copy.deepcopy(block)
    with torch.no_grad():
        other.proj.bias.add_(1)
    source, pack = tmp_path / "source", tmp_path / "rows.fp8"
    source.write_bytes(b"FP8 probe")
    build_fp8([block, other], pack, source)
    rows, layout = mapped_fp8([block, other], pack, source)
    bound = exporter.Bound(block, layout, rows.shape[1], linear="torch_fp8", rope="longlive_triton",
                           attention="torch_sdpa", cross_attention=["math"])
    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bound = bound
        def forward(self, row, x):
            return self.bound.bind(row).proj(x)
    probe = Probe()
    runners = []
    load = torch._inductor.aoti_load_package
    def remember(path):
        runner = load(path)
        runners.append(runner)
        return runner
    monkeypatch.setattr(torch._inductor, "aoti_load_package", remember)
    x = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16)
    export = exporter.Exporter(tmp_path, None, {**KERNELS, "cross_attention": ["math"]})
    export("fp8_probe", probe, (rows[0].cuda(), x))
    runner = runners[0]
    row = rows[1].cuda()
    with torch.no_grad():
        # Both rows have identical weights; the second bias is one larger.
        # This also avoids treating a second compiler as an independent oracle.
        delta = runner(row, x).float() - runner(rows[0].cuda(), x).float()
        torch.testing.assert_close(delta, torch.ones_like(delta), rtol=0, atol=.03)


@pytest.mark.parametrize("name", ["live.json", "offline.json"])
def test_run_configs_explicitly_name_every_kernel(name):
    expected = {key: value for key, value in KERNELS.items() if key != "rope"}
    expected["attention"] = "sage_rope_sm89"
    expected["cross_attention"] = ["sage_sm89"]
    assert read_kernels(ROOT / "configs" / name) == expected


@pytest.mark.parametrize("rope", ["torch_inductor", "triton"])
def test_both_rope_implementations_are_explicit_supported_choices(rope):
    assert validate_kernels({**KERNELS, "rope": rope})["rope"] == rope


def test_joint_sage_rope_owns_its_rope_instead_of_accepting_an_independent_kernel():
    fused = {k: v for k, v in KERNELS.items() if k != "rope"}
    fused["attention"] = "sage_rope_sm89"
    assert validate_kernels(fused) == fused
    with pytest.raises(ValueError, match="rope"):
        validate_kernels({**fused, "rope": "torch_inductor"})


def test_export_binds_joint_sage_rope_to_the_block(exporter, monkeypatch):
    template = exporter.torch.nn.Module()
    template.self_attn = exporter.torch.nn.Identity()
    monkeypatch.setattr(exporter.w4a4, "slot", lambda row, like, layout: copy.deepcopy(like))
    bound = exporter.Bound(template, [], 1, attention="sage_rope_sm89")
    assert bound.block.self_attn.sage_rope is True
    assert not hasattr(template.self_attn, "sage_rope")


@pytest.mark.parametrize("field", list(KERNELS))
def test_every_kernel_field_is_required(field):
    kernels = copy.deepcopy(KERNELS)
    del kernels[field]
    with pytest.raises(ValueError, match=field):
        validate_kernels(kernels)


@pytest.mark.parametrize("field", list(KERNELS))
def test_unsupported_kernels_are_refused_instead_of_exporting_something_else(field):
    kernels = copy.deepcopy(KERNELS)
    kernels[field] = ["unknown"] if field == "cross_attention" else "unknown"
    with pytest.raises(ValueError, match=field):
        validate_kernels(kernels)


@pytest.mark.parametrize("cross", [[], ["cudnn", "cudnn"], "cudnn"])
def test_cross_attention_is_an_explicit_nonempty_ordered_list(cross):
    with pytest.raises(ValueError, match="cross_attention"):
        validate_kernels({**KERNELS, "cross_attention": cross})


def test_cross_attention_can_be_restricted_to_one_backend():
    assert validate_kernels({**KERNELS, "cross_attention": ["efficient"]})["cross_attention"] == ["efficient"]


def test_sage_cross_is_a_complete_backend_instead_of_a_mixed_sdpa_fallback_list():
    assert validate_kernels({**KERNELS, "cross_attention": ["sage_sm89"]})["cross_attention"] == ["sage_sm89"]
    with pytest.raises(ValueError, match="cross_attention"):
        validate_kernels({**KERNELS, "cross_attention": ["sage_sm89", "cudnn"]})


def test_export_selects_sage_cross_before_preparing_the_prompt(exporter, monkeypatch):
    template = exporter.torch.nn.Module()
    template.self_attn = exporter.torch.nn.Identity()
    template.cross_attn = exporter.torch.nn.Identity()
    monkeypatch.setattr(exporter.w4a4, "slot", lambda row, like, layout: copy.deepcopy(like))
    bound = exporter.Bound(template, [], 1, attention="sage_rope_sm89", cross_attention=["sage_sm89"])
    assert bound.block.cross_attn.sage_cross_tokens == exporter.TEXT_LEN
    assert not hasattr(template.cross_attn, "sage_cross_tokens")
    exporter.Exporter(Path("unused"), None, {**KERNELS, "cross_attention": ["sage_sm89"]})


def test_kernel_typos_are_not_ignored():
    with pytest.raises(ValueError, match="atention"):
        validate_kernels({**KERNELS, "atention": "sage_sm89"})


def test_missing_config_or_kernel_block_is_not_defaulted(tmp_path):
    path = tmp_path / "run.json"
    with pytest.raises(FileNotFoundError):
        read_kernels(path)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="config|kernels"):
        read_kernels(path)


def test_partial_export_cannot_relabel_old_or_different_packages(tmp_path):
    path = tmp_path / "live.json"
    for previous in ({}, {"kernels": {**KERNELS, "cross_attention": ["efficient"]}}):
        path.write_text(json.dumps(previous), encoding="utf-8")
        with pytest.raises(ValueError, match="full export"):
            validate_partial_export(path, KERNELS)
    path.write_text(json.dumps({"kernels": KERNELS}), encoding="utf-8")
    validate_partial_export(path, KERNELS)
    path.unlink()
    with pytest.raises(ValueError, match="full export"):
        validate_partial_export(path, KERNELS)


def test_partial_export_cannot_mix_different_model_recipes(tmp_path):
    path = tmp_path / "live.json"
    path.write_text(json.dumps({"kernels": KERNELS, "export": {"identity": "old"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="full export"):
        validate_partial_export(path, KERNELS, identity="new")


def test_interrupted_exports_cannot_leave_a_trusted_old_manifest(tmp_path):
    from wan_dit_lab.runtime.live_kernels import begin_export, complete_export

    path = tmp_path / "live.json"
    path.write_text(json.dumps({"kernels": KERNELS, "checks": {"old": {"relative_l2": [0]}}}), encoding="utf-8")
    checks = begin_export(tmp_path, KERNELS, only=True)
    assert checks == {"old": {"relative_l2": [0]}}
    assert (tmp_path / ".exporting").is_file()
    with pytest.raises(ValueError, match="full export"):
        validate_partial_export(path, KERNELS)
    # A full retry can replace an interrupted export. Only successful publication removes the marker.
    assert begin_export(tmp_path, KERNELS, only=False) == {}
    complete_export(tmp_path, {"kernels": KERNELS, "checks": {"new": {}}})
    assert not (tmp_path / ".exporting").exists()
    assert json.loads(path.read_text(encoding="utf-8"))["kernels"] == KERNELS


@pytest.fixture(scope="module")
def exporter():
    spec = importlib.util.spec_from_file_location("export_live_kernels_test", ROOT / "scripts/conversion/export_live.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dense_export_reads_the_dense_pack_and_never_the_w4a4_pack(exporter, monkeypatch):
    from wan_dit_lab.quantization import weight_pack
    blocks, pack, checkpoint = object(), Path("dense.bf16"), Path("s2.pt")
    expected = object()
    def mapped(actual_blocks, actual_pack, actual_checkpoint):
        assert (actual_blocks, actual_pack, actual_checkpoint) == (blocks, pack, checkpoint)
        return expected
    monkeypatch.setattr(weight_pack, "mapped_rows", mapped)
    monkeypatch.setattr(exporter.w4a4, "mapped_pack", lambda *a: pytest.fail("dense export quantized weights"))
    assert exporter.block_rows(blocks, pack, checkpoint, "torch_bf16") is expected


def test_bf16_linear_can_be_selected_without_changing_other_kernels():
    assert validate_kernels({**KERNELS, "linear": "torch_bf16"})["linear"] == "torch_bf16"


def test_dense_cache_keeps_bf16_values_and_reads_only_past_and_current_chunk(exporter, monkeypatch):
    torch = exporter.torch
    monkeypatch.setattr(exporter, "store_kv_fp8", lambda *a: pytest.fail("BF16 cache quantized"))
    torch.manual_seed(7)
    k, v = [torch.randn(1, 9, 2, 4, dtype=torch.bfloat16) for _ in range(2)]
    old_k, old_v = k.clone(), v.clone()
    q, fresh_k, fresh_v = [torch.randn(1, 3, 2, 4, dtype=torch.bfloat16) for _ in range(3)]
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), torch.cat([k[:, :3], fresh_k], 1).transpose(1, 2),
        torch.cat([v[:, :3], fresh_v], 1).transpose(1, 2)).transpose(1, 2)
    scales = torch.ones(1, 9, 2, 1)
    actual = exporter.Cache(3, k, scales, v, scales.clone()).attend(q, fresh_k, fresh_v)
    assert torch.equal(actual, expected)
    assert torch.equal(k[:, :3], old_k[:, :3]) and torch.equal(v[:, :3], old_v[:, :3])
    assert torch.equal(k[:, 3:6], fresh_k) and torch.equal(v[:, 3:6], fresh_v)
    assert torch.equal(k[:, 6:], old_k[:, 6:]) and torch.equal(v[:, 6:], old_v[:, 6:])


def test_dense_attention_requires_dense_cache_and_torch_store():
    kernels = {**KERNELS, "attention": "torch_sdpa", "kv_cache": "bfloat16", "kv_store": "torch",
               "cross_attention": ["math"]}
    assert validate_kernels(kernels) == kernels
    for field, value in [("attention", "sage_sm89"), ("kv_cache", "fp8_e4m3fn"), ("kv_store", "triton")]:
        with pytest.raises(ValueError):
            validate_kernels({**kernels, field: value})


def test_dense_attention_bounds_score_memory_without_shortening_context(exporter, monkeypatch):
    torch = exporter.torch
    sdpa = torch.nn.functional.scaled_dot_product_attention
    lengths = []
    def observe(q, k, v, *args, **kwargs):
        lengths.append((q.shape[-2], k.shape[-2]))
        return sdpa(q, k, v, *args, **kwargs)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", observe)
    q, fresh_k, fresh_v = [torch.randn(1, 257, 1, 4, dtype=torch.bfloat16) for _ in range(3)]
    k, v = [torch.randn(1, 300, 1, 4, dtype=torch.bfloat16) for _ in range(2)]
    expected = sdpa(q.transpose(1, 2), torch.cat([k[:, :3], fresh_k], 1).transpose(1, 2),
                    torch.cat([v[:, :3], fresh_v], 1).transpose(1, 2)).transpose(1, 2)
    actual = exporter.Cache(3, k, torch.ones(1), v, torch.ones(1)).attend(q, fresh_k, fresh_v)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert max(length for length, _ in lengths) <= 128
    assert all(context == 260 for _, context in lengths)


@pytest.mark.gpu
def test_longlive_rope_uses_the_original_kernel_in_a_compiled_package(exporter, tmp_path):
    from wan_dit_lab.models.rope import rope
    from wan_dit_lab.kernels.longlive_rope import rope_apply, upstream
    torch = exporter.torch
    class Rotate(torch.nn.Module):
        def forward(self, x, freq):
            return rope_apply(x, ((2, 2, 3),), freq, 7)
    x = torch.randn(1, 12, 2, 12, device="cuda", dtype=torch.bfloat16)
    freq = torch.cat([rope(16, 4)] * 3, dim=-1).float().cuda()
    # Independent table construction and the original public entry point.
    cs = torch.cat([freq[:, 7:9, :2].view(2, 2, 1, 1, 2).expand(2, 2, 2, 3, 2),
                    freq[:, :2, 2:4].view(2, 1, 2, 1, 2).expand(2, 2, 2, 3, 2),
                    freq[:, :3, 4:].view(2, 1, 1, 3, 2).expand(2, 2, 2, 3, 2)], -1).reshape(2, 12, 6)
    expected = upstream.rope_apply_triton(x[0], cs[0].contiguous(), cs[1].contiguous())[None]
    export = exporter.Exporter(tmp_path, None, KERNELS)
    actual = export("longlive_rope_probe", Rotate(), (x, freq))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.gpu
def test_longlive_rope_survives_compilation_when_sizes_are_multiples_of_16():
    # Integers divisible by 16 (160 tokens, strides 96 and 48) made Inductor's analysis of the original kernel fail
    # (IndexError in set_arg_attr); it then dropped the kernel and returned an uninitialized buffer.
    import torch
    from wan_dit_lab.models.rope import rope, rope_apply as reference
    from wan_dit_lab.kernels.longlive_rope import rope_apply
    torch.manual_seed(0)
    grid, freq = ((1, 2, 80),), torch.cat([rope(160, 16)] * 3, -1).float().cuda()
    x = torch.randn(1, 160, 96, device="cuda", dtype=torch.bfloat16)
    def rotate(x):
        return rope_apply(torch.nn.functional.rms_norm(x, (96,)).reshape(1, 160, 2, 48), grid, freq, 0)
    with torch.no_grad():
        expected = reference(torch.nn.functional.rms_norm(x, (96,)).reshape(1, 160, 2, 48), grid, freq, 0)
        actual = torch.compile(rotate, fullgraph=True)(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.gpu
def test_dense_block_exports_with_original_rope_and_mutable_bf16_cache(exporter, tmp_path, monkeypatch):
    from wan_dit_lab.models.block import WanBlock
    from wan_dit_lab.models.rope import rope
    from wan_dit_lab.quantization.weight_pack import build, mapped_rows
    torch = exporter.torch
    torch.manual_seed(23)
    block = WanBlock(24, 48, 2, 1e-6).to(torch.bfloat16).eval()
    source, pack = tmp_path / "source.pt", tmp_path / "block.bf16"
    source.write_bytes(b"tiny dense checkpoint")
    other = copy.deepcopy(block)
    with torch.no_grad():
        other.ffn[2].bias.add_(0.5)
    build([block, other], pack, source)
    rows, layout = mapped_rows([block, other], pack, source)
    bound = exporter.Bound(block, layout, rows.shape[1], linear="torch_bf16", attention="torch_sdpa",
                           cross_attention=["math"], rope="longlive_triton")
    freq = torch.cat([rope(160, 4)] * 3, -1).float().cuda()
    step = exporter.BlockStep(bound, 3, (1, 1, 129), freq[:, :, 2:].contiguous())
    x = torch.randn(1, 129, 24, device="cuda", dtype=torch.bfloat16)
    e0 = torch.randn(1, 6, 24, device="cuda", dtype=torch.bfloat16) * .1
    cross_k, cross_v = [torch.randn(1, 8, 2, 12, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    k, v = [torch.zeros(1, 140, 2, 12, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    scales = torch.ones(1, 140, 2, 1, device="cuda")
    kernels = {**KERNELS, "attention": "torch_sdpa", "kv_cache": "bfloat16", "kv_store": "torch",
               "linear": "torch_bf16", "cross_attention": ["math"], "rope": "longlive_triton"}
    export = exporter.Exporter(tmp_path, None, kernels)
    runners = []
    load = torch._inductor.aoti_load_package
    def remember(path):
        runner = load(path)
        runners.append(runner)
        return runner
    monkeypatch.setattr(torch._inductor, "aoti_load_package", remember)
    export("dense_block_probe", step, (rows[0].cuda(), x, e0, cross_k, cross_v, freq[:, :1, :2].contiguous(),
                                        k, scales, v, scales.clone()), eager_rounding=True)
    assert k[:, 3:132].count_nonzero() > 0 and v[:, 3:132].count_nonzero() > 0
    assert k[:, :3].count_nonzero() == 0 and k[:, 132:].count_nonzero() == 0
    # Execute the package on a different row and fresh cache buffers: neither
    # weights nor eager cache writes may have been captured as constants.
    runner = runners[0]  # Windows keeps the extracted DLL mapped until process exit.
    second = (rows[1].cuda(), x, e0, cross_k, cross_v, freq[:, :1, :2].contiguous(),
              torch.zeros_like(k), scales, torch.zeros_like(v), scales.clone())
    expected_inputs = tuple(t.clone() for t in second)
    with torch.no_grad(), export.attention_context():
        expected = step(*expected_inputs)
        actual = runner(*second)
    assert exporter.relative(actual, expected) < .01
    torch.testing.assert_close(second[6], expected_inputs[6], rtol=.01, atol=.01)
    assert second[6][:, 3:132].count_nonzero() > 0


def test_dense_export_rebinds_bf16_rows_without_calling_quantization(exporter, monkeypatch):
    torch = exporter.torch
    template = torch.nn.Module()
    template.self_attn = torch.nn.Identity()
    template.projection = torch.nn.Linear(4, 3).to(torch.bfloat16)
    entries, parts, offset = [], [], 0
    for name, value in template.named_parameters():
        data = value.detach().reshape(-1).view(torch.uint8)
        entries.append(dict(name=name, dtype='bfloat16', shape=list(value.shape), offset=offset, bytes=data.numel()))
        parts.append(data)
        offset += data.numel()
    row = torch.cat(parts)
    monkeypatch.setattr(exporter.w4a4, 'slot', lambda *args: pytest.fail('BF16 used quantized slot'))
    bound = exporter.Bound(template, entries, row.numel(), linear='torch_bf16', rope='torch_inductor')
    x = torch.randn(2, 4, dtype=torch.bfloat16)
    assert torch.equal(bound.bind(row).projection(x), template.projection(x))


def test_dense_sdpa_rope_preserves_bf16_for_matching_qkv_dtypes(exporter):
    template = exporter.torch.nn.Module()
    template.self_attn = exporter.torch.nn.Identity()
    bound = exporter.Bound(template, [], 0, linear="torch_bf16", rope="triton", attention="torch_sdpa")
    assert not hasattr(bound.block.self_attn, "rope_input_dtype")


def test_export_uses_the_config_size_before_loading_gpu_operators(exporter, monkeypatch, tmp_path):
    config = tmp_path / "run.json"
    config.write_text((ROOT / "configs/live.json").read_text(encoding="utf-8"), encoding="utf-8")
    output = json.loads(config.read_text(encoding="utf-8"))[0]["output"]
    monkeypatch.setattr(exporter, "export_identity", lambda *a, **kw: "identity")
    monkeypatch.setattr(exporter, "ROOT", tmp_path)
    monkeypatch.setattr(exporter.sys, "argv", ["export_live.py", "--config", str(config)])

    def stop():
        assert (tmp_path / output / ".exporting").is_file()
        raise RuntimeError("stopped before GPU operators")

    monkeypatch.setattr(exporter, "load_operators", stop)
    with pytest.raises(RuntimeError, match="stopped before GPU"):
        exporter.main()


def test_export_rejects_unsupported_choices_before_loading_gpu_operators(exporter, monkeypatch, tmp_path):
    config = tmp_path / "run.json"
    value = json.loads((ROOT / "configs/live.json").read_text(encoding="utf-8"))
    value[0]["kernels"]["rope"] = "unknown"
    config.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(exporter.sys, "argv", ["export_live.py", "--config", str(config)])
    monkeypatch.setattr(exporter, "load_operators", lambda: pytest.fail("touched GPU operators"))
    with pytest.raises(ValueError, match="rope"):
        exporter.main()


def test_exporter_honors_the_cross_attention_choice_and_order(exporter, monkeypatch, tmp_path):
    chosen = []

    @contextmanager
    def sdpa(backends, *, set_priority):
        assert set_priority
        chosen.append(backends)
        yield

    monkeypatch.setattr(exporter, "sdpa_kernel", sdpa)
    kernels = {**KERNELS, "cross_attention": ["efficient", "cudnn"]}
    export = exporter.Exporter(tmp_path, {"another_package"}, kernels)
    x = exporter.torch.tensor([1.0])
    assert exporter.torch.equal(export("probe", exporter.torch.nn.Identity(), (x,)), x)
    assert chosen == [[exporter.SDPBackend.EFFICIENT_ATTENTION, exporter.SDPBackend.CUDNN_ATTENTION]]


@pytest.mark.parametrize("name", ["torch_inductor", "triton"])
def test_export_binds_the_selected_rope_to_its_block_without_global_patching(exporter, monkeypatch, name):
    from wan_dit_lab.models.rope import rope_apply as original
    from wan_dit_lab.kernels.rope_triton import rope_apply as triton

    template = exporter.torch.nn.Module()
    template.self_attn = exporter.torch.nn.Identity()
    monkeypatch.setattr(exporter.w4a4, "slot", lambda row, like, layout: copy.deepcopy(like))
    bound = exporter.Bound(template, [], 1, rope=name)
    assert bound.block.self_attn.rope_kernel is (triton if name == "triton" else original)
    assert getattr(bound.block.self_attn, "rope_input_dtype", None) == (exporter.torch.float32 if name == "triton" else None)
    assert not hasattr(template.self_attn, "rope_kernel")
    assert exporter.w4a4.rope_apply is original


@pytest.mark.parametrize("fused", [False, True])
def test_attention_calls_its_explicit_rope_for_both_q_and_k(fused):
    import torch
    from wan_dit_lab.models.multi_head_attention import MultiHeadAttention
    from wan_dit_lab.quantization.w4a4 import FusedQKVAttention

    attention = (FusedQKVAttention if fused else MultiHeadAttention)(12, 2)
    if fused:
        attention.qkv = torch.nn.Linear(12, 36)
    calls = []
    def rotate(x, grids, freq, start):
        calls.append((x.shape, grids, freq, start))
        return x
    attention.rope_kernel = rotate
    x = torch.randn(1, 4, 12)
    freq = torch.randn(2, 8, 3)
    result = attention(x, grid_sizes=((1, 2, 2),), freq=freq, start_frame=3)
    assert result.shape == x.shape
    assert len(calls) == 2
    assert all(shape == (1, 4, 2, 6) and grid == ((1, 2, 2),) and f is freq and start == 3
               for shape, grid, f, start in calls)


@pytest.mark.parametrize("fused", [False, True])
def test_rope_can_keep_normalized_q_and_k_in_fp32_until_sage_quantization(fused):
    import torch
    from wan_dit_lab.models.multi_head_attention import MultiHeadAttention
    from wan_dit_lab.quantization.w4a4 import FusedQKVAttention

    attention = (FusedQKVAttention if fused else MultiHeadAttention)(12, 2).to(torch.bfloat16)
    if fused:
        attention.qkv = torch.nn.Linear(12, 36).to(torch.bfloat16)
    attention.rope_input_dtype = torch.float32
    attention.rope_kernel = lambda x, *args: x
    observed = []
    class Cache:
        def attend(self, q, k, v):
            observed.append((q.dtype, k.dtype, v.dtype))
            return v
    attention(torch.randn(1, 4, 12, dtype=torch.bfloat16), grid_sizes=((1, 2, 2),), cache=Cache())
    assert observed == [(torch.float32, torch.float32, torch.bfloat16)]


@pytest.mark.gpu
def test_selected_triton_rope_is_packaged_and_runs_as_compiled_gpu_code(exporter, monkeypatch, tmp_path):
    import zipfile
    import torch
    from wan_dit_lab.models.rope import rope, rope_apply

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    class Rotate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = torch.nn.Identity()

        def forward(self, x, frequencies):
            return self.self_attn.rope_kernel(x, ((2, 2, 3),), frequencies, 7)

    monkeypatch.setattr(exporter.w4a4, "slot", lambda row, like, layout: copy.deepcopy(like))
    model = exporter.Bound(Rotate(), [], 1, rope="triton").block
    x = torch.randn(1, 12, 2, 6, device="cuda", dtype=torch.bfloat16)
    frequencies = torch.cat([rope(16, 2)] * 3, dim=-1).float().cuda()
    export = exporter.Exporter(tmp_path, None, {**KERNELS, "rope": "triton"})
    eager = export("rope_probe", model, (x, frequencies))
    with torch.no_grad():
        torch.testing.assert_close(eager, rope_apply(x, ((2, 2, 3),), frequencies, 7), rtol=0, atol=0)
    # Exporter already loaded and executed the package; Windows keeps that DLL mapped until process exit.
    assert export.report["rope_probe"]["relative_l2"] == [0.0]
    with zipfile.ZipFile(tmp_path / "rope_probe.pt2") as package:
        assert any(b"_rope_kernel" in package.read(name) for name in package.namelist()), "handwritten GPU kernel missing"


@pytest.mark.gpu
def test_fp8_block_with_original_rope_and_fp8_sdpa_cache(exporter, tmp_path):
    from wan_dit_lab.models.block import WanBlock
    from wan_dit_lab.models.rope import rope
    from wan_dit_lab.quantization.weight_pack import build_fp8, mapped_fp8
    torch = exporter.torch
    torch.manual_seed(23)
    block = WanBlock(96, 192, 2, 1e-6).bfloat16().eval()
    source, pack = tmp_path / "source", tmp_path / "block.fp8"
    source.write_bytes(b"combined FP8 block probe")
    build_fp8([block], pack, source)
    rows, layout = mapped_fp8([block], pack, source)
    bound = exporter.Bound(block, layout, rows.shape[1], linear="torch_fp8", attention="torch_sdpa",
                           cross_attention=["math"], rope="longlive_triton")
    freq = torch.cat([rope(160, 16)] * 3, -1).float().cuda()
    step = exporter.BlockStep(bound, 0, (1, 2, 80), freq[:, :, 8:].contiguous())
    x = torch.randn(1, 160, 96, device="cuda", dtype=torch.bfloat16)
    e0 = torch.randn(1, 6, 96, device="cuda", dtype=torch.bfloat16)
    cross_k, cross_v = [torch.randn(1, 16, 2, 48, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    k, v = [torch.zeros(1, 168, 2, 48, device="cuda", dtype=torch.float8_e4m3fn) for _ in range(2)]
    scales = torch.ones(1, 168, 2, 1, device="cuda")
    kernels = {**KERNELS, "attention": "torch_sdpa", "kv_cache": "fp8_e4m3fn", "kv_store": "triton",
               "linear": "torch_fp8", "cross_attention": ["math"], "rope": "longlive_triton"}
    export = exporter.Exporter(tmp_path, None, kernels)
    export("combined_fp8_block", step, (rows[0].cuda(), x, e0, cross_k, cross_v,
           freq[:, :1, :8].contiguous(), k, scales, v, scales.clone()))
