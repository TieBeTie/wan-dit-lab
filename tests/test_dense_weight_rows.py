import torch
import pytest
from wan_dit_lab.quantization.weight_pack import build, mapped_rows, dense_slot


def test_dense_rows_rebind_real_bf16_parameters_without_quantization(tmp_path):
    torch.manual_seed(17)
    blocks = [torch.nn.Linear(4, 3).to(torch.bfloat16) for _ in range(2)]
    checkpoint = tmp_path / 'source.pt'
    checkpoint.write_bytes(b'source identity')
    path = tmp_path / 'weights.bf16'
    # The tiny test does not need a multi-GiB reservation.
    import unittest.mock
    with unittest.mock.patch('shutil.disk_usage', return_value=type('Space', (), {'free': 10**12})()):
        build(blocks, path, checkpoint)
    rows, layout = mapped_rows(blocks, path, checkpoint)
    assert rows.dtype == torch.uint8
    x = torch.randn(2, 4, dtype=torch.bfloat16)
    for index in range(2):
        slot = dense_slot(rows[index], blocks[0], layout)
        assert torch.equal(slot(x), blocks[index](x))
        assert all(p.dtype == torch.bfloat16 for p in slot.parameters())
    assert not torch.equal(blocks[0].weight, blocks[1].weight)


def test_dense_rows_refuse_another_checkpoint(tmp_path):
    blocks = [torch.nn.Linear(4, 3)]
    path = tmp_path / 'weights.bf16'
    checkpoint = tmp_path / 'source.pt'
    checkpoint.write_bytes(b'a')
    import unittest.mock
    with unittest.mock.patch('shutil.disk_usage', return_value=type('Space', (), {'free': 10**12})()):
        build(blocks, path, checkpoint)
    checkpoint.write_bytes(b'changed')
    with pytest.raises(ValueError, match='checkpoint'):
        mapped_rows(blocks, path, checkpoint)


@pytest.mark.parametrize("free", [10**12, 1])
def test_preparation_builds_missing_dense_packs_or_refuses_before_writing(monkeypatch, tmp_path, free):
    import json
    import shutil
    from types import SimpleNamespace
    from wan_dit_lab.quantization.weight_pack import prepare_dense_weights
    from wan_dit_lab.models import text_encoder
    from wan_dit_lab.models.model import WanModel
    spec = dict(dim=24, eps=1e-6, ffn_dim=48, freq_dim=8, in_dim=4, num_heads=2, num_layers=1,
                out_dim=4, text_len=512)
    model = WanModel(**spec)
    tiny_t5 = torch.nn.Module()
    tiny_t5.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4).to(torch.bfloat16)])
    torch.save({"generator": model.state_dict()}, tmp_path / "dit.pt")
    torch.save(tiny_t5.state_dict(), tmp_path / "t5.pt")
    (tmp_path / "architecture.json").write_text(json.dumps(spec))
    monkeypatch.setattr(text_encoder, "umt5_xxl", lambda **kw: __import__('copy').deepcopy(tiny_t5).to("meta"))
    recipe = {"kernels": {"linear": "torch_bf16"}, "model": {
        "architecture": "architecture.json", "checkpoint": "dit.pt", "block_weights": "dit.bf16",
        "text_encoder": {"checkpoint": "t5.pt", "block_weights": "t5.bf16", "precision": "bfloat16"}}}
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=free))
    if free == 1:
        with pytest.raises(OSError, match="disk space"):
            prepare_dense_weights(recipe, tmp_path)
        assert not (tmp_path / "dit.bf16").exists() and not (tmp_path / "t5.bf16").exists()
    else:
        prepare_dense_weights(recipe, tmp_path)
        rows, entries = mapped_rows(model.blocks, tmp_path / "dit.bf16", tmp_path / "dit.pt")
        assert rows.shape[0] == 1
        assert torch.equal(dense_slot(rows[0], model.blocks[0], entries).modulation,
                           model.blocks[0].modulation.to(torch.bfloat16))
        mapped_rows(tiny_t5.blocks, tmp_path / "t5.bf16", tmp_path / "t5.pt")
        # Existing packs are immutable; repeated preparation must not rewrite them.
        before = [(tmp_path / name).stat().st_mtime_ns for name in ("dit.bf16", "t5.bf16")]
        prepare_dense_weights(recipe, tmp_path)
        assert before == [(tmp_path / name).stat().st_mtime_ns for name in ("dit.bf16", "t5.bf16")]
