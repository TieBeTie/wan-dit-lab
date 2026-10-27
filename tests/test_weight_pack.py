import copy
import json

import pytest
import torch

from wan_dit_lab.quantization.weight_pack import build, mapped


def test_readonly_pack_matches_inference_cast_and_preserves_source(tmp_path):
    blocks = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(3)])
    original = copy.deepcopy(blocks.state_dict())
    source = tmp_path / "checkpoint"
    source.write_bytes(b"checkpoint")
    path = tmp_path / "weights.bf16"
    build(blocks, path, source)
    result = mapped(blocks, path, source)
    for index, block in enumerate(blocks):
        expected = torch.cat([p.detach().flatten() for p in block.parameters()]).bfloat16()
        assert torch.equal(result[index], expected)
    assert all(torch.equal(value, original[key]) for key, value in blocks.state_dict().items())
    assert result.dtype == torch.bfloat16
    with pytest.raises(FileExistsError):
        build(blocks, path, source)


@pytest.mark.parametrize("failure", ["source", "layout", "size"])
def test_stale_or_incompatible_pack_is_rejected(tmp_path, failure):
    blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
    source, path = tmp_path / "source", tmp_path / "weights.bf16"
    source.write_bytes(b"source")
    build(blocks, path, source)
    if failure == "source":
        source.write_bytes(b"changed")
    elif failure == "layout":
        blocks = torch.nn.ModuleList([torch.nn.Linear(3, 2)])
    else:
        path.write_bytes(b"incomplete")
    with pytest.raises(ValueError):
        mapped(blocks, path, source)


def test_failed_pack_does_not_leave_a_partial_artifact(tmp_path, monkeypatch):
    blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
    source, path = tmp_path / "source", tmp_path / "weights.bf16"
    source.write_bytes(b"source")
    def fail(*args):
        raise OSError("injected replacement failure")
    monkeypatch.setattr("wan_dit_lab.quantization.weight_pack.os.replace", fail)
    with pytest.raises(OSError, match="injected"):
        build(blocks, path, source)
    assert list(tmp_path.iterdir()) == [source]


@pytest.mark.gpu
def test_packed_and_cast_on_demand_slots_are_bitwise_equal(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from wan_dit_lab.offloading.offload import BlockOffloader
    blocks = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(3)])
    source, path = tmp_path / "source", tmp_path / "weights.bf16"
    source.write_bytes(b"source")
    build(blocks, path, source)
    packed = mapped(blocks, path, source)
    outputs = []
    with torch.no_grad():
        for value in [None, packed]:
            offloader = BlockOffloader(blocks, 2, "cuda", ram_window=1,
                                      dtype=torch.bfloat16, packed=value)
            try:
                offloader.start()
                x = torch.ones(1, 8, device="cuda", dtype=torch.bfloat16)
                for _ in range(9):
                    block = offloader.get_next()
                    x = block(x)
                    offloader.release(block)
                outputs.append(x.cpu())
            finally:
                offloader.close()
    assert torch.equal(*outputs)
