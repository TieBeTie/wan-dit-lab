import pytest
import torch
from safetensors.torch import save_file

from wan_dit_lab import checkpoint


def test_block_loader_selects_only_requested_block(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text('{"num_layers": 2}', encoding="utf-8")
    save_file(
        {
            "blocks.0.ffn.0.weight": torch.ones(2, 3),
            "blocks.1.ffn.0.weight": torch.zeros(2, 3),
            "head.weight": torch.ones(1),
        },
        str(tmp_path / "diffusion_pytorch_model.safetensors"),
    )
    monkeypatch.setattr(checkpoint, "checkpoint_dir", lambda: tmp_path)
    weights = checkpoint.load_block_weights(0)
    assert list(weights) == ["ffn.0.weight"]
    torch.testing.assert_close(weights["ffn.0.weight"], torch.ones(2, 3))
    with pytest.raises(ValueError):
        checkpoint.load_block_weights(2)
