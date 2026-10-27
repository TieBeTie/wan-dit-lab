import sys
from pathlib import Path

import pytest
import torch

from wan_dit_lab.models.block import WanBlock
from wan_dit_lab.models.checkpoint import load_block_weights, model_config
from wan_dit_lab.models.rope import rope

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference" / "INT8-DiT" / "wan_block"))
from wan.modules.model import WanAttentionBlock, rope_params


@pytest.mark.parametrize("index", range(30))
def test_block_matches_reference(index):
    device = torch.device("cuda")
    config = model_config()
    weights = load_block_weights(index)
    mine = WanBlock(dim=config["dim"], ffn_dim=config["ffn_dim"], num_heads=config["num_heads"], eps=config["eps"])
    reference = WanAttentionBlock("t2v_cross_attn", config["dim"], config["ffn_dim"], config["num_heads"], cross_attn_norm=True, eps=config["eps"])
    mine.load_state_dict(weights, strict=True)
    reference.load_state_dict(weights, strict=True)
    mine.to(device)
    reference.to(device)
    mine.eval()
    reference.eval()

    dim, d = config["dim"], config["dim"] // config["num_heads"]
    grid = (2, 4, 4)
    length = grid[0] * grid[1] * grid[2]
    freqs = torch.cat([rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))], dim=1).to(device)
    my_freqs = torch.cat([rope(1024, d - 4 * (d // 6)), rope(1024, 2 * (d // 6)), rope(1024, 2 * (d // 6))], dim=-1).to(device)
    torch.manual_seed(index)
    for _ in range(3):
        x = torch.randn(1, length, dim, device=device)
        e = torch.randn(1, 6, dim, device=device)
        context = torch.randn(1, 16, dim, device=device)
        with torch.no_grad():
            with torch.autocast(x.device.type, dtype=torch.float16):
                expected = reference(x, e, torch.tensor([length]), torch.tensor([grid]), freqs, context, None)
                actual = mine(x, e, torch.tensor([length]), torch.tensor([grid]), my_freqs, context, None)
            torch.testing.assert_close(actual, expected)
        
