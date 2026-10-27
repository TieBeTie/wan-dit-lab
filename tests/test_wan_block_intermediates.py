import sys
from pathlib import Path

import pytest
import torch

from wan_dit_lab.models.block import WanBlock
from wan_dit_lab.models.checkpoint import load_block_weights, model_config
from wan_dit_lab.models.rope import rope

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference" / "INT8-DiT" / "wan_block"))
from wan.modules.model import WanAttentionBlock, rope_params


@pytest.mark.parametrize("index", [0, 15, 29])
@pytest.mark.parametrize("part", ["self_attn", "cross_attn", "ffn"])
def test_block_part_matches_reference(index, part):
    device = torch.device("cuda")
    config = model_config()
    weights = load_block_weights(index)
    mine = WanBlock(dim=config["dim"], ffn_dim=config["ffn_dim"], num_heads=config["num_heads"], eps=config["eps"])
    reference = WanAttentionBlock("t2v_cross_attn", config["dim"], config["ffn_dim"], config["num_heads"], cross_attn_norm=True, eps=config["eps"])
    mine.load_state_dict(weights, strict=True)
    reference.load_state_dict(weights, strict=True)
    mine.to(device).eval()
    reference.to(device).eval()

    dim, d = config["dim"], config["dim"] // config["num_heads"]
    grid = (2, 4, 4)
    length = grid[0] * grid[1] * grid[2]
    sizes = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
    freqs = torch.cat([rope_params(1024, s) for s in sizes], dim=1).to(device)
    my_freqs = torch.cat([rope(1024, s) for s in sizes], dim=-1).to(device)
    torch.manual_seed(index)
    x = torch.randn(1, length, dim, device=device)
    e = torch.randn(1, 6, dim, device=device)
    context = torch.randn(1, 16, dim, device=device)

    outputs = {"mine": [], "reference": []}
    getattr(mine, part).register_forward_hook(lambda module, args, out: outputs["mine"].append(out.detach().float()))
    getattr(reference, part).register_forward_hook(lambda module, args, out: outputs["reference"].append(out.detach().float()))
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16):
        reference(x, e, torch.tensor([length]), torch.tensor([grid]), freqs, context, None)
        mine(x, e, torch.tensor([length]), torch.tensor([grid]), my_freqs, context, None)
    torch.testing.assert_close(outputs["mine"][0], outputs["reference"][0])
