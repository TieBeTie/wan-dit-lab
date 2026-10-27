import sys
from pathlib import Path

import torch

from wan_dit_lab.models.checkpoint import load_weights, mapped_weights, model_config
from wan_dit_lab.models.model import WanModel
from wan_dit_lab.offloading.offload import BlockOffloader
from wan_dit_lab.models.rope import rope

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference" / "INT8-DiT" / "wan_block"))
from wan.modules.model import WanModel as ReferenceWanModel

KEYS = ["dim", "eps", "ffn_dim", "freq_dim", "in_dim", "num_heads", "num_layers", "out_dim", "text_len"]


def test_model_matches_reference():
    device = torch.device("cuda")
    config = model_config()
    d = config["dim"] // config["num_heads"]

    torch.manual_seed(0)
    x = torch.randn(config["in_dim"], 2, 8, 8, device=device)
    t = torch.tensor([500.0], device=device)
    context = torch.randn(1, config["text_len"], 4096, device=device)
    grid = (2, 4, 4)
    length = grid[0] * grid[1] * grid[2]

    # Mine is built on the meta device over the checkpoint file mapped read-only: its blocks stay in the file and
    # pass through two pinned buffers and two GPU slots, the rest goes to the GPU.
    # The reference is built straight on the GPU and filled one tensor at a time. One model at a time:
    # autocast keeps no half copy of the weights, 5.3 GB of float32 weights and 2.65 GB of their copies
    # do not fit into 8 GB. Mine runs first: a crash in it shows before the reference is built.
    with torch.device("meta"):
        mine = WanModel(**{key: config[key] for key in KEYS})
    mine.load_state_dict(mapped_weights(), assign=True)
    mine.eval()
    mine.block_offloader = BlockOffloader(mine.blocks, 2, device, ram_window=2)
    mine.to(device)
    freqs = torch.cat([rope(1024, d - 4 * (d // 6)), rope(1024, 2 * (d // 6)), rope(1024, 2 * (d // 6))], dim=-1).to(device)
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, cache_enabled=False):
        actual = mine(x[None], t, context, torch.tensor([grid]), freqs)[0].float().cpu()
    mine.block_offloader.stop()
    del mine
    torch.cuda.empty_cache()

    with torch.device(device):
        reference = ReferenceWanModel(**{key: config[key] for key in KEYS}, model_type="t2v")
    load_weights(reference)
    reference.eval()
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, cache_enabled=False):
        expected = reference([x], t, [context[0]], length)[0].float().cpu()

    torch.testing.assert_close(actual, expected)
