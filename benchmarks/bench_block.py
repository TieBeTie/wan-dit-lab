"""Baseline of the reference WanAttentionBlock: 32760 tokens, FP16, one row in results.jsonl labelled `reference`."""

import sys
from pathlib import Path

import torch

from wan_dit_lab.models.block import WanBlock
from wan_dit_lab.models.checkpoint import load_block_weights, model_config
from wan_dit_lab.harness import run
from wan_dit_lab.models.rope import rope

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference" / "INT8-DiT" / "wan_block"))
from wan.modules.model import WanAttentionBlock, rope_params

# 1. build the reference block with the weights of block 0 and put it on the GPU in eval mode
# 2. build the inputs for the grid (21, 30, 52): x, time input e, rope freqs, context
# 3. wrap one forward call into a function without arguments of its own, under torch.no_grad() and FP16 autocast
# 4. call run("reference", f, x, e, ..., warmup=..., number=..., repeat=...)


if __name__ == "__main__":
    config = model_config()
    mine = WanBlock(dim=config["dim"], ffn_dim=config["ffn_dim"], num_heads=config["num_heads"], eps=config["eps"])
    reference = WanAttentionBlock("t2v_cross_attn", config["dim"], config["ffn_dim"], config["num_heads"], cross_attn_norm=True, eps=config["eps"])
    weights = load_block_weights(0)
    mine.load_state_dict(weights)
    reference.load_state_dict(weights)
    device = torch.device("cuda")
    dim = config["dim"]
    head_dim = dim // config["num_heads"]
    grid = torch.tensor([[21, 30, 52]])
    length = 21 * 30 * 52  # 32760 tokens
    X = torch.randn(1, length, dim, device=device)
    time_in = torch.randn(1, 6, dim, device=device)  # six vectors: shift, scale, gate for attention and for the feed forward
    context = torch.randn(1, config["text_len"], dim, device=device)  # text tokens of Wan
    seq_lens = torch.tensor([length])
    freqs = torch.cat([rope_params(1024, head_dim - 4 * (head_dim // 6)), rope_params(1024, 2 * (head_dim // 6)), rope_params(1024, 2 * (head_dim // 6))], dim=1).to(device)
    mine.eval()
    reference.eval()
    mine.to(device)
    reference.to(device)
    mine_compiled = torch.compile(mine, mode="reduce-overhead")
    reference_compiled = torch.compile(reference, mode="reduce-overhead")
    my_freqs = torch.cat([rope(1024, head_dim - 4 * (head_dim // 6)), rope(1024, 2 * (head_dim // 6)), rope(1024, 2 * (head_dim // 6))], dim=-1).to(device)
    ref_args = (X, time_in, seq_lens, grid, freqs, context, None)
    mine_args = (X, time_in, seq_lens, grid, my_freqs, context, None)
    settings = dict(warmup=5, number=10, repeat=5)  # warmup also pays for compilation and the CUDA graph recording
    with torch.no_grad():
        with torch.autocast(device.type, dtype=torch.float16):
            run("block_reference", reference, *ref_args, **settings)
            run("block_mine", mine, *mine_args, **settings)
            run("block_reference_compiled", reference_compiled, *ref_args, **settings)
            run("block_mine_compiled", mine_compiled, *mine_args, **settings)
