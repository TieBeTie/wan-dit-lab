"""rope_apply alone: reference (complex, loop over the batch) against ours (cos/sin, no loop), eager and compiled. Grid (21, 30, 52), FP16."""

import sys
from pathlib import Path

import torch

from wan_dit_lab.harness import run
from wan_dit_lab.rope import rope, rope_apply

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference" / "INT8-DiT" / "wan_block"))
from wan.modules.model import rope_apply as reference_rope_apply, rope_params

if __name__ == "__main__":
    device = torch.device("cuda")
    heads, head_dim = 12, 128
    grid = torch.tensor([[21, 30, 52]])
    length = 21 * 30 * 52
    X = torch.randn(1, length, heads, head_dim, device=device).half()
    freqs = torch.cat([rope_params(1024, head_dim - 4 * (head_dim // 6)), rope_params(1024, 2 * (head_dim // 6)), rope_params(1024, 2 * (head_dim // 6))], dim=1).to(device)
    my_freqs = torch.cat([rope(1024, head_dim - 4 * (head_dim // 6)), rope(1024, 2 * (head_dim // 6)), rope(1024, 2 * (head_dim // 6))], dim=-1).to(device)
    settings = dict(warmup=5, number=10, repeat=5)
    run("rope_apply_reference", reference_rope_apply, X, grid, freqs, **settings)
    run("rope_apply_mine", rope_apply, X, grid, my_freqs, **settings)
    run("rope_apply_mine_compiled", torch.compile(rope_apply), X, grid, my_freqs, **settings)
