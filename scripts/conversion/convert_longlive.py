"""LongLive-1.3B (Efficient-Large-Model/LongLive-1.3B) as a checkpoint of the Self-Forcing format: the LoRA merged into
the base generator, saved under "generator_ema" with the names of WanModel after "model.".

LongLive puts a LoRA (rank 256, alpha 256, so scale 1) on every Linear of every CausalWanAttentionBlock
(reference/LongLive/LongLive1.0/utils/lora_utils.py, configs/longlive_interactive_inference.yaml) and loads the base
from "generator" (use_ema: false). Merged: W + (alpha / r) B A, in float32. Block weights are saved in bfloat16 (the
blocks run in bfloat16 anyway), the rest stays float32.

usage: python scripts/conversion/convert_longlive.py checkpoints/LongLive-1.3B/models checkpoints/longlive_merged.pt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


import torch

RANK, ALPHA = 256, 256


def main(source, target):
    source = Path(source)
    base = torch.load(source / "longlive_base.pt", map_location="cpu", mmap=True)["generator"]
    lora = torch.load(source / "lora.pt", map_location="cpu", mmap=True)
    lora = lora.get("generator_lora", lora)
    base = {key.removeprefix("model."): tensor for key, tensor in base.items()}
    pairs = {}
    for key, tensor in lora.items():
        name = key.removeprefix("base_model.model.").removeprefix("model.")
        for part in ("lora_A", "lora_B"):
            marker = f".{part}."
            if marker in name:
                module = name.split(marker)[0]
                pairs.setdefault(module, {})[part] = tensor
    out, merged = {}, 0
    for key, tensor in base.items():  # one tensor at a time: the float32 base is never copied whole
        tensor = tensor.float()
        ab = pairs.get(key.removesuffix(".weight")) if key.endswith(".weight") else None
        if ab is not None:
            delta = ab["lora_B"].float() @ ab["lora_A"].float()
            assert delta.shape == tensor.shape, (key, delta.shape, tensor.shape)
            tensor = tensor + (ALPHA / RANK) * delta
            merged += 1
        out["model." + key] = (tensor.to(torch.bfloat16) if key.startswith("blocks.") else tensor).contiguous()
    assert merged == len(pairs), (merged, len(pairs))
    torch.save({"generator_ema": out}, target)
    print(f"merged {merged} LoRA pairs into {len(base)} tensors -> {target}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
