"""Read original weights without implementing or executing the upstream model."""

import json
from pathlib import Path

from safetensors import safe_open
import math

ROOT = Path(__file__).resolve().parents[2]


def checkpoint_dir():
    spec = json.loads((ROOT / "configs/upstream.json").read_text(encoding="utf-8"))
    return ROOT / spec["local_dir"]


def model_config():
    return json.loads((checkpoint_dir() / "config.json").read_text(encoding="utf-8"))


def load_block_weights(index=0):
    """Return one block's original state_dict on CPU with 'blocks.N.' removed.

    Intended for editable source checkouts. Shape/key compatibility must be checked
    with strict=True when loading into the student's own implementation.
    """
    if not isinstance(index, int) or isinstance(index, bool):
        raise TypeError("block index must be an integer")
    if not 0 <= index < model_config()["num_layers"]:
        raise ValueError("block index is outside the configured model")
    prefix = f"blocks.{index}."
    path = checkpoint_dir() / "diffusion_pytorch_model.safetensors"
    with safe_open(path, framework="pt", device="cpu") as weights:
        result = {
            name.removeprefix(prefix): weights.get_tensor(name)
            for name in weights.keys()
            if name.startswith(prefix)
        }
    if not result:
        raise ValueError(f"No tensors found for {prefix}")
    return result


def get_weight_form():
    """Return {name: (shape, dtype)} for every tensor without loading the weights."""
    path = checkpoint_dir() / "diffusion_pytorch_model.safetensors"
    result = {}
    with safe_open(path, framework="pt", device="cpu") as weights:
        for name in weights.keys():
            tensor = weights.get_slice(name)
            result[name] = (tensor.get_shape(), tensor.get_dtype())
    return result


if __name__ == "__main__":
    sum = 0
    for name, (shape, dtype) in get_weight_form().items():
        print(name, shape, dtype)
        sum += math.prod(shape)
    print(sum)
