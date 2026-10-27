"""Read original weights without implementing or executing the upstream model."""

import io
import json
import math
import pickle
import struct
import warnings
import zipfile
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT / "reference" / "INT8-DiT" / "wan_block"  # the reference Wan code: T5, VAE, scheduler


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
    result = {name.removeprefix(prefix): tensor.clone() for name, tensor in mapped_weights().items()
              if name.startswith(prefix)}
    if not result:
        raise ValueError(f"No tensors found for {prefix}")
    return result


def mapped_weights():
    """The checkpoint as tensors that are views of the file mapped read-only: nothing is copied into RAM, a page is
    read when it is first used, and Windows drops it again when memory runs short. safe_open maps the file
    copy-on-write instead, and Windows reserves memory for the whole of such a mapping (5.4 GiB here).

    For a model built on the meta device: model.load_state_dict(mapped_weights(), assign=True). Nothing may write
    into these tensors."""
    path = checkpoint_dir() / "diffusion_pytorch_model.safetensors"
    with open(path, "rb") as f:
        size = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(size))
    header.pop("__metadata__", None)
    data = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + size)
    dtypes = {"F32": np.float32, "F16": np.float16}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # torch warns that a read-only array gives a tensor it must not write into
        return {key: torch.from_numpy(data[slice(*v["data_offsets"])].view(dtypes[v["dtype"]]).reshape(v["shape"]))
                for key, v in header.items()}


def mapped_torch_checkpoint(path):
    """A torch.save checkpoint whose tensors are views of the file mapped read-only, as in mapped_weights.

    torch.load(mmap=True) maps the file copy-on-write, and Windows charges the whole mapping to its commit limit:
    11.4 GB for T5, and the next allocation of pinned memory fails. The file is a zip whose entries are stored
    uncompressed: data.pkl describes the tensors, data/<key> holds the bytes of each storage."""
    archive = zipfile.ZipFile(path)
    root = archive.namelist()[0].split("/")[0]
    data = np.memmap(path, dtype=np.uint8, mode="r")

    def storage(key, dtype):
        info = archive.getinfo(f"{root}/data/{key}")
        assert info.compress_type == zipfile.ZIP_STORED, info
        # the local file header: 30 bytes, then the name and the extra field, then the bytes of the entry
        name, extra = struct.unpack("<HH", data[info.header_offset + 26:info.header_offset + 30].tobytes())
        start = info.header_offset + 30 + name + extra
        return torch.from_numpy(data[start:start + info.file_size]).view(dtype)

    def rebuild(storage, offset, size, stride, *_):
        return storage.as_strided(size, stride, offset)

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
                return rebuild
            return super().find_class(module, name)

        def persistent_load(self, pid):
            _, storage_type, key, _, _ = pid  # ("storage", torch.FloatStorage, key, location, numel)
            return storage(key, storage_type.dtype)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # torch warns that a read-only array gives a tensor it must not write into
        return Unpickler(io.BytesIO(archive.read(f"{root}/data.pkl"))).load()


SELF_FORCING = REFERENCE / "checkpoints" / "self_forcing_dmd.pt"
# LongLive-1.3B with its LoRA merged, in the format of SELF_FORCING (scripts/conversion/convert_longlive.py): the weights of Live
LONGLIVE = ROOT / "checkpoints" / "longlive_merged.pt"


def self_forcing_weights(path=SELF_FORCING):
    """Wan 1.3B distilled by Self-Forcing (the homework notebook), with the names of my WanModel: the checkpoint keeps
    the EMA of the generator under "model."."""
    weights = mapped_torch_checkpoint(path)
    # LongLive 2.0's model_bf16.pt (LoRA merged) may keep the generator under another key or be the state dict itself
    for key in ("generator_ema", "generator", "state_dict", "model"):
        if key in weights and isinstance(weights[key], dict):
            weights = weights[key]
            break
    return {key.removeprefix("model."): tensor for key, tensor in weights.items()}


def load_weights(model):
    """Copies the checkpoint into the model one tensor at a time, from the file mapped read-only: the whole file never
    sits in host memory. Reading through safe_open crashed now and then with an access violation while the commit
    limit of Windows was nearly used up (its mapping is copy-on-write)."""
    own = model.state_dict()
    weights = mapped_weights()
    assert set(weights) == set(own)
    for key, tensor in weights.items():
        assert tensor.shape == own[key].shape, key
        own[key].copy_(tensor)


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
