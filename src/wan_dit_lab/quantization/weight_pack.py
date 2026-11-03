"""Immutable block weights in their inference dtype, mapped read-only.

Conversion happens once, outside measurement. Runtime retains only mapped views
and its existing bounded pinned slots, rather than a full private BF16 replica.
"""
import hashlib
import json
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import torch


def layout(block):
    return [[name, list(p.shape)] for name, p in block.named_parameters()]


def source_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def build(blocks, path, source):
    path = Path(path)
    manifest = path.with_suffix(path.suffix + ".json")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if any(p.exists() for p in (path, manifest, temporary)):
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = layout(blocks[0])
    if any(layout(block) != schema for block in blocks):
        raise ValueError("Packed blocks must share one parameter layout")
    size = sum(p.numel() * 2 for block in blocks for p in block.parameters())
    if shutil.disk_usage(path.parent).free < size + 3 * 2**30:
        raise OSError("Insufficient disk space for an immutable BF16 weight pack")
    digest = hashlib.sha256()
    identity = source_identity(source)
    try:
        with temporary.open("xb") as stream, torch.no_grad():
            for block in blocks:
                for parameter in block.parameters():
                    data = parameter.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
                    payload = data.reshape(-1).view(torch.uint8).numpy().tobytes()
                    digest.update(payload)
                    stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if source_identity(source) != identity:
            raise RuntimeError("Checkpoint changed during conversion")
        if temporary.stat().st_size != size:
            raise OSError("Incomplete BF16 weight pack")
        os.replace(temporary, path)
        metadata = dict(format="block_bf16_v1", blocks=len(blocks), layout=schema,
                        bytes=size, sha256=digest.hexdigest(), source=identity)
        manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata
    except BaseException:
        temporary.unlink(missing_ok=True)
        path.unlink(missing_ok=True)
        manifest.unlink(missing_ok=True)
        raise


def mapped(blocks, path, source):
    path = Path(path)
    metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
    if metadata["format"] != "block_bf16_v1" or metadata["source"] != source_identity(source):
        raise ValueError("Weight pack has a different checkpoint or format")
    if metadata["blocks"] != len(blocks) or any(layout(b) != metadata["layout"] for b in blocks):
        raise ValueError("Weight pack has a different parameter layout")
    size = sum(p.numel() * 2 for b in blocks for p in b.parameters())
    if path.stat().st_size != size or metadata["bytes"] != size:
        raise ValueError("Incomplete BF16 weight pack")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return torch.from_numpy(data).view(torch.bfloat16).reshape(len(blocks), -1)


def mapped_rows(blocks, path, source):
    """Dense BF16 pack through the same byte-row interface as quantized packs.

    Reuse mapped() for provenance and shape checks. The native row scheduler
    deals only in bytes and does not need a separate BF16 offloading algorithm.
    """
    tensors = mapped(blocks, path, source)
    entries, offset = [], 0
    for name, parameter in blocks[0].named_parameters():
        size = parameter.numel() * 2
        entries.append(dict(name=name, dtype='bfloat16', shape=list(parameter.shape), offset=offset, bytes=size))
        offset += size
    return tensors.view(torch.uint8).reshape(len(blocks), offset), entries


def dense_slot(row, like, entries):
    """Structure-only module bound to unquantized row views; no private copy."""
    import copy
    module = copy.deepcopy(like).to('meta').requires_grad_(False)
    for entry in entries:
        parent, _, name = entry['name'].rpartition('.')
        owner = module.get_submodule(parent) if parent else module
        value = row[entry['offset']:entry['offset'] + entry['bytes']].view(torch.bfloat16).reshape(entry['shape'])
        setattr(owner, name, torch.nn.Parameter(value, requires_grad=False))
    return module


def prepare_dense_weights(recipe, root):
    """Build missing dense rows before export identity is computed, without CUDA.

    Check space for every requested pack together before writing the first one.
    Existing packs are validated by mapped_rows when the exporter opens them.
    """
    from wan_dit_lab.models.checkpoint import mapped_torch_checkpoint, self_forcing_weights
    from wan_dit_lab.models.model import WanModel
    root, requests = Path(root), []
    spec = recipe["model"]
    path = root / spec["block_weights"]
    if recipe["kernels"]["linear"] == "torch_bf16" and not path.exists():
        config = json.loads((root / spec["architecture"]).read_text(encoding="utf-8"))
        keys = ("dim", "eps", "ffn_dim", "freq_dim", "in_dim", "num_heads", "num_layers", "out_dim", "text_len")
        with torch.device("meta"):
            model = WanModel(**{k: config[k] for k in keys})
        source = root / spec["checkpoint"]
        model.load_state_dict(self_forcing_weights(source), assign=True)
        requests.append((model.blocks, path, source))
    text = spec["text_encoder"]
    path = root / text["block_weights"]
    if text.get("precision", "int8") == "bfloat16" and not path.exists():
        from wan_dit_lab.models.text_encoder import umt5_xxl
        encoder = umt5_xxl(encoder_only=True, dtype=torch.bfloat16, device="meta")
        source = root / text["checkpoint"]
        encoder.load_state_dict(mapped_torch_checkpoint(source), assign=True)
        requests.append((encoder.blocks, path, source))
    volumes = {}
    for blocks, path, _ in requests:
        volume = path.resolve().anchor
        volumes[volume] = volumes.get(volume, 0) + sum(p.numel() * 2 for b in blocks for p in b.parameters())
    for volume, size in volumes.items():
        needed, free = size + 3 * 2**30, shutil.disk_usage(volume).free
        if free < needed:
            raise OSError(f"Insufficient disk space for dense packs: need {needed / 2**30:.2f} GiB, "
                          f"free {free / 2**30:.2f} GiB on {volume}")
    for blocks, path, source in requests:
        build(blocks, path, source)


def fp8_layout(block):
    """Where each tensor of a block lies in its block_fp8_v1 row: the weight of every nn.Linear in FP8 with one
    float32 scale per tensor (fp8.quantize), every other parameter in bfloat16; each entry 16-byte aligned."""
    linears = {n for n, m in block.named_modules() if isinstance(m, torch.nn.Linear)}
    layout, offset = [], 0

    def add(name, dtype, shape, size):
        nonlocal offset
        layout.append(dict(name=name, dtype=dtype, shape=list(shape), offset=offset, bytes=size))
        offset += -(-size // 16) * 16

    for name, p in block.named_parameters():
        if name.endswith(".weight") and name[:-len(".weight")] in linears:
            add(name, "fp8", p.shape, p.numel())
            add(name[:-len(".weight")] + ".scale", "f32", (), 4)
        else:
            add(name, "bf16", p.shape, p.numel() * 2)
    return layout, offset


def build_fp8(blocks, path, source):
    """The blocks as a block_fp8_v1 pack: one uint8 row per block (fp8_layout), mapped read-only by mapped_fp8."""
    from wan_dit_lab.quantization.fp8 import quantize
    path = Path(path)
    manifest, temporary = path.with_suffix(path.suffix + ".json"), path.with_suffix(path.suffix + ".tmp")
    if any(p.exists() for p in (path, manifest, temporary)):
        raise FileExistsError(path)
    layout, size = fp8_layout(blocks[0])
    if any(fp8_layout(b) != (layout, size) for b in blocks):
        raise ValueError("Packed blocks must share one layout")
    if shutil.disk_usage(path.parent).free < size * len(blocks) + 3 * 2**30:
        raise OSError("Insufficient disk space for an FP8 weight pack")
    digest, identity = hashlib.sha256(), source_identity(source)
    try:
        with temporary.open("xb") as stream, torch.no_grad():
            for block in blocks:
                row = torch.zeros(size, dtype=torch.uint8)
                params = dict(block.named_parameters())
                for e in layout:
                    if e["dtype"] == "f32":
                        continue
                    p = params[e["name"]].detach().to("cuda" if torch.cuda.is_available() else "cpu")
                    if e["dtype"] == "fp8":
                        q, scale = quantize(p.float())
                        data = q.cpu().reshape(-1).view(torch.uint8)
                        s = next(x for x in layout if x["name"] == e["name"][:-len(".weight")] + ".scale")
                        row[s["offset"]:s["offset"] + 4] = scale.float().cpu().reshape(1).view(torch.uint8)
                    else:
                        data = p.to(torch.bfloat16).cpu().reshape(-1).view(torch.uint8)
                    row[e["offset"]:e["offset"] + e["bytes"]] = data
                payload = row.numpy().tobytes()
                digest.update(payload)
                stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if source_identity(source) != identity:
            raise RuntimeError("Checkpoint changed during conversion")
        os.replace(temporary, path)
        metadata = dict(format="block_fp8_v1", blocks=len(blocks), layout=layout, bytes_per_block=size,
                        bytes=size * len(blocks), sha256=digest.hexdigest(), source=identity,
                        quantization="fp8 e4m3, one scale per weight tensor; activations per tensor at run time")
        manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return {k: v for k, v in metadata.items() if k != "layout"}
    except BaseException:
        temporary.unlink(missing_ok=True)
        path.unlink(missing_ok=True)
        manifest.unlink(missing_ok=True)
        raise


def pack_format(path):
    path = Path(path)
    return json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))["format"]


def mapped_fp8(blocks, path, source):
    """A block_fp8_v1 pack mapped read-only: uint8 [blocks, bytes per block], and its layout."""
    path = Path(path)
    metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
    if metadata["format"] != "block_fp8_v1" or metadata["source"] != source_identity(source):
        raise ValueError("Weight pack has a different checkpoint or format")
    layout, size = fp8_layout(blocks[0])
    if metadata["blocks"] != len(blocks) or metadata["layout"] != layout or path.stat().st_size != size * len(blocks):
        raise ValueError("FP8 weight pack has a different layout or is incomplete")
    data = np.memmap(path, dtype=np.uint8, mode="r")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return torch.from_numpy(data).reshape(len(blocks), size), layout


if __name__ == "__main__":
    import argparse

    from wan_dit_lab.models.checkpoint import SELF_FORCING, model_config, self_forcing_weights
    from wan_dit_lab.models.model import WanModel
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", type=Path, default=SELF_FORCING, help="a checkpoint of the Self-Forcing format")
    parser.add_argument("--model-config", type=Path, default=None, help="config.json of another Wan model (Wan2.2-TI2V-5B)")
    parser.add_argument("--fp8", action="store_true", help="a block_fp8_v1 pack: linear weights in FP8 (build_fp8)")
    args = parser.parse_args()
    spec = json.loads(args.model_config.read_text(encoding="utf-8")) if args.model_config else model_config()
    keys = ("dim", "eps", "ffn_dim", "freq_dim", "in_dim", "num_heads", "num_layers",
            "out_dim", "text_len")
    with torch.device("meta"):
        model = WanModel(**{key: spec[key] for key in keys})
    model.load_state_dict(self_forcing_weights(args.checkpoint), assign=True)
    print(json.dumps((build_fp8 if args.fp8 else build)(model.blocks, args.output, args.checkpoint)), flush=True)
