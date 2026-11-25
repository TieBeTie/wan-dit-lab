"""Explicit kernel contract shared by the run configuration and its AOT export.

Only implementations the exporter actually uses are accepted. Adding a name here requires
implementing its export path and matching validation in crates/wan/src/config.rs.
"""

import json
from pathlib import Path

SUPPORTED = {
    "attention": ("sage_sm89", "torch_sdpa"),
    "linear": ("nunchaku_w4a4", "torch_bf16", "torch_fp8"),
    "kv_cache": ("fp8_e4m3fn", "bfloat16"),
    "kv_store": ("triton", "torch"),
    # fused_float32: RMSNorm and RoPE in float32 straight into INT8, which only the fused Sage kernel computes
    "rope": ("torch_inductor", "triton", "longlive_triton", "fused_float32"),
    "rope_dtype": ("float32",),
    # the W4A4 low-rank branch's sum over column blocks: Nunchaku's float atomics, or a fixed order (same bits every run)
    "lora_sum": ("atomic", "ordered"),
}
CROSS_BACKENDS = {"cudnn": "CUDNN_ATTENTION", "efficient": "EFFICIENT_ATTENTION", "math": "MATH"}
# Each fusion a switch of its own: rope_into_sage is baked into the block packages, qkv and gelu_mlp into the weight
# pack (block_w4a4_v2 has both, v1 neither). The fused Sage kernel rounds as the separate RoPE it replaces.
FUSIONS = ("rope_into_sage", "qkv", "gelu_mlp")
FUSED_ROPE = ("longlive_triton", "fused_float32")


def validate_kernels(kernels):
    """Return an independent JSON snapshot, refusing omissions, typos and unsupported choices."""
    if not isinstance(kernels, dict):
        raise ValueError("kernels must be an explicit object")
    expected = set(SUPPORTED) | {"cross_attention", "fuse"}
    missing, unknown = expected - kernels.keys(), kernels.keys() - expected - {"sparse"}
    if missing or unknown:
        raise ValueError(f"kernels: missing {sorted(missing)}, unknown {sorted(unknown)}")
    for field, supported in SUPPORTED.items():
        if kernels[field] not in supported:
            raise ValueError(f"kernels.{field}: {kernels[field]!r} is unsupported; export implements {supported!r}")
    fuse = kernels["fuse"]
    if not isinstance(fuse, dict) or set(fuse) != set(FUSIONS) or not all(isinstance(fuse[f], bool) for f in FUSIONS):
        raise ValueError(f"kernels.fuse must be {{{', '.join(FUSIONS)}}}, each true or false")
    joint = fuse["rope_into_sage"]
    if joint and kernels["attention"] != "sage_sm89":
        raise ValueError("kernels.fuse.rope_into_sage needs attention sage_sm89")
    if joint and kernels["rope"] not in FUSED_ROPE:
        raise ValueError(f"kernels.fuse.rope_into_sage rounds as {' or '.join(FUSED_ROPE)} only, not {kernels['rope']!r}")
    if not joint and kernels["rope"] == "fused_float32":
        raise ValueError("kernels.rope: fused_float32 exists only with kernels.fuse.rope_into_sage")
    if kernels["linear"] != "nunchaku_w4a4" and (fuse["qkv"] or fuse["gelu_mlp"]):
        raise ValueError("kernels.fuse.qkv/gelu_mlp: only W4A4 weight packs fuse layers")
    if kernels["linear"] != "nunchaku_w4a4" and kernels["lora_sum"] == "ordered":
        raise ValueError("kernels.lora_sum: ordered exists only for the W4A4 linear (nunchaku_w4a4)")
    if "sparse" in kernels:
        validate_sparse(kernels["sparse"], joint)
    cross = kernels["cross_attention"]
    dense = kernels["attention"] == "torch_sdpa"
    storage = (kernels["kv_cache"], kernels["kv_store"])
    if storage not in (("bfloat16", "torch"), ("fp8_e4m3fn", "triton")) or (not dense and storage != ("fp8_e4m3fn", "triton")):
        raise ValueError("kernels: choose BF16/torch or FP8/triton storage; Sage requires FP8/triton")
    if dense and cross == ["sage_sm89"]:
        raise ValueError("kernels.cross_attention: torch_sdpa takes SDPA backends for both attention paths")
    if cross != ["sage_sm89"] and (not isinstance(cross, list) or not cross
            or any(not isinstance(b, str) or b not in CROSS_BACKENDS for b in cross)
            or len(set(cross)) != len(cross)):
        raise ValueError("kernels.cross_attention must be [sage_sm89] or list cudnn/efficient once each in priority order")
    return {**kernels, "cross_attention": list(cross)}


def validate_sparse(sparse, joint):
    """kernels.sparse (crates/wan/src/contract.rs and config.rs hold the full contract): the joint kernel only;
    {"enabled": false}, or enabled with skip_mass in [0, 1) and frames {sink, recent_frames}."""
    if not joint:
        raise ValueError("kernels.sparse needs kernels.fuse.rope_into_sage")
    on = isinstance(sparse, dict) and sparse.get("enabled")
    keys = {"enabled", "skip_mass", "frames"} if on else {"enabled"}
    if not isinstance(sparse, dict) or not isinstance(sparse.get("enabled"), bool) or set(sparse) != keys:
        raise ValueError(f"kernels.sparse must have exactly the keys {sorted(keys)}")
    if on:
        mass, frames = sparse["skip_mass"], sparse["frames"]
        if isinstance(mass, bool) or not isinstance(mass, (int, float)) or not 0 <= mass < 1:
            raise ValueError("kernels.sparse.skip_mass must be in [0, 1)")
        if not isinstance(frames, dict) or set(frames) != {"sink", "recent_frames"} \
                or not isinstance(frames["sink"], bool) or not isinstance(frames["recent_frames"], int) \
                or frames["recent_frames"] < 0:
            raise ValueError("kernels.sparse.frames must be {sink: bool, recent_frames: frames >= 0}")


def validate_pack(kernels, layout):
    """The weight pack's layout must hold the layer fusions kernels.fuse names (w4a4.fusions: what its slot installs)."""
    from wan_dit_lab.quantization.w4a4 import fusions
    held = fusions(layout)
    for flag, fused in held.items():
        if kernels["fuse"][flag] != fused:
            raise ValueError(f"kernels.fuse.{flag} is {kernels['fuse'][flag]} but the weight pack "
                             f"{'fuses' if fused else 'does not fuse'} it: name the pack that matches")


def skip_mass(kernels):
    """The skip mass the packages implement, or None for dense attention."""
    sparse = kernels.get("sparse")
    return float(sparse["skip_mass"]) if sparse and sparse["enabled"] else None


def read_kernels(path):
    """Read the required kernel block without guessing from a preset or from existing packages."""
    from wan_dit_lab.runtime.live_pipeline import load_pipeline
    return load_pipeline(path)[0]["kernels"]


def validate_partial_export(manifest_path, kernels, *, identity=None):
    """A subset cannot establish provenance for the packages it leaves untouched."""
    path = Path(manifest_path)
    if not path.exists() or (path.parent / ".exporting").exists():
        raise ValueError(f"{path}: --only requires a full export with matching kernels first")
    previous = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(previous, dict) or previous.get("kernels") != kernels:
        raise ValueError(f"{path}: missing or different kernels; a full export without --only is required")
    if identity is not None and previous.get("export", {}).get("identity") != identity:
        raise ValueError(f"{path}: different model recipe or sources; a full export without --only is required")


def begin_export(directory, kernels, *, only, identity=None):
    """Block startup until all packages and their kernel manifest have been published successfully."""
    directory = Path(directory)
    kernels = validate_kernels(kernels)
    previous = directory / "mvp1.json"
    checks = {}
    if only:
        validate_partial_export(previous, kernels, identity=identity)
        checks = json.loads(previous.read_text(encoding="utf-8")).get("checks", {})
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ".exporting").write_text(json.dumps(kernels), encoding="utf-8")
    return checks


def complete_export(directory, manifest):
    """The marker stays after failures, including a failure to publish the manifest."""
    directory = Path(directory)
    (directory / "mvp1.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (directory / ".exporting").unlink()
