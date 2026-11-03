"""Explicit kernel contract shared by the run configuration and its AOT export.

Only implementations the exporter actually uses are accepted. Adding a name here requires
implementing its export path and matching validation in crates/wan/src/config.rs.
"""

import json
from pathlib import Path

SUPPORTED = {
    "attention": ("sage_sm89", "sage_rope_sm89", "torch_sdpa"),
    "linear": ("nunchaku_w4a4", "torch_bf16", "torch_fp8"),
    "kv_cache": ("fp8_e4m3fn", "bfloat16"),
    "kv_store": ("triton", "torch"),
    "rope": ("torch_inductor", "triton", "longlive_triton"),
    "rope_dtype": ("float32",),
}
CROSS_BACKENDS = {"cudnn": "CUDNN_ATTENTION", "efficient": "EFFICIENT_ATTENTION", "math": "MATH"}


def validate_kernels(kernels):
    """Return an independent JSON snapshot, refusing omissions, typos and unsupported choices."""
    if not isinstance(kernels, dict):
        raise ValueError("kernels must be an explicit object")
    expected = set(SUPPORTED) | {"cross_attention"}
    joint = kernels.get("attention") == "sage_rope_sm89"
    if joint:  # owns its RoPE; "longlive_triton" only names the rounding to reproduce
        expected.remove("rope")
        if kernels.get("rope", "longlive_triton") != "longlive_triton":
            raise ValueError("kernels.rope: sage_rope_sm89 takes none or longlive_triton")
    missing, unknown = expected - kernels.keys(), kernels.keys() - expected - ({"rope"} if joint else set())
    if missing or unknown:
        raise ValueError(f"kernels: missing {sorted(missing)}, unknown {sorted(unknown)}")
    for field, supported in SUPPORTED.items():
        if joint and field == "rope":
            continue
        if kernels[field] not in supported:
            raise ValueError(f"kernels.{field}: {kernels[field]!r} is unsupported; export implements {supported!r}")
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
    previous = directory / "live.json"
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
    (directory / "live.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (directory / ".exporting").unlink()
