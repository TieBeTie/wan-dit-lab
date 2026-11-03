"""Preparation of the two-stage Live pipeline. No torch import and no work during generation."""
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path

from wan_dit_lab.runtime.live_kernels import validate_kernels

ROOT = Path(__file__).resolve().parents[3]


def fields(value, required, optional=()):
    if not isinstance(value, dict):
        raise ValueError(f"expected an object with {required}")
    missing = set(required) - value.keys()
    unknown = value.keys() - set(required) - set(optional)
    if missing or unknown:
        raise ValueError(f"missing {sorted(missing)}, unknown {sorted(unknown)}")


def load_pipeline(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("config must be [export, generate]")
    export, run = value
    fields(export, ("component", "compiler", "rebuild", "output", "model", "video", "sampling", "context", "rope", "kernels"))
    fields(run, ("component", "packages_from", "mode", "prompts", "seed", "offloading"), ("output", "debug", "drain"))
    debug = run.get("debug", False)
    if type(debug) is not bool and debug != "fps" and not (
        isinstance(debug, list) and debug == ["fps"]
    ):
        raise ValueError("debug must be true, false, fps or a nonempty list of unique supported tools")
    if "drain" in run:
        fields(run["drain"], ("every",))
        every = run["drain"]["every"]
        if every != "end":
            try:
                if not isinstance(every, str) or not every.endswith("s"):
                    raise ValueError("duration needs seconds")
                seconds = float(every[:-1])
                if not math.isfinite(seconds) or seconds < 1e-9 or seconds >= 2**64:
                    raise ValueError("duration out of range")
            except (ValueError, OverflowError) as e:
                raise ValueError("drain.every must be a positive duration such as 5s, or end") from e
    if export["component"] != "export" or run["component"] != "generate" or run["packages_from"] != "export":
        raise ValueError("components must be export then generate, with packages_from=export")
    if export["compiler"] != "torch_aot_inductor" or export["rebuild"] not in ("if_missing_or_changed", "always", "never"):
        raise ValueError("unsupported compiler or rebuild policy")
    validate_kernels(export["kernels"])
    fields(export["model"], ("architecture", "checkpoint", "block_weights", "text_encoder", "decoder"))
    fields(export["model"]["text_encoder"], ("checkpoint", "block_weights", "tokenizer"), ("precision",))
    if export["model"]["text_encoder"].get("precision", "int8") not in ("int8", "bfloat16"):
        raise ValueError("text_encoder.precision must be int8 or bfloat16")
    fields(export["video"], ("size", "fps"))
    h, w = map(int, export["video"]["size"].split("x"))
    if h <= 0 or w <= 0 or h % 32 or w % 32 or export["video"]["fps"] != 24:
        raise ValueError("LongLive2 needs positive multiples of 32 and 24 fps")
    fields(export["sampling"], ("algorithm", "steps", "timestep_shift"))
    if export["sampling"]["algorithm"] != "unipc" or export["sampling"]["steps"] != 2 or export["sampling"]["timestep_shift"] != 5.0:
        raise ValueError("the current distilled checkpoint supports unipc, 2 steps, shift 5")
    fields(export["context"], ("chunk_frames", "window_frames", "sink_frames"))
    c = export["context"]
    if c != {"chunk_frames": 8, "window_frames": 32, "sink_frames": 8}:
        raise ValueError("the current model supports latent context 8/32/8")
    fields(export["rope"], ("interpolate", "native_size", "scene_offset_frames"))
    rope = export["rope"]
    if type(rope["interpolate"]) is not bool or rope["native_size"] != "704x1280" or type(rope["scene_offset_frames"]) is not int or rope["scene_offset_frames"] < 0:
        raise ValueError("invalid LongLive2 RoPE settings")
    if run["mode"] not in ("live", "offline") or type(run["seed"]) is not int or run["seed"] < 0:
        raise ValueError("invalid mode or seed")
    prompts = run["prompts"]
    if prompts.get("source") == "interactive":
        fields(prompts, ("source",), ("initial",))
        if "initial" in prompts and (not isinstance(prompts["initial"], str) or not prompts["initial"].strip()):
            raise ValueError("initial prompt must be nonempty or absent")
    elif prompts.get("source") == "script":
        fields(prompts, ("source", "file"), ("chains",))
        if not isinstance(prompts["file"], str) or not prompts["file"]:
            raise ValueError("script file is required")
        if not isinstance(prompts.get("chains", []), list) or any(not isinstance(c, str) or not c for c in prompts.get("chains", [])):
            raise ValueError("chains must be a list of identifiers")
    else:
        raise ValueError("unknown prompt source")
    off = run["offloading"]
    fields(off, ("enabled", "calibration", "resources", "copy_spare", "t5"), ("vram_margin_gib", "vram_margin_fraction", "ram_margin_fraction", "t5_activations_gib"))
    if off["calibration"] not in ("if_needed", "always", "disabled"):
        raise ValueError("invalid calibration policy")
    fields(off["resources"], ("block_weights", "kv_cache"))
    if off["resources"]["block_weights"] not in ("auto", "disk", "ram", "vram") or off["resources"]["kv_cache"] not in ("auto", "disk", "ram", "vram"):
        raise ValueError("invalid offloading tier")
    fields(off["t5"], (), ("resident_gib", "resident", "activations"))
    if any(off["t5"].get(k, "auto") != "auto" for k in ("resident", "activations")):
        raise ValueError("invalid T5 automatic policy")
    if not 0 <= off.get("ram_margin_fraction", 0) < 1:
        raise ValueError("invalid RAM fraction")
    if "vram_margin_fraction" in off and not 0 <= off["vram_margin_fraction"] < 1:
        raise ValueError("invalid VRAM fraction")
    if type(off["enabled"]) is not bool or off["copy_spare"] < 1 or any(off.get(k, 0) < 0 for k in ("vram_margin_gib", "t5_activations_gib")) or off["t5"].get("resident_gib", 0) < 0:
        raise ValueError("invalid offloading budgets")
    if run["mode"] == "offline" and "output" not in run:
        raise ValueError("offline output settings are required")
    if "output" in run:
        fields(run["output"], ("directory", "codec", "crf", "pixel_format"))
        if run["output"]["codec"] != "libx264" or run["output"]["pixel_format"] != "yuv420p" or not 0 <= run["output"]["crf"] <= 51:
            raise ValueError("unsupported output encoding")
    return export, run


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_files(root):
    paths = [root / "scripts/conversion/export_live.py", root / "ops/build.py"]
    base = root / "src/wan_dit_lab"
    for folder in ("models", "kernels", "quantization", "generation", "runtime"):
        paths.extend((base / folder).glob("*.py"))
    paths.extend((root / "ops").glob("*.cpp"))
    paths.extend((root / "ops/build").glob("*.dll"))
    # Implementations imported from the reference checkouts are inputs too.
    for folder in (root / "reference/INT8-DiT/wan_block/wan/modules", root / "reference/LongLive/LongLive2.0/wan_5b/utils"):
        paths.extend(folder.glob("*.py"))
    paths.append(root / "reference/INT8-DiT/wan_block/taehv.py")
    paths.append(root / "reference/LongLive/LongLive2.0/utils/rope_triton.py")
    return sorted(set(paths))


def export_identity(export, root=ROOT, *, sources=None, versions=None):
    root = Path(root)
    paths = []
    def inputs(value):
        for key, item in value.items():
            if key == "precision":
                continue
            if isinstance(item, dict):
                inputs(item)
            else:
                path = root / item
                paths.extend(sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path])
    inputs(export["model"])
    paths.extend(source_files(root) if sources is None else sources)
    if versions is None:
        versions = {"python": platform.python_version()}
        for name in ("torch", "triton", "triton-windows", "safetensors"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
    recipe = {k: v for k, v in export.items() if k not in ("output", "rebuild")}
    identities = {str(Path(p).resolve()): digest(p) for p in sorted(set(paths))}
    return hashlib.sha256(json.dumps([recipe, versions, identities], sort_keys=True).encode()).hexdigest()


def package_names(manifest):
    names = set(manifest["checks"])
    if "block_steps" in manifest:
        names |= {"text", "cross", "pre", "post", "decoder", "t5_block", "t5_final"}
        names |= {f"block_{k}" for k in range(manifest["block_steps"])}
    files = {f"{n}.{suffix}" for n in names for suffix in ("pt2", "safetensors")}
    if "positions" in manifest:
        files.add("time_rope.bin")
    return sorted(files)


def publish(directory, manifest, identity, export=None):
    directory = Path(directory)
    manifest["export"] = {"identity": identity, "config": export}
    manifest["files"] = {name: {"bytes": (directory / name).stat().st_size, "sha256": digest(directory / name)}
                         for name in package_names(manifest)}
    from wan_dit_lab.runtime.live_kernels import complete_export
    complete_export(directory, manifest)


def reusable(directory, identity):
    directory = Path(directory)
    if (directory / ".exporting").exists():
        return False
    try:
        manifest = json.loads((directory / "live.json").read_text(encoding="utf-8"))
        if manifest["export"]["identity"] != identity or set(manifest["files"]) != set(package_names(manifest)):
            return False
        return all((directory / n).stat().st_size == f["bytes"] and digest(directory / n) == f["sha256"]
                   for n, f in manifest["files"].items())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def prepare(export, identity, compile_packages):
    current = reusable(export["output"], identity)
    if export["rebuild"] != "always" and current:
        return False
    if export["rebuild"] == "never":
        raise ValueError("packages require export, but rebuild=never")
    directory = Path(export["output"])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ".exporting").write_text(identity, encoding="utf-8")
    compile_packages()
    if not reusable(export["output"], identity):
        raise ValueError("export did not publish complete packages with the requested identity")
    return True
