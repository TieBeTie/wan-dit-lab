"""Preparation of the two-stage Live pipeline. No torch import and no work during generation."""
import ast
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
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
    """The configuration as the Rust host has already validated it: wan.exe parses every configuration first (the
    contract of crates/wan/src/contract.rs, then its own checks) and hands this module its own normalized form. It is
    read here, not checked again."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list) or len(value) != 2 or not all(isinstance(part, dict) for part in value):
        raise ValueError("config must be [export, generate]")
    return value[0], value[1]


DIGEST_CACHE = ROOT / "artifacts/digest_cache.json"
_digests = None  # what the file at _digests_from held when it was read
_digests_from = None


def _remembered():
    global _digests, _digests_from
    if _digests is None or _digests_from != DIGEST_CACHE:
        _digests_from = DIGEST_CACHE
        try:
            value = json.loads(DIGEST_CACHE.read_text(encoding="utf-8"))
            _digests = value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            _digests = {}
    return _digests


def digest(path):
    """SHA-256 of a file, read from the disk only when the file changed: the digest is kept with the file's size and
    modification time (as a layer cache keeps a layer), and a start of Live that changed nothing reads nothing. The
    weights are tens of gigabytes; hashing them was most of the preparation."""
    path = Path(path)
    stat = path.stat()
    key = str(path.resolve())
    kept = _remembered().get(key)
    if kept and kept.get("size") == stat.st_size and kept.get("mtime_ns") == stat.st_mtime_ns:
        return kept["sha256"]
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    _remembered()[key] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": h.hexdigest()}
    _save_digests()
    return h.hexdigest()


def _save_digests():
    temporary = DIGEST_CACHE.with_name(DIGEST_CACHE.name + f".{os.getpid()}")
    try:
        temporary.write_text(json.dumps(_remembered()), encoding="utf-8")
        os.replace(temporary, DIGEST_CACHE)
    except OSError:
        pass  # the cache only saves time; a read-only disk costs the time and nothing else


def code_digest(path):
    """The digest of what a file does: for Python, its syntax tree without docstrings, so a comment, a docstring or
    blank lines change no package (a comment in multi_head_attention.py rebuilt them all on 2026-11-23); any other
    file, and Python that does not parse, by its bytes."""
    path = Path(path)
    if path.suffix != ".py":
        return digest(path)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError, ValueError):
        return digest(path)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and body
                and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return "ast:" + hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


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


def input_paths(export, root=ROOT, sources=None):
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
    return paths


def warm_listing(export, root=ROOT):
    """What the Rust host watches (crates/wan/src/stamp.rs): the files whose size or time change would change the
    identity, the package files, and the folders whose entries would. Written at the cold path; read by no Python."""
    root = Path(root)
    files = input_paths(export, root)
    try:
        import importlib.util
        files.append(Path(importlib.util.find_spec("torch").origin).parent / "version.py")
    except (ImportError, AttributeError, ValueError):
        pass
    output = root / export["output"]
    try:
        manifest = json.loads((output / "mvp1.json").read_text(encoding="utf-8"))
        files.extend([output / "mvp1.json", *(output / name for name in package_names(manifest))])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    files = sorted({str(Path(p).resolve()) for p in files})
    folders = sorted({str(Path(p).parent) for p in files})
    return {"files": files, "folders": folders}


def export_identity(export, root=ROOT, *, sources=None, versions=None):
    root = Path(root)
    paths = input_paths(export, root, sources)
    if versions is None:
        versions = {"python": platform.python_version()}
        for name in ("torch", "triton", "triton-windows", "safetensors"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
    recipe = {k: v for k, v in export.items() if k not in ("output", "rebuild")}
    # the height and width RoPE table is an input of the block packages that wan.exe computes from these keys
    # (crates/wan/src/rope.rs): they change no package (attn_logit_scale stays, it scales the time table too)
    if isinstance(recipe.get("rope"), dict):
        recipe["rope"] = {k: v for k, v in recipe["rope"].items()
                          if k not in ("interpolate", "stretch", "by_parts", "ntk_aware", "frequency_aware", "m", "ntk_pi_mix")}
    identities = {str(Path(p).resolve()): code_digest(p) for p in sorted(set(paths))}
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
        manifest = json.loads((directory / "mvp1.json").read_text(encoding="utf-8"))
        if manifest["export"]["identity"] != identity or set(manifest["files"]) != set(package_names(manifest)):
            return False
        return all((directory / n).stat().st_size == f["bytes"] and digest(directory / n) == f["sha256"]
                   for n, f in manifest["files"].items())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def prepare(export, identity, compile_packages):
    """Reuses the packages of `export` when they match `identity`; otherwise `compile_packages(folder)` builds them in
    a folder beside them, which replaces them only once complete: a failed export leaves the working packages as they
    were (on 2026-11-23 one failed at block_2 after overwriting text..block_1)."""
    directory = Path(export["output"])
    staging, previous = (directory.with_name(directory.name + suffix) for suffix in (".next", ".previous"))
    if previous.is_dir() and not directory.exists():  # a swap interrupted between its two renames
        os.replace(previous, directory)
    current = reusable(directory, identity)
    if export["rebuild"] != "always" and current:
        return False
    if export["rebuild"] == "never":
        raise ValueError("packages require export, but rebuild=never")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    (staging / ".exporting").write_text(identity, encoding="utf-8")
    try:
        compile_packages(staging)
        if not reusable(staging, identity):
            raise ValueError("export did not publish complete packages with the requested identity")
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    shutil.rmtree(previous, ignore_errors=True)
    if directory.exists():
        os.replace(directory, previous)
    os.replace(staging, directory)
    shutil.rmtree(previous, ignore_errors=True)
    return True
