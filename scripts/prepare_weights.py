"""Download pinned upstream assets and verify their SHA256 against Hub metadata."""

import hashlib
import json
from pathlib import Path

from huggingface_hub import model_info, snapshot_download
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]


def main():
    spec = json.loads((ROOT / "configs/upstream.json").read_text(encoding="utf-8"))
    target = ROOT / spec["local_dir"]
    info = model_info(spec["repo_id"], revision=spec["revision"], files_metadata=True)
    snapshot_download(
        spec["repo_id"],
        revision=spec["revision"],
        local_dir=str(target),
        allow_patterns=spec["files"],
        max_workers=3,
    )
    metadata = {entry.rfilename: entry for entry in info.siblings}
    verified = []
    for name in spec["files"]:
        path = target / name
        entry = metadata[name]
        if path.stat().st_size != entry.size:
            raise RuntimeError(f"Size mismatch: {name}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if entry.lfs and digest != entry.lfs.sha256:
            raise RuntimeError(f"SHA256 mismatch: {name}")
        verified.append({"file": name, "bytes": entry.size, "sha256": digest})
        print(f"Verified {name}", flush=True)
    # Inventory without materializing the entire DiT in CPU or GPU RAM.
    with safe_open(
        target / "diffusion_pytorch_model.safetensors", framework="pt", device="cpu"
    ) as f:
        tensors = {
            k: {"shape": f.get_slice(k).get_shape(), "dtype": f.get_slice(k).get_dtype()}
            for k in f.keys()
        }
    report = {
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "files": verified,
        "tensors": tensors,
    }
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    (artifacts / "weights-manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Ready: {target}; {len(tensors)} DiT tensors", flush=True)


if __name__ == "__main__":
    main()
