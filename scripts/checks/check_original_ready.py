"""Read-only readiness audit for the official LongLive BF16 entrypoint.

No rendering, downloads, offload substitutions or fallback to our packages.
The weight bound excludes caches/activations/VAE: passing it does not prove fit.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
DENSE_WEIGHT_BYTES = 9_999_575_424  # measured dense S2 tensors, same 5B architecture

def blockers(*, checkpoint_exists, gpu_bytes, generator_bytes, offloaded=False):
    issues = []
    if not checkpoint_exists:
        issues.append("Official BF16 checkpoint is missing; the S2 checkpoint is not a substitute.")
    if not offloaded and gpu_bytes < generator_bytes:
        issues.append("VRAM cannot hold even the dense generator weights; KV, activations and VAE need additional memory.")
    return issues

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    memory = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True)
    gpu_bytes = int(memory.strip().splitlines()[0]) * 1024 ** 2
    checkpoint = Path(config["checkpoints"]["generator_ckpt"])
    offloaded = config.get('execution', {}).get('weights') == 'cpu'
    issues = blockers(checkpoint_exists=checkpoint.is_file(), gpu_bytes=gpu_bytes,
                      generator_bytes=DENSE_WEIGHT_BYTES, offloaded=offloaded)
    cwd = ROOT / "reference/LongLive/LongLive2.0"
    if offloaded:
        cwd = Path(config['output_folder']).parent / 'original-workspace'
    required = ["models_t5_umt5-xxl-enc-bf16.pth", "google/umt5-xxl", "Wan2.2_VAE.pth", "config.json"]
    missing = [str(cwd / "wan_models/Wan2.2-TI2V-5B" / x) for x in required if not (cwd / "wan_models/Wan2.2-TI2V-5B" / x).exists()]
    if missing:
        issues.append("Official wan_models layout is not prepared in the reference working directory.")
    report = {"ready": not issues, "status": "blocked" if issues else "prerequisites_passed_not_a_memory_guarantee", "blockers": issues,
              "gpu_total_bytes": gpu_bytes, "generator_weight_bytes_lower_bound": DENSE_WEIGHT_BYTES,
              "bound_source": "dense tensors measured in local S2 checkpoint; same 5B architecture; excludes caches and workspaces",
              "checkpoint": str(checkpoint), "missing_component_paths": missing,
              "flash_attn_installed": importlib.util.find_spec("flash_attn") is not None,
              "command": (["python", str(ROOT / 'scripts/experiments/original_offload.py'), "--config", str(args.config.resolve())]
                          if offloaded else ["python", "inference.py", "--config_path", str(args.config.resolve())]), "cwd": str(cwd),
              "reference_revision": subprocess.check_output(["git", "-C", str(ROOT / 'reference/LongLive'), "rev-parse", "HEAD"], text=True).strip(),
              "checkpoint_repository": "Efficient-Large-Model/LongLive-2.0-5B", "checkpoint_revision": "8521079b863720a57c1a8d9b19c8d9e6ccb04c0f"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 2 if issues else 0

if __name__ == "__main__":
    raise SystemExit(main())
