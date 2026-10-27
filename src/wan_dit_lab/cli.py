"""Environment diagnostics and an explicitly synthetic GPU baseline."""

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


def environment():
    packages = {}
    for name in ("torch", "triton", "triton-windows", "safetensors"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if info["cuda_available"]:
        device = torch.cuda.get_device_properties(0)
        info["gpu"] = device.name
        info["vram_bytes"] = device.total_memory
        info["compute_capability"] = [device.major, device.minor]
    return info


def benchmark(config=None):
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for bench-smoke")
    from wan_dit_lab.harness.runner import run

    torch.manual_seed(0)
    x = torch.randn((256, 512), device="cuda", dtype=torch.float16)
    w = torch.randn((512, 1024), device="cuda", dtype=torch.float16)
    return run("synthetic_fp16_matmul", torch.mm, x, w, config=config) | dict(
        operation="synthetic FP16 matmul, not Wan performance", shapes=[[256, 512], [512, 1024]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("doctor", "bench-smoke", "run"))
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        if args.config is None:
            parser.error("run needs --config")
        from wan_dit_lab.runtime.pipeline import resolve_experiment

        config = resolve_experiment(json.loads(args.config.read_text(encoding="utf-8")))
        root = Path(__file__).resolve().parents[2]
        from wan_dit_lab.harness.checks import run_tests
        from wan_dit_lab.runtime.environment import gpu_environment

        env = gpu_environment()
        directory = Path.home() / "Documents/Experiments/DIT/outputs/runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        tests = run_tests(config["harness"], directory / "tests.log", env=env)
        print(f"Local checks: {tests['status']}", flush=True)
        # Live is the Rust host over the packages of export_live.py: the window, or chunks timed without one
        command = [str(root / "native/target/release/live.exe")]
        if config["run"]["mode"] != "live":
            command += ["bench", "--chunks", str(config["run"].get("chunks", 14))]
        if config["run"].get("prompt"):
            command += ["--prompt", config["run"]["prompt"]]
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "live.log", "w", encoding="utf-8") as log:
            raise SystemExit(subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT).returncode)
    result = environment() if args.command == "doctor" else benchmark(args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
