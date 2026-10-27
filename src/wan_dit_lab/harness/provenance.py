"""Environment metadata and JSONL output."""

import json
import os
import subprocess
from pathlib import Path

import torch

from wan_dit_lab.cli import environment

ROOT = Path(__file__).resolve().parents[3]
RESULTS = Path.home() / "Documents/Experiments/DIT/outputs/harness/results.jsonl"
GPU_FIELDS = ["name", "memory.used", "memory.total", "utilization.gpu", "temperature.gpu"]


def describe_inputs(*arg):
    return [
        {"shape": list(a.shape), "dtype": str(a.dtype)} if isinstance(a, torch.Tensor) else repr(a)
        for a in arg
    ]


def git_commit():
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def gpu_state(device=0):
    out = subprocess.run(
        ["nvidia-smi", f"--id={device}", f"--query-gpu={','.join(GPU_FIELDS)}",
         "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    )
    return dict(zip(GPU_FIELDS, out.stdout.strip().split(", ")))


def append_row(row, path=RESULTS):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def metadata(device=0):
    """The same provenance for forward, streaming and I/O experiments."""
    return dict(commit=git_commit(), gpu=gpu_state(device), environment=environment(),
                harness_version=3, pid=os.getpid(), quantile_method="linear interpolation")
