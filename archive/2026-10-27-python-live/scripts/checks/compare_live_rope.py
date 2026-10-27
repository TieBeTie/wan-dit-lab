"""Two sequential real Live windows, same settings, then update the open report."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wan_dit_lab.reports.rope_report import write_comparison  # noqa: E402
from wan_dit_lab.runtime.pipeline import resolve_experiment  # noqa: E402
from wan_dit_lab.runtime.environment import gpu_environment  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/rope_live.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = resolve_experiment(json.loads(args.config.read_text(encoding="utf-8")))
    report_dir = Path(config.get("report_dir", Path.home() / "Documents/Experiments/DIT/outputs"))
    directory = args.output or report_dir / "live" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    env = gpu_environment()
    records = {}
    variants = config.get("variants", [dict(label="original", pipeline=config["pipeline"])])
    for variant in variants:
        backend = variant["label"]
        output = directory / f"{backend}.json"
        command = [sys.executable, str(ROOT / "scripts/live/live_rope.py"), "--config", str(directory / "config.json"),
                   "--variant", backend, "--output", str(output)]
        if backend == "triton" and config.get("keep_open", False):
            command.append("--keep-open")
        print(f"Starting {backend} Live. Logs: {directory}", flush=True)
        with (directory / f"{backend}.log").open("w", encoding="utf-8") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
        deadline = time.perf_counter() + 600
        while not output.exists():
            if child.poll() is not None:
                raise RuntimeError(f"{backend} Live exited ({child.returncode}), see {directory / (backend + '.log')}")
            if time.perf_counter() > deadline:
                child.terminate()
                raise RuntimeError(f"{backend} Live timed out")
            time.sleep(.5)
        records[backend] = json.loads(output.read_text(encoding="utf-8"))
        if records[backend]["status"] != "passed":
            child.terminate()
            raise RuntimeError(f"{backend} measurement invalid: prompt changed")
        print(f"{backend}: FPS={records[backend]['summary'].get('generated_fps', 'disabled')}", flush=True)
        if backend == "original" or not config.get("keep_open", False):
            child.wait(timeout=30)
    print(f"Updated {write_comparison(directory, config, records)}", flush=True)
    if config.get("keep_open", False) and "triton" in records:
        print(f"Triton Live remains open, PID {records['triton']['pid']}", flush=True)


if __name__ == "__main__":
    main()
