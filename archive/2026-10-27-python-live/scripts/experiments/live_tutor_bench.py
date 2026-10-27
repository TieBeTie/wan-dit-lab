"""Live and the voice tutor on one machine at once: what each costs the other.

    .\\.venv\\Scripts\\python.exe scripts/experiments/live_tutor_bench.py --config configs/experiments/live_tutor.json

Three phases on one wall clock. Calibration: `bench.baseline_s` of the machine as it is, the tutor running and Live
not, so that every later number has its idle level beside it. Warmup: Live starts through the harness (live_rope.py),
compiles and calibrates its two streamed-block slots. Measured: the harness's measured chunks; meanwhile
`bench.prompts` go to Live's socket at their offsets (each one a T5 encoding), and whoever runs it talks to the tutor.

Written to outputs/live_tutor/<time>/: record.json (the harness: chunks with DiT and decoder GPU time, window ticks,
T5 encodings, streamed-block timestamps), live.log (Live's output, each line with its wall time), system.jsonl (CPU
and memory of Live, the tutor, OBS and browsers, RAM and commit of the machine, every `sample_interval_s`), gpu.jsonl
(nvidia-smi dmon: power, utilization, memory, PCIe traffic, each second), phases.json, and the tutor's own journal
and log lines of the same span (tutor.jsonl, tutor.log). scripts/reports/live_tutor_report.py makes the report.
"""

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wan_dit_lab.harness.config import OUTPUTS  # noqa: E402
from wan_dit_lab.runtime.environment import gpu_environment  # noqa: E402
from wan_dit_lab.runtime.pipeline import resolve_experiment  # noqa: E402

LIVE_PORT = 8765
# processes followed one by one, by a piece of their command line; browsers summed over their many processes
ROLES = dict(live=("live_rope.py", "live_video.py"), tutor=("voice_loop.py",), obs=("obs64.exe",))
BROWSERS = ("chrome.exe", "msedge.exe", "firefox.exe", "browser.exe")


def role_of(process):
    try:
        name, line = process.name().lower(), " ".join(process.cmdline())
    except (psutil.Error, OSError):
        return None
    for role, marks in ROLES.items():
        if any(mark in line or mark == name for mark in marks):
            return role
    return "browser" if name in BROWSERS else None


class SystemTrace(threading.Thread):
    """Every `interval` s one line: the machine, then each role summed over its processes. CPU in percent of one
    logical core, as psutil gives it; per-core load too, to see whether the tutor and Live share the fast cores."""

    def __init__(self, path, interval):
        super().__init__(daemon=True)
        self.path, self.interval, self.stopped = path, interval, threading.Event()
        self.processes, self.ignored = {}, set()  # pid -> (role, process); pids of no role, asked once

    def run(self):
        psutil.cpu_percent(percpu=True)
        with open(self.path, "w", encoding="utf-8") as out:
            while not self.stopped.wait(self.interval):
                memory, commit = psutil.virtual_memory(), psutil.swap_memory()
                disk = psutil.disk_io_counters()
                row = dict(t=time.time(), cores=psutil.cpu_percent(percpu=True),
                           ram_available=memory.available, ram_used=memory.used, pagefile_used=commit.used,
                           disk_read=disk.read_bytes, disk_write=disk.write_bytes, roles={})
                current = set()
                for process in psutil.process_iter():
                    if process.pid in self.ignored:
                        continue
                    if process.pid not in self.processes:
                        role = role_of(process)
                        if role is None:
                            self.ignored.add(process.pid)
                            continue
                        self.processes[process.pid] = (role, process)
                        process.cpu_percent()  # the first call only starts the count
                    current.add(process.pid)
                for pid in list(self.processes):
                    role, process = self.processes[pid]
                    if pid not in current:
                        del self.processes[pid]
                        continue
                    try:
                        with process.oneshot():
                            info, io = process.memory_info(), process.io_counters()
                            cpu = process.cpu_percent()
                    except (psutil.Error, OSError):
                        continue
                    totals = row["roles"].setdefault(role, dict(cpu=0., rss=0, private=0, read=0, faults=0, pids=[]))
                    totals["cpu"] += cpu
                    totals["rss"] += info.rss
                    totals["private"] += getattr(info, "private", 0)
                    totals["read"] += io.read_bytes
                    totals["faults"] += getattr(info, "num_page_faults", 0)
                    totals["pids"].append(pid)
                out.write(json.dumps(row) + "\n")
                out.flush()


class GpuTrace(threading.Thread):
    """nvidia-smi dmon each second: p power and temperature, u utilization, t PCIe rx/tx MB/s, m memory."""

    def __init__(self, path):
        super().__init__(daemon=True)
        self.path = path
        self.process = subprocess.Popen(["nvidia-smi", "dmon", "-s", "putm", "-d", "1"], stdout=subprocess.PIPE,
                                        text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def run(self):
        columns = None
        with open(self.path, "w", encoding="utf-8") as out:
            for line in self.process.stdout:
                if line.startswith("# gpu"):
                    columns = line[1:].split()
                elif not line.startswith("#") and columns and line.split():
                    values = line.split()
                    row = dict(zip(columns, [float(v) if re.fullmatch(r"-?[\d.]+", v) else None for v in values]))
                    out.write(json.dumps(dict(t=time.time(), **row)) + "\n")
                    out.flush()

    def stop(self):
        self.process.terminate()


def send(text):
    with socket.create_connection(("127.0.0.1", LIVE_PORT), timeout=5) as connection:
        connection.sendall((text.strip() + "\n").encode("utf-8"))


def tutor_span(folder, begin, end, out):
    """The tutor's journal events and log lines between begin and end (wall times)."""
    rows = []
    for day in sorted({datetime.fromtimestamp(t).strftime("%Y-%m-%d") for t in (begin, end)}):
        path = folder / "sessions" / f"{day}.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if begin <= row.get("t", 0) <= end:
                    rows.append(row)
    (out / "tutor.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    # voice.log stamps lines with the clock time only: those of the span, the day taken from its beginning
    log = folder / "voice.log"
    lines = []
    if log.exists():
        start_day = datetime.fromtimestamp(begin).date()
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.match(r"\[(\d\d):(\d\d):(\d\d)\] ", line)
            if match:
                at = datetime.combine(start_day, datetime.min.time()).replace(
                    hour=int(match[1]), minute=int(match[2]), second=int(match[3])).timestamp()
                if at < begin - 43200:
                    at += 86400  # after midnight
                if begin <= at <= end + 1:
                    lines.append(f"{at:.0f} {line}")
    (out / "tutor.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(rows), len(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--baseline-s", type=float, default=None, help="override bench.baseline_s")
    args = parser.parse_args()
    config = resolve_experiment(json.loads(args.config.read_text(encoding="utf-8")))
    bench = config["bench"]
    out = args.out or OUTPUTS / "live_tutor" / datetime.now().strftime("%Y%m%dT%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    phases = dict(config=str(args.config), started=time.time())
    system = SystemTrace(out / "system.jsonl", bench["sample_interval_s"])
    gpu = GpuTrace(out / "gpu.jsonl")
    system.start()
    gpu.start()

    baseline = bench["baseline_s"] if args.baseline_s is None else args.baseline_s
    print(f"calibration: {baseline:.0f} s of the machine without Live", flush=True)
    time.sleep(baseline)
    phases["live_start"] = time.time()
    env = dict(gpu_environment(), PYTHONUNBUFFERED="1")  # the CUDA toolchain Triton builds with, as cli.py run
    live = subprocess.Popen([sys.executable, str(ROOT / "scripts/live/live_rope.py"), "--config", str(args.config.resolve()),
                             "--output", str(out / "record.json")], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    measured = threading.Event()

    def prompts():
        measured.wait()
        begin = phases["measured_start"]
        for item in bench["prompts"]:
            delay = begin + item["after_s"] - time.time()
            if delay > 0:
                time.sleep(delay)
            if live.poll() is not None:
                return
            phases.setdefault("prompts", []).append(dict(t=time.time(), text=item["text"]))
            try:
                send(item["text"])
                print(f"prompt sent: {item['text']}", flush=True)
            except OSError as error:
                print(f"prompt not sent ({error}): {item['text']}", flush=True)

    threading.Thread(target=prompts, daemon=True).start()
    with open(out / "live.log", "w", encoding="utf-8") as log:
        for line in live.stdout:
            log.write(f"{time.time():.3f} {line}")
            log.flush()
            if line.startswith("Measured chunks begin"):
                phases["measured_start"] = time.time()
                measured.set()
                print("MEASURED: talk to the tutor now", flush=True)
            elif line.startswith(("Fresh prefetch", "Measured ", "Traceback", "prompt ")) or "Error" in line:
                print(line.rstrip(), flush=True)
    phases["live_exit"], phases["live_returncode"] = time.time(), live.wait()
    time.sleep(2 * bench["sample_interval_s"])
    system.stopped.set()
    gpu.stop()
    tutor = (ROOT / bench["tutor"]).resolve()
    phases["tutor_events"], phases["tutor_log_lines"] = tutor_span(tutor, phases["started"], time.time(), out)
    (out / "phases.json").write_text(json.dumps(phases, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"written to {out}", flush=True)
    report = subprocess.run([sys.executable, str(ROOT / "scripts/reports/live_tutor_report.py"), str(out)],
                            cwd=ROOT, env=env)
    raise SystemExit(phases["live_returncode"] or report.returncode)


if __name__ == "__main__":
    main()
