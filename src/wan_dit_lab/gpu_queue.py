"""DEPRECATED: the queue is now inside wan.exe (crates/wan/src/queue.rs, docs/gpu-queue.md); kept only while the NTK grid runs on it.

One GPU, one run at a time: every experiment goes through this queue.

    .venv\\Scripts\\python.exe -m wan_dit_lab.gpu_queue submit configs\\experiments\\<name>.json
    .venv\\Scripts\\python.exe -m wan_dit_lab.gpu_queue submit <config> --exe bin\\wan-x.exe -- --config {config} --chains a
    .venv\\Scripts\\python.exe -m wan_dit_lab.gpu_queue status
    .venv\\Scripts\\python.exe -m wan_dit_lab.gpu_queue wait <id>
    .venv\\Scripts\\python.exe -m wan_dit_lab.gpu_queue cancel <id>

A job is a config and a command (by default `bin/wan.exe --config <config>`; arguments after `--` replace
`--config <config>`, with `{config}` standing for its absolute path). `submit` writes the job into
artifacts/queue/jobs/<id>.json, starts a detached worker if none holds artifacts/queue/worker.lock, and then
follows the job: its position, the start, the end and the exit code (`--no-wait` returns at once). The worker takes
jobs oldest first, one at a time, and before each waits while a GPU run started outside the queue is alive
(wan*.exe, live.exe, a Python script of scripts/experiments or scripts/conversion). Records are never deleted: a job
ends as done, failed, cancelled, or interrupted (its worker died). Output of each job: artifacts/queue/logs/<id>.log.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUEUE = ROOT / "artifacts" / "queue"
WAN = ROOT / "bin" / "wan.exe"
FINAL = ("done", "failed", "cancelled", "interrupted")
GPU_SCRIPTS = re.compile(r"scripts[\\/](experiments|conversion|calibration|generation)[\\/]")


def _now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(50):  # a reader may hold the file open for a moment on Windows
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.02 * (attempt + 1))
    os.replace(tmp, path)


def _read(path: Path):
    for attempt in range(50):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (PermissionError, json.JSONDecodeError):
            time.sleep(0.02 * (attempt + 1))
    return json.loads(path.read_text(encoding="utf-8"))


def load(queue: Path, job_id: str) -> dict:
    return _read(Path(queue) / "jobs" / f"{job_id}.json")


def _update(queue: Path, job_id: str, **fields) -> dict:
    job = load(queue, job_id) | fields
    _write(Path(queue) / "jobs" / f"{job_id}.json", job)
    return job


def jobs(queue: Path) -> list[dict]:
    directory = Path(queue) / "jobs"
    if not directory.exists():
        return []
    return sorted((_read(p) for p in directory.glob("*.json")), key=lambda j: j["id"])


def position(queue: Path, job_id: str) -> int | None:
    """0 while it runs, k when k-1 jobs (the running one included) are ahead of it, None once it ended."""
    everything = jobs(queue)
    me = next(j for j in everything if j["id"] == job_id)
    if me["status"] == "running":
        return 0
    if me["status"] != "queued":
        return None
    running = any(j["status"] == "running" for j in everything)
    ahead = sum(1 for j in everything if j["status"] == "queued" and j["id"] < job_id)
    return 1 + ahead + int(running)


def submit(queue: Path, config: Path, *, exe=None, args=None, command=None, note=None, spawn=True) -> dict:
    config = Path(config).resolve()
    if not config.is_file():
        raise SystemExit(f"Нет конфига: {config}")
    try:
        json.loads(config.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"Конфиг не JSON: {config}: {error}")
    if command is None:
        tail = [a.replace("{config}", str(config)) for a in args] if args else ["--config", str(config)]
        exe = Path(exe).resolve() if exe and Path(exe).exists() else (exe or WAN)
        command = [str(exe)] + tail
    job_id = f"{datetime.now():%Y%m%d-%H%M%S}-{time.time_ns() % 10**9:09d}"
    job = dict(id=job_id, status="queued", config=str(config), command=[str(c) for c in command], note=note,
               cwd=str(ROOT), submitted=_now(), submitter_pid=os.getpid(),
               log=str(Path(queue).resolve() / "logs" / f"{job_id}.log"))
    _write(Path(queue) / "jobs" / f"{job_id}.json", job)
    if spawn:
        spawn_worker(queue)
    return job


def cancel(queue: Path, job_id: str) -> dict:
    job = load(queue, job_id)
    if job["status"] != "queued":
        raise SystemExit(f"Задание {job_id} уже {job['status']}: отменить можно только ждущее")
    return _update(queue, job_id, status="cancelled", finished=_now())


# ---------------------------------------------------------------- the worker


def acquire_worker_lock(queue: Path):
    """An open handle with its first byte locked, or None when another worker holds it; the OS frees it on exit."""
    path = Path(queue) / "worker.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+")
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def gpu_processes(procs, own=frozenset()) -> list[dict]:
    """The processes of `procs` (dicts of pid, name, cmdline) that are a GPU run, leaving out the pids in `own`."""
    found = []
    for p in procs:
        if p["pid"] in own:
            continue
        name = (p.get("name") or "").lower()
        line = " ".join(p.get("cmdline") or [])
        if re.fullmatch(r"wan.*\.exe|live\.exe", name) or GPU_SCRIPTS.search(line):
            found.append(p)
    return found


def outside_gpu_processes() -> list[dict]:
    import psutil

    me = psutil.Process()
    own = {me.pid} | {c.pid for c in me.children(recursive=True)}
    procs = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            procs.append(dict(p.info))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return [dict(pid=p["pid"], name=p["name"]) for p in gpu_processes(procs, own)]


def _next(queue: Path):
    return next((j for j in jobs(queue) if j["status"] == "queued"), None)


def _run(queue: Path, job: dict, external, poll: float):
    waiting = None
    while busy := external():
        if busy != waiting:
            waiting = busy
            _update(queue, job["id"], waiting_for=busy)
        time.sleep(poll)
        if load(queue, job["id"])["status"] != "queued":  # cancelled while it waited
            return
    log = Path(job["log"])
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ | {"PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
    with open(log, "ab") as out:
        try:
            child = subprocess.Popen(job["command"], cwd=job["cwd"], stdout=out, stderr=subprocess.STDOUT, env=env)
        except OSError as error:
            _update(queue, job["id"], status="failed", returncode=None, error=str(error), started=_now(),
                    finished=_now(), waiting_for=None)
            return
        _update(queue, job["id"], status="running", started=_now(), pid=child.pid, worker_pid=os.getpid(),
                waiting_for=None)
        code = child.wait()
    _update(queue, job["id"], status="done" if code == 0 else "failed", returncode=code, finished=_now())


def run_worker(queue: Path = QUEUE, *, idle_exit=True, external=outside_gpu_processes, poll=5.0) -> bool:
    """Run queued jobs in order until none is left; False when another worker already holds the queue."""
    queue = Path(queue)
    ran = False
    while True:
        lock = acquire_worker_lock(queue)
        if lock is None:
            return ran
        ran = True
        try:
            for stale in (j for j in jobs(queue) if j["status"] == "running"):  # its worker died
                _update(queue, stale["id"], status="interrupted", finished=_now())
            while (job := _next(queue)) is not None:
                _run(queue, job, external, poll)
        finally:
            lock.close()
        # a job submitted after the last look found the lock still held and started no worker
        if _next(queue) is None or not idle_exit:
            return ran


def spawn_worker(queue: Path = QUEUE):
    """A worker detached from the caller's console, so it outlives the shell that submitted the job."""
    queue = Path(queue)
    queue.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "wan_dit_lab.gpu_queue", "--queue", str(queue), "worker"]
    out = open(queue / "worker.log", "ab")
    env = os.environ | {"PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    kwargs = dict(cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, env=env,
                  close_fds=True)
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            subprocess.Popen(command, creationflags=flags | subprocess.CREATE_BREAKAWAY_FROM_JOB, **kwargs)
        except OSError:  # the caller's job object forbids breakaway
            subprocess.Popen(command, creationflags=flags, **kwargs)
    else:
        subprocess.Popen(command, start_new_session=True, **kwargs)
    out.close()


# ---------------------------------------------------------------- what callers see


def _short(job):
    config = Path(job["config"])
    try:
        config = config.relative_to(ROOT)
    except ValueError:
        pass
    note = f" · {job['note']}" if job.get("note") else ""
    return f"{job['id']}  {config}{note}"


def status_text(queue: Path = QUEUE) -> str:
    everything = jobs(queue)
    running = [j for j in everything if j["status"] == "running"]
    queued = [j for j in everything if j["status"] == "queued"]
    lines = []
    for j in running:
        lines.append(f"Идёт: {_short(j)}  (с {j['started']}, pid {j.get('pid')}, лог {j['log']})")
    offset = len(running)
    for k, j in enumerate(queued, 1):
        wait = f"  ждёт внешний прогон {j['waiting_for']}" if j.get("waiting_for") else ""
        lines.append(f"{k + offset}. {_short(j)}{wait}")
    if not running and not queued:
        lines.append("Очередь пуста.")
    ended = [j for j in everything if j["status"] in FINAL][-5:]
    if ended:
        lines.append("Последние завершённые:")
        lines += [f"  {j['status']} (код {j.get('returncode')}) {_short(j)}  {j.get('finished', '')}" for j in ended]
    return "\n".join(lines)


def follow(queue: Path, job_id: str, poll=2.0, say=print, revive=True) -> int:
    """Print what happens to the job until it ends; its exit code, or 1 when it did not run to the end."""
    seen = None
    while True:
        job = load(queue, job_id)
        state = (job["status"], position(queue, job_id), repr(job.get("waiting_for")))
        if state != seen:
            seen = state
            status, pos, _ = state
            if status == "queued" and job.get("waiting_for"):
                say(f"[очередь] {job_id}: позиция {pos}, ждёт, пока закончится внешний прогон {job['waiting_for']}")
            elif status == "queued":
                say(f"[очередь] {job_id}: позиция {pos}, " + ("ждёт своей очереди" if pos > 1 else "следующее, ждёт запуска"))
            elif status == "running":
                say(f"[очередь] {job_id}: запущено {job['started']}, лог {job['log']}")
            else:
                say(f"[очередь] {job_id}: {status}, код {job.get('returncode')}, {job.get('finished', '')}, лог {job['log']}")
                return job.get("returncode") if status in ("done", "failed") and job.get("returncode") is not None else 1
        if job["status"] == "queued" and revive and (lock := acquire_worker_lock(queue)) is not None:
            lock.close()  # no worker holds the queue (it died or never started): start one
            spawn_worker(queue)
        time.sleep(poll)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    tail = None
    if "--" in argv:
        tail, argv = argv[argv.index("--") + 1:], argv[:argv.index("--")]
    parser = argparse.ArgumentParser(prog="python -m wan_dit_lab.gpu_queue", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queue", type=Path, default=QUEUE)
    sub = parser.add_subparsers(dest="action", required=True)
    s = sub.add_parser("submit", help="поставить конфиг в очередь")
    s.add_argument("config", type=Path)
    s.add_argument("--exe", help="раннер вместо bin/wan.exe")
    s.add_argument("--note", help="кто и зачем, видно в status")
    s.add_argument("--no-wait", action="store_true", help="не следить за заданием")
    sub.add_parser("status")
    for name in ("wait", "cancel"):
        sub.add_parser(name).add_argument("id")
    sub.add_parser("worker")
    args = parser.parse_args(argv)
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.action == "submit":
        config = args.config if args.config.is_absolute() else Path.cwd() / args.config
        job = submit(args.queue, config, exe=args.exe, args=tail, note=args.note)
        print(f"[очередь] поставлено {job['id']}: {' '.join(job['command'])}", flush=True)
        if args.no_wait:
            print(f"[очередь] позиция {position(args.queue, job['id'])}; следить: wait {job['id']}")
            return 0
        return follow(args.queue, job["id"], say=lambda m: print(m, flush=True))
    if args.action == "status":
        print(status_text(args.queue))
        busy = outside_gpu_processes()
        if busy:
            print(f"GPU занят вне очереди: {busy}")
        return 0
    if args.action == "wait":
        return follow(args.queue, args.id, say=lambda m: print(m, flush=True))
    if args.action == "cancel":
        cancel(args.queue, args.id)
        print(f"[очередь] {args.id} отменено")
        return 0
    run_worker(args.queue)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
