"""The GPU queue: one worker, jobs in order, positions and notices; no GPU, the jobs are short Python commands."""

import json
import sys
import threading
import time

import pytest

from wan_dit_lab import gpu_queue as q

PY = sys.executable


def job_command(seconds=0.0, code=0):
    return [PY, "-c", f"import time,sys; time.sleep({seconds}); print('ran'); sys.exit({code})"]


@pytest.fixture
def root(tmp_path):
    config = tmp_path / "exp.json"
    config.write_text(json.dumps({"seed": 0}), encoding="utf-8")
    return tmp_path / "queue", config


def test_submit_on_an_idle_queue_is_position_one(root):
    queue, config = root
    job = q.submit(queue, config, command=job_command(), spawn=False)
    assert job["status"] == "queued"
    assert q.position(queue, job["id"]) == 1


def test_second_submit_while_one_waits_gets_position_two(root):
    queue, config = root
    q.submit(queue, config, command=job_command(), spawn=False)
    second = q.submit(queue, config, command=job_command(), spawn=False)
    assert q.position(queue, second["id"]) == 2


def test_submit_refuses_a_missing_or_broken_config(root, tmp_path):
    queue, _ = root
    with pytest.raises(SystemExit):
        q.submit(queue, tmp_path / "absent.json", command=job_command(), spawn=False)
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(SystemExit):
        q.submit(queue, broken, command=job_command(), spawn=False)


def test_default_command_is_wan_exe_with_the_config(root):
    queue, config = root
    job = q.submit(queue, config, spawn=False)
    assert job["command"][0].endswith("wan.exe")
    assert job["command"][1:] == ["--config", str(config.resolve())]


def test_config_placeholder_is_filled_in_given_arguments(root):
    queue, config = root
    job = q.submit(queue, config, exe="python.exe", args=["run.py", "--config", "{config}"], spawn=False)
    assert job["command"] == ["python.exe", "run.py", "--config", str(config.resolve())]


def test_worker_runs_jobs_in_order_and_records_exit_codes(root):
    queue, config = root
    a = q.submit(queue, config, command=job_command(code=0), spawn=False)
    b = q.submit(queue, config, command=job_command(code=3), spawn=False)
    q.run_worker(queue, idle_exit=True, external=lambda: [])
    ja, jb = q.load(queue, a["id"]), q.load(queue, b["id"])
    assert (ja["status"], ja["returncode"]) == ("done", 0)
    assert (jb["status"], jb["returncode"]) == ("failed", 3)
    assert ja["finished"] <= jb["started"]
    assert "ran" in (queue / "logs" / f"{a['id']}.log").read_text(encoding="utf-8")


def test_while_a_job_runs_the_next_waits_at_position_two(root):
    queue, config = root
    a = q.submit(queue, config, command=job_command(seconds=1.5), spawn=False)
    worker = threading.Thread(target=q.run_worker, args=(queue,), kwargs=dict(idle_exit=True, external=lambda: []))
    worker.start()
    deadline = time.time() + 10
    while q.load(queue, a["id"])["status"] != "running" and time.time() < deadline:
        time.sleep(0.05)
    b = q.submit(queue, config, command=job_command(), spawn=False)
    assert q.position(queue, a["id"]) == 0  # running
    assert q.position(queue, b["id"]) == 2  # one ahead of it
    worker.join(timeout=20)
    assert q.load(queue, b["id"])["status"] == "done"


def test_only_one_worker_holds_the_lock(root):
    queue, _ = root
    first = q.acquire_worker_lock(queue)
    assert first is not None
    assert q.acquire_worker_lock(queue) is None
    first.close()
    again = q.acquire_worker_lock(queue)
    assert again is not None
    again.close()


def test_worker_waits_for_a_gpu_run_started_outside_the_queue(root):
    queue, config = root
    job = q.submit(queue, config, command=job_command(), spawn=False)
    seen = []

    def external():  # an outside wan.exe for the first two looks, then free
        seen.append(1)
        return [{"pid": 4242, "name": "wan-ntk-sweep.exe"}] if len(seen) <= 2 else []

    q.run_worker(queue, idle_exit=True, external=external, poll=0.01)
    record = q.load(queue, job["id"])
    assert record["status"] == "done"
    assert len(seen) >= 3


def test_outside_gpu_processes_are_recognised_by_name_and_command_line():
    procs = [
        {"pid": 1, "name": "wan.exe", "cmdline": ["wan.exe"]},
        {"pid": 2, "name": "wan-ntk-sweep.exe", "cmdline": []},
        {"pid": 3, "name": "python.exe", "cmdline": ["python", "scripts/experiments/run_experiment.py"]},
        {"pid": 4, "name": "python.exe", "cmdline": ["python", "-m", "pytest"]},
        {"pid": 5, "name": "wan-debug.exe", "cmdline": ["wan-debug.exe", "check"]},
        {"pid": 6, "name": "notepad.exe", "cmdline": []},
    ]
    found = q.gpu_processes(procs, own={5})
    assert [p["pid"] for p in found] == [1, 2, 3]


def test_status_lists_running_then_queued_with_positions(root):
    queue, config = root
    a = q.submit(queue, config, command=job_command(), spawn=False, note="first")
    q.submit(queue, config, command=job_command(), spawn=False)
    text = q.status_text(queue)
    assert "1." in text and "2." in text and "first" in text


def test_worker_command_puts_the_queue_option_before_the_action(root, monkeypatch):
    queue, _ = root
    seen = []
    monkeypatch.setattr(q.subprocess, "Popen", lambda command, **kw: seen.append(command))
    q.spawn_worker(queue)
    args = seen[0][3:]  # after python -m wan_dit_lab.gpu_queue
    assert q.main.__module__ and args[-1] == "worker" and args[:2] == ["--queue", str(queue)]


def test_cancel_marks_a_queued_job_and_keeps_its_file(root):
    queue, config = root
    a = q.submit(queue, config, command=job_command(), spawn=False)
    b = q.submit(queue, config, command=job_command(), spawn=False)
    q.cancel(queue, a["id"])
    assert q.load(queue, a["id"])["status"] == "cancelled"
    assert q.position(queue, b["id"]) == 1
