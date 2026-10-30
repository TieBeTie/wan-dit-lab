"""artifacts/start_live.ps1 only starts bin/wan.exe, the one entry point, over a configuration: wan.exe reads the rest
of it itself. build.ps1 puts both builds into bin/."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "artifacts" / "start_live.ps1"


def powershell(*args, check=True):
    return subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", *map(str, args)],
                          capture_output=True, text=True, check=check)


def test_it_starts_wan_exe_over_the_config_and_the_prompt():
    out = powershell(SCRIPT, "-Config", r"configs\live.json", "-Prompt", "a red fox", "-Print").stdout.strip()
    assert out == r'bin\wan.exe --config configs\live.json --prompt "a red fox"'


def test_the_config_is_live_json_by_default():
    assert powershell(SCRIPT, "-Prompt", "a fox", "-Print").stdout.strip() == \
        r'bin\wan.exe --config configs\live.json --prompt "a fox"'


def test_unknown_flag_is_refused_and_starts_nothing():
    # without [CmdletBinding()] an unknown flag lands in $args and Live starts
    assert powershell(SCRIPT, "-NoSuchFlag", "-Print", check=False).returncode != 0


def test_build_puts_both_builds_into_bin():
    out = powershell(ROOT / "build.ps1", "-Print").stdout.splitlines()
    assert out == [r"cargo build --release -p wan -> bin\wan.exe",
                   r"cargo build --release -p wan --features wan/debug-tools --target-dir target\debug-tools"
                   r" -> bin\wan-debug.exe"]
