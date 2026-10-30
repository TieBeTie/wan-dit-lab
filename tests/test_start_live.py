"""artifacts/start_live.ps1 picks the live.exe of the build named by "build" in the config."""

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "artifacts" / "start_live.ps1"


def chosen_exe(tmp_path: Path, config: dict) -> str:
    path = tmp_path / "live.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    out = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT),
         "-Config", str(path), "-PrintExe"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


@pytest.mark.parametrize(
    "config, exe",
    [
        ({"build": "debug"}, r"native\target\debug-tools\release\live.exe"),
        ({"build": "release"}, r"native\target\release\live.exe"),
        ({}, r"native\target\release\live.exe"),
    ],
)
def test_build_of_config_picks_exe(tmp_path, config, exe):
    assert chosen_exe(tmp_path, config) == exe


def test_unknown_flag_is_refused_and_starts_nothing(tmp_path):
    # without [CmdletBinding()] an unknown flag lands in $args and Live starts
    out = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT), "-NoSuchFlag", "-PrintExe"],
        capture_output=True, text=True,
    )
    assert out.returncode != 0
