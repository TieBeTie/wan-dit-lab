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


def test_no_hidden_prompt_overrides_the_config():
    assert powershell(SCRIPT, "-Print").stdout.strip() == r'bin\wan.exe --config configs\live.json'


def test_unknown_flag_is_refused_and_starts_nothing():
    # without [CmdletBinding()] an unknown flag lands in $args and Live starts
    assert powershell(SCRIPT, "-NoSuchFlag", "-Print", check=False).returncode != 0


def test_build_puts_both_builds_into_bin():
    out = powershell(ROOT / "build.ps1", "-Print").stdout.splitlines()
    assert out == [r"cargo build --release -p wan -> bin\wan.exe",
                   r"cargo build --release -p wan --features wan/debug-tools --target-dir target\debug-tools"
                   r" -> bin\wan-debug.exe"]


def test_build_does_not_report_success_if_installing_the_executable_fails():
    script = str(ROOT / "build.ps1").replace("'", "''")
    result = subprocess.run(["powershell", "-NoProfile", "-Command",
        "function cargo { $global:LASTEXITCODE = 0 }; "
        "function Copy-Item { Write-Error 'copy failed' }; "
        f"& '{script}'"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "built bin" not in result.stdout


def test_stopping_live_closes_the_window_and_waits_for_end_metrics():
    script = str(ROOT / "artifacts/stop_live.ps1").replace("'", "''")
    result = subprocess.run(["powershell", "-NoProfile", "-Command",
        "$p = [pscustomobject]@{ Id = 1234 }; "
        "$p | Add-Member ScriptMethod CloseMainWindow { Write-Host 'closed normally'; return $true }; "
        "$p | Add-Member ScriptMethod WaitForExit { param($ms) Write-Host 'waited for exit'; return $true }; "
        "function Get-Process { $p }; "
        "function Stop-Process { throw 'force stop loses end metrics' }; "
        f"& '{script}'"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "closed normally" in result.stdout and "waited for exit" in result.stdout


def test_force_stop_is_only_used_when_explicitly_requested():
    script = str(ROOT / "artifacts/stop_live.ps1").replace("'", "''")
    result = subprocess.run(["powershell", "-NoProfile", "-Command",
        "$p = [pscustomobject]@{ Id = 1234 }; "
        "$p | Add-Member ScriptMethod CloseMainWindow { throw 'must not close gracefully with Force' }; "
        "function Get-Process { $p }; "
        "function Stop-Process { param($Id, [switch]$Force) if (-not $Force) { throw 'missing Force' }; Write-Host \"forced $Id\" }; "
        f"& '{script}' -Force"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "forced 1234" in result.stdout


def test_an_unavailable_window_is_not_silently_force_killed():
    script = str(ROOT / "artifacts/stop_live.ps1").replace("'", "''")
    result = subprocess.run(["powershell", "-NoProfile", "-Command",
        "$p = [pscustomobject]@{ Id = 1234 }; "
        "$p | Add-Member ScriptMethod CloseMainWindow { return $false }; "
        "function Get-Process { $p }; function Stop-Process { throw 'unexpected force kill' }; "
        f"& '{script}'"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "window is not ready" in result.stderr and "unexpected force kill" not in result.stderr
