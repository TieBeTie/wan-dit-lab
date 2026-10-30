# Builds the one entry point of wan-dit-lab, wan.exe (crates/wan), and puts it into bin\: bin\wan.exe (release) and
# bin\wan-debug.exe (with live check and live render, --features wan/debug-tools). Nothing else runs a video:
#   bin\wan.exe                                    the window (configs\live.json, "mode": "live")
#   bin\wan.exe --config configs\offline.json      chains of prompts into mp4 files ("mode": "offline")
#   powershell -NoProfile -ExecutionPolicy Bypass -File build.ps1 [-DebugTools] [-Print]
# -DebugTools builds only wan-debug.exe; -Print prints what it would build and builds nothing.
[CmdletBinding()]
param([switch]$DebugTools, [switch]$Print)
Set-Location $PSScriptRoot
$builds = @(
  @{ args = @("--release", "-p", "wan"); target = "target"; out = "bin\wan.exe" },
  @{ args = @("--release", "-p", "wan", "--features", "wan/debug-tools", "--target-dir", "target\debug-tools"); target = "target\debug-tools"; out = "bin\wan-debug.exe" }
)
if ($DebugTools) { $builds = @($builds[1]) }
foreach ($b in $builds) {
  if ($Print) { "cargo build $($b.args -join ' ') -> $($b.out)"; continue }
  & cargo build @($b.args)
  if ($LASTEXITCODE -ne 0) { throw "cargo build $($b.args -join ' ') failed" }
  New-Item -ItemType Directory -Force bin | Out-Null
  Copy-Item -Force "$($b.target)\release\wan.exe" $b.out
  "built $($b.out)"
}
