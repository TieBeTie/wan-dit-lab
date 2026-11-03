# Closes Live normally, so the generator and end metrics finish before the GPU is released.
#   powershell -NoProfile -ExecutionPolicy Bypass -File artifacts\stop_live.ps1
[CmdletBinding()]
param([switch]$Force)
$ErrorActionPreference = 'Stop'
$running = Get-Process -Name wan,wan-debug -ErrorAction SilentlyContinue
if (-not $running) { "Live is not running"; exit 0 }
foreach ($process in $running) {
  if ($Force) { Stop-Process -Id $process.Id -Force; continue }
  if (-not $process.CloseMainWindow()) {
    throw "Live window is not ready (pid $($process.Id)); retry or use -Force"
  }
  if (-not $process.WaitForExit(30000)) {
    throw "Live is still finishing (pid $($process.Id)); retry or use -Force"
  }
}
"Live stopped (pid $(($running | ForEach-Object { $_.Id }) -join ', '))"
