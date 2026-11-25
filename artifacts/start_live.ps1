# Starts Live: bin\wan.exe (built by build.ps1, the one entry point) over a configuration, configs\mvp2.json by
# default, with an optional initial prompt override; returns at once. wan.exe reads the rest of the configuration
# itself (export, prompts, generation and offloading). The log goes to artifacts\live.log. A Live already running holds the GPU and
# port 8765: then nothing is started. -Print prints the command and starts nothing.
#   powershell -NoProfile -ExecutionPolicy Bypass -File artifacts\start_live.ps1 -Prompt "A red fox in a snowy forest..."
[CmdletBinding()]
param([string]$Config = "configs\mvp2.json", [switch]$Print, [string]$Prompt)
Set-Location (Split-Path $PSScriptRoot)
$exe = "bin\wan.exe"
$arguments = @('--config', $Config)
if ($PSBoundParameters.ContainsKey('Prompt')) {
  if ([string]::IsNullOrWhiteSpace($Prompt)) { throw 'Prompt must be nonempty or omitted' }
  $arguments += @('--prompt', ('"' + $Prompt.Replace('"', "'") + '"'))
}
if ($Print) { "$exe $($arguments -join ' ')"; exit 0 }
$running = Get-Process -Name wan,wan-debug -ErrorAction SilentlyContinue
if ($running) { "Live already running (pid $($running[0].Id))"; exit 0 }
if (-not (Test-Path $exe)) { "Live is not built: powershell -File build.ps1"; exit 1 }
$p = Start-Process -FilePath $exe -ArgumentList $arguments -WindowStyle Hidden -RedirectStandardOutput artifacts\live.log -RedirectStandardError artifacts\live.err.log -PassThru
"Live started (pid $($p.Id)); progress: artifacts\live.log"
