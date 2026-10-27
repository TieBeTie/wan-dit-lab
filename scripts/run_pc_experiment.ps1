param([string]$Config = 'configs/experiments/pc_fifo_480p.json')
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $taskRoot
$taskProtocol = Get-Content -LiteralPath (Join-Path $taskRoot $Config) -Raw -Encoding utf8 | ConvertFrom-Json
$taskOutput = Join-Path $taskRoot $taskProtocol.output
$taskMeta = Join-Path $taskOutput '_data'
New-Item -ItemType Directory -Path $taskMeta -Force | Out-Null
$taskWorker = Join-Path $taskMeta 'worker.json'
$taskStatus = Join-Path $taskMeta 'status.json'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONUTF8 = '1'
$env:CUDA_HOME = Join-Path $taskRoot 'artifacts/cuda_toolkit'
$env:CUDA_PATH = $env:CUDA_HOME
$env:PATH = "$env:CUDA_HOME/bin;" + $env:PATH
$env:TORCHINDUCTOR_COMPILE_THREADS = '1'
$taskPython = Join-Path $taskRoot '.venv/Scripts/python.exe'
@{status='running';pid=$PID;started=(Get-Date -Format o);config=$Config} |
    ConvertTo-Json | Set-Content -LiteralPath $taskWorker -Encoding utf8
try {
    @('generate', 'evaluate') | ForEach-Object {
        $taskAction = $_
        & $taskPython -u scripts/experiments/pc_ablation.py $taskAction --config $Config *> (Join-Path $taskMeta "$taskAction.log")
        if ($LASTEXITCODE -ne 0) { throw "$taskAction failed with exit code $LASTEXITCODE; see $taskAction.log" }
    }
    & $taskPython -u scripts/reports/pc_ablation_report.py --config $Config *> (Join-Path $taskMeta 'report.log')
    if ($LASTEXITCODE -ne 0) { throw "Report failed with exit code $LASTEXITCODE" }
    @{status='complete';finished=(Get-Date -Format o);config=$Config} |
        ConvertTo-Json | Set-Content -LiteralPath $taskWorker -Encoding utf8
} catch {
    $taskFailure = @{status='failed';error=$_.ToString();finished=(Get-Date -Format o);config=$Config}
    $taskFailure | ConvertTo-Json | Set-Content -LiteralPath $taskWorker -Encoding utf8
    # Keep the last per-chunk status intact; record failure separately for resumption.
    throw
}
