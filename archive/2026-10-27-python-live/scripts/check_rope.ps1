param([switch]$TestsOnly, [switch]$OpenReport, [switch]$Live, [switch]$CacheStore)

$ErrorActionPreference = 'Stop'
$ropeRoot = Split-Path $PSScriptRoot -Parent
$ropeWslRoot = '/mnt/' + $ropeRoot.Substring(0, 1).ToLower() + '/' + $ropeRoot.Substring(3).Replace('\', '/')
$ropeArguments = @(
    '-d', 'Ubuntu-24.04', '--cd', $ropeWslRoot, '--', 'env',
    'TORCHINDUCTOR_CACHE_DIR=/home/rama/.cache/wan-rope-inductor',
    '/home/rama/.venvs/wan-rope/bin/python', 'scripts/checks/check_rope.py'
)
if ($TestsOnly) {
    $ropeArguments += '--tests-only'
}
if ($Live -or $CacheStore) {
    if (Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue) {
        throw 'Live is already running. Close its window before starting another comparison.'
    }
    $ropeNativePython = Join-Path $ropeRoot '.venv\Scripts\python.exe'
    $env:CUDA_HOME = Join-Path $ropeRoot 'artifacts\cuda_toolkit'
    $env:CUDA_PATH = $env:CUDA_HOME
    $env:PATH = "$env:CUDA_HOME\bin;$env:PATH"
    $env:PYTHONPATH = Join-Path $ropeRoot 'src'
    $env:CC = & $ropeNativePython -c "import pathlib,triton; print(pathlib.Path(triton.__file__).parent / 'runtime/tcc/tcc.exe')"
    if ($CacheStore) {
        $cacheArguments = @((Join-Path $ropeRoot 'scripts/checks/check_cache_fp8.py'))
        if ($TestsOnly) { $cacheArguments += '--tests-only' }
        & $ropeNativePython @cacheArguments
    } else {
        & $ropeNativePython -m pytest tests/test_rope_triton.py tests/test_rope_metrics.py tests/test_rope_report.py tests/test_live_rope_metrics.py -q
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        if (-not $TestsOnly) { & $ropeNativePython scripts/checks/compare_live_rope.py }
    }
    $ropeExit = $LASTEXITCODE
} else {
    & wsl @ropeArguments
    $ropeExit = $LASTEXITCODE
}
if ($OpenReport) {
    $ropeConfig = Get-Content (Join-Path $ropeRoot 'configs\experiments\rope_triton.json') -Raw | ConvertFrom-Json
    $ropeLatest = Get-Content (Join-Path $ropeConfig.output_dir 'latest.json') -Raw | ConvertFrom-Json
    $ropeHtml = $ropeLatest.report_html -replace '^/mnt/([a-z])/', '$1:/'
    Start-Process -FilePath $ropeHtml
}
exit $ropeExit
