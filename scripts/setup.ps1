$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path '.venv/Scripts/python.exe')) {
    python -m venv --system-site-packages .venv
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
}
# Reuse the host CUDA stack; install development tools into the project venv.
& .venv/Scripts/python.exe -m pip install -e '.[dev]'
if ($LASTEXITCODE -ne 0) { throw 'Package installation failed' }
& .venv/Scripts/python.exe -m wan_dit_lab.cli doctor
if ($LASTEXITCODE -ne 0) { throw 'Environment check failed' }
