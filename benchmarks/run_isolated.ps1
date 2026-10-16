# Stops the voice loop (it holds VRAM), runs the full model benchmark, then restarts the voice loop.
$voiceDir = "C:\Inner Firmament\Growth Space\Projects\tutor-mcp\voice"
$voicePython = Join-Path $voiceDir ".venv\Scripts\python.exe"
$project = "C:\Inner Firmament\Growth Space\Projects\wan-dit-lab"
$log = Join-Path $project "benchmarks\bench_model.log"

Start-Sleep -Seconds 15  # lets the tutor finish speaking

$voice = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*voice_loop.py*" -and $_.Name -eq "python.exe" }
foreach ($p in $voice) { Stop-Process -Id $p.ProcessId -Force }
Start-Sleep -Seconds 5  # VRAM is released

try {
    Set-Location $project
    & python benchmarks\bench_model.py *> $log
}
finally {
    Start-Process -FilePath $voicePython -ArgumentList "-u", (Join-Path $voiceDir "voice_loop.py"), "--muted" -WorkingDirectory $voiceDir
}
