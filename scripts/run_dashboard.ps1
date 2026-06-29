# Run browser dashboard (Windows PowerShell)
# Run from cispa_final root: .\scripts\run_dashboard.ps1
# Or from client/: npm run dashboard
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Port = 8080

function Get-PythonExe {
    $venv = Join-Path $Root ".venv\Scripts\python.exe"
    if (Test-Path $venv) { return $venv }
    return "python"
}

function Stop-ListenerOnPort {
    param([int]$TargetPort)
    $listeners = @(Get-NetTCPConnection -LocalPort $TargetPort -State Listen -ErrorAction SilentlyContinue)
    if ($listeners.Count -eq 0) { return }

    $pids = $listeners | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($procId in $pids) {
        $name = (Get-Process -Id $procId -ErrorAction SilentlyContinue).ProcessName
        Write-Host "Port $TargetPort busy ($name PID $procId) - stopping previous dashboard..."
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 1
}

$Python = Get-PythonExe
Write-Host "Using Python: $Python"
Write-Host "Installing dashboard Python dependencies..."
& $Python -m pip install -r dashboard/requirements.txt -q

$DistPath = Join-Path $Root "client\dist\index.html"
if (-not (Test-Path $DistPath)) {
    Write-Host "Client not built yet - running build..."
    & (Join-Path $PSScriptRoot "build_dashboard.ps1")
    Set-Location $Root
}

Stop-ListenerOnPort -TargetPort $Port

Write-Host ""
Write-Host "Starting dashboard (MODE from dashboard/config.py)..."
Write-Host "Open http://127.0.0.1:$Port"
Write-Host 'Press Ctrl+C to stop'
Write-Host ""

& $Python -m uvicorn dashboard.server:app --host 127.0.0.1 --port $Port
