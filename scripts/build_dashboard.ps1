# Build React client (Windows PowerShell)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location (Join-Path $Root "client")

Write-Host "Installing client dependencies..."
npm install

Write-Host "Building client..."
npm run build

Write-Host ""
Write-Host "Done. Start the dashboard from cispa_final root (NOT from client/):"
Write-Host "  cd $Root"
Write-Host "  .\scripts\run_dashboard.ps1"
Write-Host ""
Write-Host "Or from client/:"
Write-Host "  npm run dashboard"
Write-Host ""
Write-Host "Open http://127.0.0.1:8080"
