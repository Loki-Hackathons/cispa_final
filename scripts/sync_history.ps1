# Pull submission history + team state from JURECA to the laptop.
# Single ssh call (one TOTP code). Usage: .\scripts\sync_history.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$RemoteBase = "/p/scratch/training2625/ansart1/loki"
$Key = "$env:USERPROFILE\.ssh\id_ed25519"

New-Item -ItemType Directory -Force -Path "$Root\history" | Out-Null

Write-Host "Pulling history + team_state from JURECA (one TOTP prompt)..."
# tar over ssh: one connection for both files; missing files are skipped
ssh -i $Key -o Ciphers=aes256-ctr -o MACs=hmac-sha2-256-etm@openssh.com `
    ansart1@jureca.fz-juelich.de `
    "cd $RemoteBase && tar -cf - --ignore-failed-read history/submissions.jsonl team_state.json 2>/dev/null" `
    | tar -xf - -C $Root

Write-Host ""
Write-Host "Done. Read with:  python shared/history.py list"
