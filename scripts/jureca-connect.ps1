# Establish SSH ControlMaster to JURECA via WSL (TOTP once). Does not run remote commands.
# Usage:
#   .\scripts\jureca-connect.ps1
# First call needs TOTP_CODE in env (agent asks user once in chat):
#   $env:TOTP_CODE="123456"; .\scripts\jureca-connect.ps1

$ErrorActionPreference = "Stop"
. "$PSScriptRoot/jureca-lib.ps1"

Ensure-JurecaMaster
$check = Invoke-Wsl "ssh -o ControlPath=~/.ssh/sockets/ctl-%r@%h-%p -O check jureca 2>&1"
Write-Output $check
