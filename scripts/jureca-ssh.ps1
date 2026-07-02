# Run a remote command on JURECA via WSL ControlMaster.
# Prefer Phase 2 from AGENTS.md (WSL ssh/scp direct) for routine agent work — this script
# adds an extra PowerShell quoting layer and is kept for backward compatibility.
#
# Usage:
#   .\scripts\jureca-ssh.ps1 "squeue -u ansart1"
# Establish master first (TOTP once):
#   $env:TOTP_CODE="123456"; .\scripts\jureca-connect.ps1

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$RemoteCommand
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot/jureca-lib.ps1"

Ensure-JurecaMaster
$rc = Invoke-JurecaRemote $RemoteCommand
if ($rc -ne 0) {
    Write-Error "Remote command failed (exit $rc). SSH master is still alive — fix the command and retry."
    exit $rc
}
exit 0
