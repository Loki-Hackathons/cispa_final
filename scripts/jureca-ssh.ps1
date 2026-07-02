# SSH to JURECA via Git Bash (ControlMaster works; Windows OpenSSH does not).
# Usage:
#   .\scripts\jureca-ssh.ps1 hostname
#   .\scripts\jureca-ssh.ps1 "squeue -u ansart1"
# First call in a session needs TOTP_CODE in env (agent asks user once):
#   $env:TOTP_CODE="123456"; .\scripts\jureca-ssh.ps1 hostname

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$RemoteCommand
)

$ErrorActionPreference = "Stop"

$GitBash = "C:\Program Files\Git\bin\bash.exe"
if (-not (Test-Path $GitBash)) {
    throw "Git Bash not found at $GitBash. Install Git for Windows."
}

$SshConfig = "/c/Users/super/.ssh/config"
$ControlPath = "/c/Users/super/.ssh/sockets/ctl-%r@%h-%p"
$AskPass = "/c/Users/super/.ssh/askpass_totp.sh"
$SocketsDir = "$env:USERPROFILE\.ssh\sockets"

New-Item -ItemType Directory -Force -Path $SocketsDir | Out-Null

function Invoke-JurecaBash {
    param([string]$BashCommand)
    & $GitBash -lc $BashCommand
    return $LASTEXITCODE
}

$checkCmd = "/usr/bin/ssh -F $SshConfig -o ControlPath=$ControlPath -O check jureca 2>&1"
$checkOut = & $GitBash -lc $checkCmd 2>&1 | Out-String
$masterRunning = $checkOut -match "Master running"

if (-not $masterRunning) {
    if (-not $env:TOTP_CODE) {
        Write-Error "No SSH master and TOTP_CODE not set. Ask the user for a fresh 6-digit TOTP, then:`n  `$env:TOTP_CODE='<code>'; .\scripts\jureca-ssh.ps1 hostname"
        exit 1
    }

    $masterCmd = @"
export TOTP_CODE='$($env:TOTP_CODE)' SSH_ASKPASS='$AskPass' SSH_ASKPASS_REQUIRE=force DISPLAY=:0
/usr/bin/ssh -F '$SshConfig' -o ControlMaster=yes -o ControlPath='$ControlPath' -o ControlPersist=24h jureca 'echo JURECA_MASTER_OK && hostname'
"@

    $rc = Invoke-JurecaBash $masterCmd
    if ($rc -ne 0) {
        Write-Error "Failed to establish SSH control master (exit $rc). TOTP may be invalid or expired."
        exit $rc
    }
}

$escaped = $RemoteCommand -replace "'", "'\\''"
$slaveCmd = "/usr/bin/ssh -F '$SshConfig' -o ControlMaster=no -o ControlPath='$ControlPath' jureca '$escaped'"
$rc = Invoke-JurecaBash $slaveCmd
exit $rc
