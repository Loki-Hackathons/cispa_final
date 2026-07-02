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
$AskPass = "/c/Users/super/.ssh/askpass_totp_once.sh"
$TotpFile = "/c/Users/super/.ssh/totp_once.txt"
$TotpLock = "/c/Users/super/.ssh/totp_once.lock"
$SocketsDir = "$env:USERPROFILE\.ssh\sockets"

New-Item -ItemType Directory -Force -Path $SocketsDir | Out-Null

function Invoke-JurecaBash {
    param([string]$BashCommand)
    $out = & $GitBash -lc $BashCommand 2>&1
    $text = ($out | Out-String).Trim()
    if ($text) { Write-Output $text }
    return $LASTEXITCODE
}

function Clear-TotpOnce {
    $null = Invoke-JurecaBash "rm -f '$TotpFile' '$TotpLock' 2>/dev/null || true"
}

function Test-JurecaMaster {
    $cmd = "/usr/bin/ssh -F '$SshConfig' -o ControlPath='$ControlPath' -O check jureca 2>&1"
    $out = (& $GitBash -lc $cmd 2>&1 | Out-String)
    return ($out -match "Master running")
}

function Stop-JurecaMaster {
    $null = Invoke-JurecaBash "/usr/bin/ssh -F '$SshConfig' -o ControlPath='$ControlPath' -O exit jureca 2>/dev/null || true"
    $null = Invoke-JurecaBash "rm -f /c/Users/super/.ssh/sockets/* 2>/dev/null || true"
}

function Start-JurecaMaster {
    if (-not $env:TOTP_CODE) {
        Write-Error "No SSH master and TOTP_CODE not set. Ask the user for a fresh 6-digit TOTP, then:`n  `$env:TOTP_CODE='<code>'; .\scripts\jureca-ssh.ps1 hostname"
        exit 1
    }

    Stop-JurecaMaster
    Clear-TotpOnce

    $masterCmd = @"
printf '%s' '$($env:TOTP_CODE)' > '$TotpFile'
chmod +x '$AskPass'
export SSH_ASKPASS='$AskPass' SSH_ASKPASS_REQUIRE=force DISPLAY=:0
/usr/bin/ssh -F '$SshConfig' -o NumberOfPasswordPrompts=1 -o ControlMaster=yes -o ControlPath='$ControlPath' -o ControlPersist=24h -MNf jureca
rc=`$?
rm -f '$TotpFile' '$TotpLock' 2>/dev/null || true
exit `$rc
"@

    $rc = Invoke-JurecaBash $masterCmd
    if ($rc -ne 0) {
        Write-Error "Failed to establish SSH control master (exit $rc). TOTP may be invalid or expired."
        exit $rc
    }

    Start-Sleep -Seconds 1
    if (-not (Test-JurecaMaster)) {
        Write-Error "SSH master did not start. TOTP may be invalid or expired."
        exit 1
    }
}

if (-not (Test-JurecaMaster)) {
    Start-JurecaMaster
}

$escaped = $RemoteCommand -replace "'", "'\\''"
$slaveCmd = "/usr/bin/ssh -F '$SshConfig' -o ControlMaster=no -o ControlPath='$ControlPath' jureca '$escaped'"
$rc = Invoke-JurecaBash $slaveCmd
exit $rc
