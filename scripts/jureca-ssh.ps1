# SSH to JURECA via WSL Ubuntu (ControlMaster works on Linux).
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
$WslDistro = "Ubuntu-26.04"
$script:WslExitCode = 0

function Invoke-Wsl {
    param([string]$Script)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $out = wsl -d $WslDistro -- bash -lc $Script 2>&1
        $script:WslExitCode = $LASTEXITCODE
        return ($out | Out-String).Trim()
    } finally {
        $ErrorActionPreference = $prev
    }
}

function Write-WslOutput {
    param([string]$Text)
    if ($Text) { Write-Output $Text }
}

function Get-JurecaMasterPid {
    $out = wsl -d $WslDistro -- bash -lc 'ssh -o ControlPath=~/.ssh/sockets/ctl-%r@%h-%p -O check jureca 2>&1' 2>&1 | Out-String
    if ($out -match 'Master running \(pid=(\d+)\)') { return $Matches[1] }
    return $null
}

function Stop-JurecaMaster {
    $masterPid = Get-JurecaMasterPid
    if ($masterPid) {
        $null = Invoke-Wsl "kill -9 $masterPid 2>/dev/null || true"
    }
    $null = Invoke-Wsl 'pkill -f "ssh.*ControlMaster.*jureca" 2>/dev/null || true'
    $null = Invoke-Wsl 'rm -f ~/.ssh/sockets/* ~/.ssh/totp_once.txt 2>/dev/null || true'
}

function Test-JurecaMasterAlive {
    $out = wsl -d $WslDistro -- bash -lc 'ssh -o ControlPath=~/.ssh/sockets/ctl-%r@%h-%p -O check jureca 2>&1' 2>&1 | Out-String
    if ($out -notmatch 'Master running') { return $false }
    $null = Invoke-Wsl "timeout 15 ssh -o ControlMaster=no jureca true"
    return ($script:WslExitCode -eq 0)
}

function Start-JurecaMaster {
    if (-not $env:TOTP_CODE) {
        Write-Error "No SSH master and TOTP_CODE not set. Ask the user for a fresh 6-digit TOTP, then:`n  `$env:TOTP_CODE='<code>'; .\scripts\jureca-ssh.ps1 hostname"
        exit 1
    }

    Stop-JurecaMaster

    $totp = $env:TOTP_CODE -replace "'", "'\\''"
    $null = Invoke-Wsl "printf '%s' '$totp' > ~/.ssh/totp_once.txt && ~/.local/bin/jureca-master.sh"
    if ($script:WslExitCode -ne 0) {
        Write-Error "Failed to establish SSH control master (exit $($script:WslExitCode)). TOTP may be invalid or expired."
        exit $script:WslExitCode
    }

    Start-Sleep -Seconds 1
    if (-not (Test-JurecaMasterAlive)) {
        Stop-JurecaMaster
        Write-Error "Failed to establish SSH control master. TOTP may be invalid or expired."
        exit 1
    }

    Remove-Item Env:TOTP_CODE -ErrorAction SilentlyContinue
}

function Invoke-JurecaRemote {
    param([string]$Command)
    $escaped = $Command -replace "'", "'\\''"
    $text = Invoke-Wsl "bash ~/.local/bin/jureca-run.sh '$escaped'"
    Write-WslOutput $text
}

if (-not (Test-JurecaMasterAlive)) {
    if (Get-JurecaMasterPid) { Stop-JurecaMaster }
    Start-JurecaMaster
}

Invoke-JurecaRemote $RemoteCommand
$rc = $script:WslExitCode
if ($rc -ne 0) {
    Stop-JurecaMaster
    Write-Error "SSH command failed (exit $rc). If master was lost, provide a fresh TOTP and retry."
    exit $rc
}
exit 0
