<#
.SYNOPSIS
    Start the local control panel.

.DESCRIPTION
    Checks prerequisites, frees the port if a previous run is still holding it,
    starts the server and opens your browser.

    The panel runs project commands by design, so it binds to 127.0.0.1 only and
    the live submit button stays disabled unless you ask for it explicitly.

.PARAMETER Port
    Default 8787.

.PARAMETER AllowSubmit
    Arm the live submit button. It still requires typing SUBMIT in the browser.

.PARAMETER NoBrowser
    Do not open a browser window.

.PARAMETER Stop
    Stop a dashboard already running on this port and exit.

.EXAMPLE
    .\scripts\dashboard.ps1

.EXAMPLE
    .\scripts\dashboard.ps1 -AllowSubmit -Port 9000

.EXAMPLE
    .\scripts\dashboard.ps1 -Stop
#>
[CmdletBinding()]
param(
    [int]$Port = 8787,
    [switch]$AllowSubmit,
    [switch]$NoBrowser,
    [switch]$Stop
)

. (Join-Path $PSScriptRoot '_lib.ps1')
$ErrorActionPreference = 'Stop'

function Get-PortOwner {
    param([int]$P)
    try {
        $conn = Get-NetTCPConnection -LocalPort $P -State Listen -ErrorAction SilentlyContinue |
                Select-Object -First 1
        if ($conn) { return Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue }
    } catch { }
    return $null
}

Write-Banner 'Kaggriculture - control panel' "http://127.0.0.1:$Port"

try {
    $owner = Get-PortOwner $Port

    if ($Stop) {
        if ($owner) {
            Write-Step "Stopping PID $($owner.Id) ($($owner.ProcessName)) on port $Port"
            Stop-Process -Id $owner.Id -Force
            Write-Ok 'stopped'
        } else {
            Write-Note "nothing is listening on port $Port"
        }
        exit 0
    }

    if ($owner) {
        Write-Warn2 "port $Port is already in use by PID $($owner.Id) ($($owner.ProcessName))"
        Write-Note "stop it:  .\scripts\dashboard.ps1 -Stop -Port $Port"
        Write-Note "or pick another:  .\scripts\dashboard.ps1 -Port $($Port + 1)"
        exit 1
    }

    Test-Prereq

    $serve = Join-Path $Root 'dashboard\serve.py'
    if (-not (Test-Path $serve)) { throw "not found: $serve" }

    $a = @('--port', $Port)
    if ($AllowSubmit) { $a += '--allow-submit' }
    if ($NoBrowser)   { $a += '--no-browser' }

    if ($AllowSubmit) {
        Write-Host ''
        Write-Host '  Live submit is ARMED. The browser will still require you to type' -ForegroundColor Yellow
        Write-Host '  SUBMIT. Five submissions a day; only the latest two stay active.' -ForegroundColor Yellow
    } else {
        Write-Note 'live submit is locked; re-run with -AllowSubmit to arm it'
    }

    Write-Step 'Starting the server (Ctrl-C to stop)'
    # Foreground on purpose: Ctrl-C should stop it, and its output is the log.
    Invoke-Py -Script 'dashboard\serve.py' -Arguments $a | Out-Null
} catch {
    Write-Fail $_.Exception.Message
    exit 1
}
