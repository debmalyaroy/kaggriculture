<#
.SYNOPSIS
    Download Kaggriculture episode replays: the top-20 ladder's games and all of
    your own.

.DESCRIPTION
    Wraps download_data.py, which streams: fetch one replay (~27 MB), reduce it
    to ~60 numbers, delete it, next. Disk never grows and re-running only pulls
    what is new -- state lives in data\fetch_state.json.

.PARAMETER Days
    How many daily top-episode datasets to mine (Kaggle publishes one per day).

.PARAMETER PerDay
    Top episodes to take from each day.

.PARAMETER Own
    Your own episodes. 0 means all of them.

.PARAMETER TopN
    Restrict ladder episodes to the top-N leaderboard teams.

.PARAMETER Jobs
    Parallel replay downloads. Each replay is ~27 MB and every one pays a fresh
    kaggle-CLI startup (SDK import + TLS + auth, 1-2s), so serial downloading
    spent most of its time idle. 8 is a good default; 12-16 helps on a fast link.

.PARAMETER AnyTop
    Skip the team filter and just take the highest-rated episodes. The daily
    dataset is already ranked by agent rating, so this is very nearly the same
    set -- use it if leaderboard name resolution is failing.

.PARAMETER Check
    Preflight only: verify CLI, auth, competition entry and disk, then stop.

.PARAMETER Diagnose
    Show the raw leaderboard CLI output and what gets parsed out of it. Use this
    when the run reports 0 team names resolved.

.PARAMETER Report
    Rebuild docs\eda-report.html afterwards.

.PARAMETER Schedule
    Install (or, with -Remove, uninstall) the hourly Windows scheduled task.

.EXAMPLE
    .\scripts\download.ps1 -Check

.EXAMPLE
    .\scripts\download.ps1 -PerDay 60 -Days 5 -Verbose2 -Report

.EXAMPLE
    .\scripts\download.ps1 -AnyTop          # ignore leaderboard names
#>
[CmdletBinding()]
param(
    [int]$Days = 3,
    [int]$PerDay = 40,
    [int]$Own = 0,
    [int]$TopN = 20,
    [int]$Jobs = 8,
    [switch]$AnyTop,
    [switch]$Check,
    [switch]$Diagnose,
    [switch]$Report,
    [switch]$Verbose2,
    [switch]$Schedule,
    [switch]$Remove
)

. (Join-Path $PSScriptRoot '_lib.ps1')
$ErrorActionPreference = 'Stop'

Write-Banner 'Kaggriculture - data download' 'top-20 ladder replays + all of your own'
$log = Start-Log 'download'
$exit = 0

try {
    if ($Schedule) {
        Write-Step 'Installing the hourly fetch task'
        $psArgs = @('-ExecutionPolicy', 'Bypass', '-File', (Join-Path $Root 'src\install_scheduler.ps1'))
        if ($Remove) { $psArgs += '-Remove' }
        & powershell @psArgs
        Stop-Log 0
        return
    }

    Test-Prereq -NeedKaggle

    $a = @('--days', $Days, '--per-day', $PerDay, '--own', $Own,
           '--top-n', $TopN, '--jobs', $Jobs)
    if ($AnyTop)   { $a += '--any-top' }
    if ($Check)    { $a += '--check' }
    if ($Diagnose) { $a += '--leaderboard' }
    if ($Report)   { $a += '--report' }
    if ($Verbose2) { $a += '--verbose' }

    Invoke-Py -Script 'download_data.py' -Arguments $a -Title 'Downloading episodes' | Out-Null

    if (-not ($Check -or $Diagnose)) {
        $rows = Invoke-PyInline @'
import os
p = os.path.join("data", "episodes.csv")
print(sum(1 for _ in open(p, encoding="utf-8")) - 1 if os.path.exists(p) else 0)
'@
        Write-Ok "data\episodes.csv holds $rows player-rows"
    }

    Show-Next @(
        '.\scripts\download.ps1 -Schedule        # keep it fresh hourly',
        '.\scripts\train.ps1 -Algo learn         # what predicts winning',
        'python src\kaggriculture\pipeline\eda_report.py              # docs\eda-report.html'
    )
} catch {
    Write-Fail $_.Exception.Message
    $exit = 1
}

Stop-Log $exit
exit $exit
