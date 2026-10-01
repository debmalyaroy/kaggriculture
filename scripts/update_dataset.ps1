# Refresh the mined data and publish a new version of the Kaggle dataset.
#
#   .\scripts\update_dataset.ps1                    # one refresh, now
#   .\scripts\update_dataset.ps1 -Episodes 120      # mine more this run
#   .\scripts\update_dataset.ps1 -Schedule          # daily, unattended
#   .\scripts\update_dataset.ps1 -Schedule -Remove  # uninstall
#   .\scripts\update_dataset.ps1 -Notebook          # also re-run it on Kaggle
#
# What one refresh does, in order and never in parallel:
#
#   1. src\kaggriculture\data\mine_top.py         stream new top-rated episodes, replay each
#                                sampled state through the current agent, and
#                                append to data\*.csv. Replays are deleted as
#                                they are used, so disk stays flat.
#   2. src\kaggriculture\pipeline\schedule.py         re-derive the SELL_SCHEDULE prior from the
#                                field's freshly measured sale timing
#   3. src\kaggriculture\data\kaggle_dataset.py   push a new version of the private dataset
#   4. (-Notebook) push the analysis kernel so it re-runs on Kaggle's compute
#
# Stages are sequential on purpose. Mining saturates every core, and an
# evaluation run on a saturated box is not just slow, it is wrong: the same
# agent on the same seed recorded $16,348 instead of $80,289 because turns that
# miss actTimeout get a substituted action.

param(
    [int]$Episodes = 60,
    [int]$Days = 7,
    [int]$Jobs = 8,
    [switch]$Notebook,
    [switch]$Schedule,
    [switch]$Remove,
    [string]$At = "03:30"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$TaskName = "KaggricultureDatasetUpdate"
$LogDir = Join-Path $Root ".local\logs"

if ($Schedule) {
    if ($Remove) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
        Write-Host "Removed scheduled task '$TaskName'."
        exit 0
    }
    $pwshExe = (Get-Command powershell).Source
    $self = Join-Path $Root "scripts\update_dataset.ps1"
    $args = "-ExecutionPolicy Bypass -NoProfile -File `"$self`" -Episodes $Episodes -Days $Days -Jobs $Jobs"
    if ($Notebook) { $args += " -Notebook" }

    $action = New-ScheduledTaskAction -Execute $pwshExe -Argument $args -WorkingDirectory $Root
    $trigger = New-ScheduledTaskTrigger -Daily -At $At
    # Mining is long and network-bound; give it room but never let it wedge.
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
        -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries `
        -ExecutionTimeLimit (New-TimeSpan -Hours 4) `
        -MultipleInstances IgnoreNew
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
        -Settings $settings -Force `
        -Description "Nightly Kaggriculture mine + Kaggle dataset version" | Out-Null

    Write-Host "Registered '$TaskName' - daily at $At."
    Write-Host "  mines   : $Episodes episode(s) across $Days day(s), $Jobs at a time"
    Write-Host "  log     : $LogDir\dataset_update.log"
    Write-Host "  run now : Start-ScheduledTask -TaskName $TaskName"
    Write-Host "  remove  : .\scripts\update_dataset.ps1 -Schedule -Remove"
    exit 0
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$log = Join-Path $LogDir "dataset_update.log"
$py = (Get-Command python).Source
$env:PYTHONUTF8 = "1"

function Stage([string]$name, [string[]]$argv) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] === $name ===" | Tee-Object -FilePath $log -Append | Write-Host
    & $py @argv 2>&1 | Tee-Object -FilePath $log -Append | Select-Object -Last 6
    if ($LASTEXITCODE -ne 0) {
        "[$stamp] $name exited $LASTEXITCODE" | Tee-Object -FilePath $log -Append | Write-Host
    }
}

Push-Location $Root
try {
    Stage "mine" @("-u", "src/kaggriculture/data/mine_top.py", "--episodes", "$Episodes",
                   "--days", "$Days", "--jobs", "$Jobs", "--sample-every", "6",
                   "--agent", "agents/v9_cem.py")

    Stage "derive schedule" @("-u", "src/kaggriculture/pipeline/schedule.py", "--derive",
                              "--base", "agents/v9_cem.py",
                              "--out", ".local/loop/seeded.py")

    Stage "publish dataset" @("-u", "src/kaggriculture/data/kaggle_dataset.py", "--update",
                              "--note", "nightly mine $(Get-Date -Format yyyy-MM-dd)")

    if ($Notebook) {
        Stage "push notebook" @("-u", "src/kaggriculture/agentbuild/build_kaggle_notebook.py",
                                "--agent", "agents/v9_cem.py", "--push")
    }

    Write-Host ""
    Write-Host "Next:"
    Write-Host "  python src\kaggriculture\data\mine_top.py --report        # what the field does vs what we do"
    Write-Host "  python src\kaggriculture\pipeline\schedule.py --show          # the derived sale timing"
    Write-Host "  python src\kaggriculture\pipeline\loop.py --cycles 1          # mine, search, gate, promote"
}
finally {
    Pop-Location
}
