# Run the full improvement cycle, or schedule it to run itself.
#
#   .\scripts\autopilot.ps1                     # one cycle, now
#   .\scripts\autopilot.ps1 -Quick              # small budgets, ~15 min
#   .\scripts\autopilot.ps1 -Stages train,build,gate
#   .\scripts\autopilot.ps1 -Schedule           # daily at 03:00, unattended
#   .\scripts\autopilot.ps1 -Schedule -At 05:30 -Quick
#   .\scripts\autopilot.ps1 -Schedule -Remove   # uninstall
#   .\scripts\autopilot.ps1 -Status
#
# A cycle is: mine fresh routes from the top-200 -> re-derive the sale-timing
# prior and search the parameter vector -> package and run the submission
# contract -> play the candidate against the incumbent on seeds it was never
# tuned on. It promotes only on a win, and it never uploads: the submit stage
# shells out to src\kaggriculture\pipeline\submit.py, which asks for a typed SUBMIT.
#
# Scheduled runs deliberately omit the submit stage entirely. Five submissions
# a day and only the latest two active make an unattended upload a way to evict
# a better agent while you sleep.
#
# The task runs one instance at a time (-MultipleInstances IgnoreNew) because
# stages must not overlap: a search sharing the box with a download does not
# just run slower, it records the wrong numbers. The same agent on the same
# seed has read $16,348 and $80,289 depending on what else was running.

param(
    [switch]$Schedule,
    [switch]$Remove,
    [switch]$Status,
    [switch]$Quick,
    [string]$Stages = "fetch,train,build,gate",
    [int]$Top = 200,
    [int]$PerTeam = 3,
    [int]$Jobs = 8,
    [string]$At = "03:00"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$TaskName = "KaggricultureAutopilot"
$LogDir = Join-Path $Root ".local\logs"
$py = (Get-Command python).Source

if ($Status) {
    Push-Location $Root
    try {
        $env:PYTHONUTF8 = "1"
        & $py -u src\kaggriculture\pipeline\autopilot.py --status
        $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        if ($task) {
            $info = Get-ScheduledTaskInfo -TaskName $TaskName
            Write-Host ""
            Write-Host "scheduled : $TaskName ($($task.State))"
            Write-Host "  last run: $($info.LastRunTime)  result $($info.LastTaskResult)"
            Write-Host "  next run: $($info.NextRunTime)"
        } else {
            Write-Host ""
            Write-Host "not scheduled. Install with: .\scripts\autopilot.ps1 -Schedule"
        }
    } finally { Pop-Location }
    exit 0
}

if ($Schedule) {
    if ($Remove) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
        Write-Host "Removed scheduled task '$TaskName'."
        exit 0
    }

    if ($Stages -like "*submit*") {
        throw "refusing to schedule the submit stage. Only the latest 2 submissions stay active, so an unattended upload can evict a better agent. Run submit by hand."
    }

    $self = Join-Path $Root "scripts\autopilot.ps1"
    $argline = "-ExecutionPolicy Bypass -NoProfile -File `"$self`" -Stages $Stages -Top $Top -PerTeam $PerTeam -Jobs $Jobs"
    if ($Quick) { $argline += " -Quick" }

    $action = New-ScheduledTaskAction -Execute (Get-Command powershell).Source `
        -Argument $argline -WorkingDirectory $Root
    $trigger = New-ScheduledTaskTrigger -Daily -At $At
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
        -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries `
        -ExecutionTimeLimit (New-TimeSpan -Hours 8) `
        -MultipleInstances IgnoreNew
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
        -Settings $settings -Force `
        -Description "Kaggriculture: mine fresh routes, retrain, gate. Never submits." | Out-Null

    Write-Host "Registered '$TaskName' - daily at $At."
    Write-Host "  stages  : $Stages   (submit is refused for scheduled runs)"
    Write-Host "  slice   : top $Top teams, $PerTeam route(s) each"
    Write-Host "  log     : $LogDir\autopilot\"
    Write-Host "  run now : Start-ScheduledTask -TaskName $TaskName"
    Write-Host "  status  : .\scripts\autopilot.ps1 -Status"
    Write-Host "  remove  : .\scripts\autopilot.ps1 -Schedule -Remove"
    exit 0
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$env:PYTHONUTF8 = "1"
Push-Location $Root
try {
    $argv = @("-u", "src\kaggriculture\pipeline\autopilot.py", "--stages", $Stages,
              "--top", "$Top", "--per-team", "$PerTeam", "--jobs", "$Jobs")
    if ($Quick) { $argv += "--quick" }
    & $py @argv
    $rc = $LASTEXITCODE
    Write-Host ""
    Write-Host "Next:"
    Write-Host "  .\scripts\autopilot.ps1 -Status            # what the cycle did"
    Write-Host "  python src\kaggriculture\pipeline\submit.py --dry-run           # validate the promoted agent"
    Write-Host "  .\scripts\autopilot.ps1 -Schedule          # make it run itself daily"
    exit $rc
} finally { Pop-Location }
