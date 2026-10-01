<#
O5: register (or run) the daily Slot-2 pipeline as a Windows Scheduled Task.
It builds + gates both seats and writes a report; it NEVER submits.

  .\scripts\schedule_daily_slot2.ps1 -Register -At 06:00   # daily at 06:00
  .\scripts\schedule_daily_slot2.ps1 -Run                  # run once now
  .\scripts\schedule_daily_slot2.ps1 -Unregister
#>
param(
  [switch]$Register,
  [switch]$Unregister,
  [switch]$Run,
  [string]$At = "06:00",
  [int]$RlRounds = 2
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Py = "C:/ProgramData/anaconda3/envs/llm/python.exe"
$TaskName = "KaggricultureSlot2Daily"

# the command: run after the daily data fetch; never submits
$Inner = "cd `"$Root`"; `$env:PYTHONPATH='src;vendor'; `$env:PYTHONUTF8='1'; " +
         "& `"$Py`" -m kaggriculture.pipeline.daily_slot2 --rl-rounds $RlRounds " +
         "*> .local/scratch/daily_slot2_cron.log"

if ($Run) {
  Write-Host "[schedule] running daily_slot2 once (rounds=$RlRounds)..."
  powershell -NoProfile -Command $Inner
  Write-Host "[schedule] done -> .local/candidates/release_report.md"
  exit 0
}

if ($Unregister) {
  Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
  Write-Host "[schedule] unregistered $TaskName"
  exit 0
}

if ($Register) {
  $action  = New-ScheduledTaskAction -Execute "powershell.exe" `
             -Argument "-NoProfile -WindowStyle Hidden -Command `"$Inner`""
  $trigger = New-ScheduledTaskTrigger -Daily -At $At
  $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
              -DontStopOnIdleEnd -ExecutionTimeLimit (New-TimeSpan -Hours 4)
  Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Description "Daily Slot-2 build+gate (never submits)" -Force | Out-Null
  Write-Host "[schedule] registered $TaskName daily at $At (rounds=$RlRounds). It NEVER submits."
  exit 0
}

Write-Host "Usage: -Register [-At HH:mm] [-RlRounds N] | -Run | -Unregister"
