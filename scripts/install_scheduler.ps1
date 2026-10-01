# Register an hourly Kaggriculture data fetch as a Windows Scheduled Task.
#
#   powershell -ExecutionPolicy Bypass -File tools\install_scheduler.ps1
#   powershell -ExecutionPolicy Bypass -File tools\install_scheduler.ps1 -Remove
#
# The task runs tools\scheduled_fetch.py, which is incremental and timestamped:
# each run resumes from data\fetch_state.json, so nothing is re-downloaded.

param([switch]$Remove, [int]$IntervalHours = 1)

$TaskName = "KaggricultureFetch"
$Root     = Split-Path -Parent $PSScriptRoot
$Script   = Join-Path $Root "tools\scheduled_fetch.py"
$LogDir   = Join-Path $Root ".local\logs"

if ($Remove) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Removed scheduled task '$TaskName'."
    exit 0
}

if (-not (Test-Path $Script)) { Write-Error "not found: $Script"; exit 1 }
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { Write-Error "python is not on PATH"; exit 1 }

# cmd wrapper so stdout/stderr land in a dated log
$cmd = "/c `"cd /d `"$Root`" && `"$py`" `"$Script`" >> `"$LogDir\fetch.log`" 2>&1`""

$action    = New-ScheduledTaskAction -Execute "cmd.exe" -Argument $cmd
$trigger   = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2) `
             -RepetitionInterval (New-TimeSpan -Hours $IntervalHours)
$settings  = New-ScheduledTaskSettingsSet -StartWhenAvailable `
             -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries `
             -ExecutionTimeLimit (New-TimeSpan -Minutes 50)

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Description "Hourly incremental Kaggriculture episode fetch" `
    -Force | Out-Null

Write-Host "Registered '$TaskName' - every $IntervalHours h, starting in 2 minutes."
Write-Host "  log    : $LogDir\fetch.log"
Write-Host "  state  : $Root\data\fetch_state.json"
Write-Host "  run now: Start-ScheduledTask -TaskName $TaskName"
Write-Host "  remove : powershell -File tools\install_scheduler.ps1 -Remove"
