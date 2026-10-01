# Install/repair the unattended daily-release schedule.
# Idempotent: run any time to restore the intended configuration.
#
# Why 06:30 local (IST, +05:30): Kaggle's 5/day submission quota resets at
# UTC midnight = 05:30 IST, and the daily episode dataset publishes ~00:09
# UTC = ~05:39 IST. 06:30 runs with a fresh quota AND the newest archive day.
# (The old 05:00 trigger was 23:30 UTC -- inside the PREVIOUS quota day.)

$ErrorActionPreference = 'Stop'

$action = New-ScheduledTaskAction -Execute 'D:\codebase\kaggriculture\scripts\daily_release.bat'
# 04:30 IST: the de-duplicated pipeline (refresh_cycle --no-fetch + hourly
# KaggricultureSameDay keeping the index current) measures ~60-75 min, and
# daily_release's quota floor holds stage-5 uploads until 00:05 UTC = 05:35
# IST -- so the pair lands 05:35-06:00 IST with a fresh quota, per the
# operator's 06:00 push target (set 2026-08-12).
$trigger = New-ScheduledTaskTrigger -Daily -At '04:30'
# Run every day up to and including the ENTRY DEADLINE, 2026-09-23.
# NO runs after the deadline: the trigger expires at Sep 24 00:00, so the
# last run is Sep 23, 06:30.
$trigger.EndBoundary = '2026-09-24T00:00:00'
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -WakeToRun `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 10) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 30) `
    -MultipleInstances IgnoreNew

# Principal: the USER, not SYSTEM. Kaggle CLI 2.2.4 authenticates via
# ~/.kaggle/access_token + credentials.json resolved from USERPROFILE, so a
# SYSTEM run finds no credentials and dies at stage_detect ("Save the token
# to ~/.kaggle/access_token", observed 2026-08-12 06:30). S4U = runs
# logged-out without a stored password; HTTPS + local files work under it.
$principal = New-ScheduledTaskPrincipal -UserId "$env:COMPUTERNAME\debma" `
    -LogonType S4U -RunLevel Limited

Set-ScheduledTask -TaskName 'KaggricultureRefreshCycle' `
    -Action $action -Trigger $trigger -Settings $settings `
    -Principal $principal | Out-Null

$t = Get-ScheduledTask -TaskName 'KaggricultureRefreshCycle'
$i = Get-ScheduledTaskInfo -TaskName 'KaggricultureRefreshCycle'
Write-Host 'State:              ' $t.State
Write-Host 'Action:             ' ($t.Actions | ForEach-Object { $_.Execute })
Write-Host 'NextRun:            ' $i.NextRunTime
Write-Host 'StartWhenAvailable: ' $t.Settings.StartWhenAvailable
Write-Host 'WakeToRun:          ' $t.Settings.WakeToRun
Write-Host 'RestartCount:       ' $t.Settings.RestartCount 'every' $t.Settings.RestartInterval
Write-Host 'ExecutionTimeLimit: ' $t.Settings.ExecutionTimeLimit
Write-Host 'BatteryStart OK:    ' (-not $t.Settings.DisallowStartIfOnBatteries)
Write-Host 'MultipleInstances:  ' $t.Settings.MultipleInstances
