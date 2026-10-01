# Prints the number of HEAVY project jobs currently running.
#
# Called by sameday_scrape.bat's skip guard. Inline batch->powershell quoting
# broke once ('^|' and escaped quotes mangled -> exit 255, every hourly run died
# before its first log line, 2026-08-13), which is why this lives in a file.
#
# Broadened 2026-08-13 from "daily_release only" after enabling every scheduled
# task exposed the gap: KaggricultureAutopilot runs at 04:00 with
# `routes.py --mine --jobs 8`, and KaggricultureRefreshCycle at 04:30, but the
# guard could not see the autopilot at all. The 04:35 hourly scrape would then
# start fetching against the same endpoints while the mine was still going --
# which is precisely the 429 contention that once left a cycle with nothing
# built. Skipping one hourly sip is cheap; a throttle storm during the release
# window is not.
#
# Matches on the COMMAND LINE rather than the process name because every job is
# python.exe or powershell.exe. Deliberately does not match this script itself.
$patterns = 'daily_release', 'refresh_cycle', 'autopilot', 'routes\.py.*--mine'
$rx = ($patterns -join '|')

$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='powershell.exe' OR Name='pwsh.exe'" |
         Where-Object {
            $_.CommandLine -and
            $_.CommandLine -match $rx -and
            $_.CommandLine -notmatch 'release_running'
         }

Write-Output ([int]($procs | Measure-Object).Count)
