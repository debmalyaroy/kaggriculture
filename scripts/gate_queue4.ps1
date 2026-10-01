# Fourth gate queue (2026-09-24): fixed-profile field gates for the rsa neighbourhood (v2 table):
# 23 rsa8_l12, 21 rsa10. Waits for queue 1 (gate_queue.log GATE-QUEUE-DONE) to keep RAM in budget.
$ErrorActionPreference = 'Continue'
$rl = Split-Path -Parent $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
$exe = Join-Path $rl 'data\bin\agent-stdio-g4.exe'
$tab = Join-Path $rl 'configs\profiles\v2.json'
$q1 = Join-Path $rl 'data\recordings\gate_queue.log'
$log = Join-Path $rl 'data\recordings\gate_queue4.log'
while (-not (Select-String -Path $q1 -Pattern 'GATE-QUEUE-DONE' -Quiet)) { Start-Sleep -Seconds 60 }
foreach ($p in @('23', '21')) {
  & python (Join-Path $rl 'python\run_profile_gates.py') --profiles $p --table $tab --workers 5 --exe $exe --ref rust_v611_p0_v61.1__064226311a6a 2>&1 | ForEach-Object { "$_" } | Out-File -FilePath $log -Append -Encoding utf8
}
'GATE-QUEUE4-DONE' | Out-File -FilePath $log -Append -Encoding utf8
