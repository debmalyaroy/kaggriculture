# Third gate queue (2026-09-24): STRICT clone-gated controllers (positions + similarity, no step-1
# cash mirror) targeting profile 19 (ad_rsa12_l24) and 13 (aggr_deep), v2 table, ref = Rust p0.
param([int]$Clone = 19)
$ErrorActionPreference = 'Continue'
$rl = Split-Path -Parent $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
$exe = Join-Path $rl 'data\bin\agent-stdio-g3.exe'
$tab = Join-Path $rl 'configs\profiles\v2.json'
$log = Join-Path $rl "data\recordings\gate_queue3_$Clone.log"
& python (Join-Path $rl 'python\run_profile_gates.py') --profiles 0 --clone $Clone --strict --table $tab --workers 5 --exe $exe --ref rust_v611_p0_v61.1__064226311a6a 2>&1 | ForEach-Object { "$_" } | Out-File -FilePath $log -Append -Encoding utf8
"GATE-QUEUE3-DONE $Clone" | Out-File -FilePath $log -Append -Encoding utf8
