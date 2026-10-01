# Second gate queue (2026-09-24): clone-gated controller targeting profile 19 (ad_rsa10_l20, v2 table),
# then fixed profile 19 (its cost vs non-clones = the controller's false-positive exposure).
$ErrorActionPreference = 'Continue'
$rl = Split-Path -Parent $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
$exe = Join-Path $rl 'data\bin\agent-stdio-g2.exe'
$tab = Join-Path $rl 'configs\profiles\v2.json'
$log = Join-Path $rl 'data\recordings\gate_queue2.log'
function Run($argsList) {
  & python (Join-Path $rl 'python\run_profile_gates.py') @argsList 2>&1 | ForEach-Object { "$_" } | Out-File -FilePath $log -Append -Encoding utf8
}
Run @('--profiles', '0', '--clone', '19', '--table', $tab, '--workers', '6', '--exe', $exe, '--ref', 'rust_v611_p0_v61.1__064226311a6a')
Run @('--profiles', '19', '--table', $tab, '--workers', '6', '--exe', $exe)
'GATE-QUEUE2-DONE' | Out-File -FilePath $log -Append -Encoding utf8
