# Sequential field-gate queue for Rust-agent profiles (resumable: each gate skips recorded games).
# Order: p0 sanity (= v61.1), clone-gated controller (clone days -> profile 13 aggr_deep),
# fixed aggr_deep, then the single-knob profiles.
$ErrorActionPreference = 'Continue'
$rl = Split-Path -Parent $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
$lat = Join-Path $rl 'target-lat\release\agent-stdio.exe'
$log = Join-Path $rl 'data\recordings\gate_queue.log'
function Run($argsList) {
  & python (Join-Path $rl 'python\run_profile_gates.py') @argsList 2>&1 | ForEach-Object { "$_" } | Out-File -FilePath $log -Append -Encoding utf8
}
Run @('--profiles', '0', '--workers', '6')
Run @('--profiles', '0', '--clone', '13', '--workers', '6', '--exe', $lat)
Run @('--profiles', '13', '--workers', '6', '--exe', $lat)
Run @('--profiles', '12,2,10', '--workers', '6', '--exe', $lat)
'GATE-QUEUE-DONE' | Out-File -FilePath $log -Append -Encoding utf8

