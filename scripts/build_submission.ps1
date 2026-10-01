<#
.SYNOPSIS
  Build a Kaggle submission tarball of the Rust v61.x agent (task P8.1/P8.2). NEVER submits.

.DESCRIPTION
  1. Cross-builds crates/agent `agent-stdio` for x86_64-unknown-linux-musl (static) in a
     rust:latest container (target dir target-linux, shared with the corpus build).
  2. Stages main.py (from kaggle/submission/main_template.py with the profile settings filled in),
     the binary, base/ (routes.json + router.json), profiles.json, fallback.py (the Python v61.1).
  3. Packs data/builds/<Name>/submission.tar.gz inside the container so the exec bit survives.

.EXAMPLE
  .\scripts\build_submission.ps1 -Name v62_ctrl13 -CloneProfile 13
  .\scripts\build_submission.ps1 -Name v64rl_lead32copy -Profile 35 -Group 35,35,36 -Profiles configs\profiles\v64rl.json
  .\scripts\build_submission.ps1 -Name v611_rust                   # profile 0 == v61.1
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]$Name,
    [int]$Profile = -1,
    [int]$CloneProfile = -1,
    [switch]$CloneStrict,
    [switch]$Disguise,
    [string]$ChainOff = "",   # comma list of chain stages off for the whole game (agent-stdio --chain-off), e.g. "r127,sm,r95"
    [string]$Shell = "",      # learned sales shell (shell.json from python/top50/train_shell.py + eval_shell.py)        # stream disguise (no-op orders; results unchanged)
    [string]$RShell = "",     # reactive shell v2 config json (its net_file / lineage_file are shipped beside it in rshell/)
    [string]$Preempt = "",    # sale-control layer config json (its "clusters" file is shipped as lineages.json)
    [string]$Lineages = "",   # lineage decision tables (configs/opp/lineages_v1.json)
    [string]$GroupKnobs = "", # knob overlay applied only against rival groups -GroupKnobsFor (0 DIFFERENT, 1 PARTIAL, 2 COPY)
    [string]$GroupKnobsFor = "0,1",
    [string]$KnobOver = "",   # chain-constant overrides json (tuned with the shell)
    [string]$Endg = "",       # learned endgame model json
    [string]$Gt = "",         # game-theoretic liquidation layer json (v63.12)
    [string]$Dispatch = "",   # specialist dispatcher json; every file it names is shipped beside it in dispatch/ (v63.12)
    [string]$Group = "",      # opponent-group controller "P_DIFF,P_PARTIAL,P_COPY" (needs -Profile for day 0), e.g. "35,35,36"
    [string]$Policy = "",
    [string]$Base = "configs\bases\v61.1",
    [string]$Profiles = "",   # default: configs\profiles\rl3.json with -Policy (the table it was trained on), else v1.json
    [string]$Shield = "configs\shield\v1.json",   # shipped and passed with -Policy (trained/gated under it); "" = none
    [string]$Fallback = "agents\v61.1_bandit.py",   # a generated agent: build it first (the repo ships no built agents)
    [switch]$SkipDocker,
    [string]$Image = "rust:latest"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Target = "x86_64-unknown-linux-musl"
$Out = Join-Path $Root "data\builds\$Name"
$Stage = Join-Path $Out "stage"
if (-not $Profiles) { $Profiles = if ($Policy) { "configs\profiles\rl3.json" } else { "configs\profiles\v1.json" } }
if ($Policy) {
    # a policy only makes sense with the action table it was trained on: its output width must equal the table size
    $nProf = ((Get-Content (Join-Path $Root $Profiles) -Raw | ConvertFrom-Json).profiles).Count
    Write-Host "policy build: profiles $Profiles ($nProf), shield $(if ($Shield) { $Shield } else { 'none' })"
}
$mount = "$($Root -replace '\\','/'):/work"

if (-not $SkipDocker) {
    Write-Host "[1/3] cross-building agent-stdio ($Target)"
    $script = @"
set -e
rustup target add $Target >/dev/null 2>&1 || true
cd /work
cargo build --release -p agent --bin agent-stdio --target $Target --target-dir /work/target-linux
strip /work/target-linux/$Target/release/agent-stdio || true
ls -l /work/target-linux/$Target/release/agent-stdio
"@
    docker run --rm -v $mount -w /work $Image bash -c $script
    if ($LASTEXITCODE -ne 0) { throw "container build failed" }
}
$Bin = Join-Path $Root "target-linux\$Target\release\agent-stdio"
if (-not (Test-Path $Bin)) { throw "no Linux binary at $Bin" }

Write-Host "[2/3] staging $Stage"
if (Test-Path $Stage) { Remove-Item -Recurse -Force $Stage }
New-Item -ItemType Directory -Force (Join-Path $Stage "base") | Out-Null
$sha = (Get-FileHash $Bin -Algorithm SHA256).Hash.Substring(0, 12).ToLower()
$pv = if ($Profile -ge 0) { "$Profile" } else { "None" }
$cv = if ($CloneProfile -ge 0) { "$CloneProfile" } else { "None" }
$main = (Get-Content (Join-Path $Root "kaggle\submission\main_template.py") -Raw)
$main = $main.Replace("__PROFILE__", $pv).Replace("__CLONE_PROFILE__", $cv).Replace("__CLONE_STRICT__", $(if ($CloneStrict) { "True" } else { "False" })).Replace("__POLICY__", $(if ($Policy) { "`"policy.bin`"" } else { "None" })).Replace("__SHIELD__", $(if ($Policy -and $Shield) { "`"shield.json`"" } else { "None" })).Replace("__GROUP__", $(if ($Group) { "`"$Group`"" } else { "None" })).Replace("__DISGUISE__", $(if ($Disguise) { "True" } else { "False" })).Replace("__SHELL__", $(if ($Shell) { "`"shell.json`"" } else { "None" })).Replace("__RSHELL__", $(if ($RShell) { "`"rshell/rshell.json`"" } else { "None" })).Replace("__PREEMPT__", $(if ($Preempt) { "`"preempt.json`"" } else { "None" })).Replace("__GROUP_KNOBS__", $(if ($Preempt) { Copy-Item (Join-Path $Root $Preempt) (Join-Path $Stage "preempt.json") }
if ($Lineages) { Copy-Item (Join-Path $Root $Lineages) (Join-Path $Stage "lineages.json") }
if ($GroupKnobs) { "`"group_knobs.json`"" } else { "None" })).Replace("__GROUP_KNOBS_FOR__", "`"$GroupKnobsFor`"").Replace("__KNOB_OVER__", $(if ($KnobOver) { "`"knobs.json`"" } else { "None" })).Replace("__ENDG__", $(if ($Endg) { "`"endg.json`"" } else { "None" })).Replace("__GT__", $(if ($Gt) { "`"gt.json`"" } else { "None" })).Replace("__DISPATCH__", $(if ($Dispatch) { "`"dispatch/dispatch.json`"" } else { "None" })).Replace("__CHAIN_OFF__", $(if ($ChainOff) { "`"$ChainOff`"" } else { "None" })).Replace("__BUILD__", "$Name bin=$sha")
[IO.File]::WriteAllText((Join-Path $Stage "main.py"), $main.Replace("`r`n", "`n"))
Copy-Item $Bin (Join-Path $Stage "agent-stdio")
Copy-Item (Join-Path $Root "$Base\routes.json") (Join-Path $Stage "base\routes.json")
Copy-Item (Join-Path $Root "$Base\router.json") (Join-Path $Stage "base\router.json")
Copy-Item (Join-Path $Root $Profiles) (Join-Path $Stage "profiles.json")
Copy-Item (Join-Path $Root $Fallback) (Join-Path $Stage "fallback.py")
if ($Policy) { Copy-Item (Join-Path $Root $Policy) (Join-Path $Stage "policy.bin") }
if ($Policy -and $Shield) { Copy-Item (Join-Path $Root $Shield) (Join-Path $Stage "shield.json") }
if ($Shell) { Copy-Item (Join-Path $Root $Shell) (Join-Path $Stage "shell.json") }
if ($RShell) {
    $rsrc = Join-Path $Root $RShell
    $rdir = Join-Path $Stage "rshell"
    New-Item -ItemType Directory -Force $rdir | Out-Null
    Copy-Item $rsrc (Join-Path $rdir "rshell.json")
    $rj = Get-Content $rsrc -Raw | ConvertFrom-Json
    foreach ($f in @($rj.net_file, $rj.lineage_file)) { if ($f) { Copy-Item (Join-Path (Split-Path $rsrc) $f) (Join-Path $rdir $f) } }
    # per-group replacement configs ("group_cfg": {"2": "rshell_copy.json"}), shipped beside rshell.json
    if ($rj.group_cfg) { foreach ($g in $rj.group_cfg.PSObject.Properties) { Copy-Item (Join-Path (Split-Path $rsrc) $g.Value) (Join-Path $rdir $g.Value) } }
}
if ($KnobOver) { Copy-Item (Join-Path $Root $KnobOver) (Join-Path $Stage "knobs.json") }
if ($GroupKnobs) { Copy-Item (Join-Path $Root $GroupKnobs) (Join-Path $Stage "group_knobs.json") }
if ($Endg) { Copy-Item (Join-Path $Root $Endg) (Join-Path $Stage "endg.json") }
if ($Gt) { Copy-Item (Join-Path $Root $Gt) (Join-Path $Stage "gt.json") }
if ($Dispatch) {
    $dsrc = Join-Path $Root $Dispatch
    $ddir = Join-Path $Stage "dispatch"
    New-Item -ItemType Directory -Force $ddir | Out-Null
    Copy-Item $dsrc (Join-Path $ddir "dispatch.json")
    $dj = Get-Content $dsrc -Raw | ConvertFrom-Json
    foreach ($pk in $dj.packages.PSObject.Properties) { foreach ($f in $pk.Value.PSObject.Properties) { Copy-Item (Join-Path (Split-Path $dsrc) $f.Value) (Join-Path $ddir $f.Value) -Force } }
}

Write-Host "[3/3] packing"
$rel = ($Stage.Substring($Root.Length) -replace '\\','/').TrimStart('/')
$outRel = ($Out.Substring($Root.Length) -replace '\\','/').TrimStart('/')
$pack = @"
set -e
cd /work/$rel
chmod 755 agent-stdio
chmod 644 main.py fallback.py profiles.json base/*.json; [ -f policy.bin ] && chmod 644 policy.bin || true; [ -f shield.json ] && chmod 644 shield.json || true; [ -f shell.json ] && chmod 644 shell.json || true; [ -f knobs.json ] && chmod 644 knobs.json || true; [ -f group_knobs.json ] && chmod 644 group_knobs.json || true; [ -f preempt.json ] && chmod 644 preempt.json || true; [ -f lineages.json ] && chmod 644 lineages.json || true; [ -f endg.json ] && chmod 644 endg.json || true; [ -f gt.json ] && chmod 644 gt.json || true; [ -d dispatch ] && chmod 755 dispatch && chmod 644 dispatch/*.json || true; [ -d rshell ] && chmod 755 rshell && chmod 644 rshell/*.json || true
tar -czf /work/$outRel/submission.tar.gz main.py agent-stdio fallback.py profiles.json base `$(ls -d policy.bin shield.json shell.json knobs.json group_knobs.json preempt.json lineages.json endg.json gt.json dispatch rshell 2>/dev/null)
ls -l /work/$outRel/submission.tar.gz
tar -tzvf /work/$outRel/submission.tar.gz
"@
docker run --rm -v $mount -w /work $Image bash -c $pack
if ($LASTEXITCODE -ne 0) { throw "packing failed" }
@{ name = $Name; profile = $pv; profiles = $Profiles; shield = $(if ($Policy) { $Shield } else { "" }); policy = $Policy; clone_profile = $cv; clone_strict = [bool]$CloneStrict; group = $Group; disguise = [bool]$Disguise; shell = $Shell; rshell = $RShell; knob_over = $KnobOver; preempt = $Preempt; lineages = $Lineages; group_knobs = $GroupKnobs; group_knobs_for = $GroupKnobsFor; endg = $Endg; gt = $Gt; dispatch = $Dispatch; chain_off = $ChainOff; binary_sha = $sha; built = (Get-Date).ToUniversalTime().ToString("o") } |
    ConvertTo-Json | Set-Content (Join-Path $Out "build.json") -Encoding utf8
Write-Host "built $Out\submission.tar.gz (binary $sha). NOT submitted."
