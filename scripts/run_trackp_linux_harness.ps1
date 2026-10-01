<#
.SYNOPSIS
  Play the assembled Linux artefact on the OFFICIAL engine, inside Linux.

.DESCRIPTION
  Unpacks the built submission.tar.gz into a scratch directory INSIDE the
  container -- so what is measured is literally the archive contents, mode bits
  and all -- then runs src/kaggriculture/trackp/compiled/harness.py against the named
  opponents on the vendored official interpreter.

  Reports bank, engine status, mean/p95/worst turn latency and the fallback and
  validator-repair counts. Never submits.

.EXAMPLE
  .\scripts\run_trackp_linux_harness.ps1
  .\scripts\run_trackp_linux_harness.ps1 -Seeds 3,4,5,6 -Seats 0,1
#>
[CmdletBinding()]
param(
    [string]$Seeds = "3,4,5",
    [string]$Seats = "0,1",
    [string]$Image = "kagg-harness:2",
    [string[]]$Vs = @("agents/v43.0_bandit.py", "data/gauntlet/pub_v16rc5.py"),
    [string]$JsonOut = ".local/candidates/trackp_compiled/harness_linux.json",
    # Phase-B search budget per turn, in ms. -1 = whatever bridge.py ships with.
    [int]$BudgetMs = -1,
    # Cells played concurrently. The budget is WALL CLOCK, so oversubscribing
    # the box measures a weaker searcher than the one that ships.
    [int]$Workers = 1,
    [double]$WorstGate = 250.0,
    # Play this plain Python agent in our seat instead of the artefact, over the
    # SAME cells, so the two runs pair for src/kaggriculture/trackp/compiled/verdict.py.
    [string]$Ref = "",
    # Cores the container may use. Left unset the container sees the whole box,
    # which is fine for a serial run and wrong for a parallel one.
    [string]$Cpus = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$mount = "$($Root -replace '\\','/'):/work"

docker image inspect $Image *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "building $Image ..." -ForegroundColor Cyan
    # Build from a SCRATCH context holding only the Dockerfile. The repo root
    # carries tens of GB of episode data; using it as the build context uploads
    # all of it to the daemon and appears to hang.
    $ctx = Join-Path $Root ".local\build\harness_ctx"
    New-Item -ItemType Directory -Force -Path $ctx | Out-Null
    Copy-Item (Join-Path $Root "scripts\trackp_harness.Dockerfile") `
        (Join-Path $ctx "Dockerfile") -Force
    docker build -t $Image $ctx
    if ($LASTEXITCODE -ne 0) { throw "image build failed" }
}

$vsArgs = ($Vs | ForEach-Object { "/work/$_" }) -join " "
$extra = ""
if ($BudgetMs -ge 0) { $extra += " --budget-ms $BudgetMs" }
if ($Workers -gt 1)  { $extra += " --workers $Workers" }
if ($Ref)            { $extra += " --ref /work/$Ref" }
$extra += " --worst-gate $WorstGate"

$script = @"
set -e
rm -rf /tmp/artefact && mkdir -p /tmp/artefact
tar -xzf /work/.local/candidates/trackp_compiled/submission.tar.gz -C /tmp/artefact
ls -l /tmp/artefact
echo '--- binary smoke test ---'
/tmp/artefact/kagg play-selftest
echo '--- official-engine harness ---'
cd /work
python src/kaggriculture/trackp/compiled/harness.py --stage /tmp/artefact \
    --vs $vsArgs --seeds $Seeds --seats $Seats --json /work/$JsonOut$extra
"@

$docker = @("run", "--rm", "-v", $mount, "-w", "/work")
if ($Cpus) { $docker += @("--cpus", $Cpus) }
$docker += @($Image, "bash", "-c", $script)
docker @docker
if ($LASTEXITCODE -ne 0) { throw "harness failed" }
