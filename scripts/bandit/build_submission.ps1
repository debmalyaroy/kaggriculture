<#
.SYNOPSIS
  Build a Kaggle submission tarball of the v62 Rust bandit (rustengine/v62). NEVER submits.

.DESCRIPTION
  Bandit copy of the v62 / v62.1 build (scripts/build_submission.ps1), pointed at the
  bandit's home in this repo.
  1. Cross-builds rustengine/v62 crates/agent `agent-stdio` for x86_64-unknown-linux-musl (static) in a
     rust:latest container (target dir rustengine/v62/target-linux).
  2. Stages main.py (from configs/bandit/submission/main_template.py with the profile filled in), the
     binary, base/ (routes.json + router.json), profiles.json, fallback.py (the Python v61.1).
  3. Packs data/bandit_builds/<Name>/submission.tar.gz inside the container so the exec bit survives.

.EXAMPLE
  .\scripts\bandit\build_submission.ps1 -Name v622_c13 -Profile 48
  .\scripts\bandit\build_submission.ps1 -Name v621 -Profile 19 -Profiles configs\bandit\profiles\v2.json
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]$Name,
    [int]$Profile = -1,
    [int]$CloneProfile = -1,
    [switch]$CloneStrict,
    [string]$Base = "configs\bandit\bases\v61.1",
    [string]$Profiles = "configs\bandit\profiles\v3.json",
    [string]$Fallback = "agents\v61.1_bandit.py",
    [switch]$SkipDocker,
    [string]$Image = "rust:latest"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Target = "x86_64-unknown-linux-musl"
$Out = Join-Path $Root "data\bandit_builds\$Name"
$Stage = Join-Path $Out "stage"
$mount = "$($Root -replace '\\','/'):/work"

if (-not $SkipDocker) {
    Write-Host "[1/3] cross-building agent-stdio ($Target)"
    $script = @"
set -e
rustup target add $Target >/dev/null 2>&1 || true
cd /work/rustengine/v62
cargo build --release -p agent --bin agent-stdio --target $Target --target-dir /work/rustengine/v62/target-linux
strip /work/rustengine/v62/target-linux/$Target/release/agent-stdio || true
ls -l /work/rustengine/v62/target-linux/$Target/release/agent-stdio
"@
    docker run --rm -v $mount -w /work $Image bash -c $script
    if ($LASTEXITCODE -ne 0) { throw "container build failed" }
}
$Bin = Join-Path $Root "rustengine\v62\target-linux\$Target\release\agent-stdio"
if (-not (Test-Path $Bin)) { throw "no Linux binary at $Bin" }

Write-Host "[2/3] staging $Stage"
if (Test-Path $Stage) { Remove-Item -Recurse -Force $Stage }
New-Item -ItemType Directory -Force (Join-Path $Stage "base") | Out-Null
$sha = (Get-FileHash $Bin -Algorithm SHA256).Hash.Substring(0, 12).ToLower()
$pv = if ($Profile -ge 0) { "$Profile" } else { "None" }
$cv = if ($CloneProfile -ge 0) { "$CloneProfile" } else { "None" }
$main = (Get-Content (Join-Path $Root "configs\bandit\submission\main_template.py") -Raw)
$main = $main.Replace("__PROFILE__", $pv).Replace("__CLONE_PROFILE__", $cv).Replace("__CLONE_STRICT__", $(if ($CloneStrict) { "True" } else { "False" })).Replace("__POLICY__", "None").Replace("__BUILD__", "$Name bin=$sha")
[IO.File]::WriteAllText((Join-Path $Stage "main.py"), $main.Replace("`r`n", "`n"))
Copy-Item $Bin (Join-Path $Stage "agent-stdio")
Copy-Item (Join-Path $Root "$Base\routes.json") (Join-Path $Stage "base\routes.json")
Copy-Item (Join-Path $Root "$Base\router.json") (Join-Path $Stage "base\router.json")
Copy-Item (Join-Path $Root $Profiles) (Join-Path $Stage "profiles.json")
Copy-Item (Join-Path $Root $Fallback) (Join-Path $Stage "fallback.py")

Write-Host "[3/3] packing"
$rel = ($Stage.Substring($Root.Length) -replace '\\','/').TrimStart('/')
$outRel = ($Out.Substring($Root.Length) -replace '\\','/').TrimStart('/')
$pack = @"
set -e
cd /work/$rel
chmod 755 agent-stdio
chmod 644 main.py fallback.py profiles.json base/*.json
tar -czf /work/$outRel/submission.tar.gz main.py agent-stdio fallback.py profiles.json base
ls -l /work/$outRel/submission.tar.gz
tar -tzvf /work/$outRel/submission.tar.gz
"@
docker run --rm -v $mount -w /work $Image bash -c $pack
if ($LASTEXITCODE -ne 0) { throw "packing failed" }
@{ name = $Name; profile = $pv; profiles = $Profiles; clone_profile = $cv; clone_strict = [bool]$CloneStrict; binary_sha = $sha; built = (Get-Date).ToUniversalTime().ToString("o") } |
    ConvertTo-Json | Set-Content (Join-Path $Out "build.json") -Encoding utf8
Write-Host "built $Out\submission.tar.gz (binary $sha). NOT submitted."
