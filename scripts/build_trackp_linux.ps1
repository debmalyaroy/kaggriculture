<#
.SYNOPSIS
  Build the Track-P COMPILED agent artefact: a Linux x86_64 binary + main.py,
  packed as submission.tar.gz with main.py at the ROOT of the archive.

.DESCRIPTION
  Kaggle runs Linux x86_64; this box is Windows, so the binary is cross-built
  inside a `rust:latest` container with the crate bind-mounted.

  The target is **x86_64-unknown-linux-musl**, not gnu, and that is deliberate:
  a musl build is fully static, so it cannot be broken by whatever glibc the
  competition image happens to ship. A dynamically linked agent that fails to
  load on the ladder looks exactly like a lazy agent -- it just PASSes for 720
  turns while the Python fallback covers for it, and we would never see it in
  local testing.

  NEVER submits. It prints the operator's submit command and stops.

.EXAMPLE
  .\scripts\build_trackp_linux.ps1
  .\scripts\build_trackp_linux.ps1 -SkipDocker      # repack with an existing binary
#>
[CmdletBinding()]
param(
    [switch]$SkipDocker,
    [string]$Image = "rust:latest",
    [string]$OutDir
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not $OutDir) { $OutDir = Join-Path $Root ".local\candidates\trackp_compiled" }
$Stage = Join-Path $OutDir "stage_linux"
$Target = "x86_64-unknown-linux-musl"
$LinuxTargetDir = "/work/.local/build/rust-linux"

Write-Host "== Track-P compiled agent: Linux build ==" -ForegroundColor Cyan
Write-Host "root:  $Root"
Write-Host "out:   $OutDir"

# ---------------------------------------------------------------- main.py --
Write-Host "`n[1/4] assembling main.py (bridge + inlined Python fallback)"
& python (Join-Path $Root "src\kaggriculture\trackp\compiled\build_main.py")
if ($LASTEXITCODE -ne 0) { throw "build_main.py failed" }

# ------------------------------------------------------------- the binary --
$BinOut = Join-Path $Root ".local\build\rust-linux\$Target\release\kagg"
if (-not $SkipDocker) {
    Write-Host "`n[2/4] cross-building the searcher in $Image ($Target)"
    docker version --format '{{.Server.Version}}' | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Docker is not available. Use -SkipDocker, or build on Kaggle (see docs/history/trackp-compiled-2026-09-03.md)." }

    $mount = "$($Root -replace '\\','/'):/work"
    $script = @"
set -e
rustup target add $Target >/dev/null 2>&1 || true
cd /work/rustengine
# --bin kagg ONLY: the crate also hosts the Phase-B search lib and its
# offline bins, which are edited in parallel. The shipped artefact must not
# be able to fail to build because an unrelated target is mid-change.
cargo build --release --bin kagg --target $Target --target-dir $LinuxTargetDir
strip $LinuxTargetDir/$Target/release/kagg || true
ls -l $LinuxTargetDir/$Target/release/kagg
file $LinuxTargetDir/$Target/release/kagg 2>/dev/null || true
"@
    docker run --rm -v $mount -w /work $Image bash -c $script
    if ($LASTEXITCODE -ne 0) { throw "container build failed" }
} else {
    Write-Host "`n[2/4] -SkipDocker: reusing $BinOut"
}
if (-not (Test-Path $BinOut)) { throw "no Linux binary at $BinOut" }

# ------------------------------------------------------------------ stage --
Write-Host "`n[3/4] staging"
if (Test-Path $Stage) { Remove-Item -Recurse -Force $Stage }
New-Item -ItemType Directory -Force -Path $Stage | Out-Null
Copy-Item (Join-Path $Root "src\kaggriculture\trackp\compiled\main.py") (Join-Path $Stage "main.py")
Copy-Item $BinOut (Join-Path $Stage "kagg")

# --------------------------------------------------------------- tar.gz ----
# main.py must sit at the ARCHIVE ROOT (docs/history/submission.md). tar is run INSIDE
# the container so the executable bit survives -- Windows has no mode bit to
# preserve, and a non-executable kagg would silently fall back forever.
Write-Host "`n[4/4] packing submission.tar.gz"
$mount = "$($Root -replace '\\','/'):/work"
$rel = ($Stage.Substring($Root.Length) -replace '\\','/').TrimStart('/')
$out = ($OutDir.Substring($Root.Length) -replace '\\','/').TrimStart('/')
$pack = @"
set -e
cd /work/$rel
chmod 755 kagg
chmod 644 main.py
tar -czf /work/$out/submission.tar.gz main.py kagg
cd /work/$out
ls -l submission.tar.gz
tar -tzvf submission.tar.gz
"@
docker run --rm -v $mount -w /work $Image bash -c $pack
if ($LASTEXITCODE -ne 0) { throw "packing failed" }

$tar = Join-Path $OutDir "submission.tar.gz"
$mb = [math]::Round((Get-Item $tar).Length / 1MB, 2)
Write-Host "`nartefact: $tar  ($mb MiB, limit 100 MiB)" -ForegroundColor Green
Write-Host "`nOPERATOR submit command (this script never submits):" -ForegroundColor Yellow
Write-Host "  kaggle competitions submit kaggriculture -f `"$tar`" -m `"trackp compiled phase A`""
