# Run the RL corpus delta pipeline (python/delta.py run) with logging. See docs/delta.md.
$ErrorActionPreference = 'Continue'
$rl = Split-Path -Parent $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
$log = Join-Path $rl 'data\kaggle_out\delta.log'
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
Set-Location $rl
& python (Join-Path $rl 'python\delta.py') run 2>&1 | ForEach-Object { "$_" } | Out-File -FilePath $log -Append -Encoding utf8
exit $LASTEXITCODE
