<#
    Shared helpers for the Kaggriculture PowerShell scripts.

    Dot-source it, never run it:

        . (Join-Path $PSScriptRoot '_lib.ps1')

    What it gives every script:
      * $Root      - the project directory, resolved from this file's location,
                     so the scripts work from any working directory
      * $Python    - a real python.exe, resolved once (python, then py -3)
      * Invoke-Py  - run a project script with unbuffered output, wall-clock
                     timing, a tee'd log and a non-zero exit that actually stops
                     the caller
      * Start-Log  - one timestamped log per run under .local\logs\

    Everything writes to .local\ for scratch and never outside the project.
#>

Set-StrictMode -Version Latest

$script:Root   = Split-Path -Parent $PSScriptRoot
$script:LogDir = Join-Path $script:Root '.local\logs'
$script:LogFile = $null
$script:RunStart = Get-Date

# --------------------------------------------------------------- appearance --

function Write-Banner {
    param([string]$Title, [string]$Subtitle = '')
    $line = '=' * 72
    Write-Host ''
    Write-Host $line -ForegroundColor DarkCyan
    Write-Host "  $Title" -ForegroundColor Cyan
    if ($Subtitle) { Write-Host "  $Subtitle" -ForegroundColor DarkGray }
    Write-Host $line -ForegroundColor DarkCyan
}

function Write-Step { param([string]$Message) Write-Host "`n>> $Message" -ForegroundColor White }
function Write-Ok   { param([string]$Message) Write-Host "   [ok]   $Message" -ForegroundColor Green }
function Write-Warn2{ param([string]$Message) Write-Host "   [warn] $Message" -ForegroundColor Yellow }
function Write-Fail { param([string]$Message) Write-Host "   [FAIL] $Message" -ForegroundColor Red }
function Write-Note { param([string]$Message) Write-Host "   $Message" -ForegroundColor DarkGray }

function Format-Duration {
    param([TimeSpan]$Span)
    if ($Span.TotalHours -ge 1) { return ('{0:0}h{1:00}m{2:00}s' -f [int]$Span.TotalHours, $Span.Minutes, $Span.Seconds) }
    if ($Span.TotalMinutes -ge 1) { return ('{0:0}m{1:00}s' -f [int]$Span.TotalMinutes, $Span.Seconds) }
    return ('{0:0.0}s' -f $Span.TotalSeconds)
}

# ------------------------------------------------------------------ logging --

function Start-Log {
    <#  One log file per run: .local\logs\<name>-<yyyyMMdd-HHmmss>.log
        Returns the path. Every Invoke-Py call appends to it. #>
    param([Parameter(Mandatory)][string]$Name)
    New-Item -ItemType Directory -Force -Path $script:LogDir | Out-Null
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $script:LogFile = Join-Path $script:LogDir "$Name-$stamp.log"
    "=== $Name  started $(Get-Date -Format 'u')" | Out-File $script:LogFile -Encoding utf8
    $script:RunStart = Get-Date
    return $script:LogFile
}

function Stop-Log {
    param([int]$ExitCode = 0)
    $span = (Get-Date) - $script:RunStart
    $msg = "=== finished $(Get-Date -Format 'u')  exit=$ExitCode  elapsed=$(Format-Duration $span)"
    if ($script:LogFile) { $msg | Out-File $script:LogFile -Append -Encoding utf8 }
    Write-Host ''
    if ($ExitCode -eq 0) { Write-Ok "done in $(Format-Duration $span)" }
    else                 { Write-Fail "exit code $ExitCode after $(Format-Duration $span)" }
    if ($script:LogFile) { Write-Note "log: $($script:LogFile.Replace($script:Root + '\', ''))" }
}

# ------------------------------------------------------------------- python --

function Resolve-Python {
    <#  A python.exe that can import this project.

        Prefers `python` on PATH; falls back to the launcher (`py -3`), which is
        what a machine with only the Store/launcher install has. Returns an
        object with .Exe and .Prefix (extra args, e.g. @('-3')).  #>
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source -and (Test-Path $cmd.Source)) {
        return [pscustomobject]@{ Exe = $cmd.Source; Prefix = @() }
    }
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) { return [pscustomobject]@{ Exe = $py.Source; Prefix = @('-3') } }
    throw "python was not found on PATH. Install Python 3.11+ and reopen this terminal."
}

$script:Py = Resolve-Python
$script:Python = $script:Py.Exe

function Invoke-Py {
    <#  Run a python script inside the project, streaming its output.

        -u is always passed: without it Python block-buffers stdout when it is
        piped, which is exactly why long downloads looked frozen.

        Throws on a non-zero exit unless -AllowFail, so a broken stage stops the
        script instead of quietly poisoning the next one.  #>
    param(
        [Parameter(Mandatory)][string]$Script,
        [string[]]$Arguments = @(),
        [string]$Title = '',
        [switch]$AllowFail
    )
    if ($Title) { Write-Step $Title }
    $full = @($script:Py.Prefix) + @('-u', $Script) + $Arguments
    Write-Note ("$ python " + (@($Script) + $Arguments -join ' '))
    if ($script:LogFile) {
        "`n--- $ python $($Script) $($Arguments -join ' ')" |
            Out-File $script:LogFile -Append -Encoding utf8
    }

    $t0 = Get-Date
    Push-Location $script:Root
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'      # native stderr is not a PS error
    try {
        if ($script:LogFile) {
            & $script:Python @full 2>&1 | Tee-Object -FilePath $script:LogFile -Append | Out-Host
        } else {
            & $script:Python @full 2>&1 | Out-Host
        }
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prev
        Pop-Location
    }
    $span = (Get-Date) - $t0

    if ($code -ne 0) {
        if ($AllowFail) { Write-Warn2 "$Script exited $code after $(Format-Duration $span)" }
        else { Write-Fail "$Script exited $code after $(Format-Duration $span)"; throw "stage failed: $Script (exit $code)" }
    } else {
        Write-Ok "$(Split-Path -Leaf $Script) finished in $(Format-Duration $span)"
    }
    return $code
}

function Invoke-PyModule {
    <#  Same as Invoke-Py but for `python -m <module>` (compileall, pytest). #>
    param(
        [Parameter(Mandatory)][string]$Module,
        [string[]]$Arguments = @(),
        [string]$Title = '',
        [switch]$AllowFail
    )
    if ($Title) { Write-Step $Title }
    $full = @($script:Py.Prefix) + @('-u', '-m', $Module) + $Arguments
    Write-Note ("$ python -m $Module " + ($Arguments -join ' '))
    if ($script:LogFile) {
        "`n--- $ python -m $Module $($Arguments -join ' ')" |
            Out-File $script:LogFile -Append -Encoding utf8
    }
    $t0 = Get-Date
    Push-Location $script:Root
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        if ($script:LogFile) {
            & $script:Python @full 2>&1 | Tee-Object -FilePath $script:LogFile -Append | Out-Host
        } else {
            & $script:Python @full 2>&1 | Out-Host
        }
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prev
        Pop-Location
    }
    $span = (Get-Date) - $t0
    if ($code -ne 0) {
        if ($AllowFail) { Write-Warn2 "python -m $Module exited $code" }
        else { Write-Fail "python -m $Module exited $code"; throw "stage failed: $Module (exit $code)" }
    } else {
        Write-Ok "$Module finished in $(Format-Duration $span)"
    }
    return $code
}

function Invoke-PyCode {
    <#  Run a python snippet with `-c`, streaming and tee'ing its output.

        Use this when the snippet is real work whose output the user should see;
        Invoke-PyInline is for one-line lookups whose value you want back. #>
    param(
        [Parameter(Mandatory)][string]$Code,
        [string]$Title = '',
        [switch]$AllowFail
    )
    if ($Title) { Write-Step $Title }
    $full = @($script:Py.Prefix) + @('-u', '-c', $Code)
    Push-Location $script:Root
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        if ($script:LogFile) {
            & $script:Python @full 2>&1 | Tee-Object -FilePath $script:LogFile -Append | Out-Host
        } else {
            & $script:Python @full 2>&1 | Out-Host
        }
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prev
        Pop-Location
    }
    if ($code -ne 0 -and -not $AllowFail) { throw "python snippet failed (exit $code)" }
    return $code
}

function Invoke-PyInline {
    <#  Run a short python snippet and return its stdout as text (no tee).
        For lookups -- newest agent, row counts -- not for long work. #>
    param([Parameter(Mandatory)][string]$Code)
    Push-Location $script:Root
    try {
        $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
        $out = & $script:Python @($script:Py.Prefix) '-c' $Code 2>&1
        $ErrorActionPreference = $prev
    } finally { Pop-Location }
    return ($out | Out-String).Trim()
}

# ------------------------------------------------------------------ project --

function Get-NewestAgent {
    <#  The newest agents\agent_v*.py, matching what the python tools default to.
        Falls back to v2_tuned.py so a fresh clone still works. #>
    $dir = Join-Path $script:Root 'agents'
    $cand = Get-ChildItem -Path $dir -Filter 'agent_v*.py' -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending | Select-Object -First 1
    if ($cand) { return "agents\$($cand.Name)" }
    if (Test-Path (Join-Path $dir 'v2_tuned.py')) { return 'agents\v2_tuned.py' }
    throw "no agent found in $dir"
}

function Test-Prereq {
    <#  Fail fast and say what to fix, rather than dying three stages in. #>
    param([switch]$NeedKaggle)
    Write-Step 'Checking prerequisites'
    Write-Ok "project: $script:Root"
    $ver = Invoke-PyInline 'import sys;print(".".join(map(str,sys.version_info[:3])))'
    Write-Ok "python : $script:Python ($ver)"

    $envCheck = Invoke-PyInline @'
import sys, os
sys.path.insert(0, "tools")
try:
    import _vendor  # puts .local/vendor on the path when there is no system install
except Exception:
    pass
try:
    import kaggle_environments as k
    print("ok " + getattr(k, "version", "?"))
except Exception as exc:
    print("MISSING " + str(exc)[:120])
'@
    if ($envCheck -like 'ok*') { Write-Ok "kaggle-environments $($envCheck.Substring(3))" }
    else { Write-Fail "kaggle-environments not importable: $envCheck"; throw 'kaggle-environments is required' }

    if ($NeedKaggle) {
        $cli = Invoke-PyInline @'
import sys
sys.path.insert(0, "tools")
import episodes as ep
try:
    print("ok " + " ".join(ep.kaggle_cmd()))
except Exception as exc:
    print("MISSING " + str(exc).splitlines()[0])
'@
        if ($cli -like 'ok*') { Write-Ok "kaggle CLI: $($cli.Substring(3))" }
        else { Write-Fail "kaggle CLI unusable: $cli"; throw 'the Kaggle CLI is required for this script' }
    }
}

function Show-Next {
    param([string[]]$Lines)
    Write-Host "`nNext:" -ForegroundColor White
    foreach ($l in $Lines) { Write-Host "  $l" -ForegroundColor DarkGray }
}
