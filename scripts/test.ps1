<#
.SYNOPSIS
    Contract tests for the agents: the things Kaggle punishes silently.

.DESCRIPTION
    Four layers, cheapest first, so a syntax error costs a second rather than
    ten minutes:

      1. compile   - every .py in the project parses
      2. import    - each agent module loads and exposes agent(obs, config)
      3. contract  - tests\test_agents.py: legal action dicts, legal ops, and
                     per-turn latency against the real 1-second actTimeout
      4. baseline  - each agent must beat the three built-ins (pass, random,
                     starter). Slow -- full 720-turn matches -- so it is opt-in
                     via -Full, and it is the check that actually catches a
                     regression that "runs fine" but plays badly.

.PARAMETER Full
    Include layer 4 (beats-the-built-ins). Adds a few minutes.

.PARAMETER Agent
    Test one agent instead of every file in agents\.

.EXAMPLE
    .\scripts\test.ps1

.EXAMPLE
    .\scripts\test.ps1 -Full
#>
[CmdletBinding()]
param(
    [switch]$Full,
    [string]$Agent
)

. (Join-Path $PSScriptRoot '_lib.ps1')
$ErrorActionPreference = 'Stop'

Write-Banner 'Kaggriculture - tests' 'legality, latency, and does-it-still-win'
$log = Start-Log 'test'
$exit = 0
$failed = @()

try {
    Test-Prereq

    # ---- 1. everything compiles -----------------------------------------
    $code = Invoke-PyModule -Module 'compileall' `
        -Arguments @('-q', 'tools', 'agents', 'tests', 'download_data.py') `
        -Title 'Compiling every module' -AllowFail
    if ($code -ne 0) { $failed += 'compile' }

    # ---- 2. each agent imports and exposes agent() ----------------------
    Write-Step 'Importing agents'
    $targets = if ($Agent) { @($Agent) } else {
        @(Get-ChildItem (Join-Path $Root 'agents') -Filter '*.py' |
            Where-Object { $_.Name -ne '__init__.py' } |
            ForEach-Object { "agents\$($_.Name)" })
    }
    foreach ($t in $targets) {
        $res = Invoke-PyInline @"
import importlib.util, os, sys
sys.path.insert(0, "agents"); sys.path.insert(0, "tools")
p = r"$t"
try:
    spec = importlib.util.spec_from_file_location(os.path.basename(p)[:-3], p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    print("ok" if callable(getattr(m, "agent", None)) else "NO agent() callable")
except Exception as exc:
    print("FAILED " + type(exc).__name__ + ": " + str(exc)[:120])
"@
        if ($res -eq 'ok') { Write-Ok $t } else { Write-Fail "$t -> $res"; $failed += $t }
    }

    # ---- 3. contract tests ----------------------------------------------
    if ($Full) {
        $code = Invoke-Py -Script 'tests\test_agents.py' `
            -Title 'Contract tests + beats-the-built-ins (slow)' -AllowFail
    } else {
        # Just the fast half: legality + latency, no 720-turn matches.
        $code = Invoke-PyCode -Title 'Contract tests (legal actions, 1s actTimeout)' -AllowFail -Code @'
import sys
sys.path.insert(0, "tests")
import test_agents as t
t.test_agents_are_legal_and_fast()
print("legality + latency ok")
'@
    }
    if ($code -ne 0) { Write-Fail 'contract tests failed'; $failed += 'contract' }
    else { Write-Ok 'contract tests passed' }

    # ---- 4. smoke match --------------------------------------------------
    if (-not $Full) {
        $newest = if ($Agent) { $Agent } else { Get-NewestAgent }
        Invoke-Py -Script 'src\kaggriculture\engine\run_match.py' `
            -Arguments @('agents\v0_baseline.py', $newest, '--steps', '120', '--seed', '11') `
            -Title "Smoke match: v0_baseline vs $newest (120 turns)" -AllowFail | Out-Null
    }

    if ($failed.Count) {
        Write-Fail ('failed: ' + ($failed -join ', '))
        $exit = 1
    } else {
        Write-Ok 'everything passed'
        if (-not $Full) { Write-Note 'run with -Full for the beats-the-built-ins check' }
    }

    Show-Next @(
        '.\scripts\test.ps1 -Full                # add the beats-built-ins check',
        '.\scripts\simulate.ps1                  # head-to-head + Elo'
    )
} catch {
    Write-Fail $_.Exception.Message
    $exit = 1
}

Stop-Log $exit
exit $exit
