<#
.SYNOPSIS
    Run simulations: head-to-head win rates, a Kaggle-identical episode, the
    local Elo ladder, and the diff against the top-20's play.

.DESCRIPTION
    Five modes. The default (-Mode standard) runs win-rate + official + elo,
    which is the set worth looking at before deciding whether a change helped.

      quick     evaluate.py  - N seeds x both seats against the opponent set.
                               Win rate is the number that matters: the Kaggle
                               leaderboard is a skill rating, so coin margin is
                               irrelevant except as a tie-break.
      official  official_eval.py - one episode under the stock configuration
                               with the real 1-second actTimeout, i.e. what
                               Kaggle actually runs.
      elo       elo.py       - round-robin over agents\, maintained in
                               .local\elo\ladder.json so ratings accumulate
                               across iterations rather than resetting.
      match     run_match.py - a single match, optionally with a replay file.
      diff      diff_sim.py  - replays downloaded top-20 games and asks what our
                               agent would have done from the identical state.
      graph     model_graph.py - regenerates agents\<model>.html, the per-model
                               knowledge graph.
      all       everything above.

.PARAMETER Agent
    Defaults to the newest agents\agent_v*.py.

.PARAMETER Vs
    Opponents for -Mode quick. Defaults to the previous generation plus the
    built-ins.

.PARAMETER Matches
    Seeds per opponent (each is played twice, once per seat).

.EXAMPLE
    .\scripts\simulate.ps1

.EXAMPLE
    .\scripts\simulate.ps1 -Mode quick -Vs agents\v2_tuned.py -Matches 16

.EXAMPLE
    .\scripts\simulate.ps1 -Mode match -Agent agents\v2_tuned.py -Vs agents\v1_heuristic.py -Replay
#>
[CmdletBinding()]
param(
    [ValidateSet('standard', 'quick', 'official', 'elo', 'match', 'diff', 'graph', 'all')]
    [string]$Mode = 'standard',
    [string]$Agent,
    [string[]]$Vs,
    [int]$Matches = 8,
    [int]$Rounds = 4,
    [int]$Seed0 = 0,
    [int]$Workers = 0,
    [switch]$Replay,
    [switch]$IncludeBuiltins
)

. (Join-Path $PSScriptRoot '_lib.ps1')
$ErrorActionPreference = 'Stop'

Write-Banner 'Kaggriculture - simulation' 'win rate is the score; coin margin is only a tie-break'
$log = Start-Log 'simulate'
$exit = 0

function Get-Workers {
    if ($Workers -gt 0) { return $Workers }
    $n = [int]$env:NUMBER_OF_PROCESSORS
    if ($n -lt 2) { return 1 }
    return [Math]::Max(1, $n - 1)
}

try {
    Test-Prereq

    if (-not $Agent) { $Agent = Get-NewestAgent }
    Write-Ok "agent under test: $Agent"
    $w = Get-Workers
    Write-Ok "workers: $w"

    $run = @{
        quick    = ($Mode -in @('standard', 'quick', 'all'))
        official = ($Mode -in @('standard', 'official', 'all'))
        elo      = ($Mode -in @('standard', 'elo', 'all'))
        match    = ($Mode -in @('match', 'all'))
        diff     = ($Mode -in @('diff', 'all'))
        graph    = ($Mode -in @('graph', 'all'))
    }

    # ---- head-to-head win rate ------------------------------------------
    if ($run.quick) {
        $opponents = if ($Vs) { $Vs } else {
            @(Get-ChildItem (Join-Path $Root 'agents') -Filter '*.py' |
                Where-Object { "agents\$($_.Name)" -ne $Agent -and $_.Name -notmatch '^ml_' } |
                Sort-Object Name -Descending | Select-Object -First 2 |
                ForEach-Object { "agents\$($_.Name)" })
        }
        $a = @($Agent, '--vs') + $opponents + @('-n', $Matches, '--workers', $w, '--seed0', $Seed0)
        Invoke-Py -Script 'src\kaggriculture\measure\evaluate.py' -Arguments $a `
            -Title "Head-to-head: $Matches seeds x 2 seats vs $($opponents.Count) opponent(s)" | Out-Null
    }

    # ---- the episode Kaggle actually runs -------------------------------
    if ($run.official) {
        $a = @('--agent', $Agent)
        if ($Vs) { $a += @('--vs') + $Vs }
        if ($Replay) { $a += '--replay' }
        Invoke-Py -Script 'src\kaggriculture\engine\official_eval.py' -Arguments $a `
            -Title 'Official evaluation (stock config, real 1s actTimeout)' -AllowFail | Out-Null
    }

    # ---- Elo ladder ------------------------------------------------------
    if ($run.elo) {
        $a = @('--rounds', $Rounds, '--workers', $w, '--seed0', $Seed0)
        if ($IncludeBuiltins) { $a += '--include-builtins' }
        Invoke-Py -Script 'src\kaggriculture\measure\elo.py' -Arguments $a -Title 'Elo ladder' -AllowFail | Out-Null
        Invoke-Py -Script 'src\kaggriculture\measure\elo.py' -Arguments @('--show') -AllowFail | Out-Null
    }

    # ---- a single match --------------------------------------------------
    if ($run.match) {
        $opp = if ($Vs) { $Vs[0] } else { 'agents\v2_tuned.py' }
        $a = @($Agent, $opp, '--seed', $Seed0)
        if ($Replay) {
            $dir = Join-Path $Root '.local\replays'
            New-Item -ItemType Directory -Force -Path $dir | Out-Null
            $a += @('--replay', ".local\replays\match-$(Get-Date -Format 'yyyyMMdd-HHmmss').json")
        }
        Invoke-Py -Script 'src\kaggriculture\engine\run_match.py' -Arguments $a `
            -Title "Single match: $Agent vs $opp" -AllowFail | Out-Null
    }

    # ---- us vs the top-20 on identical states ---------------------------
    if ($run.diff) {
        $replays = Join-Path $Root '.local\episodes'
        if (-not (Test-Path $replays)) {
            Write-Warn2 'no downloaded replays yet -- run .\scripts\download.ps1 first'
        } else {
            Invoke-Py -Script 'src\kaggriculture\engine\diff_sim.py' `
                -Arguments @('--agent', $Agent, '--out', 'docs\diff-simulation.md') `
                -Title 'Diff vs the top-20 on identical states' -AllowFail | Out-Null
        }
    }

    # ---- per-model knowledge graph --------------------------------------
    if ($run.graph) {
        Invoke-Py -Script 'src\kaggriculture\agentbuild\model_graph.py' -Arguments @($Agent) `
            -Title 'Knowledge graph for this model' -AllowFail | Out-Null
    }

    Show-Next @(
        '.\scripts\train.ps1 -Algo cmaes -Minutes 30   # search for better params',
        '.\scripts\simulate.ps1 -Mode all              # everything, including diff + graph',
        '.\scripts\submit.ps1 -DryRun                  # validate exactly as Kaggle will'
    )
} catch {
    Write-Fail $_.Exception.Message
    $exit = 1
}

Stop-Log $exit
exit $exit
