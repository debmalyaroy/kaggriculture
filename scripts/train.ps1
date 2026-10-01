<#
.SYNOPSIS
    Run the machine-learning and search algorithms that produce a better agent.

.DESCRIPTION
    Five algorithms, each writing a new agents\agent_v<timestamp>.py rather than
    overwriting its input -- so a bad run costs nothing and every generation
    stays reproducible.

      learn   learn_params.py     Supervised attribution over the downloaded
                                  ladder dataset: which state features actually
                                  predict winning. Read-only -- it produces a
                                  ranking, not an agent. Start here, it tells
                                  you which knobs are worth searching.

      tune    tune.py             Coordinate descent over PARAMS. One knob at a
                                  time, paired both-seat matches on common
                                  random numbers. Cheap, interpretable, and
                                  gets stuck in local optima.

      cmaes   cmaes.py            CMA-ES over all 28 knobs at once. The right
                                  optimiser for this problem: correlated,
                                  noisy, no gradients. Checkpointed, so
                                  -Resume picks up where it stopped.

      ridge   train_supervised.py Ridge regression on decision-level outcomes,
                                  written into the ASSET_BIAS hook.

      rl      train_rl.py         Monte-Carlo control over the same hook.
                                  Slowest and highest variance; needs a few
                                  hundred episodes before it beats the
                                  heuristic it started from.

    Objective throughout is WIN RATE, not coin margin -- the Kaggle leaderboard
    is a skill rating, so a 10-coin win and a 10,000-coin win score identically.

    After a run that produced an agent, the script rates it on the local Elo
    ladder and regenerates its knowledge graph, so agents\<model>.html always
    matches the model.

.PARAMETER Algo
    learn | tune | cmaes | ridge | rl | all. Default cmaes.

.PARAMETER Minutes
    Budget for -Algo tune.

.PARAMETER Generations
    CMA-ES generations.

.PARAMETER Episodes
    Training episodes for ridge / rl.

.PARAMETER Base
    Starting agent. Defaults to the newest agents\agent_v*.py.

.PARAMETER Resume
    Continue from the checkpoint instead of starting over.

.EXAMPLE
    .\scripts\train.ps1 -Algo learn

.EXAMPLE
    .\scripts\train.ps1 -Algo cmaes -Generations 12 -Seeds 6

.EXAMPLE
    .\scripts\train.ps1 -Algo all -Minutes 20
#>
[CmdletBinding()]
param(
    [ValidateSet('learn', 'tune', 'cmaes', 'ridge', 'rl', 'all')]
    [string]$Algo = 'cmaes',
    [string]$Base,
    [int]$Minutes = 20,
    [int]$Generations = 8,
    [int]$Popsize = 8,
    [int]$Seeds = 4,
    [int]$Seed0 = 0,
    [int]$Episodes = 200,
    [int]$Workers = 0,
    [int]$Top = 25,
    [switch]$Resume,
    [switch]$NoGraph
)

. (Join-Path $PSScriptRoot '_lib.ps1')
$ErrorActionPreference = 'Stop'

Write-Banner 'Kaggriculture - training' 'objective: win rate, not coin margin'
$log = Start-Log "train-$Algo"
$exit = 0
$produced = @()

function Get-Workers {
    if ($Workers -gt 0) { return $Workers }
    $n = [int]$env:NUMBER_OF_PROCESSORS
    if ($n -lt 2) { return 1 }
    return [Math]::Max(1, $n - 1)
}

function New-AgentPath {
    param([string]$Tag)
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    return "agents\agent_v$($Tag)_$stamp.py"
}

try {
    Test-Prereq

    if (-not $Base) { $Base = Get-NewestAgent }
    $w = Get-Workers
    Write-Ok "base agent: $Base"
    Write-Ok "workers   : $w"

    $run = @{
        learn = ($Algo -in @('learn', 'all'))
        tune  = ($Algo -in @('tune', 'all'))
        cmaes = ($Algo -in @('cmaes', 'all'))
        ridge = ($Algo -in @('ridge', 'all'))
        rl    = ($Algo -in @('rl', 'all'))
    }

    # ---- what predicts winning ------------------------------------------
    if ($run.learn) {
        $data = Join-Path $Root 'data\episodes.csv'
        if (-not (Test-Path $data)) {
            Write-Warn2 'data\episodes.csv is missing -- run .\scripts\download.ps1 first'
        } else {
            Invoke-Py -Script 'src\kaggriculture\train\learn_params.py' -Arguments @('--top', $Top) `
                -Title 'Attribution: what predicts winning' -AllowFail | Out-Null
        }
    }

    # ---- coordinate descent ---------------------------------------------
    if ($run.tune) {
        $out = New-AgentPath 'tuned'
        Invoke-Py -Script 'src\kaggriculture\train\tune.py' `
            -Arguments @('--base', $Base, '--out', $out, '--seeds', $Seeds,
                         '--seed0', $Seed0, '--workers', $w, '--budget-min', $Minutes) `
            -Title "Coordinate descent ($Minutes min budget)" -AllowFail | Out-Null
        if (Test-Path (Join-Path $Root $out)) { $produced += $out }
    }

    # ---- CMA-ES ----------------------------------------------------------
    if ($run.cmaes) {
        $out = New-AgentPath 'cma'
        $a = @('--base', $Base, '--out', $out, '--generations', $Generations,
               '--popsize', $Popsize, '--seeds', $Seeds, '--seed0', $Seed0,
               '--workers', $w)
        if ($Resume) { $a += '--resume' }
        Invoke-Py -Script 'src\kaggriculture\train\cmaes.py' -Arguments $a `
            -Title "CMA-ES ($Generations generations x $Popsize candidates)" -AllowFail | Out-Null
        if (Test-Path (Join-Path $Root $out)) { $produced += $out }
    }

    # ---- ridge regression -------------------------------------------------
    if ($run.ridge) {
        $out = New-AgentPath 'ridge'
        $a = @('--base', $Base, '--out', $out, '--episodes', $Episodes, '--seed0', $Seed0)
        if ($Resume) { $a += '--resume' }
        Invoke-Py -Script 'src\kaggriculture\train\train_supervised.py' -Arguments $a `
            -Title "Ridge regression ($Episodes episodes)" -AllowFail | Out-Null
        if (Test-Path (Join-Path $Root $out)) { $produced += $out }
    }

    # ---- reinforcement learning -------------------------------------------
    if ($run.rl) {
        $out = New-AgentPath 'rl'
        $a = @('--base', $Base, '--out', $out, '--episodes', $Episodes, '--seed0', $Seed0)
        if ($Resume) { $a += '--resume' }
        Invoke-Py -Script 'src\kaggriculture\train\train_rl.py' -Arguments $a `
            -Title "Monte-Carlo control ($Episodes episodes)" -AllowFail | Out-Null
        if (Test-Path (Join-Path $Root $out)) { $produced += $out }
    }

    # ---- rate and document whatever came out -----------------------------
    if ($produced.Count) {
        Write-Step 'Rating the new agents'
        foreach ($p in $produced) {
            Write-Ok "produced $p"
            Invoke-Py -Script 'src\kaggriculture\measure\evaluate.py' `
                -Arguments @($p, '--vs', $Base, '-n', 8, '--workers', $w) `
                -Title "$p vs its base" -AllowFail | Out-Null
            Invoke-Py -Script 'src\kaggriculture\measure\elo.py' -Arguments @('--only', $p, '--workers', $w) `
                -Title "Elo for $p" -AllowFail | Out-Null
            if (-not $NoGraph) {
                Invoke-Py -Script 'src\kaggriculture\agentbuild\model_graph.py' -Arguments @($p) `
                    -Title "Knowledge graph for $p" -AllowFail | Out-Null
            }
        }
        Invoke-Py -Script 'src\kaggriculture\measure\elo.py' -Arguments @('--show') -AllowFail | Out-Null
    } else {
        Write-Warn2 'no new agent was produced'
    }

    Show-Next @(
        '.\scripts\simulate.ps1 -Mode all        # confirm it really is better',
        '.\scripts\test.ps1 -Full                # legality + beats-built-ins',
        '.\scripts\submit.ps1 -DryRun            # validate exactly as Kaggle will'
    )
} catch {
    Write-Fail $_.Exception.Message
    $exit = 1
}

Stop-Log $exit
exit $exit
