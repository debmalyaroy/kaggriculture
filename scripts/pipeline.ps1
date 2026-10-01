<#
.SYNOPSIS
    The whole loop: download -> analyse -> train -> simulate -> test -> submit.

.DESCRIPTION
    One command that takes the project from "there is new ladder data" to "there
    is a validated agent ready to upload". Each stage is one of the other
    scripts, so anything here can also be run on its own.

      1. download   fresh top-20 ladder replays + all of your own games
      2. analyse    attribution over the new data, EDA report, diff against the
                    top-20's play on identical states
      3. train      the search/ML algorithm you chose, producing a new agent
      4. simulate   head-to-head win rate, official episode, Elo ladder
      5. test       legality, latency, beats-the-built-ins
      6. submit     validation only, unless you pass -Submit -Live

    NOTHING IS UPLOADED unless you pass BOTH -Submit and -Live, and even then
    src\kaggriculture\pipeline\submit.py still asks you to type SUBMIT. A stage that fails stops the
    run rather than feeding a broken artefact to the next stage -- except the
    ones marked optional, which warn and continue.

.PARAMETER Algo
    Which trainer stage 3 runs. Default cmaes.

.PARAMETER Minutes
    Budget for the training stage.

.PARAMETER SkipDownload
    Reuse the data already in data\ and .local\episodes.

.PARAMETER Submit
    Run stage 6. Still a dry run unless -Live is also passed.

.PARAMETER Live
    Allow stage 6 to actually upload (after the typed SUBMIT confirmation).

.PARAMETER Native
    Run src\kaggriculture\pipeline\pipeline.py instead -- the Python orchestration, which does
    fetch -> analyze -> refine -> evaluate -> improve in one process with its
    own interactive menu. Useful when you want its refine step specifically.

.EXAMPLE
    .\scripts\pipeline.ps1

.EXAMPLE
    .\scripts\pipeline.ps1 -Algo cmaes -Minutes 45 -PerDay 120 -Days 5

.EXAMPLE
    .\scripts\pipeline.ps1 -SkipDownload -Algo tune -Submit        # validate, do not upload
#>
[CmdletBinding()]
param(
    [ValidateSet('learn', 'tune', 'cmaes', 'ridge', 'rl', 'all')]
    [string]$Algo = 'cmaes',
    [int]$Days = 3,
    [int]$PerDay = 60,
    [int]$Own = 0,
    [int]$Jobs = 8,
    [int]$Minutes = 20,
    [int]$Generations = 8,
    [int]$Seeds = 4,
    [int]$Matches = 8,
    [int]$Workers = 0,
    [switch]$SkipDownload,
    [switch]$SkipTrain,
    [switch]$AnyTop,
    [switch]$Submit,
    [switch]$Live,
    [switch]$Native,
    [string]$Message = ''
)

. (Join-Path $PSScriptRoot '_lib.ps1')
$ErrorActionPreference = 'Stop'

Write-Banner 'Kaggriculture - full pipeline' 'download -> analyse -> train -> simulate -> test -> submit'
$log = Start-Log 'pipeline'
$exit = 0
$stages = @()

function Invoke-Stage {
    param([string]$Name, [scriptblock]$Body, [switch]$Optional)
    Write-Host ''
    Write-Host ('-' * 72) -ForegroundColor DarkCyan
    Write-Host "  STAGE: $Name" -ForegroundColor Cyan
    Write-Host ('-' * 72) -ForegroundColor DarkCyan
    $t0 = Get-Date
    try {
        & $Body
        $script:stages += [pscustomobject]@{ Stage = $Name; Result = 'ok'; Took = (Format-Duration ((Get-Date) - $t0)) }
    } catch {
        $script:stages += [pscustomobject]@{ Stage = $Name; Result = 'FAILED'; Took = (Format-Duration ((Get-Date) - $t0)) }
        if ($Optional) { Write-Warn2 "$Name failed but is optional: $($_.Exception.Message)" }
        else { throw }
    }
}

try {
    Test-Prereq

    if ($Native) {
        $a = @('--days', $Days, '--per-day', $PerDay, '--own', $Own,
               '--seeds', $Seeds, '--improve-budget', $Minutes, '--non-interactive')
        if ($SkipDownload) { $a += '--skip-fetch' }
        if ($Message) { $a += @('--message', $Message) }
        Invoke-Py -Script 'src\kaggriculture\pipeline\pipeline.py' -Arguments $a `
            -Title 'Native Python pipeline (never auto-submits)' | Out-Null
        Stop-Log 0
        return
    }

    function Invoke-Script {
        <#  Run a sibling script with hashtable splatting and propagate failure. #>
        param([string]$Name, [hashtable]$Params = @{})
        & (Join-Path $PSScriptRoot $Name) @Params
        if ($LASTEXITCODE -ne 0) { throw "$Name exited $LASTEXITCODE" }
    }

    # ---- 1. data ---------------------------------------------------------
    if (-not $SkipDownload) {
        Invoke-Stage 'download' -Optional {
            $p = @{ Days = $Days; PerDay = $PerDay; Own = $Own; Jobs = $Jobs }
            if ($AnyTop) { $p.AnyTop = $true }
            Invoke-Script 'download.ps1' $p
        }
    } else { Write-Note 'skipping download (-SkipDownload)' }

    # ---- 2. analysis -----------------------------------------------------
    Invoke-Stage 'analyse' -Optional {
        Invoke-Py -Script 'src\kaggriculture\train\learn_params.py' -Arguments @('--top', 25) `
            -Title 'What predicts winning' -AllowFail | Out-Null
        Invoke-Py -Script 'src\kaggriculture\pipeline\eda_report.py' -Title 'EDA report' -AllowFail | Out-Null
        Invoke-Py -Script 'src\kaggriculture\engine\diff_sim.py' `
            -Arguments @('--out', 'docs\diff-simulation.md') `
            -Title 'Diff vs the top-20 on identical states' -AllowFail | Out-Null
    }

    # ---- 3. training -----------------------------------------------------
    $before = Get-NewestAgent
    if (-not $SkipTrain) {
        Invoke-Stage 'train' {
            Invoke-Script 'train.ps1' @{
                Algo = $Algo; Minutes = $Minutes
                Generations = $Generations; Seeds = $Seeds
            }
        }
    } else { Write-Note 'skipping training (-SkipTrain)' }
    $after = Get-NewestAgent
    if ($after -ne $before) { Write-Ok "new agent: $after" }
    else { Write-Warn2 "training produced nothing new; still on $after" }

    # ---- 4. simulation ---------------------------------------------------
    Invoke-Stage 'simulate' {
        Invoke-Script 'simulate.ps1' @{ Mode = 'standard'; Agent = $after; Matches = $Matches }
    }

    # ---- 5. tests --------------------------------------------------------
    Invoke-Stage 'test' {
        Invoke-Script 'test.ps1' @{ Full = $true }
    }

    # ---- 6. submission ---------------------------------------------------
    if ($Submit) {
        Invoke-Stage 'submit' {
            $p = @{ Agent = $after; Seeds = 6 }
            if ($Live)    { $p.Live = $true }
            if ($Message) { $p.Message = $Message }
            Invoke-Script 'submit.ps1' $p
        }
    } else {
        Write-Note 'stage 6 skipped: pass -Submit to validate, -Submit -Live to upload'
    }

    # ---- summary ---------------------------------------------------------
    Write-Host ''
    Write-Host '  Pipeline summary' -ForegroundColor Cyan
    $stages | Format-Table -AutoSize | Out-String | Write-Host
    Write-Ok "current best agent: $after"

    Show-Next @(
        ".\scripts\submit.ps1 -Agent $after -Live   # upload it",
        '.\scripts\pipeline.ps1 -SkipDownload -Algo tune   # another improvement round'
    )
} catch {
    Write-Fail $_.Exception.Message
    if ($stages.Count) { $stages | Format-Table -AutoSize | Out-String | Write-Host }
    $exit = 1
}

Stop-Log $exit
exit $exit
