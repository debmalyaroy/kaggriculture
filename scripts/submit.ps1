<#
.SYNOPSIS
    Validate a specific bot exactly the way Kaggle will, then (only if you
    explicitly say so) upload it.

.DESCRIPTION
    SAFE BY DEFAULT. Without -Live this is a dry run: everything except the
    upload. Uploading needs BOTH -Live on the command line AND typing SUBMIT at
    the prompt inside src\kaggriculture\pipeline\submit.py. Two gates, because a submission is
    irreversible, you get 5 a day, and only the latest 2 stay active.

    Three ways to choose the bot:

      -Agent <name>   submit exactly this one. Accepts any of:
                          v2_tuned
                          v2_tuned.py
                          agents\v2_tuned.py
                          D:\codebase\kaggriculture\agents\v2_tuned.py
      -Best           take the top-rated bot off the Elo ladder. Refuses if the
                      ladder says the lead is not settled -- override with
                      -Force once you have read why.
      (neither)       src\kaggriculture\pipeline\submit.py ranks agents\ by win rate and picks.

    -List prints the ladder and exits, so you can see what you are choosing
    between before committing a slot.

    Validation covers, in Kaggle's own order:
      1. rank        by win rate (not coin margin -- the leaderboard is a skill
                     rating), unless you named a bot
      2. size        under 100 MiB
      3. latency     per-turn time against the real 1-second actTimeout
      4. self-play   the Validation Episode Kaggle runs on upload, run locally
                     first so a failure costs seconds, not a submission slot
      5. approval    stats printed, you type SUBMIT

.PARAMETER Agent
    The bot to submit. Bare name, file name, or path.

.PARAMETER Best
    Pick the top-rated bot from .local\elo\ladder.json.

.PARAMETER Force
    Allow -Best to proceed when the ladder lead is inside the error bars.

.PARAMETER List
    Print the ladder and exit.

.PARAMETER Live
    Actually upload. Without it, nothing is sent.

.EXAMPLE
    .\scripts\submit.ps1 -List

.EXAMPLE
    .\scripts\submit.ps1 -Agent v2_tuned                    # dry run

.EXAMPLE
    .\scripts\submit.ps1 -Agent agent_v3_20260804_160019 -Live -Message "cma gen12"

.EXAMPLE
    .\scripts\submit.ps1 -Best -Live
#>
[CmdletBinding()]
param(
    [string]$Agent,
    [switch]$Best,
    [switch]$Force,
    [switch]$List,
    [switch]$Live,
    [int]$Seeds = 6,
    [int]$Seed0 = 0,
    [int]$Workers = 0,
    [string]$Message = '',
    [string[]]$Exclude
)

. (Join-Path $PSScriptRoot '_lib.ps1')
$ErrorActionPreference = 'Stop'

function Resolve-AgentPath {
    <#  Accept a bare bot name and turn it into a repo-relative agents\ path.
        Anything that already resolves to a file is passed through. #>
    param([string]$Name)
    if (-not $Name) { return $null }

    $tries = @(
        $Name,
        (Join-Path $Root $Name),
        (Join-Path $Root "agents\$Name"),
        (Join-Path $Root "agents\$Name.py")
    )
    if ($Name -notlike '*.py') { $tries += (Join-Path $Root "$Name.py") }

    foreach ($t in $tries) {
        if (Test-Path $t -PathType Leaf) {
            $full = (Resolve-Path $t).Path
            return $full.Substring($Root.Length).TrimStart('\')
        }
    }

    # Nothing matched -- fail with the list, not just "not found".
    $have = Get-ChildItem (Join-Path $Root 'agents') -Filter '*.py' |
            ForEach-Object { '  ' + $_.BaseName }
    throw ("no agent matches '$Name'. Available:`n" + ($have -join "`n"))
}

function Get-Ladder {
    $raw = Invoke-PyInline 'import runpy,sys;sys.argv=["dashboard","--json"];runpy.run_path("src/kaggriculture/pipeline/dashboard.py",run_name="__main__")'
    try { return $raw | ConvertFrom-Json } catch { return $null }
}

Write-Banner 'Kaggriculture - submission' $(if ($Live) { 'LIVE - this can upload' } else { 'DRY RUN - nothing will be uploaded' })
$log = Start-Log 'submit'
$exit = 0

try {
    # ---- -List: show the ladder and stop ---------------------------------
    if ($List) {
        Invoke-Py -Script 'src\kaggriculture\pipeline\dashboard.py' -Title 'Elo ladder' | Out-Null
        Show-Next @(
            '.\scripts\submit.ps1 -Agent <name>      # submit a specific bot',
            '.\scripts\submit.ps1 -Best              # submit the top-rated one',
            'docs\elo-dashboard.html                 # the full table'
        )
        Stop-Log 0
        exit 0
    }

    Test-Prereq -NeedKaggle

    # ---- pick the bot ----------------------------------------------------
    if ($Best) {
        if ($Agent) { throw '-Agent and -Best are mutually exclusive' }
        $lad = Get-Ladder
        if (-not $lad -or -not $lad.recommendation.pick) {
            throw 'the Elo ladder has no rated bots yet. Run: python src\kaggriculture\measure\elo.py --rounds 4'
        }
        $pick = $lad.recommendation.pick
        Write-Note $lad.recommendation.why
        if (-not $lad.recommendation.confident -and -not $Force) {
            Write-Fail 'the ladder lead is inside the error bars'
            Write-Note 'add more rated games (python src\kaggriculture\measure\elo.py --rounds 4),'
            Write-Note 'or pass -Force if you have read the reason above and still want it.'
            throw 'refusing to spend a submission slot on an unsettled lead'
        }
        $Agent = $pick
        Write-Ok "-Best selected: $Agent"
    }

    if ($Agent) {
        $Agent = Resolve-AgentPath $Agent
        Write-Ok "submitting: $Agent"
    } else {
        Write-Note 'no bot named: src\kaggriculture\pipeline\submit.py will rank agents\ by win rate and pick'
    }

    if (-not $Live) {
        Write-Warn2 'dry run: add -Live to actually upload'
    } else {
        Write-Host ''
        Write-Host '  You passed -Live. src\kaggriculture\pipeline\submit.py will still require you to type' -ForegroundColor Yellow
        Write-Host '  SUBMIT before anything is uploaded. 5 submissions per day; only' -ForegroundColor Yellow
        Write-Host '  the latest 2 stay active.' -ForegroundColor Yellow
    }

    $w = if ($Workers -gt 0) { $Workers } else {
        [Math]::Max(1, [int]$env:NUMBER_OF_PROCESSORS - 1)
    }

    # A submission is worth 60 seconds of checking first.
    Invoke-Py -Script 'tests\test_agents.py' -Title 'Contract tests before submitting' -AllowFail | Out-Null

    $a = @('--seeds', $Seeds, '--seed0', $Seed0, '--workers', $w)
    if ($Agent)   { $a += @('--agent', $Agent) }
    if ($Message) { $a += @('--message', $Message) }
    if ($Exclude) { $a += @('--exclude') + $Exclude }
    if (-not $Live) { $a += '--dry-run' }

    Invoke-Py -Script 'src\kaggriculture\pipeline\submit.py' -Arguments $a `
        -Title $(if ($Live) { 'Validate and submit' } else { 'Validate (no upload)' }) | Out-Null

    # Keep the dashboard honest about what is now on Kaggle.
    if ($Live) {
        Invoke-Py -Script 'src\kaggriculture\pipeline\dashboard.py' -Title 'Refreshing the dashboard' -AllowFail | Out-Null
    }

    if (-not $Live) {
        Show-Next @(
            ".\scripts\submit.ps1 -Agent $Agent -Live   # same run, but it can upload",
            'https://www.kaggle.com/competitions/kaggriculture/submissions'
        )
    } else {
        Show-Next @(
            'https://www.kaggle.com/competitions/kaggriculture/submissions',
            '.\scripts\download.ps1                  # pull the resulting games back in ~1h'
        )
    }
} catch {
    Write-Fail $_.Exception.Message
    $exit = 1
}

Stop-Log $exit
exit $exit
