# Hands-off watcher for the Kaggle slim-extraction notebooks (resumable: state in data/kaggle_out/state.json).
#   - pushes queued kernels when a CPU slot is free (Kaggle allows 5 concurrent sessions)
#   - downloads each COMPLETE kernel's output once, then files it into data/slim/s1 with RUNID naming
#   - prints one line per event (for a Monitor); exits when every kernel is filed or failed
param(
  [string[]]$Kernels = @('krl-daily-a','krl-daily-b','krl-daily-c','krl-daily-d','krl-gm-sep2','krl-gm-sep1','krl-gm-aug'),
  [int]$PollSec = 120,
  # kernels that may be filed up to 2 files short (recorded as missing_files in the run json)
  [string[]]$AllowShort = @()
)
$ErrorActionPreference = 'Continue'
# `-File` passes "a,b,c" as ONE string: split comma lists ourselves
$Kernels = @($Kernels | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
$AllowShort = @($AllowShort | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
$root = Split-Path -Parent $PSScriptRoot
$outRoot = Join-Path $root 'data\kaggle_out'
$slim = Join-Path $root 'data\slim\s1'
New-Item -ItemType Directory -Force $outRoot, $slim | Out-Null
$statePath = Join-Path $outRoot 'state.json'
$state = @{}
if (Test-Path $statePath) { (Get-Content $statePath -Raw | ConvertFrom-Json).psobject.Properties | ForEach-Object { $state[$_.Name] = $_.Value } }
function Save { $state | ConvertTo-Json | Set-Content -Encoding utf8 $statePath }
function Status($k) { $s = kaggle kernels status "debmalya84/$k" 2>&1 | Out-String; if ($s -match 'KernelWorkerStatus\.(\w+)') { $Matches[1] } elseif ($s -match '404|not found') { 'MISSING' } else { 'UNKNOWN' } }
function Emit($m) { Write-Output ("{0} {1}" -f (Get-Date).ToUniversalTime().ToString('HH:mm:ssZ'), $m) }

function Download($k, $dl) {
  # kernel output listing is paginated (default 20 files/page): follow every page
  $tok = $null
  for ($i = 0; $i -lt 100; $i++) {
    $a = @('kernels','output',"debmalya84/$k",'-p',$dl,'--page-size','200')
    if ($tok) { $a += @('--page-token', $tok) }
    $r = & kaggle @a 2>&1 | Out-String
    if ($r -match 'Next Page Token\s*=\s*(\S+)') { $tok = $Matches[1] } else { break }
  }
}

function FileOutput($k) {
  $dl = Join-Path $outRoot $k
  New-Item -ItemType Directory -Force $dl | Out-Null
  # skip the (slow, per-file up-to-date check) paged download when every file is already here
  $pre = Get-Content (Join-Path $dl 'run.json') -Raw -ErrorAction SilentlyContinue | ConvertFrom-Json
  $got = (Get-ChildItem (Join-Path $dl 'slim') -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count
  if (-not $pre -or $got -lt $pre.files) { Download $k $dl }
  $run = Get-Content (Join-Path $dl 'run.json') -Raw -ErrorAction SilentlyContinue | ConvertFrom-Json
  if (-not $run -or $run.rc -ne 0) { return "no run.json or rc!=0" }
  $have = (Get-ChildItem (Join-Path $dl 'slim') -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count
  $short = $run.files - $have
  if ($short -gt 0 -and -not ($k -in $AllowShort -and $short -le 2)) { return "INCOMPLETE download: $have of $($run.files) files" }
  $run | Add-Member -NotePropertyName missing_files -NotePropertyValue ([math]::Max(0, $short)) -Force
  $src = if ($k -like 'krl-gm*') { 'gm' } else { 'official' }
  $sha7 = ($run.binary_sha -replace 'sha256\s*','').Substring(0,7)
  $runid = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmZ') + "-$k-$sha7"
  $n = 0
  Get-ChildItem (Join-Path $dl 'slim\slim') -Recurse -Filter *.parquet -ErrorAction SilentlyContinue | ForEach-Object {
    $date = $_.Directory.Name  # date=YYYY-MM-DD
    $dst = Join-Path $slim "source=$src\$date"
    New-Item -ItemType Directory -Force $dst | Out-Null
    $tag = $_.BaseName -replace '^part-',''
    Move-Item $_.FullName (Join-Path $dst "part-$runid-$tag.parquet") -Force; $n++
  }
  $ldst = Join-Path $slim 'ledger'; New-Item -ItemType Directory -Force $ldst | Out-Null
  Get-ChildItem (Join-Path $dl 'slim\ledger') -Filter *.parquet -ErrorAction SilentlyContinue | ForEach-Object {
    Move-Item $_.FullName (Join-Path $ldst ("ledger-$runid-" + ($_.BaseName -replace '^part-','') + '.parquet')) -Force
  }
  $runs = Join-Path $slim 'runs'; New-Item -ItemType Directory -Force $runs | Out-Null
  $run | Add-Member -NotePropertyName runid -NotePropertyValue $runid -Force
  $run | Add-Member -NotePropertyName kernel -NotePropertyValue $k -Force
  $run | Add-Member -NotePropertyName source -NotePropertyValue $src -Force
  $run | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 (Join-Path $runs "$runid.json")
  Get-ChildItem $dl -Filter *.log | Copy-Item -Destination (Join-Path $runs "$runid.log") -ErrorAction SilentlyContinue
  return "filed $n parquet files as $runid ($([math]::Round($run.bytes/1MB)) MB, $($run.secs)s)"
}

while ($true) {
  $open = 0
  foreach ($k in $Kernels) {
    $st = $state[$k]
    if ($st -in @('FILED','FAILED')) { continue }
    $open++
    $s = Status $k
    if ($s -eq 'MISSING' -or $st -eq 'QUEUED' -or ($s -eq 'UNKNOWN' -and -not $st)) {
      $r = kaggle kernels push -p (Join-Path $root "kaggle\kernels\$k") 2>&1 | Out-String
      if ($r -match 'successfully pushed') { $state[$k] = 'PUSHED'; Emit "PUSHED $k" }
      elseif ($r -match 'Maximum batch CPU session') { if ($st -ne 'QUEUED') { Emit "QUEUED $k (5 sessions busy)" }; $state[$k] = 'QUEUED' }
      else { Emit "PUSH-ERROR $k $($r.Trim())" }
    } elseif ($s -eq 'COMPLETE') {
      $msg = FileOutput $k
      if ($msg -like 'filed*') {
        $state[$k] = 'FILED'; Emit "FILED $k $msg"
        # rebuild the deduplicated, rank-annotated episode index (data/slim/s1/index/)
        $ix = & python (Join-Path $root 'python\corpus_index.py') 2>&1 | Select-Object -First 1
        Emit "INDEX $ix"
      }
      elseif ($msg -like 'INCOMPLETE*') { $state[$k] = 'DOWNLOADING'; Emit "RETRY $k $msg" }
      else { $state[$k] = 'FAILED'; Emit "FAILED $k $msg" }
    } elseif ($s -in @('ERROR','CANCEL_ACKNOWLEDGED','CANCELLED')) {
      $state[$k] = 'FAILED'; Emit "FAILED $k status=$s"
    } elseif ($s -ne $st) { $state[$k] = $s; if ($s -ne 'RUNNING' -or $st -ne 'PUSHED') { Emit "STATUS $k $s" } }
  }
  Save
  if ($open -eq 0) { Emit "ALL-DONE"; break }
  Start-Sleep -Seconds $PollSec
}
