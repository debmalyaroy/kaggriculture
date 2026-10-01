# kaggriculture-rl: architecture (2026-09-25)

This supersedes the Kaggle-offload parts of `docs/queue.md` and `aws/README.md` (§15–18d of `PLAN.md`).
The plan and schedule are in `PLAN.md` §20–22. The research behind the choices is in
`docs/research-2026-09-25.md`.

**Objective:** 0 losses to opponents rated below 2500, and at least 80% wins at 2500+ (target 90%).

## 1. System at a glance

```
                ┌───────────────────────────── KAGGLE ─────────────────────────────┐
                │ daily episode datasets + GM dataset ──▶ krl-delta-* notebooks     │
                │   (private; corpus-extract --mode slim; engine 1.32.7 only)       │
                │ our older notebooks' outputs (krl-daily-a..f, krl-gm-*)            │
                │ private submission notebook  ◀── operator only                    │
                └────────▲──────────────────────────┬──────────────────────────────┘
                         │ kaggle CLI (kjob.py)     │ slim parts (~300 MB/day)
┌────────────────────────┴──────────────────────────▼──────────────────────────────┐
│ AWS c8g.16xlarge (ARM64, 64 cores) — systemd `krl-queue` runs ops/queue.py         │
│                                                                                   │
│  corpus ─▶ features/day_obs ─▶ caches ─▶ BC ─▶ PPO (+oracle) ─▶ checkpoints        │
│  league ─┘                                     │                                   │
│                                                ▼                                   │
│                        gate / tournament / public-panel pre-filter                 │
│                                                │ policy.bin + report               │
│                                                ▼                                   │
│                                  data/candidates/RC_PENDING.json                   │
└────────────────────────────────────────────────┬──────────────────────────────────┘
                                                 │ aws/pull_rc.sh (KB, not GB)
┌────────────────────────────────────────────────▼──────────────────────────────────┐
│ LAPTOP (x86, Windows) — harness copied into kaggriculture-rl/harness/              │
│  Rust==torch on x86 → x86 musl tar.gz → FULL tournament (band-scored) →           │
│  official-engine spot check → RC report → operator approves → private notebook    │
└───────────────────────────────────────────────────────────────────────────────────┘
```

Three rules hold everywhere:
- **Nothing is submitted automatically.**
- **Every number is paired** (same world, seed and seat) and reported by rating band.
- **This track's code, data and memory live only in `kaggriculture-rl`.** Pieces from the root repo
  are copied in, never imported.

## 2. The agent (what ships)

One Rust binary (`crates/agent`) inside `submission.tar.gz`, with `main.py` as a thin stdio
bridge. Per turn it takes one observation in and returns one action out, in under 20 ms.

```
 observation (step t)
   │
   ├─▶ dayobs Builder.observe()   (crates/dayobs; the SAME code as the corpus adapter)
   │
   ├─▶ at hour 1 of each day (t = 24d+1):  DAILY CONTROLLER
   │       ┌──────────────────────────────────────────────────────────────┐
   │       │ 1. SHIELD (rules, from v62 GroupCtl, copied)                 │
   │       │    group at step 25: DIFFERENT / PARTIAL / COPY              │
   │       │    D24 mirror guard; allowed-profile mask per group/day      │
   │       │ 2. POLICY  x = dayobs v2 (90 + group one-hot + sell timing)  │
   │       │    Linear(→64) → GRU(64) → heads:                             │
   │       │      pi[32] profiles (masked by the shield)                   │
   │       │      v  (win prob)          aux: band[5], family[K], rnext[9] │
   │       │ 3. choose: argmax(pi ∘ mask); fall back to the group profile  │
   │       │ 4. JITTER: per-game hash offset on market timing knobs        │
   │       └───────────────────────────┬──────────────────────────────────┘
   │                                   ▼  today's PROFILE (market-side knobs)
   ├─▶ route tape action (fixed plan: planting, hiring, land, animals; router picks the route)
   ├─▶ CHASSIS (every turn; v61.1 settings): hand_align · weed_repair · sell_lead + R36 debts ·
   │     RACEPX   (budget/room/clamp/dead_stock/terminal chassis guards are OFF in v61.1: that work
   │     is done by the post chain below)
   ├─▶ POST CHAIN, 64 stages in v61.1 order (always on, never under RL control; the profile only
   │     sets the market-side knobs): room guard (hour 23) · sales-first · V219/V231/V233/R51/R85/
   │     R95/R97/V9/CA/HD2/Y production overlays · R36 reserve · R37/R44 · RACE · OR2/V44Y/CXD
   │     ordering · ADV/FX/EV/DP/MP/MPX lead-sells · E410/E402 · MG/IG queue · RSA · AFR
   └─▶ ≤ 10 legal market orders, hands aligned ─▶ action
```

**Why this shape.** The public evidence settles it:
- A tape's market orders on another farm go bankrupt, so the channels can't be separated.
- Pure RL reaches about $81k against tapes that bank $107–160k.
- Per-turn cloning doesn't track money.

So RL never writes an order. Once a day it chooses one of 32 market-side profiles, and the farm
plays its tape exactly. That keeps every branch in sync with the base.

**Opponent identification.**

| layer | what | when |
|---|---|---|
| hard rule (shield) | day-0 position equality (at least 10%) and step-1 cash equality → DIFFERENT, PARTIAL or COPY; 81% correct by day 1.5 | step 25, fixed |
| D24 mirror guard | ≥ 90% same squares on day 23 and a small money gap → keep the day 1–23 profile | day 24 |
| GRU family head | aux classifier on the shared memory, labelled by league truth and corpus hindsight; target ≥ 80% by day 3 | every day |
| baseline | 3-day MLP, used only for comparison | offline |

**Sell schedule.**
- *The rival's* sales come from public inventory: `ΔI + town draw − own fills + own buys`, per step.
  This matches the corpus with 0 mismatches. The observation carries:
  - daily totals;
  - (v2) the rival's mean sell hour and the share sold before our scheduled sale;
  - the `rnext` head, which predicts the rival's next-day sales.
- *Ours* is the tape's planned sales, which the guard chain shifts every turn. The profile sets:
  - `rsa_look` (sell ahead);
  - AFR on/extra (pull sales forward when the rival pre-empts us);
  - EV/DP/MP horizons;
  - race depth;
  - dump/hold.

**Deployed invariants** (each checked by a test that can't be skipped):
- profile 0 is bank-identical to v61.1;
- live dayobs equals corpus dayobs;
- the Rust forward pass equals torch on 1,000 states, on ARM and on x86;
- mean turn under 20 ms, worst under 100 ms;
- only legal actions and at most 10 orders.

## 3. Code map

| crate / module | role |
|---|---|
| `crates/engine` | full step function, bit-exact vs the official engine; state is `Clone`, so branching is exact |
| `crates/agent` | chassis + layers (v61.1 port) + profiles (`layers/knobs.rs`) + shield (`layers/group.rs`, copied from v62) + `PolicyCtl` hook |
| `crates/dayobs` | online per-day observation vector (v1: 90 floats; v2 adds group + sell timing) |
| `crates/features` | corpus features + the dayobs adapter (the same code path as live play) |
| `crates/policy` | pure-Rust GRU forward pass; `policy-check` |
| `crates/corpus` | `corpus-extract`: slim / features / obs modes over GM, official and league sources |
| `crates/runner` | `league-rand`, `ppo-rollout`, `tapeplay` (incl. `--guarded` band tapes and `--policy/--shield` candidates), `selfplay`; in-process opponents; the branch oracle (new) |
| `python/learn` | `cache.py`, `model.py`, `bc.py`, `ppo.py`, `gate.py`, `rc.py`, `export_check.py`, `profile_audit.py` |
| `python/delta.py`, `kjob.py` | Kaggle delta: plan → push → wait → download by manifest → file → prune → index |
| `python/panel_gate.py` | public-panel gate (Python opponents through the copied harness) |
| `harness/` (new, copied) | faithful Python-agent harness + public roster + v61/v61.1 for the laptop tournament |
| `ops/queue.py`, `tasks.json` | the autopilot: dependencies, lanes, RAM floor, `every_h`, `follow`, exit records |
| `aws/` | `sync_up`, `bootstrap`, `verify`, `status`, `pull_rc` (new), `shutdown_at` |

## 4. Training system

### 4.1 Data

| data | source | size | refresh |
|---|---|---|---|
| slim corpus | Kaggle delta notebooks → `data/slim/s1` | 12 GB, 183k episodes | every 12 h |
| features, day_obs | `corpus-extract --mode features/obs` | 1 GB | after each delta |
| corpus cache | `cache.py corpus` (memmap, band + rival-next + win labels) | 1.2 GB | after each delta |
| league games | `league-rand` on the box: sticky random per-day profile schedules vs the opponent mix | 57.5k games, 12 MB | once, then per new opponent set |
| top-band tapes | 2500+ team medoid per world → `slim_to_tape.py` | small | every delta |

### 4.2 Opponent pool (in-process Rust unless noted)

| opponent | share of PPO games |
|---|---|
| top-band guarded tapes (2500+ medoids, played by the v61.1 chassis) | 30% |
| our lineage: v61.1, v62 / v62.1 (with shield), escalated / AFR / random-knob variants | 25% |
| families that beat us (herd_safe first, then pioneers, population_robust), ported as chassis configs with parity | 15% (Python host until ported, capped) |
| frozen self snapshots | 20% |
| random-schedule learners | 10% |

Worlds are sampled with priority on the current loss rate: at least 25% of games in the five worst
worlds. One seed per world is held out for validation.

### 4.3 Learning

1. **BC** (`bc.py`): advantage-weighted on league outcomes, plus the corpus aux heads (band,
   family, rnext). 6,000 steps, 15–30 min on CPU.
2. **PPO** (`ppo.py`), recurrent over whole 30-day episodes:
   - 4,096 games per iteration;
   - clip 0.2, GAE λ 0.95;
   - **adaptive KL leash to the BC anchor** (target KL; entropy 0);
   - **value head on a detached trunk**;
   - reward **Φ(margin / σ_world)**.
3. **Branch oracle** (expert iteration, new, from day 2):
   - on key days (D0–3, D12, D20–29), clone the engine and chassis;
   - play the top-8 profiles and the shield's profile to the end against the same in-process
     opponent;
   - the label is the best profile, weighted by its gain;
   - a supervised loss runs beside PPO, then DAgger on the policy's own states;
   - at most 30% of the box.
4. **Teacher hot-swap:** a new BC after each delta becomes the new anchor at the next iteration
   boundary.
5. **Checkpoints** every 10 iterations. `best.bin` is kept, and two validation drops in a row roll
   back to it.

### 4.4 Measurement (ladder order)

| gate | where | panel | pass |
|---|---|---|---|
| Q14 BC gate | box | 8-opponent Rust panel × 100 seeds × 2 seats, paired vs v61.1 | not worse (McNemar) |
| Q17 tournament (every 8 h) | box | BC, PPO best, last 3 snapshots vs the pool, band-scored | ranks candidates |
| Q18 public-panel pre-filter | box | 12 random public agents + v61, v61.1, v62, v62.1; 24 worlds × 2 seeds × 2 seats | band rule + paired ≥ v62.1 |
| RC tournament | laptop | all 88 public + v61, v61.1, v62, v62.1; 24 worlds × 2 seats, paired vs v62.1 | 0 sub-2500 losses, ≥ 80% at 2500+, McNemar vs v62.1 not worse, no family significantly worse |
| official spot check | laptop | 10 seeds on kaggle-environments 1.32.7 | exact banks |

**Band scoring (G1).** Rating bands come from the ladder itself: `python/band_tapes.py` takes recent
real players from the corpus (1-2 games per team, every seed used once, so one world per match as on
the ladder) in five bands (<2100 · 2100-2300 · 2300-2500 · 2500-2700 · 2700+), and
`python/band_gate.py` plays the candidate and the reference against each player's recorded stream on
its own seed via `tapeplay --guarded`. The guards kept for a tape (hand_align, weed_repair,
clamp_sells) reproduce 60/60 real games exactly when nothing is off; budget_guard broke 41/60, so it
is excluded. Mapping public notebooks to ladder ratings by author covered only 7 of 88, so the public
roster stays an unbanded field check.

## 5. The hands-off pipeline

### 5.1 Setup (once, about 60 min)

```
 laptop: aws/up.sh HOST
   1 sync_up code (1 MB) + index/ledgers/runs/state (35 MB) ............ seconds
   2 bootstrap: rustup + cargo build --release (native ARM), venv, torch CPU,
     systemd krl-queue, shutdown timer ..................................... 10–15 min
   3 delta.py pull-corpus: download our notebooks' outputs by name, check vs the index 15–30 min
   4 Q02 features → Q09 day_obs → Q09c cache ............................... ~10 min
   5 verify.sh: parity 600 games · 180 exact replays · obs equality · Rust==torch ·
     v61.1 banks == laptop (24 seeds) · PPO smoke + games/s ................ 5–10 min
     └─ any failure → the queue never starts; status.sh shows why
```

### 5.2 The queue (autopilot)

```
 Q07 league (60 thr, ~40 min) ─▶ Q07c cache ─▶ Q05 profile audit (masks)
                                      │
 Q09c corpus cache ───────────────────┼─▶ Q11 BC ─▶ Q10c Rust==torch ─▶ Q15c PPO smoke
                                      │                                  │
                                      │                   Q14 BC gate ◀──┤
                                      ▼                                  ▼
                              ┌──────────────────── Q16 PPO (nonstop, ~3.5–4 min/iter) ───────────┐
                              │  + Q19 branch oracle labels (≤30% cores, from day 2)             │
                              └──────▲──────────────────────────────┬─────────────────────────────┘
               new teacher (hot-swap) │                              │ checkpoints
 every 12 h: Q01 delta ─▶ Q02 ─▶ Q09 ─▶ Q09c ─▶ Q12 BC fine-tune     │
             └▶ Q20 top-band tape refresh (opponents)                ▼
                                                    every 8 h: Q17 tournament (box)
                                                                     │ top PASS
                                                                     ▼
                                                    Q18 public-panel pre-filter (box)
                                                                     │ pass
                                                                     ▼
                                                    RC_PENDING.json (policy.bin + reports)
```

Queue mechanics (`ops/queue.py`):
- A job starts when its dependencies are done, its lane has a free slot, and there is enough free
  RAM.
- Jobs are detached and write `data/ops/logs/<id>.{log,exit}`.
- A process that vanishes without an exit record is marked `lost`, never done.
- Every job resumes from what is on disk.
- `every_h` handles periodic jobs; `follow` re-runs a job for each new round of its upstream.
- systemd restarts the runner.

CPU lanes on 64 cores:

| lane | cores |
|---|---|
| ppo | 44 |
| oracle | up to 16 (yields to gates) |
| gate / tournament | 60 while running (PPO pauses between iterations) |
| io | 1 |

### 5.3 Laptop leg (per candidate)

```
 aws/pull_rc.sh HOST ─▶ data/candidates/<name>/{policy.bin, report.json}
   ─▶ export check on x86 (1,000 states) ─▶ build_submission.ps1 (Docker, x86_64 musl tar.gz)
   ─▶ harness tournament: 88 public + v61, v61.1, v62, v62.1, 24 worlds × 2 seats, paired vs v62.1
      (~8.8k games, 1.5–2 h on 14 workers), band-scored
   ─▶ official-engine spot check (10 seeds, exact banks) ─▶ RC report ─▶ OPERATOR
```

### 5.4 Safety

| failure | response |
|---|---|
| verify fails on ARM | the queue never starts; status shows which check |
| a job crashes or is lost | marked failed/lost; `retry` resumes from disk; never counted as done |
| validation drops twice | roll back to `best.bin` |
| a harness error in a gate game | counted as an error, never as a win |
| spot instance reclaimed | relaunch and `up.sh`; everything resumes from S3-free on-disk state (re-pull from Kaggle if the disk is lost) |
| cost | auto-shutdown on 29 Sep at 22:00Z; budget alert |
| anything outward-facing | submission, publishing and dataset pushes all need the operator |
