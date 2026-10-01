# ⚠ CRITICAL 2026-09-17: v58 SHIPPED BROKEN — root cause + fix

v58 (sub 56293043) rates ~600 and loses to opponents v56y/v57 beat. Diagnosed on
the OFFICIAL engine (vendored kaggle-environments, in Docker):
- v58 vs v56y = **0-8**, v58 banks **35-44k** vs v56y's **145k**.
- v58 COMPILED == v58 PYTHON (43903=43903) → **NOT a binary bug**; the economy is weak.

THE REAL BUG IS A CODING BUG (not the binary, not "serve is bad"):

**The bandit engine MUST always take its base economy FROM THE CONFIG — never
from any other reference.** Today `build_rust_bandit`/`build_bandit_harness`
silently FALL BACK to `our_route()` when no `--base-tape` is given, and
`our_route()` hardcodes `agents/v45.0_bandit.py` (the OLD v45 tape). So a
"config-driven" harness shipped a hardcoded v45 economy that the config never
named. That is the defect. The PROVEN v56y economy (145k official, rates 2140)
is `rustengine/tapes/bandit_v57_egg.tape` + `bandit_v57_yarn.tape`.

Serve is NOT the villain — it exists for FAST iteration and stays. But it
currently mis-ranks (v58 18-14 on serve vs 0-8 official for the SAME matchup):
that is a SEPARATE coding bug in the serve port to be FIXED so serve stays a
faithful fast proxy, not abandoned.

**THE FIX (do before any improvement work):**
- **A. CONFIG-DRIVEN BASE (the core fix).** Add a `base_tape` field to
  `configs/bandit_config*.json` pointing at the PROVEN economy (egg/yarn). Make
  `build_rust_bandit.py` and `build_bandit_harness.py` read the base FROM CONFIG,
  and **REFUSE to build (hard error) if no base is specified** — delete the
  silent `our_route()` fallback entirely. `--base-tape` may override for
  experiments, but the default MUST be the config's base. Result: the bandit can
  only ever ship the economy the config names.
- **B. FIX SERVE ↔ OFFICIAL divergence (keep serve).** Find why the same matchup
  ranks opposite on serve vs official (v58 18-14 serve / 0-8 official) — likely a
  reactive-obs or market-mechanic difference in the serve port (the bit-exactness
  was verified for TAPE replay, not necessarily for reactive agents reading obs).
  Fix the serve port so serve matches official; keep serve as the fast gate,
  with an official-engine spot-check on ship candidates.
- **C. Ship the PROVEN economy.** With A in place, point the config's `base_tape`
  at `bandit_v57_egg.tape` (+ yarn fork); RE-EVOLVE branches on that base (the
  current branches were evolved on the v45 base — invalid). Verify on the
  official engine (bank ≈145k, no loss to v56y) with the agent loaded WITHOUT
  `__file__`, then re-submit to replace the broken v58.
- **D.** v57 (2140) is still active as one of the latest-2, so the pair isn't
  fully broken — but 56248907 (2251) was evicted; re-ship the strong economy to
  restore it.

The improvement plan below still holds; gating stays on serve for speed once B is
fixed, with an official-engine confirmation before every submit.

---

# Bandit improvement plan (from v58, 2026-09-17)

Two pillars, worked in lockstep:
- **Pillar A — a local panel that gives a REALISTIC picture** (so we trust a
  number before spending a submission slot on it).
- **Pillar B — bandit improvement TOWARD THE GOAL** (top-10, ~2900-2970; we are
  ~2140, so the gap is ~800 rating = economy).

Grounded entirely in what this session MEASURED, not a wish list.

## What we know for certain (constraints any plan must respect)
1. The gap is ECONOMY (production volume), not architecture/dispatch/sells: v58≈
   v56y loses 0-16 to the live frontier in EVERY world, wins only vs weak/old
   agents. Our D6/D12/D15/D21/D27 dispatch already beats the field's single fork.
2. Concrete deficit: v56y caps herd ~12 / land 3; real 2700+ winners run herd
   ~17 and outplant us, and CASH is not the blocker at divergence (banks even at
   D12). The tape just stops expanding.
3. Open-loop tape transplant is DEAD (lifting desyncs −56k; 0/18 winner tapes
   win-robust; recorded flexon loses its reactive edge under contention).
4. **Offline tape panels don't predict the ladder** (0.947 offline yet 0-16
   live), and we have NO faithful ~2400 opponent — lifted tapes desync to weak
   (we beat them 1.000, fake), reactive publics are all frontier (we lose 0.000,
   saturated). Evolution against a tape panel OVERFITS.
5. Reactive rails DON'T desync (they react to live state). Sell rails (`_r37`)
   are inert until production is high enough to glut a market.
6. Metric = per-world win rate + paired McNemar; never margin. Ship every
   NON-REGRESSION; the ladder is the final oracle.

---

# PILLAR A — a local panel that gives a realistic picture

The current panel is useless for ranking: lifted mid tapes saturate at 1.000,
the frontier saturates at 0.000, and offline scores don't track the ladder.
Fix it in four concrete steps.

### A1. Faithful opponents ONLY — purge the desyncing tapes
A panel opponent must PLAY CONSISTENTLY on our engine. That means:
- KEEP: reactive `agent(obs)` files (the public frontier — flexon, tschinkel,
  prvsiyan, tetsutani, nathanjacob…) and our OWN versioned agents (v52, v56y,
  v58 — they run their own tape+shell coherently).
- DROP: `data/gauntlet/mid_*.py` and any lifted single-episode tape — they
  desync to weak and score a fake 1.000 (proven this session).
Rule going forward: no opponent enters the panel unless it is either a reactive
agent file or a tape verified win-robust across worlds (0/18 real winner tapes
passed — so effectively "reactive agents only").

### A2. Fill the mid-level (2200-2700) gap — the missing rungs
We have weak (djamila) + our own (~2100-2300) + frontier (~2700). The 2400-2600
rung, where the ladder actually pairs us, is empty. Fill it with FAITHFUL
reactive agents:
- Download 20-40 more PUBLIC agents spanning the rating spectrum (not just the
  top): `kaggle kernels list --competition kaggriculture --sort-by voteCount`
  then pull a spread; extract each to a standalone `agent(obs)`; keep only the
  self-consistent ones (import-clean, 720 legal steps vs PASS, no crash).
- Label each opponent with its REAL ladder rating (join the notebook's
  submission id → `data.registry --sync-kaggle` / the LB), so the roster is
  BANDED, not a bag.
- Our own version history (v40…v58) gives free, faithful rungs at 1900-2300.
Target roster: ~6 per band × {sub2100, 2100-2300, 2300-2500, 2500-2700, 2700+}.

### A3. Ladder-CALIBRATE the panel (the only proof it's realistic)
A panel is "realistic" iff it predicts the ladder. Certify it:
- For every agent WE have submitted (v52/v56y/v57/v58…), compute its panel win
  rate AND its REAL ladder win rate (`ourgames --by-submission`, banded by
  `op_rating_pre`). Keep only the opponents/cells where panel ≈ ladder (extend
  the "hard band instrument", which already matched 56/58 exact). Prune the rest.
- Report the panel's predictive error (panel wr − ladder wr) per band. A band
  with high error is not trusted for ship decisions until fixed.
- Weight the aggregate by the ladder's real matchmaking distribution (we mostly
  meet near-rating opponents), not uniformly.

### A4. The gate (the deliverable)
`bandit_gate.py`: per-world win rate (`win_metric`, money regime cancels) vs the
banded faithful roster, paired McNemar vs the incumbent, reported per band +
ladder-weighted aggregate. Ship rule: NON-REGRESSION in every band + no band
regressed at p<0.05. The 2500-2700 band trending up is the top-10 signal.

**Why this matters:** without A, every B number is a coin flip (v58 read 0.947
offline and lost live). A is what turns "built an agent" into "know it's better."

---

# PILLAR B — bandit improvement toward the goal (top-10)

Close the ~800-rating economy gap by growing production through REACTIVE rails
(no desync), validated on the Pillar-A panel, shipped as non-regressing v58.x.

### B1. Reactive economy expansion (the real lever — start here)
Directly attacks the measured herd-12→17 / under-planting deficit, world-robust:
- **`herd_grow` rail (highest value):** when `cash > reserve` AND a pasture/coop
  slot is free AND herd < target(shops), buy animal (sheep→YARN, cow→milk,
  goose→egg) + build pasture + FEED/CARE. Grows herd 12→~17 REACTIVELY across
  D6-D18 wherever cash/land allow. Tunable target + reserve.
- **`land_grab` rail:** buy the next quadrant when cash-rich + tile-constrained.
- **`fill_plant` rail:** fill empty owned tiles with the best-ROI crop for the
  world (tomato in pizza worlds) when labour capacity remains.
Each = one flag in `mbandit.rs` (+ Python fallback), swept, gated by Pillar A,
shipped v58.1/58.2/58.3. These use the mechanism that already works (rails),
pointed at production instead of sells — the one path proven not to desync.

### B2. Construct a stronger BASE economy (if B1 plateaus below frontier)
- **Tape constructor:** a policy that plays the target economy (herd 17, winners'
  crop mix from `episode_features` signatures) on OUR engine, RECORDED →
  self-consistent base tape ("recording solves desync" applied to our OWN target
  policy, not a foreign replay). Feed to `build_bandit_harness --base-tape`.
- Re-evolve branches on the stronger base, gated on Pillar A (not the saturated
  tape panel that caused v58's overfit).

### B3. Activate the sell layer (once production is strong)
`_r37 scarcity_sell` is built and inert only because we under-produce. Once B1/B2
raise volume enough to glut WOOL/MILK/STRAW, re-tune `hold_frac` (≈1.0 prototype
sweet spot) + add anti-clone front-run (our own route mirror is the biggest live
loss source). Gate + ship.

### B4. Learned dispatcher (the one place ML earns a seat)
Train a small CART/tree (Tschinkel-style) on real 2600+ ladder replays
(georgymamarin shards) to pick the economy per world at D6/D12/D24 from public
state. Low-dim, robust; the correct use of the BC/IL datasets — NOT a per-turn
policy seat (measured dead).

---

## Sequencing (interleave A and B)
1. **A1+A2+A4 first** (purge tapes, pull spanning-rating faithful publics, stand
   up `bandit_gate.py`) — one focused pass; without it B is blind.
2. **A3** calibrate against `ourgames` as v58's real games accumulate.
3. **B1a `herd_grow`** → gate on the new panel → ship v58.1 if non-regression.
4. Repeat B1b/B1c as v58.2/58.3; re-calibrate A3 after each ship.
5. Escalate to B2 only if reactive rails plateau below the frontier; B3/B4 ride
   on top once the economy is competitive.

**Cadence per ship:** paired win-rate gate (banded) → Linux musl rebuild via
Docker (PowerShell, not Git Bash) → push to the PRIVATE notebook → validate →
submit → `ourgames` re-calibration. The 2500-2700 band is the top-10 dial.

## STOP doing (paid-for negatives)
- Lifting/curating foreign winner tapes as an open-loop base (desync, 0/18).
- Ranking by mean margin (ship on wins, per world, paired).
- Trusting the synthetic tape panel or lifted mid-tapes as an oracle.
- Treating "beats the frontier" as the ship bar — ship non-regressions.

## Datasets/models
- **georgymamarin episodes** → mine winner economy TARGETS (herd/crop signatures)
  for B1/B2, opponent-rating LABELS for A2/A3, the B4 dispatcher training set.
- **KiroSamurai IL** → portable BC corpus for B4 only.
- **spatial-BC / hesoponyo / buddy** → not seat material; study only.
