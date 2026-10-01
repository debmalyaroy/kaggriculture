# v23.2_bandit — postmortem of the v23/v23.1 arm-commit collapse (2026-08-11)

## What happened

The live pair submitted 04:33 UTC 2026-08-11 split hard by ~15h in:
`v23_route` (55423272) developed to **2208** while `v23.1_bandit`
(55423269) stalled at **1056** — predecessor bandit v22.2 rated 2776.

Mined all 45 of the bandit's ladder games (replays via
`kaggleusercontent.com/episodes/<id>.json`; the CLI replay endpoint was
429-limited): 26W-19L, **no timeouts, no errors** — every game
DONE/DONE. Loss-bank median **$52k** against a win-bank median of $87k
and the route's loss-bank median of $98k: in its losses the bandit's own
economy halved.

## Root cause

**14 of the 19 losses carry the identical signature**: at t≈217 — one
turn after `_COMMIT_MIN_STEP = 216` — the agent commits the `raj`
counter-arm and leaves the crowned base route. The arm is a **market-only
override** (`v22_agent.market_overrides`) grafted onto the base farm
plan, and the two are incoherent: the scheduled `BUY_LAND` at t=240 is
dropped, off-plan animal/seed buys appear, plantings outrun watering
(weed `DIG` repairs from t≈497), final banks land at $17–66k.

Two compounding failures:

1. **The identifier over-matches class 1.** 14 distinct mainstream
   opponents were "decisively" (p ≥ 0.85, streak 2) classified into the
   class that maps to `raj`. Class 1 is a giant mainstream cluster, not
   a fingerprint of one program.
2. **The arm path went live with zero field evidence.**
   `models/v22/identifier/gates.json` said `n_commits: 0 … need >=25`.
   v22.2 — the 2776 bandit — shipped `_CLASS2ARM = {}` and never
   committed an arm. v23 flipped it to `{'1': 'raj'}` untested. This is
   the 7th confirmation of the documented arm lesson
   (`docs/history/agent-v19_1-v22_1-v23.md`): unvalidated schedule-retiming arms
   do not generalize.

## Reproduction (local, exact)

Tape built from the ladder loss 91888503 (Jay Gautam), 3 seeds x 2 seats:

| agent | vs tape_91888503_s1 | mean bank |
|---|---|---|
| v23.1_bandit (broken) | **0%**, −38,363 | $43,952 |
| v23.2_bandit (fixed) | **100%**, +55,982 | $112,117 |

## The fix — v23.2_bandit (notebook v4, sha f2a90c23ef9e3b14)

Rebuilt with `python -m kaggriculture.agentbuild.v22_agent --base 91464294_s1 --arms --out
agents/v23.2_bandit.py`. Byte-identical to v23.1 except `_ARMS = {}` and
`_CLASS2ARM = {}` (docstring aside): identifier, relay v2, and the
M2-RL/M3-RL in-game relay are all retained — exactly the v22.2 arm
configuration on the v23.1 feature set.

Validation: gate passed (self-play DONE, banks 66,811/66,811, latency
mean 0.69 ms / worst 89.26 ms, 52,663 bytes); v22_2_bandit 100%/+331;
v23_route 67%/+151 (mirror near-tie, expected); tape_90036815_s1
100%/+20,578; the reproduction table above.

## Prevention

* **ARM GUARD in `src/kaggriculture/agentbuild/v22_agent.py`** (the only place `_CLASS2ARM` is
  emitted, so it covers `refresh_cycle`'s daily builds too): a non-empty
  class→arm mapping is forced to `{}` unless `gates.json` records
  **≥ 25 judged real-ladder commits** — the threshold the gate estimator
  was already designed around. The guard prints what it blocked.
* The commit audit (`src/kaggriculture/measure/commit_audit.py`) accumulates judged commits
  from mined games; arms can earn their way back in with evidence.
* **Sell-only overrides** (`v22_agent.market_overrides`, guarded by
  `tests/test_arm_overrides.py`): even when evidence eventually unlocks
  arms, an override may only retime SELL orders. Structural orders
  (BUY_LAND, HIRE, BUY_ANIMAL, BUY_SEED, BUY_PRODUCT) always come from
  the base plan, and under the 10-order cap base structure wins. The
  t=240 BUY_LAND drop that halved v23.1's banks is now unreachable —
  verified against the actual raj arm: 155 override turns, 0 structural
  losses.
* Standing rule reaffirmed: any new *behavior* (not tuning) must be run
  against the fixed held-out panel with the behavior **actually
  firing** before it ships. v23's certification measured the relay paths
  but never a single arm-committed game.
