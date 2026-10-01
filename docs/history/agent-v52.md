# v52 — the mirror-war release (2026-09-12)

> **STATUS: SHIPPED as the GUARDED winner** (reactive floor in the
> accept loop; the earlier unguarded draft was pulled — see
> issues-and-improvements 2026-09-12 addendum, agents kept as
> `v52draft_*` for the record). Guarded winner: 283 market-channel
> re-timings, STRAWBERRY channel untouched, 2 reactive vetoes logged.
>
> **SEAT DECISION (second-slot rule applied): only v52_trackp ships.**
> The bandit's route-emission cannot carry the evolved tape's timing —
> even with EVERY overlay flag off it loses the yhay81 mirror 2-14
> while the same tape in the trackp template sweeps 16-0. Fixing the
> emission = code change = forbidden by the architecture order. Active
> pair after submission: 56168005 (v51.1 trackp, climbing) + v52_trackp;
> the pinned bandit seat (56168002, 2473 equilibrium) is evicted.
>
> Guarded v52_trackp gates: mirror 1.000 holdout (+3,920); fresh
> publics **128-0** with collapse margins intact (yhay81 margin +3,314
> vs v51.1's +2,078); loss panel 0.658 vs 0.553 (p=0.125 ns, upset-risk
> 16→10... see log); oppwins-44 **0.977** (YARN 1.000, 0 upset-risk);
> ab_test vs live v51.1 **31-1 (p≈0)**; Rust equiv 15,099/0; self-play
> DONE/DONE; latency 0.05/0.27ms; tarball sha b9f7c69ce601f717.

## Why

Both v51.1-era seats stalled at 2417-2472. The 38 certified loss cells of
the live pair showed 21/38 are >80% identical-stream MIRROR games: half
the 2450-2500 band forks the same public yhay81 kernel v51 rebased on, and
those games are coin flips decided on day 29 by sell micro-timing (close
losses: +1,398 ahead at day 27, −2,305 at day 30). A shared schedule caps
at its own coin-flip equilibrium. Full diagnosis:
`docs/history/issues-and-improvements.md` 2026-09-12 entry.

## What changed (architecture FROZEN — tape content + existing flags only)

- **Both seats' base tape** = the mirror-war evolution winner
  (`.local/livepool/mirror_egg.json`): (1+λ) search over the v51 base
  tape's MARKET channel, fitness = beat the unmodified yhay81 schedule
  head-to-head (both seats × 12 seeds), hard floor = no regression on 8
  certified live-loss cells. 259 micro re-timings (±1-2 unit sell
  nudges that win the engine's per-unit lockstep races).
  Holdout: mirror **1.000**, +$3,596/game vs the schedule the fork band
  plays. Field-ops and econ variants measured WORSE in-agent (econ's
  +27k own-bank exploited the mirror partner; refuted).
- **trackp (v52_trackp.py)**: v51.1 code byte-identical, EGG slot = the
  evolved tape, YARN slot = Kaggledew (unchanged).
- **bandit (v52_bandit.py)**: v51 build (overlays kept, tape-swap layers
  off) with _ROUTE = the evolved tape. A YARN d6-fork variant
  (v52y_bandit: existing d6_fork layer re-enabled with _DAY6 = 17 YARN
  worlds → Kaggledew continuation) was gated head-to-head and REFUTED:
  0.250 vs 0.553, p=0.011 — re-enabling d6_fork degrades even no-YARN
  play (0.554→0.196). Deleted; the bandit's YARN compensation stays its
  overlays + the new tape (YARN 0.550 on the loss panel, up from 0.150).

## Measurements (UNGUARDED DRAFT — kept for the record; pulled, see banner)

| gate | v52draft | live baseline | verdict |
|---|---|---|---|
| 38-cell live-loss panel (in-agent) trackp | 0.816 (YARN 0.800) | 0.553 | +0.263 p=0.002 |
| 38-cell live-loss panel (in-agent) bandit | 0.553 | 0.079 | +0.474 p<0.0001 |
| ab_test vs live trackp (full fidelity) | 25-5-2 (0.813) | — | p=0.0003 |
| ab_test vs live bandit | 24-6-2 (0.781) | — | p=0.0014 |
| 58-cell combined panel, trackp | 0.828 (upset-risk 6/36) | 0.741 (8/36) | +0.086 p=0.125 ns, no regression |
| 44-cell adaptive-oppwins panel, trackp | 0.909 (YARN 1.000, upset-risk 0) | — | |
| upset-risk cells lost, trackp | 4/32 | 16/32 | |
| Rust binary equivalence | 15,099 steps 0 mismatch | | |
| latency (Python fallback) | trackp 0.03/0.09ms, bandit 1.79/5.62ms | | limits 20/100ms |
| official-engine self-play | DONE/DONE both artifacts | | |
| ladder parity audit | 6/6 exact (2026-09-12) | | |

Elite recorded streams (39 games, both players 3000+, all certified
bank-exact) were REFUTED as base tapes: 5W-73L vs our base even at home
seeds — adaptive play desyncs when replayed. Our agents sweep the 78-cell
elite recorded panel 0.942 — the elite edge is pure in-game reactivity.

## Artifacts (GUARDED, shipping)

- `agents/v52_trackp.py` + `agents/v52_trackp.html` (guarded tape;
  `agents/v52draft_*` = the pulled draft, never shipped)
- `agents/v52_bandit.py` — built but NOT shipping (seat decision above)
- `.local/candidates/v52_trackp_compiled/submission.tar.gz` (328,162 B,
  sha256 b9f7c69ce601f717…; musl static-pie agent + Python fallback)
- `rustengine/agentdata/trackp_egg.json` = guarded tape (v51 backup kept)
- search: `.local/livepool/mirror_egg_r.json`, log `mw_reactive.log`
  (275 accepts, 2 reactive vetoes); floor probe `reactive_probe.py`

## The treadmill (standing posture)

Our episodes are public: forks will re-mine this tape with days of lag.
Daily loop: refresh the loss panel (`build_panel.py`, LP_SUBS env),
re-run `mirror_search.py` against OUR OWN latest tape as the mirror
opponent, re-gate, ship on a sign-tested PASS. The search machinery, not
any single tape, is the moat.
