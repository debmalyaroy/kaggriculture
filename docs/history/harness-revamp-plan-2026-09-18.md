# Rust test-harness revamp plan (2026-09-18)

Task: revamp the Rust evaluation harness, informed by
`research/Rust Test Harness Architecture for Kaggriculture Simulation.md`.
Grounded in a full read of the current `rustengine/` + `src/kaggriculture/engine/`.

## TL;DR — the plan in one paragraph

Our harness is **not** short on speed or structure: the Rust engine is bit-exact
vs official (per-step full-state digest, `test_rust_engine.py`), `kagg serve` is
**12× faster** than official (0.93 s vs 11.08 s/game), and seat-swapping +
mirror-dedupe + held-out splits are already implemented in every orchestrator.
It has exactly **one** structural defect: the **reactive-agent observation
contract** on `kagg serve` is a hand-rolled, human-mirrored *subset* of the
official observation, delivered with the **wrong agent call signature**. That —
not performance — is why serve mis-ranks reactive agents (v58 18-14 serve / 0-8
official) while staying bit-exact for tape agents. **The research doc optimizes
throughput (PyO3 zero-copy, GIL strategies, batched inference); our bottleneck is
fidelity.** So: **fix the observation contract first (days, high value), adopt
the doc's PyO3 embedding second (optional, post-deadline throughput upgrade).**

## Why the divergence exists (grounded)

Tape-replay agents ignore `obs` → bit-exact. Reactive agents read `obs` and
adapt → they drift on any field that isn't byte-identical to the official
per-seat observation. There are **two independent, unshared obs code paths**
(they can and do drift):
- **Serve reactive path** (what reactive agents see): `service.rs::json_state`
  (225-244) → `json_farm`/`json_tile`/`json_private`/`json_omap` — a **hand-rolled
  JSON emitter**, then re-sliced in Python by `serve_match.py::obs_for` (60-63).
- **Searcher path**: `obsstate.rs::from_obs` — different direction, deliberately
  leaves opponent private empty (a belief model), not used by reactive serve.

Official per-seat obs is built in `vendor/.../kaggriculture.py::_initialize`
(244-276) + interpreter tail (948-956).

### The authoritative observation contract (from the competition README/AGENTS.md)
The competition docs (now local at `.local/scratch/compdata/{README,AGENTS}.md`)
define the contract precisely, and it **corrects an earlier assumption**: the
agent is **single-arg `agent(obs)`** (both worked examples use `def agent(obs):`),
with **`step` carried INSIDE obs** ("supplied by the framework"). There is no
documented second `configuration` argument. Exact obs schema to match byte-for-byte:
```
{ player:int, step:int, day:int, hour:int,
  farms:[farm,farm],                       # both farms public: money, tiles[y][x],
                                            #   farmer[x,y], hands[[x,y]], unlocked_quadrants, hires_today
  market:{ inventory:{PROD:int}, prices:{PROD:int} },   # BOTH sub-dicts, per-product
  town:{ unlocked_shops:[str,...] },        # may repeat
  private:{ shed:{item:int}, seeds:{crop:int}, inventories:[farmer_inv, hand_inv,...] } }
```

### Concrete divergence sources — CORRECTED by the 2026-09-18 bug audit
**The bug audit verified the obs CONTENT is FAITHFUL** — schema, values, numeric types,
and **dict ORDER (PRODUCTS order, correct)** all match vendor ground truth field-by-field
(`docs/history/bug-findings-2026-09-18.md`). So the reactive mis-rank is **NOT** an obs-content /
ordering problem (my earlier ranking was wrong). It is the **invocation/harness layer**:
1. **[MOST LIKELY] Extra final turn (S5).** Serve invokes steps 0..719 (**720 calls**);
   official stops at 0..718 (**719 calls**). The step-719 action is applied in serve but
   NOT official → **end-game liquidators bank differently** → `--compare-official`
   mismatch. Hits reactive AND tape agents on the last turn. FIX: stop serve at step 719.
2. **[MED] Config-less invocation (S3).** `serve_match.py:95` calls `agent(obs)`; official
   `env.run` calls `agent(obs, configuration)`. Agents that read `config` diverge. FIX:
   pass a configuration mapping with arity handling (single-arg stays default, 2-arg supported).
3. **[MED] Shared mutable obs (S4).** `obs_for` aliases `farms`/`market`/`town` into both
   seats without copying; an agent mutating `obs["market"]["prices"]` in place corrupts the
   other seat AND the shared `js` used to read final banks. FIX: deep-copy per seat.
4. **[KNOWN, separate] Python-harness sell divergence.** `build_bandit_harness` sells ~1434
   WOOL/game vs the Rust `kagg bandit`'s ~7198 on the same worlds
   (`.local/memory/faithful-testbed-and-offline-gap-2026-09-16.md`) — this is the
   **bandit fallback ≠ Rust** bug (C1/H4/H5), NOT a serve-obs issue. A reactive
   agent SELLING LESS ⇒ it's reading a different market obs. Exactly the fidelity
   failure Phase 0 must pin to a field.

### The gate blind spot
`serve_equiv.json` (12/12, **dated 2026-09-03**) only certified **tape-shell**
agents (v43.0_bandit, v42.1_trackp, raw_105017780). Tape shells are exactly the
agents that CAN'T expose the obs bug. So `serve_allowed()` reads green while
reactive ranking is broken — the certification never tested a truly reactive
agent. **This is the single most important thing to fix.**

## Plan

### Phase 0 — reproduce & pinpoint (now, hours)
Build a per-turn diff harness: take 2-3 genuinely reactive agents (a public
reactive like flexon, one of ours), run the **same (seed, seat)** on serve and on
the official vendored engine, and dump BOTH the per-turn observation the agent
receives AND the action it returns. Diff field-by-field / action-by-action; find
the **first turn** they differ. Output = the exact divergent field(s), turning
"serve mis-ranks" into a named bug list. (Reuses `serve_match.py --compare-official`
plumbing + `evaluate.py` official path.)

### Phase 1 — fix the serve INVOCATION layer (days, HIGH value; the audit-pinpointed fixes)
> ✅ **IMPLEMENTED + VERIFIED 2026-09-18** (S5/S3/S4 in `serve_match.py`; reactive `--compare-official`
> 3/3 exact, was diverging). Controlled A/B proved S5 load-bearing (banks off ~440/~1020 on 2/3
> seeds pre-fix). Reactive parity gate `reactive_parity.py` + `serve_allowed()` reactive-coverage
> requirement added. See tasklist STATUS LOG + `.local/memory/serve-fidelity-fix-2026-09-18.md`.
The obs content is already faithful; fix the three invocation-layer bugs the audit found:
1. **S5 — stop serve at step 719 (last-turn parity).** Serve currently invokes 720 calls
   (0..719) vs official's 719 (0..718); the extra step-719 action changes end-game
   liquidator banks and breaks `--compare-official`. Gate the final action / set `done`
   one step earlier to match official exactly.
2. **S3 — pass `configuration` with arity handling.** Default single-arg `agent(obs)`
   (documented), but inspect arity and pass a `configuration` mapping (turnsPerDay,
   startingMoney, episodeSteps, actTimeout, marketParams) to 2-arg agents, matching
   `env.run`. So config-reading reactive agents don't diverge.
3. **S4 — deep-copy the per-seat obs.** `obs_for` must give each seat an isolated copy of
   `farms`/`market`/`town` so an agent mutating obs in place can't corrupt the other seat
   or the shared `js` used for final banks.
4. **(Kept) One canonical obs builder + reactive parity gate.** Keep a single obs
   serializer (already faithful), and **extend `serve_equiv` to ≥6 truly-reactive agents**
   (not tape shells) at 0 mismatches before `serve_allowed()` trusts reactive tournaments,
   wired into `serve_spot_check()`. Re-date it — the current 12/12 only certified tape shells.

**Outcome:** a fast (12×) harness we can *trust* to rank reactive candidates —
which is the actual lever on shipping better submissions this week.

### Phase 2 — PyO3 embedding (optional, post-deadline; the doc's vision)
Add `pyo3`, embed CPython, run agents **in-process** against the real `State`
with zero-copy observation views. This eliminates stdio serialization AND the
dual obs path (one source of truth by construction). **But** weigh honestly:
- Our agents are **heuristic, not ML** — the doc's headline wins (zero-copy tensor
  passing, batched GIL inference for NN agents) are **low value** for us.
- Serve is already 12× official; **throughput is not the bottleneck**, fidelity is.
- Cost: `pyo3` dep (currently Cargo.toml has **zero** external crates), GIL
  management, `catch_unwind` boundary rework, warm-up handling.
→ Do this **only after Phase 1 locks fidelity**, and only if throughput becomes
limiting. Keep stdio serve as the fallback. Framed as an architecture upgrade,
**not** a 1-week task.

### Phase 3 — cheap wins the doc surfaces that we lack
- **Warm-up phase**: run each agent against dummy obs before timed eval so first-map
  latency isn't skewed (doc §Function Invocation Profiling). Add to `arena.py`.
- **Replay-agent opponents**: serve already supports `OPP <seat> <tape>` scripting.
  Formalize a "replay opponent" panel from the **top-100/200 replays** now being
  downloaded (ties the download + public-agent harvest into the gate) — clean-room
  battles vs the meta from the opposing seat (doc §Replay-Driven Validation).

## Non-goals / risks
- **Don't rewrite the engine** — it's proven bit-exact; touching it risks parity.
- **Don't chase PyO3 throughput before fidelity** — that just builds a faithful-but-
  fast version of a mis-ranking harness.
- **Don't trust `serve_equiv` green** until it contains reactive agents.

## 1-week sequencing (deadline is ~2026-09-23/25)
- **Day 1:** Phase 0 diff → exact divergent fields.
- **Day 1-3:** Phase 1 obs-contract fix + reactive parity gate + re-cert serve_equiv.
- **Day 3+:** use the now-faithful harness + the new high-scoring public panel
  (Task 1) + top-100/200 replay opponents to gate submissions properly → ship.
- **Only if time remains:** Phase 2 PyO3.

The deliverable that moves the ladder this week is **Phase 1** — a serve substrate
we can trust to rank reactive agents. Everything else is secondary.
