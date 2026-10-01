# Bug findings — codebase correctness audit (2026-09-18)

Two bug hunts on the shipping-critical surface. **Headline: the shipped bandit "Rust
path" and its "action-faithful" Python fallback are DIFFERENT AGENTS, and the packaging
+ watchdog make it likely the weaker fallback is what actually plays on Kaggle.** This
plausibly explains v58's 463 (it may never have run its Rust economy) and why local vs
ladder numbers keep disagreeing.

Severity: CRITICAL (ship-breaking) → HIGH → MED → LOW. Each has file:line + repro in the
audit; condensed here. Fix tasks tracked in `implementation-tasklist-2026-09-18.md` §G7.

## BANDIT ship path

### CRITICAL
- **C1 — Rust and Python fallback are different agents.** `mbandit.rs::sells` generates a
  premium SELL overlay from LIVE shed inventory (`SELL_PRIORITY` sweep). The Python
  fallback (`build_bandit_harness.py::_run` / `_RAILS`) has **no such rail** — it only
  emits the tape row's own market orders. Economies diverge by tens of thousands. The
  DRIVER calls the fallback "action-faithful"; it is NOT.
- **C2 — Packaging ships the Windows `kagg.exe`, not a musl ELF.** `build_rust_bandit.py`
  (~281-299) runs `cargo build --release` with **no `--target x86_64-unknown-linux-musl`**
  and the copy loop prefers `kagg.exe`. On Kaggle Linux that binary can't exec →
  `_BIN["dead"]=True` → the C1 fallback plays **every turn**. **The "Rust shipping path"
  may never run its Rust on Kaggle.** → very likely a big part of v58's 463.

### HIGH (all Rust↔fallback divergences → the fallback plays a different, weaker agent)
- **H3 — `--branches` override dropped for the fallback:** Rust gets the override,
  the embedded fallback bakes in `cfg["branches"]` → different continuations spliced.
- **H4 — `weed_repair` weaker in fallback:** farmer-only, no hands, no reinstate ledger
  (Rust does farmer+hands + DIG→reinstate). Displaced plant lost forever → tape desync.
- **H5 — `scarcity_sell` is a different algorithm each side, and default `hold_frac`
  differs (Rust 1.0 vs Python 0.0). ACTIVE in v58.** Rust caps sweep qty; Python drops+
  reorders tape SELLs. Python ignores `press`; Rust disables under pressure.
- **H6 — `hand_align` enabled in v58 but absent from fallback `_RAILS`** (no-op), and the
  fallback output bypasses the DRIVER's `_validate` → misaligned/oversized hand lists.
- **H7 — `front_run` in fallback:** no `fr_from` gate (front-runs from step 0 vs Rust
  145), stores product not qty, not shed-clamped, and never reduces the future tape SELL
  → **double-sells** (front-run now + full amount later).
- **H8 — Watchdog terminal on first miss (0.40/0.60s).** One slow turn / cold start /
  large `branches.json` load on step 0 → `_kill` → divergent fallback for the WHOLE game.
  Makes the fallback the common case, not the rare one.

### MEDIUM
- **M9 — Dead config CONFIRMED + extends:** `load_config` keeps only scalar numbers
  (mbandit.rs:198-204); `anti_dump.premium` array AND `scarcity_i0` are dropped; Rust uses
  a hardcoded premium set + fixed I0=10000. `anti_dump` algorithm also differs from the
  fallback. Config silently lies on the Rust path.
- **M10 — Endgame differs:** Rust `SELL p 1000` in ALLP order vs fallback exact-qty,
  quantity-sorted. Different realized liquidation if the engine rejects over-qty.
- **M11 — `_r37_price` rounding:** Rust `f64::round` (half-away) vs Python `int(round())`
  (half-to-even) → boundary-inventory decisions differ.

### LOW / verified-clean
- L12 `base_tape[:719]` truncation is deliberate & consistent (final step plays PASS+overlay).
- L13 `.take(8)` shopkey — harmless (only 8 shop types). **No bug.**
- L14 **`__file__` guard is CORRECT** — v58 import crash is genuinely fixed. **No bug.**
- Verified clean: 10-order cap (≤10, sells before buys), splice/dispatch mechanics match,
  `step%24`/escalation logic matches bandit.rs, DRIVER reader/queue concurrency safe.

**Bandit bottom line:** fix **C2 (musl ELF + hard-fail if not ELF)** and **H8 (don't retire
the bridge on one miss)** FIRST so the Rust binary actually runs; THEN either genuinely
unify the fallback with `mbandit.rs` (port the shed sweep, weed_repair, scarcity_sell,
hand_align, front_run, endgame, rounding) or stop advertising equivalence and make the
fallback a loud legal-PASS safety net only. This is also why the refactor (§G) must make
one config-driven path with a single source of truth.

## TRACKP ship path

### HIGH
- **T1 — Rust and Python use DIFFERENT labour-assignment algorithms** (same class as
  bandit C1: the shipped binary ≠ the reference fallback). Rust (`policy.rs:840-856`,
  `take()` :1067-1136) = one LINEAR global pass, no quadrant awareness. Python
  (`build_econ_agent.py:790-864`) = QUADRANT-MATCHED per-unit assignment + nearest-first
  sweep. Unconditional (not genome-gated) → fires every day with tasks → different
  farmer/hand op streams → different banks. **The shipped binary plays the WEAKER older
  linear assignment.** Undetected because `tests/test_compiled_agent.py` **SKIPs (returns
  0) when no `kagg` binary is staged** (:122-127) — the identity gate silently passes.
  Fix: port quadrant-matched `_take`/`uq` into `policy.rs` (or revert Python to linear),
  and run the identity gate with a staged binary in CI (no silent skip).
- **T2 — [latent] Rust binary ignores the genome entirely.** `policy.rs` hardcodes
  `const SKELETON` and never implements `hire_slack`/`adaptive_herd`/`herd_gate`/
  `demand_crops`/`drop_daily`/`hour0_sells`/`sell_order`/`fert_buy`/`hold_price`/
  `sell_floor`. Skeleton matches today, but if a TUNED genome is ever shipped via the
  fallback, the binary silently plays the skeleton. Fix: feed the genome to the binary
  (ties to G2 genome externalization) or assert build_main ships only skeleton.

## SERVE ↔ OFFICIAL fidelity — the reactive mis-rank root cause (CORRECTS Item A)

**Obs content is FAITHFUL** — verified field-by-field vs vendor ground truth: schema,
values, **dict ORDER (PRODUCTS order, correct)**, int prices (refreshed after each order
index + town-consume), shed/seeds order all match. **So the reactive mis-rank is NOT a
missing/mistyped/mis-ordered obs field** (my earlier §A ranking was wrong). It is the
**invocation/harness layer**, ranked by likely impact:
- **S5 [most likely] — serve gives one EXTRA final turn.** `run_match` loops while not
  done; `json_state` sets done only at `st.step >= 720` → agent invoked on steps 0..719
  (**720 calls**). Official stops at `episodeSteps-1` → 719 calls (0..718). The step-719
  action IS applied in serve but NOT official → **end-game liquidators bank differently**
  → `--compare-official` MISMATCH. Hits reactive AND tape agents on the last turn. Fix:
  stop serve at step 719 (or gate the last action).
- **S3 — single-arg agent call drops `configuration`.** `serve_match.py:95` calls
  `agent(obs)`; official `env.run` calls `agent(obs, configuration)`. Agents reading
  `config` (turnsPerDay, startingMoney, marketParams…) diverge under serve. Fix: pass a
  configuration mapping with arity handling (matches the documented single-arg default
  but supports 2-arg agents).
- **S4 — both seats share the SAME obs sub-objects** (`obs_for` aliases `js["farms"]`/
  `["market"]`/`["town"]` into both seats, no copy). An agent that mutates
  `obs["market"]["prices"]` in place corrupts the other seat's view AND the shared `js`
  used to read final banks. Fix: deep-copy per seat.

**Item A correction:** the faithful-obs work (byte-identical schema/ordering) is largely
ALREADY correct in the Rust emitter; the real fixes are **S5 (last-turn parity), S3
(config arg), S4 (per-seat obs copy)** in the serve harness, not the obs schema.

### CLEAN (verified)
- `from_obs` empty-opponent-belief — only reachable via the searcher (`--budget-ms>0`);
  shipped budget is 0 (skeleton reads raw obs) → cannot drive a wrong shipped decision.
- **`bridge.py` (trackp) watchdog is CLEAN** — sized safely under actTimeout, degrades to
  the validated fallback, `agent()` never raises. (Contrast: the BANDIT DRIVER's watchdog
  is the buggy H8 — two different watchdog implementations, only the bandit one is broken.)
- policy.rs vs build_econ_agent skeleton match everywhere EXCEPT T1 (assignment);
  `PLANT_ORDER` differs but TOMATO target=0 so no effect. No illegal-action bug beyond T1.

## SYSTEMIC finding (both hunts)
**In BOTH tracks the shipped Rust binary and the Python fallback are DIFFERENT agents**
(bandit C1, trackp T1), and the parity/identity tests DON'T catch it (bandit: fallback
never truly tested for equivalence; trackp: identity gate SKIPs without a staged binary).
Combined with the bandit C2 (Windows-binary → fallback runs on Kaggle), **what we ship and
what we measure locally can be two different agents.** This is the deepest reason local
numbers disagree with the ladder — and why the refactor (§G) must make ONE source of truth
per track with a parity test that cannot silently skip.
