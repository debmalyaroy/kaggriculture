# Track P — the COMPILED agent, Phase A (transport spike), 2026-09-03

> **SUPERSEDED AS A STRENGTH CLAIM (2026-09-03).** Phase A and Phase B were
> joined and measured against real opponents on the official interpreter:
> **0 wins in 40 cells**, and significantly worse than the seat it would
> replace. Read `docs/history/trackp-phase-ab-integration-2026-09-03.md` before
> believing any number in this document as evidence of ladder strength.
> The transport and engineering claims here still hold.
>
> **§0 BELOW IS THE SHIP BUILD (2026-09-03 23:0x) AND IT OVERRIDES THE REST OF
> THIS DOCUMENT** on three specifics: the shipping search budget is **0, not
> 150**; `FIRST_BUDGET` is **0.50 s, not 1.50 s**; and §7's "do not submit this
> artefact" is superseded by an explicit operator order. Every other number
> below was taken on the pre-21:22 economy and the 17:26 tarball.

---

## 0. THE SHIP BUILD — rebuilt, re-measured, frozen

The 17:26 tarball was **stale**: the base-economy fix landed at 21:22
(`rustengine/src/policy.rs`, `src/kaggriculture/trackp/build_econ_agent.py` — land as early
as cash allows, `land_reserve_lead 0`, the mined top-30 lean hire ramp) and the
artefact did not carry it. Rebuilt, re-measured end to end, and frozen.

**Full machine-readable record: `models/release_2026-09-03_trackp.json`.**

```
.local/candidates/trackp_compiled/submission.tar.gz
  sha256 21a6a02469202041d01ad6c30a9f13087fe02793d616fb02651d9bbdf4c1fab9
  502,912 bytes  (0.48 MiB, limit 100 MiB)
  -rw-r--r--  main.py    43,795   (mode 644, archive ROOT)
  -rwxr-xr-x  kagg      987,872   (mode 755, ELF x86-64 static-pie, stripped,
                                   BuildID 7d5acf21e7a2385db1344870b3e16e41b3b420a0)
```

### 0.1 The shipping budget is 0, and that is a measurement

`SEARCH_BUDGET_MS` was 150. It is now **0**. Same rebuilt binary, the same 40
official-engine cells (5 gauntlet opponents × seeds 3,4,5,6 × both seats), the
real tarball untarred inside Linux:

| budget | W-D-L | own median | own mean | opp median | share of all money | worst turn (6 workers) |
|---|---|---|---|---|---|---|
| **0 (SHIPPED)** | 0-0-40 | **63,857** | **64,791** | 135,090 | **33.0%** | **110.2 ms** |
| 150 | 0-0-40 | 48,543 | 50,036 | 115,840 | 29.7% | 266.7 ms |

Paired cell for cell, the searcher is worth **−14,755 of own bank on the mean
and is richer on only 11 of the 40 cells (two-sided sign test p = 0.0064)**,
for exactly the same zero wins. It also **breaks the latency bar** — 266.7 ms
against the 250 ms line that keeps 4× margin on `actTimeout`.

The mechanism is not mysterious: the searcher's value function and knob space
were fitted around the OLD, weak skeleton, so after the economy fix it
hill-climbs *away* from a better economy. Raising the budget cannot rescue it
on Kaggle either — `budget_ms` is wall clock, so 1.6 vCPU buys **fewer
rollouts**, i.e. an even weaker search than the row that already measures
negative here.

Three things budget 0 buys that matter more than a negative search:

1. **A blocked `fork`/`exec` on Kaggle costs NOTHING.** At budget 0 the
   compiled path is byte-identical to the inlined Python fallback
   (`tests/test_compiled_agent.py`, the identity gate). Nobody has proven the
   agent sandbox permits spawning a subprocess; this makes the question free to
   ask.
2. **It is reproducible.** A wall-clock budget is not, and `src/kaggriculture/measure/determinism.py`
   is explicit that a paired A/B over a non-reproducible agent is *invalid*,
   not merely noisy.
3. **9× worst-turn margin under load**, and that worst turn is a single process
   spawn — p95 is 3.3 ms.

The searcher is not deleted. `--budget-ms N` / `TRACKP_BUDGET_MS` still turn it
on for measurement, and `search_selftest` still proves a zero-budget searcher
is action-for-action the skeleton. **Do not raise the constant without a fresh
paired sign test against budget 0 on the gauntlet.**

### 0.2 Latency, on the shipped configuration

The real `submission.tar.gz` untarred inside `python:3.11-slim`, played on the
**vendored official interpreter**, 719 agent turns per cell.

| | mean | p95 | worst |
|---|---|---|---|
| **shipped (budget 0), 40 cells / 28,760 turns, 6 workers** | **2.09** | **3.34** | **110.15** |
| shipped (budget 0), quiet box | 1.06 | 1.3 | **9.14** |

The 110 ms worst is the **first turn of one cell** — process spawn under
6-way contention. Every non-first turn is ~2 ms. Against a 1,000 ms
`actTimeout` that is **9.1× margin loaded, 109× quiet**; doubling it for
Kaggle's 1.6 vCPU still leaves >4×.

Budget sweep on the rebuilt binary, quiet box, for the record:

| budget | mean | p95 | worst |
|---|---|---|---|
| 0 | 1.06 | 1.3 | 9.14 |
| 100 | 99.22 | 103.9 | 105.15 |
| 150 | 148.24 | 155.6 | 159.78 |

**The honest caveat, restated because it is the thing people get wrong:** the
overhead above a non-zero budget is a flat 7–10 ms, not a multiple of it. The
budget is *wall clock*, checked after every rollout, so a slower core buys
fewer rollouts rather than more milliseconds — **Kaggle runs a weaker search
than any dev row shows.** At budget 0 the caveat is void, because there are no
rollouts to lose.

### 0.3 Safety — re-verified, and one real bug found

Everything from §3 re-checked on the rebuilt artefact, plus three breakage
cases that had never been exercised:

| deliberate breakage | fallback | status | bank | note |
|---|---|---|---|---|
| binary removed | 719/719 | DONE | 63,943 | `binary not found` |
| `chmod 000` | 0 | DONE | 63,943 | the bridge `chmod 755`'d it back |
| replaced with garbage | 719/719 | DONE | 63,943 | `Exec format error` |
| **binary HANGS (new)** | 719/719 | DONE | 63,943 | watchdog fired, **1 timeout**, bridge retired permanently, worst turn **503 ms** |
| **answers non-JSON (new)** | 719/719 | DONE | 63,943 | `malformed reply` |
| **answers ILLEGAL actions (new)** | **0** | DONE | 0 | **2,077 validator repairs**, bridge stayed ALIVE, and the engine saw **0** misaligned hands, **0** over-10 order lists and **0** unknown ops |
| normal | 0 | DONE | 63,943 | |

Five of the six bank **exactly** what the healthy artefact banks. That is the
payoff of shipping at budget 0: the two implementations are one policy, so a
transport failure costs speed and nothing else.

**`FIRST_BUDGET = 1.50 s` was a bug, and firing the timeout path is what found
it.** `actTimeout` is 1.0 s. A watchdog longer than the engine's own limit
cannot protect anything: in the single situation it exists for — the binary
spawns and never answers — it sat on turn 1 for a measured **1,503 ms** and
would have handed the engine a turn the engine had already timed out, turning a
harmless fallback into a possible forfeit. The old value rested on an
*assumption* written into §3 ("the engine has not started the clock on turn 1
in the same way") that was never measured.

It is now **0.50 s**, sized from measurement instead: spawn + the first day plan
is ≤110 ms with 6 cells in flight, so 0.50 s is ~4.5× the observed cost and
leaves half of `actTimeout` in hand. Re-measured, case D's worst turn is
**503 ms** and the episode still completes DONE with the identical bank. If it
ever does fire, the cost is zero (same policy) and the beacon's `reason` field
separates `turn exceeded 0.50s` from `spawn: …` and `binary not found`, so a
slow spawn can never be misread as a blocked `fork`/`exec`.

`F` is the first time the validator has ever had to do anything: a stand-in
binary emitting an unknown farmer op, 40 hand ops regardless of the real crew,
and 24 market orders including malformed ones, **every turn**, reached the
engine as a legal action 719 times out of 719.

### 0.4 The strength number, and the honest expected level

Official vendored interpreter, the shipped artefact at its default budget,
5 real opponents × 4 seeds × both seats.

| opponent | cells | W-D-L | our median | their median |
|---|---|---|---|---|
| `data/gauntlet/pub_v16rc5.py` | 8 | 0-0-8 | 79,160 | 142,684 |
| `data/gauntlet/kaito_v48.py` | 8 | 0-0-8 | 69,554 | 141,046 |
| `data/gauntlet/pub_rayk_c94.py` | 8 | 0-0-8 | 55,040 | 105,473 |
| `agents/v42.1_trackp.py` | 8 | 0-0-8 | 64,060 | 139,156 |
| `agents/v43.0_bandit.py` | 8 | 0-0-8 | 55,700 | 124,800 |
| **ALL** | **40** | **0-0-40** | **63,857** | **135,090** |

Fallback fires **0**, validator repairs **0**, engine-visible illegal ops
**0**, misaligned hands **0**, over-10 order lists **0**, 40/40 DONE.

Against the pre-rebuild artefact, paired over the same cells:

| | own median | own mean | share | W-D-L |
|---|---|---|---|---|
| 17:26 tarball, search@150 | 59,264 | 57,946 | 31.9% | 0-0-40 |
| 17:26 tarball, budget 0 (old economy) | 44,205 | 50,733 | 30.0% | 0-0-40 |
| **rebuilt, budget 0 (SHIPPED)** | **63,857** | **64,791** | **33.0%** | **0-0-40** |

**+14,058 of own bank on the mean over the old economy (richer on 30 of 40),
+6,845 over the stale shipping artefact — and 0 discordant pairs in every
comparison, p = 1.0000. The rebuild is worth real dollars and exactly zero
wins.** That is now the fourth time this lane has produced a large, real,
reproducible economic gain that converts to nothing.

**Against the seat it would replace, on the 32 shared cells:**

| | score | W-D-L | own median | opp median | share |
|---|---|---|---|---|---|
| SHIPPED | 0.000 | 0-0-32 | 63,857 | 135,090 | 33.0% |
| `agents/v42.1_trackp.py` | **0.688** | 22-0-10 | 68,526 | **79,086** | **43.0%** |

22 discordant pairs, every one against us, exact **p < 0.0001 — significantly
worse**.

**Read the shape, because it changed.** Our own bank has essentially *caught
up* with the live seat: the paired per-cell own-bank delta is −3,218 on the
mean, **+2,340 on the median, and we are richer on 18 of 32 cells**. We still
lose 32 of 32. The reason is entirely the other column: the opponent banks
**135,090 against us and 79,086 against the live seat in the same world** —
about 56,000 more. Growing this economy grows the pie and hands the opponent
most of the increment. **The remaining gap is market share, not farm output,
and the base-economy work did not touch it.**

#### Expected ladder level — blunt

**This is weaker than both live seats and it should be expected to rate below
them.** `v42.1_trackp` currently rates ~1933 against a median-1771 field and
`v42.0_bandit` ~1919; a seat that loses *every* paired world to `v42.1` should
settle materially below 1900. There is no offline evidence supporting a number
tighter than "below the live pair". Remember also that a fresh submission's
`publicScore` starts provisional near ~600 and takes ~24 h of games to develop
— the first-day number is not the level.

**And the caveat cuts both ways.** This expectation comes from exactly the
offline instrument the operator has correctly judged anti-predictive: a
candidate won its selection panel 77-11 today and then lost 84 of 86 held-out
cells, and this lane's own base-economy work watched a +14,058 gain on the
report seeds shrink to **−434** on a reserved seed set. Treat "below 1900" as a
*direction*, not a forecast.

### 0.5 Why it ships anyway

The operator's call, and it is about **information**, not strength.

* **Nobody has proven Kaggle's agent sandbox permits `fork`/`exec`.** The
  `TRACKP start` / `TRACKP end` stderr beacons land in the replay's per-agent
  `logs` and answer that from ONE replay: `"fallback": 0` in the `end` beacon
  means the compiled path ran on the ladder; 719 with `reason` `spawn: …` or
  `binary not found` means it did not. Shipping at budget 0 makes the question
  **free** — the fallback is the same policy, so the answer costs nothing in
  strategy either way.
* **A live per-band, per-regime readout on a genuinely closed-loop seat** is
  the second piece of information, and no local panel can produce it.

Nothing here was submitted and no kernel was pushed. Only the latest 2
submissions stay active, so shipping this evicts a live agent — the operator's
call, not this session's.

```
kaggle competitions submit kaggriculture \
  -f ".local/candidates/trackp_compiled/submission.tar.gz" \
  -m "trackp compiled, closed-loop, budget 0"
```

### 0.6 Gates re-run on the ship build

| gate | result |
|---|---|
| `cargo build --release --bin kagg` | **OK** |
| `python tests/test_rust_engine.py` | **6/6 bit-identical, 0 diverged** |
| `python tests/test_compiled_agent.py --seeds 3,4,5` (shipped default = identity gate) | **PASS** — 0 differing actions over 4,320 turns, 0 fallbacks, 0 repairs, **seed 3 bank 63,943** (the post-fix number, so the stage is not stale) |
| `python tests/test_trackp.py` | **65 passed, 0 failed** (was 56/1 — see §0.7) |
| tarball layout | `main.py` mode 644 at the archive ROOT, `kagg` mode 755, static-pie ELF, verified by untarring inside Linux |

### 0.7 The isolation regression, fixed properly

`tests/test_trackp.py::test_isolation` was failing: `elite_fit.py`,
`factory.py`, `genome_ga.py` and `shop_branches.py` imported `routes` (and
`factory.py` imported `win_metric`), which CLAUDE.md's lane rule forbids —
Track P shares zero code with the bandit/route lane. Importing `src/kaggriculture/data/routes.py`
also drags in `_vendor`, `episodes` and `opponents`, i.e. the whole scraping
stack, for what those four modules actually want: two file reads and a paired
test.

Fixed by decoupling, not by weakening the test:

* **`src/kaggriculture/trackp/routes_io.py`** (new) — a read-only reader for the route store,
  implemented against the on-disk layout rather than wrapping `routes.py`.
* **`src/kaggriculture/trackp/guard.py`** gains `paired_test` / `expected_score`, a
  deliberate re-implementation of `win_metric`'s, reusing the `_binom_tail` the
  lane already had.
* **`tests/test_trackp.py::test_lane_copies`** (new) **pins both copies against
  the originals** — identical `route_path` for adversarial ids, identical
  `INDEX`, and identical `paired_test` output on five hand-picked score
  vectors — and reports a skip rather than passing silently if the bandit lane
  is not importable. Duplication is only safe when it is pinned.

`src/kaggriculture/trackp/compiled/verdict.py` still imports `win_metric` **on purpose** and
`test_isolation` is deliberately scoped to `src/kaggriculture/trackp/*.py`: it is a *judge*,
not lane code, and CLAUDE.md's measurement discipline says verdicts are read in
`win_metric`'s currency. That reasoning is now written into the test.

---

Phase A of the revised plan of record
(`docs/history/plan-2800-2026-09-03.md`, "Sep 4-6 (b)"). The goal was **not** a strong
agent. It was to prove end-to-end that we can ship a compiled binary inside a
Kaggle submission and to measure the real per-turn budget.

**Verdict: the transport works and is safe. The policy is not.**
4 of the 5 gate conditions pass with large margins; the bank condition fails,
and it fails on the skeleton policy, which is exactly what Phase B's search is
for.

---

## 1. What was built

| file | what it is |
|---|---|
| `rustengine/src/json.rs` | dependency-free JSON reader (the crate still has zero crates.io deps) |
| `rustengine/src/policy.rs` | `kagg play` — the compiled closed-loop economy policy |
| `rustengine/src/main.rs` | `play` and `play-selftest` subcommands wired in |
| `src/kaggriculture/trackp/compiled/bridge.py` | the subprocess transport, action validator, watchdog, fallback |
| `src/kaggriculture/trackp/compiled/build_main.py` | assembles `main.py` = bridge + inlined Python fallback |
| `src/kaggriculture/trackp/compiled/main.py` | **generated** — the shipped agent file |
| `src/kaggriculture/trackp/compiled/harness.py` | plays the assembled artefact on the OFFICIAL engine, reports latency/legality/bank |
| `scripts/build_trackp_linux.ps1` | Docker cross-build (musl) + `submission.tar.gz` |
| `scripts/trackp_harness.Dockerfile` | the Linux measurement image |
| `scripts/run_trackp_linux_harness.ps1` | unpacks the tarball inside Linux and runs the harness on it |
| `tests/test_compiled_agent.py` | differential test: Rust policy vs the Python fallback, action for action |

---

## 2. Architecture

```
Kaggle unpacks submission.tar.gz into /kaggle_simulations/agent/
    main.py     pure stdlib, exports agent(obs, cfg)
    kagg        static x86_64 Linux binary (musl), 857 KB

  agent(obs) --first call--> Popen([<dir of __file__>/kagg, "play"])
             --each turn---> json.dumps(obs) + "\n"  -> stdin
             <--------------  one action JSON line   <- stdout
             validate -> return dict
             on ANY failure -> _fallback_agent(obs)  (pure Python, same policy)
```

**Subprocess over stdio, not cdylib + ctypes.** The crate is already a binary
and `service.rs` already speaks a line protocol, so this reuses tested code and
has no ABI surface to get wrong: a mistake in a ctypes signature is a segfault
that takes the whole Python process with it, whereas a mistake here is a bad
line of JSON that the validator rejects. Process start-up is paid once per
episode (measured 14.7 ms including the first plan), and the per-turn cost is
one pipe write plus one pipe read.

**The binary is found relative to `__file__`**, not at a hard-coded
`/kaggle_simulations/agent/`. That path is documented but not contractual, and
hard-coding it makes the artefact untestable anywhere else — every measurement
below was taken by unpacking the real tarball into `/tmp` and running it from
there. The documented Kaggle path is kept only as a second probe.

**musl, not glibc.** `x86_64-unknown-linux-musl` produces a fully static
`static-pie` executable, so nothing about the competition image's glibc version
can break it. This matters more than it sounds: a binary that fails to load
does not raise — the Python fallback quietly covers for it and the agent looks
merely weak.

### The policy, and why it is a line-by-line port

`rustengine/src/policy.rs` is a faithful port of the planner
`src/kaggriculture/trackp/build_econ_agent.py --genome skeleton` emits, and that same Python
planner is inlined into `main.py` as `_fallback_agent`. So the compiled path
and the fallback path are **two implementations of one policy**.

`tests/test_compiled_agent.py` asserts they emit byte-identical actions over
full episodes. That is the correctness proof for the transport: if the streams
are identical, any bank difference between the artefact and the reference
planner is a transport bug, not a policy difference — and a fallback costs
speed and nothing else.

```
$ python tests/test_compiled_agent.py --seeds 3,4,5
seed 3 seat 0  turns 720  bank 55008  bridge 720 fallback 0 repairs 0  diffs 0  OK
seed 3 seat 1  turns 720  bank 55008  bridge 720 fallback 0 repairs 0  diffs 0  OK
seed 4 seat 0  turns 720  bank 43474  bridge 720 fallback 0 repairs 0  diffs 0  OK
seed 4 seat 1  turns 720  bank 43474  bridge 720 fallback 0 repairs 0  diffs 0  OK
seed 5 seat 0  turns 720  bank 59007  bridge 720 fallback 0 repairs 0  diffs 0  OK
seed 5 seat 1  turns 720  bank 59007  bridge 720 fallback 0 repairs 0  diffs 0  OK
PASS
```

The policy executes the measured elite skeleton: 2 land buys, cows 1→9 and
sheep 1→5 cash-gated, ~17 pastures, hires to 10/day in the hour-0 market batch,
melon 12 tiles / strawberry 38 (fertilised) / wheat elsewhere, feed+care daily,
sell milk/wool/fertiliser on production and wheat continuously, liquidate from
day 28.

---

## 3. Fallback semantics

`agent()` **never raises**. A raise costs the whole submission — the upload
Validation Episode marks it Error and the slot is spent.

| failure | what happens |
|---|---|
| binary missing | `spawn_fail`, bridge marked dead, Python planner for the whole episode |
| binary not executable | `chmod 755` is attempted first; if that fails, dead |
| binary is not an ELF / crashes on exec | `spawn` raises, caught, dead |
| process dies mid-episode | `poll()` is not None → dead |
| turn exceeds the watchdog | **dead for the rest of the episode**, not just that turn |
| reply is not JSON / not an object | dead |
| reply contains illegal actions | repaired by the validator (the bridge stays alive) |
| anything else inside `agent()` | caught; worst case a legal all-PASS turn |

**The watchdog retires the bridge permanently rather than for one turn.** A
timed-out turn leaves a reply in flight and the protocol carries no
request/response tag, so a late answer would silently become the *next* turn's
action. One correct weaker agent beats one fast desynchronised one. Phase B
should add a sequence tag if per-turn recovery becomes worth it.

Budgets: `FIRST_BUDGET = 1.50 s` (pays process start-up, and the engine has not
started the clock on turn 1 in the same way), `TURN_BUDGET = 0.25 s` — a
quarter of `actTimeout`, against a measured worst turn of 7.85 ms.

> **WRONG, AND FIXED — see §0.3.** The parenthetical above is an assumption
> that was never measured, and `1.50 s` is longer than the 1.0 s `actTimeout`,
> so in the one situation the watchdog exists for it would have handed the
> engine a turn it had already timed out (measured: a 1,503 ms first turn).
> `FIRST_BUDGET` is now **0.50 s**, sized from the measured ≤110 ms spawn.
> `TURN_BUDGET` is unchanged at 0.25 s for the shipped budget of 0.

### Validator

Everything the binary returns is checked in Python before it is returned:

* `hands` is forced to align **positionally** with `farms[me]["hands"]`
  (padded with `PASS`, truncated if too long);
* at most **10** market orders, in the binary's priority order;
* op names checked against the interpreter's actual vocabulary, unit ops and
  market ops separately; unknown ops become `PASS`, unknown orders are dropped;
* market order counts must parse as a positive int.

`repairs` counts every fix. It was **0** across all 11,504 measured turns —
i.e. the Rust side has never yet produced anything the validator had to touch.

### Proven, not asserted

Unpacking the real tarball and breaking the binary three ways
(`.local/build/fallback_test.sh`):

| mode | fallback fires | status | bank |
|---|---|---|---|
| binary removed | 719 / 719 | DONE | 55,008 |
| `chmod 000` | 0 (bridge chmod'd it back) | DONE | 55,008 |
| binary replaced with garbage | 719 / 719 (`Exec format error`) | DONE | 55,008 |
| normal | 0 | DONE | 55,008 |

**Identical bank in every mode.** That is the payoff of keeping the two
implementations identical.

### Ladder beacon

`agent()` writes one line to stderr on the first turn and one on the last:

```
TRACKP start {"turns": 1, "bridge": 1, "fallback": 0, ...}
TRACKP end   {"turns": 719, "bridge": 719, "fallback": 0, ...}
```

kaggle-environments captures agent stderr into the episode's per-agent `logs`,
so **this is how we find out whether the subprocess transport actually worked
on Kaggle**. Without it a blocked `fork/exec` would look like a normal game:
the fallback covers for it, the agent plays a legal episode, and we would never
know the binary never ran. Check the beacon in the first replay after any
submission of this artefact.

---

## 4. Build

Docker was verified working (Docker Desktop 4.84.0, engine 29.6.2, WSL2
backend).

```powershell
# Linux artefact (main.py + static musl binary) -> submission.tar.gz
.\scripts\build_trackp_linux.ps1

# repack without rebuilding the binary
.\scripts\build_trackp_linux.ps1 -SkipDocker
```

What it does:

1. `python src/kaggriculture/trackp/compiled/build_main.py` — assembles `main.py`.
2. `docker run --rm -v <repo>:/work -w /work rust:latest bash -c
   'rustup target add x86_64-unknown-linux-musl && cd /work/rustengine &&
   cargo build --release --bin kagg --target x86_64-unknown-linux-musl
   --target-dir /work/.local/build/rust-linux && strip …'`
3. stages `main.py` + `kagg`
4. `tar -czf submission.tar.gz main.py kagg` **inside the container**, so the
   executable bit survives — Windows has no mode bit to preserve and a
   non-executable `kagg` would fall back silently for 719 turns.

Output:

```
.local/candidates/trackp_compiled/submission.tar.gz   0.42 MiB   (limit 100 MiB)
  -rw-r--r-- main.py   37,178
  -rwxr-xr-x kagg     856,768   ELF 64-bit LSB pie, x86-64, static-pie, stripped
```

`--bin kagg` is deliberate: the crate also hosts the Phase-B search library and
its offline bins, which are being edited in parallel, and the shipped artefact
must not be able to fail to build because an unrelated target is mid-change.

### Measurement

```powershell
# unpacks the real tarball INSIDE Linux and plays it on the official engine
.\scripts\run_trackp_linux_harness.ps1 -Seeds "3,4,5,6" -Seats "0,1"

# the Rust-vs-Python identity check (Windows, on the kagg serve engine)
python tests\test_compiled_agent.py --seeds 3,4,5
```

The harness plays the **vendored official interpreter**, never the Rust engine.
A compiled agent measured on the engine it embeds would be marking its own
homework.

---

## 5. Measured

Official engine, inside `python:3.11-slim` on Docker Desktop, 8 seeds×seats ×
2 opponents = 16 episodes = **11,504 timed turns**.

### Latency (ms per `agent()` call, including json.dumps + pipe + validate)

| path | mean | p95 | worst |
|---|---|---|---|
| compiled bridge | **1.36** | 2.1 | **7.85** |
| pure-Python fallback | 0.09 | 0.2 | 1.38 |
| first turn (process spawn + first day plan) | — | — | 14.7 |

Against a **1,000 ms** `actTimeout`: the worst turn uses **0.8%** of the
budget — **127x headroom**. Kaggle allocates 1.6 vCPU; even a 10x slower single
core leaves 12x margin. The house rule of "mean under 20 ms, worst under
100 ms" is met with room to spare by the bridge, and the fallback is faster than
the bridge (it skips the JSON round trip) so a degraded episode is never a
timeout risk.

Note the shape: the compiled path is *slower* than the fallback at this policy,
because at ~1.4 ms/turn the cost is entirely `json.dumps(obs)` and the pipe, not
the planning. That inverts the moment Phase B puts a real search behind it —
which is the whole point of the spike — but it means the transport buys nothing
today.

### Bank

| opponent | seeds 3-6, both seats | our bank | their bank | wins |
|---|---|---|---|---|
| `agents/v43.0_bandit.py` (our live slot-2 agent) | 8 cells | 42,155 – 59,007 (median ~49k) | 107,203 – 136,785 | 0/8 |
| `data/gauntlet/pub_v16rc5.py` (public V16-RC5) | 8 cells | 30,811 – 57,046 (median ~42k) | 78,438 – 117,550 | 0/8 |
| **all** | 16 cells | **median 43,905, min 30,811** | | **0/16** |

### Gate

| condition | required | measured | |
|---|---|---|---|
| full legal 720-step episode | 720 steps, status DONE | 719 agent turns, DONE, 16/16 | **PASS** |
| worst turn latency | well under 1 s with 1.6-vCPU margin | 7.85 ms (0.8% of budget) | **PASS** |
| fallback fires | 0 | 0 / 11,504 | **PASS** |
| action legality | validator repairs 0, hands aligned, ≤10 orders | 0 repairs, 0 mismatches | **PASS** |
| own bank vs a reactive opponent | ≥ 85,000 | median 43,905 | **FAIL** |

---

## 6. Why the bank fails, in numbers

This is the useful output of the spike. Per-day ledger of one episode
(`.local/build/ledger.py`, seed 3 vs v43.0_bandit) against the top-10 medians
from `docs/history/plan-2800-2026-09-03.md`:

| symptom | ours | top-10 |
|---|---|---|
| idle (PASS) unit-turns | 2,340 | 513 |
| movement as a share of non-PASS ops | 52% (2,457 of 4,744) | — |
| standing crops, days 1-11 | 18 | — |
| PLANT ops, whole season | 168 | 243 plants |
| 2nd quadrant bought | day 10 (schedule says day 4) | day 5 usable |
| undug weeds standing from day 21 | 11-13 | — |
| shed at cap (100/100, dusk overflow discarded) | days 23 and 27 | — |
| final MILK / FERTILIZER quote | $1 / $1 (glutted) | — |

The single largest item was land timing, and it was a **failure to execute the
documented skeleton**, not a different strategy: the herd ramp (a cow a day from
day 2) spent the money the day-4 quadrant needed, so the farm sat on NW's 24
usable tiles for a third of the season.

Fixed by a new genome knob `land_reserve_lead` in
`src/kaggriculture/trackp/build_econ_agent.py` (and mirrored in `policy.rs`): withhold the
next scheduled quadrant's cost from the herd. Measured over 24 official-engine
cells (6 seeds × 2 seats × 2 opponents):

| `land_reserve_lead` | median bank | mean bank |
|---|---|---|
| -1 (off, the old behaviour) | 48,368 | 47,796 |
| **0 (new default)** | **55,008** | **52,099** |
| 1 | 48,513 | 50,340 |
| 2 and 6 | 41,732 | 43,397 |

Reserving *too far* ahead is worse than not reserving: SW costs 2,000 and
holding that from day 5 starves the herd for six days. `valve_hi` / `valve_lo`
were also promoted from hard-coded 55/45 to genome keys so the two
implementations parameterise identically.

That is +6.6k of median. The remaining ~40k gap is labour utilisation, glutted
sell scheduling and shed overflow — i.e. per-turn decisions. **That is Phase B's
job and it is precisely what the compiled substrate was built to afford.**

---

## 7. How the operator submits

Nothing in this repo submits. Build, then:

```
kaggle competitions submit kaggriculture \
  -f ".local/candidates/trackp_compiled/submission.tar.gz" \
  -m "trackp compiled phase A"
```

**~~Do not submit this artefact as a competition entry.~~ SUPERSEDED by an
explicit operator order, 2026-09-03 evening — see §0.5.** The strength verdict
has not changed (§0.4: 0 wins in 40, significantly worse than the seat it
replaces); what changed is that the operator judged the *information* worth the
slot, because our offline panels have proven anti-predictive and the stderr
beacon is the only way to learn whether Kaggle's sandbox permits `fork`/`exec`.
The eviction cost is real and is the operator's to weigh.

The paragraph below describes the ORIGINAL Phase-A numbers (~44k, 16 cells) and
is kept for the record; §0.4 has the ship build's.

If the operator wants to validate the transport on real Kaggle hardware before
Phase B — which is worth doing, see risk 1 — the cheap way is to submit it on a
day when a slot is genuinely free, then read the first replay's agent logs for
the `TRACKP start` / `TRACKP end` beacon. `"fallback": 0` in the `end` beacon
means the compiled path ran on the ladder.

---

## 8. Risks

1. **Untested on Kaggle. Nothing was submitted.** The open question is whether
   the agent sandbox permits `fork`/`exec`. yhay81's ctypes approach only needs
   `dlopen`, which is a weaker requirement. If subprocess spawning is blocked we
   fall back for all 719 turns and score the Python planner — safe, but the
   whole spike buys nothing. The stderr beacon exists to answer this from one
   replay. **Contingency if it is blocked:** add a cdylib target and a ctypes
   path as a second transport, keeping this one as the preferred path and the
   Python planner as the third.

2. **The strength gate fails and Phase B has to close a ~40k gap**, not a
   rounding error. The transport is proven; the policy is not remotely
   competitive, and the diagnosis in §6 says the deficit is in per-turn
   labour and sell decisions.

3. **`kagg serve` grants the agent ONE MORE TURN than the official
   interpreter** — 720 calls against 719 (measured,
   `.local/build/turncount.py`). The official engine ends when the pre-step
   counter reaches `episodeSteps - 2`; `service.rs::serve()` ends at
   `s.step >= EPISODE_STEPS`. For an agent that acts on the final turn this
   changes the bank: the pre-fix skeleton banked 52,909 on serve and 50,771 on
   the official engine for the same seed. Tape agents PASS at the end, which is
   why `models/serve_equiv.json` never caught it and why
   `v43.0_bandit vs pub_v16rc5` still matches 2/2.
   **This matters beyond Track P: the tournament runs on `serve` whenever
   `refresh_cycle.serve_allowed()` is green.** The fix is one line —
   `let done = s.step >= EPISODE_STEPS - 1;` in `service.rs` — but it
   invalidates the existing `serve_equiv` evidence and would shift tournament
   numbers, so it was NOT applied here. It needs a deliberate re-audit.
   The engines themselves are fine: driven by the same tape they are
   bit-identical for 719 steps including this policy's action stream
   (`.local/build/divergence.py`).

4. **A harness that wraps an agent in a callable OBJECT silently cripples it.**
   kaggle-environments introspects the agent's signature to decide whether to
   pass `configuration`; a class instance made it pass only the observation, and
   `pub_v16rc5` then banked exactly its 3,000 starting money while our candidate
   "won" 100k games. Caught here only because a bank of exactly the starting
   money is suspicious. `harness.py` now uses a plain two-argument function and
   flags any cell whose opponent banked ≤ 3,000 as `DEAD OPPONENT`. Worth
   copying that guard into every panel.

5. **`rustengine/` is being edited concurrently** by the Phase-B search work
   (`lib.rs`, `plan.rs`, `search.rs`, `value.rs`, `src/bin/` all appeared during
   this task). `main.rs` keeps its own `mod` declarations and the build is
   pinned to `--bin kagg`, so the artefact is insulated, but two cargo builds in
   the same target dir do collide (`Access is denied` on a locked exe). Prefer
   `--target-dir` separation when both lanes build at once.

6. **`panic = "abort"` was removed from the release profile** so `kagg play`
   can `catch_unwind` each turn and answer a legal PASS instead of dying
   mid-episode. This changes the profile for every binary in the crate,
   including the pre-ranker. It does not change arithmetic; the engine
   differential test still passes 5/5 bit-identical.

7. ~~**The 250 ms watchdog is untested in anger**~~ — **CLOSED 2026-09-03,
   see §0.3.** Fired deliberately (breakage case D: a `kagg` that reads the
   request and never answers). The permanent-retirement semantics behave as
   argued — 1 timeout, then 719 fallback turns, DONE, identical bank — and
   firing it **found a real bug**: `FIRST_BUDGET` was 1.50 s against a 1.0 s
   `actTimeout`. Now 0.50 s. Cases E (non-JSON reply) and F (illegal actions,
   2,077 validator repairs, bridge stays alive) were added at the same time,
   so every documented failure mode in §3 is now empirically exercised rather
   than asserted.

---

## Appendix — commands

### Reproducing the SHIP build (§0)

```powershell
# rebuild the artefact
.\scripts\build_trackp_linux.ps1

# THE number: the shipped tarball at its default budget, official engine
.\scripts\run_trackp_linux_harness.ps1 -Seeds "3,4,5,6" -Seats "0,1" `
  -Workers 6 -WorstGate 250 `
  -Vs @("data/gauntlet/pub_v16rc5.py","data/gauntlet/kaito_v48.py",
        "data/gauntlet/pub_rayk_c94.py","agents/v43.0_bandit.py",
        "agents/v42.1_trackp.py") `
  -JsonOut ".local/candidates/trackp_compiled/g3_ship.json"

# the same 40 cells with the search forced on, for the budget decision
... -BudgetMs 150 -WorstGate 1000 -JsonOut ".../g2_b150.json"

# clean latency sweep on the rebuilt binary (Workers 1, quiet box)
foreach ($b in 0,100,150) {
  .\scripts\run_trackp_linux_harness.ps1 -Seeds "3" -Seats "0,1" `
    -Vs @("data/gauntlet/pub_v16rc5.py") -BudgetMs $b -Workers 1 `
    -WorstGate 1000 -JsonOut ".local/candidates/trackp_compiled/lat2_$b.json"
}

# deliberate breakage A-F on the real tarball, inside Linux
docker run --rm -v "D:/codebase/kaggriculture:/work" -w /work kagg-harness:2 `
  bash /work/.local/build/fallback_test.sh
docker run --rm -v "D:/codebase/kaggriculture:/work" -w /work kagg-harness:2 `
  bash /work/.local/build/fallback_df.sh     # D and F again, with --json

# summarise / pair any two harness runs (W-D-L, banks, share, legality audit)
python .local/build/report.py <cand.json> "label" [<ref.json> "ref label"]
python src/kaggriculture/trackp/compiled/verdict.py <cand.json> --ref <ref.json>

# regenerate models/release_2026-09-03_trackp.json from the measurements
python .local/build/make_release.py .local/build/release_spec.json `
  models/release_2026-09-03_trackp.json
```

### Phase-A commands (original)

```bash
# build the Linux artefact
.\scripts\build_trackp_linux.ps1

# measure the assembled artefact on the official engine, inside Linux
.\scripts\run_trackp_linux_harness.ps1 -Seeds "3,4,5,6" -Seats "0,1"

# Rust policy == Python fallback, action for action
python tests\test_compiled_agent.py --seeds 3,4,5

# the binary alone, no Python, no engine
rustengine\target\release\kagg.exe play-selftest

# the assembled agent alone: one observation in, one action out
cd <stage> && python main.py --selftest < obs.json
```
