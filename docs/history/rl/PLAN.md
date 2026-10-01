# kaggriculture-rl: plan of record (2026-09-24)

Everything for the macro RL track lives in this folder: corpus extraction, the Rust agent,
self-play and clone leagues, BC, PPO, validation, the agent builder, and the Kaggle notebooks.
Nothing here depends on code elsewhere in the repo at run time. Code copied from the repo gets
a provenance note.

Hard requirements (operator): **Rust wherever possible, very low memory, all code/configs here,
the agent itself in Rust.** Deadline: final submissions by 30 Sep; lock 28 Sep.

> **2026-09-25:** the current architecture is `docs/ARCHITECTURE.md`; the current plan is §20–22 below (they override earlier sections where they differ). Research: `docs/research-2026-09-25.md`.

## 0. What we are building

A Rust agent that plays exactly like v61.1: the chassis tape executor plus its reactive
layers. The difference is one extra component, a **macro policy**. Once per game day it picks
one of about 12 **lever profiles**, i.e. settings for the market-side layers. The policy is
trained with BC and then PPO, on a corpus of ladder games plus our own leagues. All training,
rollouts and evaluation run in Rust on the Rust engine.

Why this shape: every per-turn BC/RL attempt (ours, and all 9 public threads) broke the causal
chain. Every home-made executor capped the economy ($0, or $51k vs $115k+). The chassis is the
proven executor; the policy only tunes it.

## 1. Layout

```
kaggriculture-rl/
  Cargo.toml                 workspace, members = crates/*
  crates/
    engine/                  Rust engine (copied from kaggriculture-sim kagg-engine, certified vs 1.32.7)
    obs/                     observation structs + JSON (de)serialisation shared by everything
    features/                per-day features and behaviour (shared by the corpus AND the agent: no train/serve skew)
    corpus/                  bin corpus-extract: GM / daily / replay-dir adapters, ledger + delta, Parquet
    chassis/                 Rust port of the tape executor (Chassis, routes, pending queues, sell ledger)
    layers/                  Rust ports of the reactive layers, one module per layer; LayerChain from config
    policy/                  macro policy: GRU/MLP forward pass (plain f32, no ML framework), weight loader, profiles
    agent/                   the agent = chassis + layer chain + policy; bin `agent` speaks the stdio protocol
    runner/                  bin `league`: games on the Rust engine, Rust agents in-process, Python opponents via host
    learn/                   bin `bc`, bin `ppo` (candle CPU; tiny nets), datasets over mmap'd Arrow
    gate/                    bin `gate`: paired McNemar, per-family tables, real-seed suite (same rules as winplan)
    build/                   bin `build-agent`: config + weights -> submission dir/tarball; verify; manifest
  python/
    bridge/main.py           Kaggle entry: spawns the static `agent` binary once, relays obs/actions (stdlib only)
    host/                    minimal Python opponent host for public agents (last-callable loader, S3/S4/S5 rules)
    ref/                     parity references: Python v61.1 + truncated-chain builder
  configs/
    profiles/v1.json         the lever profiles (profile 0 = v61.1 exactly)
    agents/*.json            agent build configs (chain, knobs, weights path, version)
    league/*.json            panels, seed ranges, family weights, held-out validation panel
    bc.json, ppo.json        training hyper-parameters
    corpus.json              filters, sources, date window
  kaggle/
    extract_official/        private notebook: official daily datasets -> corpus (kernel output)
    extract_gm/              private notebook: GM dataset backfill -> corpus (kernel output)
    submit/                  private submission notebook template (assembles submission.tar.gz)
  scripts/                   thin PowerShell wrappers (daily delta, nightly league, build+gate)
  opponents/                 (gitignored) copies of the harvested public agents + stress variants
  data/                      (gitignored) corpus/, ledger, recordings/, ckpts/, leagues/, gates/
  docs/                      PLAN.md, corpus.md, port.md (layer inventory + parity status), runbook.md
```

## 2. Memory budget (hard caps, checked in tests)

| Process | Cap | How |
|---|---|---|
| corpus-extract | <= 200 MB/thread, 2 threads | stream one replay at a time; compact id maps for the CSVs |
| agent (one instance) | <= 30 MB | routes table decoded once, shared via `Arc` across in-process games |
| league/ppo runner | <= 1.5 GB total at 16 threads | Rust agents in-process; engine state is small; no per-game process |
| Python opponent host | <= 250 MB each, at most 4 alive | only for opponents not ported; one module per host, reused |
| bc / ppo learner | <= 1.5 GB | Arrow memory-mapped, sequences sampled on demand; model ~32k params |
| Kaggle notebooks | remote | not counted |

The whole RL stack at peak (runner + learner + 4 hosts) stays under about 4 GB, against today's
9-worker Python gate at about 12 GB.

## 3. The Rust agent (critical path)

**Target:** action-for-action identical to Python v61.1 (`configs/agents/v61.1.json` in the
main repo = cha22 + RSA 5 + glut guard 0.5), on the same observations.

**Structure.** `Chassis` holds the players, routes, pending queues and sell_state/r36_debts
exactly as in Python. Each reactive layer is a `Layer` with
`fn apply(&mut self, obs: &Obs, action: Action, ctx: &mut Ctx) -> Action`. The chain order and
the knobs come from config. The port follows the Python chain order from `layermap`: 69 captures,
some of them telemetry-only.

**Scope trim (measured first, tonight).** The cha22 ablation found IG, MG, BD, SM, WL and EV
neutral or negative against v61. We build **v61.1-lite** in Python (those layers switched off
through config) and gate it against v61.1 on the full roster. If it is not worse, the parity
target becomes lite, which saves about 15% of the port.

**Parity method.** Agents are deterministic given the observation history, so:
1. **Recordings:** our seat's observation stream plus the Python agent's actions, from about 600
   games (49 public agents + stress + self, stratified worlds, both seats). Stored as compact
   JSONL.zst in `data/recordings`, about 1.5 MB per game.
2. **Open-loop parity:** feed each recorded stream to the Rust agent and compare every action
   (normalised JSON). The report shows the first divergence per game and per layer.
3. **Layer-by-layer:** `python/ref` builds the Python agent **truncated after layer k** (entry
   = layer k's wrapper). The Rust chain up to k must match it. This lets layers be ported and
   verified in parallel.
4. **Closed-loop:** Rust agent against the same opponents on the Rust engine gives banks
   identical to the Python agent (the existing gate rows).
5. **Official check:** the Rust agent through `python/bridge/main.py` on the official
   kaggle-environments 1.32.7 gives identical banks on 10 seeds.

**Port order** (value first, dependencies respected):
chassis core + tape executor -> opening/route selection -> V9 courier/carrot/herd/fert ->
R36/R37/RELEASE/race layers -> CA/OR2/CH/SR/HD2/CS -> RACE/R127/PG/V44Y/Y/E334/E335/V11/V13V ->
cha22 tail (E343_WL, ADV, T62A, PIPE, MA, WB3, FX, EV, DP, MP, BD, MPX, SM, CXD, E410, E402, MG, IG) -> RSA.

**Parallelism:** 3 porting agents on disjoint layer groups, all using the shared parity
harness. At most 2 concurrent cargo builds (memory).

**Acceptance:** 100% action parity on all recordings and identical closed-loop banks.

**Fallback:** if parity misses by 26 Sep 23:59 UTC, PPO runs with the Python agent through the
Rust runner (slower), and the final pair comes from the Python line.

## 4. Macro policy inside the agent

- **Decision:** at step 24*d + 1 (hour 1; hour 0 has no hands), for d = 0..29. The chosen
  profile holds for the whole day. Switching only at day boundaries keeps the chassis ledgers
  consistent.
- **Observation:** about 96 floats from `crates/features` (the same code the corpus uses), plus
  chassis state (route id, tape position) and clone signals.
- **Action:** a profile id from `configs/profiles/v1.json`. Profiles change market-side/timing
  knobs only: RSA look/guard, DP/MP/MPX/EV on/off and horizons, CXD pass, race horizons. They
  never change tape planting, hiring, land or animals, so the structural tape plays as written.
- **Net:** Linear(96->64) -> GRU(64) over days -> heads: policy (12), value (win prob), aux
  opponent-family (K) and aux rival next-day sales (9). About 32k parameters.
- **Guards:** telemetry per episode for failed buys, layer errors and pending-queue growth. A
  profile that desyncs is masked in training and flagged at the gate. Profile 0 must stay
  bank-identical to v61.1.

## 5. Corpus (crates/corpus + kaggle/)

- **Sources:**
  - GM dataset (local, delta daily);
  - official daily datasets (private notebook, kernel output, delta daily);
  - our leagues (replay-format JSON).
- **Top-10 archive:** not used.
- **Filters:** engine 1.32.7 only. Ratings and outcomes are kept as columns; filtering by
  player strength happens at training time. Newest first. Window: 16 Aug onward, backfilled
  newest first; training starts on 1 Sep+ as it lands.
- **Tables:** `episodes`, `daily_states`, `daily_behaviour` (schema in docs/corpus.md,
  SCHEMA_VERSION 1).
- **Ledger:** episode_id -> source, file sha, engine, date, extractor version, status. Deduped
  across sources.
- **Delta:**
  - GM: per-file (name, size, head sha); frozen shards are skipped; for the growing shard and new
    shards, only unknown ids are extracted.
  - Official: new dates in `kaggle/kaggriculture-episodes-index`.
  - The notebooks attach their own previous output and skip processed days.

## 6. Leagues (crates/runner)

| League | Purpose | Games |
|---|---|---|
| cross-play | BC labels, payoff matrix, non-transitivity | top public agents + cha22 + v61 + v61.1 in round-robin pairs x stratified worlds x both seats; our Rust agent with random per-day profiles |
| clone | ground-truth clone/family labels | each agent vs itself and vs its stress variants, many worlds |
| validation (held out) | checkpoint selection | fixed panel (stress variants + unseen public agents) + real-loss seeds + a seed range never used in training |

All games are written as replay-format JSON and go through `corpus-extract`, so there is one
schema for everything. Opponents that exist in Rust (our own chassis family configs, see §9)
run in-process. Python-only opponents run through at most 4 host processes.

## 7. BC (crates/learn, candle, CPU)

- **Data:** `daily_states` + `daily_behaviour` -> profile labels:
  - league games: the profile actually played, weighted by outcome (advantage-weighted BC);
  - ladder winners from our lineage: nearest profile to their behaviour vs the known tape.
- **Aux targets:** opponent family (fingerprint/league truth) and the rival's next-day sales.
- **Model:** GRU(64) + heads (main); MLP over the last 3 days (baseline).
- **Metrics:** per-head accuracy on held-out games (split by episode sha); family classifier
  accuracy by day 3 (target >= 80%).
- **Cost:** about 9M day-steps; minutes per epoch on CPU; about 15-30 min per run.

## 8. PPO (crates/learn + crates/runner, one process)

- **Rollouts:** 16 threads; each game samples a random opponent (ladder-family weights, 20%
  frozen self snapshots), a random world/seed from the training range, and a random seat.
- **Reward:** win 1 / draw 0.5 / loss 0, plus a small margin term annealed to 0.
- **Algorithm:** recurrent PPO over whole 30-day episodes. gamma 1, GAE lambda 0.95, clip 0.2,
  entropy annealed, KL early stop, normalised values, separate LRs per head, action masking for
  flagged profiles. About 1,024 games per iteration.
- **Validation** every 10 iterations on the held-out panel, paired against v61.1 on the same
  seeds. It picks checkpoints, drives early stopping, and reweights the training panel toward
  families where validation drops. It never enters the gradient.
- **Throughput target:** all-Rust games about 0.1-0.3 s each -> about 50-150 games/s on 16
  threads. Python-opponent games are much slower, so they are capped to a share of each
  iteration.

## 9. Rust opponents (phase 2, speed multiplier)

Most of the ladder is our chassis lineage (fingerprints: herd_safe, harvest_ledger, v15stack,
v54, pioneers). With the Rust chassis + layer library, a family is **a config**: its chain and
knobs. Each family config must pass the same open-loop parity against its Python original on
recordings before it joins the in-process pool. Order: herd_safe_v3 -> harvest_ledger ->
v15stack -> v54 -> pioneers (ladder frequency).

## 10. Builder and submission (crates/build + python/bridge + kaggle/submit)

- **One static binary** `agent` (x86_64-unknown-linux-musl, opt-level 3, LTO, stripped), built
  once per code version. Target size about 2-4 MB.
- **Config-driven sub-versions:** a submission = binary + `config.json` (chain/knobs) +
  `weights.bin` (checkpoint) + `profiles.json` + `main.py` bridge. Sub-versions from different
  checkpoints differ only in `weights.bin`, with no recompilation. Each has a manifest (shas,
  versions) and is gated separately.
- **Verification:** the builder refuses mismatched arch/feature/profile versions and runs an
  export-equivalence test (Rust forward pass == candle on 1,000 stored states). The bridge runs
  the official-engine check (DONE/DONE, latency p99/max) before packaging.
- **Kaggle:** notebook payloads over about 1.6 MB fail to push. So the binary goes in a
  **private** Kaggle dataset; the private submission notebook copies it from /kaggle/input, adds
  the small files, checks every sha, and writes `submission.tar.gz`. Submit with
  `-k <slug> -v <N> -f submission.tar.gz`, operator approval each time. Naming follows house
  rules: v{x}.{y}_bandit, daily major.
- **Bridge robustness:** the binary is spawned once and kept alive. On a crash or timeout the
  bridge restarts it and replays the observation history to rebuild state. The per-turn budget
  is checked against the 1 s actTimeout.

## 11. Gate (crates/gate)

The same promotion rule as winplan:
- paired McNemar vs the current best on the same games: full roster + stress + self + real-loss
  suite;
- no losing family;
- at least 60 paired seeds per claim;
- the official-engine check;
- operator approval.

The Rust gate reproduces `winplan.gate.compare` on the same jsonl (a test).

## 12. Timeline (UTC)

| When | Deliverable |
|---|---|
| 24 Sep night | workspace scaffold; corpus-extract (in progress); v61.1-lite gate; parity recordings (600 games); port.md layer inventory |
| 25 Sep | Kaggle extraction notebooks run (official ~1-2 h, GM ~2-4 h); engine + obs + chassis core in Rust; 3 port agents start; policy crate; BC data ready |
| 26 Sep | remaining layers; open-loop parity 100%, closed-loop banks identical; runner + leagues; BC trained; builder + bridge; Rust v61.1 (no policy) gated bank-identical, speed measured |
| 27 Sep | PPO (all-Rust pool + capped Python opponents); validation every 10 iterations; checkpoints -> sub-versions |
| 28 Sep | gate the best checkpoints; build; submit with approval; choose the final pair |
| 29-30 Sep | buffer; ladder watch; no new builds |

Decision points: 25 Sep 12:00 (corpus quality); 26 Sep 23:59 (port parity -> fallback);
27 Sep 20:00 (PPO beats BC and bandit baselines on validation?).

## 13. Risks

| Risk | Mitigation |
|---|---|
| Port parity is slower than planned | layer-by-layer truncation tests; 3 agents; lite scope; fallback at 26 Sep |
| candle on Windows | CPU only (tiny net); PyTorch kept only as an emergency trainer that reads the same Arrow data |
| Notebook size | binary via a private dataset |
| Non-transitive league | family-conditional policy; diverse league; held-out validation |
| Harness crashes counted as wins | errors are counted separately, never as wins (kagsym red flag) |

## 14. Naming conventions (append-only; nothing is overwritten)

- `RUNID = <YYYYMMDD>T<HHMM>Z-<source>-<gitsha7>`, e.g. `20260925T0400Z-gm-3fa91c2`.
- **data/**
  - `raw/gm/<kaggle-version-UTC>/...`
  - `corpus/s<SCHEMA>/source=<gm|official|league>/[league=<name>/]date=<YYYY-MM-DD>/part-<RUNID>-<NNNN>.parquet`
  - `corpus/s<SCHEMA>/ledger/ledger-<RUNID>.parquet`
  - `corpus/s<SCHEMA>/runs/<RUNID>.json` (a delta = the parts and ledger rows of one RUNID)
  - `recordings/r<V>/<agent-version>/<RUNID>/*.jsonl.zst`
  - `leagues/<name>/<RUNID>/`, `gates/<candidate>__<RUNID>.jsonl`
- **weights/**
  - `registry.json` (id, parent, data window, teacher, panel version, metrics, status)
  - `bc/<BCID>/{model.safetensors, meta.json, metrics.json}`, with
    `BCID = bc-<arch>-f<featV>-p<profV>-s<schema>-to<LASTDATE>-<YYYYMMDD>T<HHMM>Z`
  - `ppo/<PPORUN>/ckpt-i<ITER:06>-g<GAMES_K>k.safetensors` (+ meta.json), with
    `PPORUN = ppo-<arch>-f<featV>-p<profV>-from-<parent>-<YYYYMMDD>T<HHMM>Z`
  - `release/<agent-version>/{weights.bin, manifest.json}`: exactly what shipped
- The builder refuses weights whose arch/feature/profile/schema versions do not match the agent
  config.

## 15. Continuous RL and the daily BC feed

- **One long-running PPO learner** (weights and optimizer state persist).
- **Daily at 04:00 UTC:**
  1. corpus delta sync;
  2. BC fine-tuned from yesterday's BC on the delta + a recency-weighted replay sample (~15 min)
     -> a new BC teacher in the registry.
- The learner hot-swaps the teacher at the next iteration boundary.
- **Loss:** `L_PPO + lambda * KL(pi || pi_BC)`, with lambda decaying within the day. The aux heads
  (opponent family, rival sales) keep training on corpus mini-batches mixed into each update.
- The league panel file gains new opponents (harvest, real-loss opponents, frozen self snapshots)
  and is reloaded between iterations. The validation panel stays fixed per version.
- **Guard:** if validation drops twice in a row after a teacher swap, roll back to the last good
  checkpoint and halve lambda.

## 16. Step budget

- 1 game = 30 policy decisions.
- Iteration = 4,096 games; validation and a checkpoint every 25 iterations (~100k games, ~2,000
  paired validation games).
- **All-Rust:** ~50-150 games/s -> 36 h of PPO (27 Sep 00:00 to 28 Sep 12:00) = ~5-15M games
  (150-450M decisions).
- **Fallback (Python agent):** ~300k games.
- Milestones at 0.25M / 1M / 3M / 5M / 10M games vs the BC and bandit baselines. Expected
  convergence within 1-5M games (12 actions x 30 days).

## 17. Revisions (operator, 2026-09-24 late) — these override earlier sections

1. **Extraction is two-stage.** The Kaggle notebooks do NOT compute features. They convert each
   replay into a lossless **slim record**:
   - every step: both actions, the market (inventory, prices, params patches, shops), both
     farms' money / farmer and hands positions / hires_today / quadrants, and each seat's private
     shed / inventories / seeds;
   - full tiles only at hour 1 of each day and at step 719.

   Stored as zstd JSON per episode in Parquet (`data/slim/s1/...`, same naming as §14).
   Features and labels are computed **locally** from the slim corpus in Rust, in minutes, so
   feature changes never re-read the ~5 TB of raw replays.
2. **The partial crates are kept** and built on (crates/features, crates/corpus).
3. **Port all 69 layers.** Every layer has an on/off switch and a typed knob struct. The lite
   gate is an optional experiment only.
4. **The base agent is a config** (`configs/bases/*.json`: chassis family, route table, opening
   mode, ordered chain with knobs). v61, v61.1 and the lineage families are configs over one Rust
   chassis/layer library. Hot-swap levels:
   - profile: knob overrides at every day boundary;
   - base config within a chassis family: at steps 0, 72 and 144 (the world-reveal checkpoints
     where the chassis already re-selects routes), carrying the chassis state over;
   - different chassis family: game start only.

   Base selection at steps 72 and 144 is a policy action.
5. **Sync rule:** overrides may change WHEN and HOW MUCH we sell, never WHAT the farm does.
   - Structural actions (moves, plant/water/harvest/build, hire, land, animals) always come from
     the tape + reactive layers; the tape is time-indexed.
   - Market-side deviations go through the chassis ledger (r36_debts, funding protection, budget
     guard).
   - Per-turn invariants: tape index == step, projected shed >= 0, cash covers planned spend or a
     funding sell is queued, pending queue bounded, no layer errors. An invariant-breaking profile
     is masked in that state.
   - Sync test: for each profile, profile k vs profile 0 must give identical structural action
     streams until the first reactive divergence, with the cause reported.
6. **Labour:** no structural labour learning before 30 Sep. Only safe on/off knobs (the labour
   reserve's hire-skip, input/fertiliser appliers, feed-load matcher) inside profiles. Tape
   transforms + a reassigner come after 30 Sep.
7. **Price model + rules model in `features`,** built on the engine crate (single source):
   - price model: per-item quote, marginal revenue of k units, price pressure, glut depth,
     forecast quotes at +1/4/8/24 turns (town draw schedule minus estimated rival flow, with
     optimistic and pessimistic bands), hold value;
   - rules model: realized shops and next unlock day, a distribution over the next shop,
     expected demand per item, world key.
   - Optional label: hindsight-best sell hour per item-day (an aux head).
8. **Latency gate:** agent binary max <= 500 us/turn on this machine; bridge + binary p99 <= 1 ms.
   Measured over ~1M turns in the parity harness; the build fails if exceeded. Heavy layers (CXD,
   E410, MPX) get precomputed price tables, fixed buffers and hard evaluation caps. The bridge
   sends a compact subset of the observation.
9. **Work runs in the main session only.** No subagents unless the operator names one.

## 18. Task list (IDs match the War Room board, stage "rl")

**P0: Scaffold**
- P0.1 keep and review the partial crates
- P0.2 workspace skeleton + memory-cap tests
- P0.3 engine crate copied from kaggriculture-sim + 20/20 official parity

**P1: Corpus**
- P1.1 slim format spec + Rust `slim` converter (built on crates/corpus)
- P1.2 static Linux build + container smoke test
- P1.3 private Kaggle dataset holding the binary
- P1.4 private notebook, official daily datasets 16 Aug-> (kernel output)
- P1.5 private notebook, GM backfill (kernel output, split Aug / Sep)
- P1.6 download the outputs into data/slim with RUNID naming + ledger
- P1.7 `obs` crate and the slim reader
- P1.8 features: economy, market, rival flow, clone signals, tape-relative
- P1.9 price model + rules model (engine-backed)
- P1.10 Python reference parity on 50 games
- P1.11 daily delta: GM file diff + notebook re-run on new days
- P1.12 corpus quality report

**P2: Rust agent port (all 69 layers)**
- P2.1 layer inventory (docs/port.md)
- P2.2 lite gate (optional experiment)
- P2.3 parity recordings (~600 Python v61.1 games)
- P2.4 truncated-chain Python reference
- P2.5 parity harness + latency recorder
- P2.6 chassis core
- P2.7-P2.12 layer groups in Python order, each with parity at its cut point
- P2.13 closed-loop bank parity
- P2.14 official check through the bridge
- P2.15 latency gate (max <= 500 us)
- P2.16 base configs + hot-swap at steps 0/72/144

**P3: Profiles and controller**
- P3.1 profile table v1 (incl. safe labour knobs)
- P3.2 knob table + day-boundary switching (profile 0 == v61.1)
- P3.3 MacroController (deploy / train / fixed modes)
- P3.4 profile signature library
- P3.5 base selection as an action (steps 72/144)
- P3.6 invariants, masking, sync test

**P4: Leagues**
- P4.1 Rust runner
- P4.2 Python opponent host
- P4.3 panels + held-out validation
- P4.4 cross-play league
- P4.5 clone league
- P4.6 Rust lineage-family configs, each parity-checked

**P5: Labels**
- P5.1 profile labels (exact + nearest-signature)
- P5.2 family labels
- P5.3 rival-flow and hindsight-sell labels
- P5.4 weights and splits

**P6: BC**
- P6.1 learn crate (candle CPU, GRU64 + heads, MLP)
- P6.2 pure-Rust forward pass + export equivalence
- P6.3 BC runs + metrics
- P6.4 BC agent gate

**P7: Continuous PPO**
- P7.1 rollout collector
- P7.2 recurrent PPO learner
- P7.3 teacher KL + hot-swap + panel reload
- P7.4 validation + promotion + rollback
- P7.5 the run (milestones at 0.25M / 1M / 3M / 5M / 10M games)

**P8: Builder and submission**
- P8.1 build-agent (config + weights -> submission, version checks)
- P8.2 bridge main.py (compact protocol, restart + replay)
- P8.3 private binary dataset + private submission notebook
- P8.4 submit with approval

**P9: Gate and ops**
- P9.1 Rust gate reproducing winplan.compare
- P9.2 daily scheduled job
- P9.3 board wiring + runbook

**Order:** P1.1-P1.6 first (notebooks running), then P0.2-P0.3, P1.7-P1.12, P2 (critical
path), P3, P4, P5, P6, P7, P8, P9 in parallel with the long runs.

## 18b. Objective and ranking (operator, 2026-09-24)

**Objective (operator, revised 2026-09-25):** NO losses to opponents rated below 2500; against 2500+ a
**>= 80% win rate is the floor and 90% the target**.

Every gate reports:
- **per-band results by opponent rating** (the public agents' ladder ratings; real-loss
  opponents from `ourladder`). Pass: 0 losses in every band below 2500, and the 2500+ win rate
  >= 80% (lower confidence bound reported; 90% is the target);
- **Elo** (online, per game, for trend curves during PPO);
- **Bradley-Terry** (batch MLE over every league/gate game), anchored to the public agents'
  known ladder ratings. This gives each candidate / checkpoint a **predicted ladder rating**
  with a confidence interval, plus a pairwise win-probability matrix (non-transitivity
  visible).

Promotion = paired McNemar vs the current best + no losing family + the band objective (or,
while no candidate meets it, the smallest number of sub-2500 losses and the best 2500+ rate).
Task P9.4.

## 18c. Checkpoint tournament (operator, 2026-09-24) — task P7.6

At each milestone (every 25 PPO iterations, and on every new BC teacher), the builder makes
agents from several weights:
- the last 3 PPO checkpoints and the best so far;
- the current BC;
- the bandit/CEM baseline;
- v61.1 (reference).

They play a round robin among themselves plus the held-out validation panel on the same seeds,
both seats. Bradley-Terry (anchored to the public agents' ladder ratings) ranks them, and the
band objective (§18b) is checked. The winner becomes the promotion candidate and the reference
for the next milestone; the rest are archived with their scores in `weights/registry.json`. All
of it is automated and resumable (per-game jsonl, skip recorded games).

## 18d. Hands-off operation (operator, 2026-09-24) — task P9.2 extended

The whole RL track runs unattended:
- **Orchestrator** (`scripts/`, driven by a Windows scheduled task + a watchdog): a state
  machine over the steps (sync -> notebooks -> download -> features -> BC -> PPO -> tournament ->
  gate -> build).
  - Every step is idempotent and keyed by RUNID.
  - A lock file prevents two instances.
  - On restart it resumes the last incomplete step.
- **Training is resumable and safe:**
  - checkpoints write optimizer + RNG + iteration + replay-buffer cursor atomically (temp file +
    rename);
  - resume reloads the newest complete checkpoint;
  - NaN/divergence guards roll back to the last good checkpoint;
  - disk and memory guards pause the run before the box starves;
  - every long job appends heartbeats; the watchdog restarts stalled jobs.
- **Board:** every step writes its status/progress/result to the War Room automatically.
- **Never submits:** a candidate that passes the gate is staged and the operator is asked.

## 19. Rules

- Every bug fix is checked against kaggriculture-sim (public) and kaggle-sim-framework
  (private), per CLAUDE.md.
- Generic pieces (runner, league, corpus ledger, PPO) are candidates for ksf after 30 Sep.
- Nothing is submitted, published or pushed without explicit operator approval.

## 20. Improvement items from the public-code research (2026-09-25)

Evidence: `docs/research-2026-09-25.md`, which covers the 640 Code-tab notebooks, master-engine
v1-25, meta-atlas v1-4, kagsym, kaggicultureRL, and v62.1's 3,349 tournament games. The objective
is §18b: 0 losses below 2500, and at least 80% at 2500+ (90% is the target). Items are in priority
order. **P0** = before the AWS launch, **P1** = during the first PPO day, **P2** = if time allows.

### G. Measure against the objective

- **G1 (P0) Band-weighted panel.** Map every public roster agent to its ladder team and rating,
  through the kernel author and the corpus index team names and submission ids.
  - Report the gate in two parts: losses vs agents under 2500 (must be 0) and the win rate vs
    2500+ agents (>= 80%).
  - Unmapped agents are reported separately, never pooled.
  - Replaces the pooled score in Q14, Q17 and Q18.
- **G2 (P0) World-level reporting and sampling.** v62.1's losses sit in 5 of 24 worlds: 166 of
  194, including 35% losses on BAKERY|FARMERS_MARKET.
  - Every gate prints per-world win, loss and draw counts.
  - PPO and the league oversample worlds by their current loss rate: prioritised replay over
    worlds, with at least 25% of games in the bottom-5 worlds.
  - Validation keeps one held-out seed per world.
- **G3 (P1) Baseline integrity.** A check that fails the gate unless profile 0 plays
  bank-identical to v61.1 on 24 seeds (zhincez trap 4, the mis-indexed "neutral"). The reference
  arm of every paired test is the stored v62.1 row, never re-derived.

### O. Opponents that look like the ladder

- **O1 (P0) The families that beat us, in-process.** herd_safe (sale_window_race ca20/ca25,
  shepherds_ledger, herd_safe_v3), pioneers and population_robust_economy take most of v62.1's
  losses.
  - Port them as chassis configs (§9 order, herd_safe first), each passing open-loop parity
    against its Python original.
  - Until a family is ported, it plays through the Python host at a capped share (at most 15% of
    each iteration).
- **O2 (P0) Top-band tape opponents.** For each current 2500+ team, the medoid episode per world
  from the corpus goes through `python/slim_to_tape.py` into the in-process tape executor. It is
  loaded as the single route of a v61.1 chassis, so it plays reactively with the full guard set,
  as meta-atlas v2 does.
  - Refresh on every delta.
  - About 30% of the PPO opponent mix.
  - The field is scripted (a shared schedule on 94% of seats at turn 24), so these opponents are
    the actual 2500+ opponents.
- **O3 (P1) v62 / v62.1 in-process.** GroupCtl is copied (§21 C2) so our own live agents are
  opponents in the league and PPO, with 20% frozen self-snapshots on top.

### L. Learning method

- **L1 (P0) PPO fixes for the traps the public runs hit.**
  - Replace `--ent 0.01` plus a KL-to-teacher weight that halves every 40 iterations with an
    **adaptive KL leash** (target KL to the BC anchor, coefficient raised or lowered to hold it).
    Entropy 0.
  - The value head reads `h.detach()`, or uses its own small GRU, so critic updates can't move
    the policy.
  - Log each iteration: mass on profile 0 and on the BC argmax, and KL to the anchor.
- **L2 (P0) Reward = Phi(margin / sigma_world)** on the terminal bank gap. Wins compress while
  margins separate, and Phi is the win currency made smooth (destbreso §7.1).
  - sigma per world, from the v62.1 panel spread.
  - Validation and gates still score W/D/L only.
- **L3 (P1) Counterfactual branch oracle (expert iteration).** The engine and chassis state are
  `Clone`, and the action is one of 32 profiles per day. So at a day boundary we can branch the
  exact game against an in-process opponent:
  - try each candidate profile for day d, then continue with the current policy;
  - score by final W/L plus L2;
  - label with the best profile, weighted by its gain over the policy's choice.

  Details:
  - Branch the policy's top-8 profiles plus the shield's profile, on key days: D0-D3, D12, and
    D20-D29.
  - Distil the labels into the GRU with a supervised loss next to PPO, then DAgger on the
    policy's own states.
  - Two public teams (andrewsokolovsky, kagsym) converged on this. Our discrete action space and
    43k steps/s engine make it cheaper for us.
  - Budget: at most 30% of the box; measure its gain before scaling it up.

### A. Action space (all market-side or state-consistent, so the base stays in sync)

- **A1 (P0) Shield + group features** (§21 C2, C4): GroupCtl, the D24 mirror guard and jitter,
  copied from v62, plus rival sell-timing features (dayobs v2).
- **A2 (P1) Conserved livestock substitution as a profile knob.** Test meta-atlas's shop-pressure
  rule (above) against v61.1's V231 cattle switch.
  - It relabels only registered COW/SHEEP purchase bundles and keeps pasture, count and service
    route.
  - It joins the profile table only if a paired gate passes.
- **A3 (P2) Latent-pasture activation.** One extra animal plus a hand into an already serviced
  empty pasture, as a gated profile bit. Only if A2 shows structural knobs pay.
- **Rejected by the evidence:** per-turn imitation of farm plus market (the channels can't be
  separated), end-to-end pure RL (hesoponyo: $81k vs $107-160k tapes), route switching away from a
  shared prefix, and quantity MPC (meta-atlas's own holdouts found no gain).

### I. Infrastructure

- **I1 (P0) Determinism across machines.** The export check (Rust == torch) runs on the ARM box
  and again on the x86 laptop before the tournament. The deployed forward pass is pure Rust, so
  the torch thread count kagsym hit can't reach the agent. The check still pins
  `torch.set_num_threads` for training reproducibility.
- **I2 (P2, post-lock) An owned base schedule** (island GA over a schedule compiler). It raises
  the floor (a relaxed roof of about $229k vs about $73k for the consensus route, both against an
  idle opponent) and is uncopyable while unfielded. It is too big before the 28 Sep lock and is
  recorded for after.

## 21. AWS autopilot and the reactive agent (design, 2026-09-25; not yet implemented)

**Where things run.**
- **AWS box** (c8g.16xlarge, ARM): the whole training loop, autonomously.
  - Setup: sync 1 MB of code and 35 MB of index, ledgers and run records. Build natively. Pull the
    12 GB corpus from our own Kaggle notebook outputs, checked against the index. Rebuild the
    derived data. Run `verify.sh`.
  - Loop: league (Q07, 60 threads, about 40 min), BC, gate (Q14), PPO (Q16), delta every 12 h,
    tournament every 8 h (Q17, on the box, not Kaggle), and the public-panel pre-filter (Q18).
- **Kaggle** keeps only delta extraction and the final submission.
- **Laptop** gets candidates. The box writes `policy.bin` plus a report to `RC_PENDING.json`, and
  `aws/pull_rc.sh` fetches them. The laptop then:
  - runs Rust == torch on x86;
  - builds the x86 tar.gz;
  - runs the FULL tournament: 88 public agents, v61, v61.1, v62 and v62.1, 24 worlds, both seats,
    paired and band-scored (§20 G1);
  - runs an official-engine spot check;
  - hands the result to the operator. Submission is never automatic.
- The laptop harness is COPIED into `kaggriculture-rl/harness/`, never imported from the root
  repo.

**The agent.**
- **C1** The chassis and the 64-stage post chain run every turn and are never under RL control (the
  profile only sets market-side knobs). In v61.1 the chassis budget/room/clamp/dead-stock guards are
  off; the post chain does that work.
- **C2** Once a day, the choice is: shield (GroupCtl copied from v62: DIFFERENT, PARTIAL or COPY
  at step 25; D24 mirror guard; hashed per-game jitter) → mask → GRU policy.
- **C3** Opponent identification:
  - the hard rule's group as 3 one-hot features;
  - an aux **family head** on the GRU, labelled from league truth and corpus hindsight (target
    >= 80% by day 3);
  - a separate 3-day MLP only as a baseline.
- **C4** Sell schedule:
  - the rival's sales are exact from public inventory (`rival_sold_*`, 0 mismatches), plus the
    `rnext` head;
  - dayobs v2 adds the rival's sell-hour and "sold before our scheduled sale" features;
  - our own schedule is the tape's, and RL sets only how aggressive RSA, AFR and the race
    horizons are.

## 22. Execution plan to the lock (written 2026-09-25 16:00Z; supersedes §12 and the Kaggle-offload rows of §15–18d)

Architecture: `docs/ARCHITECTURE.md`. Lock: **28 Sep 12:00Z** for the final RC decision; the
submission deadline is 30 Sep.

### 22.1 Build before launch (P0, about 12 h of work, 25 Sep 16:00Z – 26 Sep 04:00Z)

| ID | task | est | proof |
|---|---|---|---|
| B1 | Copy GroupCtl (groups, D24 mirror guard, jitter) from v62 into `crates/agent/src/layers/group.rs`; wire the shield (mask → policy → group fallback) into `switch_profile` | 1.5 h | v62 and v62.1 flag replays give identical banks through the copy |
| B2 | dayobs v2: group one-hot, rival sell-hour, share sold before our scheduled sale; FEAT_VERSION 2 | 1.5 h | live == corpus on 180 games (Q03c) |
| B3 | family head + labels (league truth, corpus hindsight) in `model.py`, `bc.py`, `cache.py`; export includes the head | 1 h | Q10c Rust==torch |
| B4 | PPO fixes: adaptive KL leash, entropy 0, value on detached trunk, Φ(margin/σ_world) reward, anchor-mass logging | 1 h | Q15c smoke + a unit test that the critic step leaves pi unchanged |
| B5 | Opponent pool: v62/v62.1 in-process; top-band medoid tapes (Q20) played by the v61.1 chassis; world priority sampling | 2 h | tapeplay --verify on 20 tapes; pool mix logged |
| B6 | Band mapping of the public roster (G1) and band + world reporting in `gate.py`, `panel_gate.py` and the laptop tournament | 1 h | report shows <2500 losses, 2500+ rate, per-world table |
| B7 | Box pipeline: Q07 league local (60 thr), Q14/Q17 local, `delta.py pull-corpus`, `aws/pull_rc.sh`, lane sizes, RC build removed from the box | 1.5 h | dry run of `queue.py` on the laptop with a scratch root |
| B8 | `harness/` copy (faithful harness, roster, v61/v61.1, worlds) + `harness/rc_tournament.py` for the laptop | 1 h | reproduces 50 stored v62.2 rows exactly |
| B9 | Profile-0 integrity gate (G3) | 0.5 h | 24 seeds bank-identical to v61.1 |
| B10 | Docs: `aws/README.md` and `docs/queue.md` rewritten to match | 0.5 h | — |

After B1–B10: rerun Q03c, Q10c and Q15c on the laptop (about 10 min) before the operator
launches the box.

**Status 2026-09-25 19:50Z (every item verified as stated):**

| ID | status | proof |
|---|---|---|
| B1 | done | GroupCtl copy: v62 banks identical on 72 games (3 configs); shield mask: 240/240 decisions inside their mask |
| B2 | done | dayobs v2 (93 floats: + pos_equal_day0, rival_sell_hour, rival_sell_early): live == corpus on 10,800 rows × 93 = 1,004,400 cells, 180/180 exact replays |
| B3 | done | fam head (5 classes, league truth), torch-only (not exported); BC logs day-3 accuracy |
| B4 | done | adaptive KL leash, entropy 0, detached value (value-only step moves pi by 0 vs 0.117 without), Φ reward, anchor-mass logging, shield mask in the learner's ratio; PPO smoke passes |
| B5 | done (tapes) | real players as PPO opponents: guarded tapes on their own seeds (3,000 training tapes, 10-17 Sep); v62/v62.1 already fixed-profile opponents (13/19). Loss-world oversampling of non-tape games: P1 |
| B6 | done | band gate (988 real players since 18 Sep, one world each): v62.1 baseline 32 losses below 2500, 91.4% at 2500+; wired into the candidate step with the reference-relative rule |
| B7 | done | box pipeline: dry run full chain, jobs survive runner restarts, candidate hand-off tested (hash identical) |
| B8 | done | harness/ copy: 16 games identical to the root repo on the laptop, and identical on ARM |
| B9 | done | profile 0 bank-identical to v61.1 on 24 seeds (and in verify.sh) |
| B10 | done | aws/README.md, docs/queue.md, docs/ARCHITECTURE.md |


### 22.2 Launch and first model (26 Sep about 04:00–07:00Z)

| step | est |
|---|---|
| operator launches c8g.16xlarge (spot) + kaggle.json | 10 min |
| `aws/up.sh`: sync, bootstrap, pull corpus, rebuild, verify | about 60 min |
| league 57.5k (with new opponents) → cache → audit | about 50 min |
| BC → Rust==torch → PPO smoke → BC gate | about 40 min |

### 22.3 Train (26 Sep 07:00Z – 28 Sep 12:00Z, about 53 h)

- PPO nonstop: about 800 iterations, about 3.3M games.
- The branch oracle (B11, P1, 3 h of work, built during day 1) switches on from about 26 Sep 18:00Z
  if the smoke test passes. It is capped at 30% of the box, and its gain is measured on validation
  before it is scaled.
- Delta every 12 h (BC fine-tune + tape refresh).
- Tournament every 8 h, then the pre-filter.
- A candidate that passes the pre-filter goes to the laptop tournament (about 2 h).

P1 during training: B12 herd_safe family as chassis configs, with parity (3 h); B13 livestock
substitution knob and its paired gate (2 h).

### 22.4 Decision points

| when (UTC) | question | if no |
|---|---|---|
| 26 Sep 04:00 | B1–B10 done and laptop checks green? | launch with what passes; slipped items go P1 |
| 26 Sep 07:00 | verify green on ARM? | fix and pin the difference; nothing trains meanwhile |
| 26 Sep 12:00 | BC not worse than v61.1 on the Q14 panel? | retrain BC (league weights / labels) before PPO continues |
| 27 Sep 12:00 | PPO best ≥ v62.1 paired on validation, and on track for the band rule? | turn up oracle share, re-weight the loss worlds; keep the bandit line as the fallback |
| 28 Sep 06:00 | an RC passes the laptop tournament (0 sub-2500 losses, ≥ 80% at 2500+, McNemar vs v62.1 not worse)? | no RL submission; the bandit track's v62.x stays live |
| 28 Sep 12:00 | final pair chosen with operator approval | — |

### 22.5 After the lock

29–30 Sep: ladder watch only, no new builds. The box auto-stops at 29 Sep 22:00Z. The owned-schedule
search (I2) is recorded for later.

### 22.6 Training log (IST)

- **26 Sep ~08:00 IST: PPO stall diagnosed and fixed.**
  - **Symptoms:** after 307 iterations (1.25M games), entropy had moved only 3.33 → 3.30 (uniform is 3.47), and the BC model's top choice still held 5.6% of probability. Validation against v61.1 sat at 0.99, so `best` never moved. The band gate put PPO best level with v62.1 (paired +6/-6).
  - **A.** Validation now plays v62.1 (fixed:19) on its own seeds (600,000+, 800 games), so it no longer shares seeds with the gate.
  - **B.** KL target 0.05 → 0.3. The BC anchor is nearly uniform, so a tight leash pinned the policy to near-random.
  - **C.** Branch oracle (expert iteration, Q19). Every game is deterministic given its seed and choices, so an exact counterfactual is a replay with one day forced. Forcing the policy's own choice reproduced the base game to the dollar in 64/64 cases on the laptop and 24/24 on ARM.
    - Finding: the day's profile changes the margin on only ~16-37% of decisions and flips W/L on ~1.5%. That is why plain PPO's signal is flat.
    - PPO now also takes 16 supervised steps per iteration toward the oracle's best candidate, on decisions where the choice mattered.
- **Bug found by the new validation.** A shield-only group controller supplied profile 0 as a fallback, and that fallback outranked fixed schedules. So every fixed-profile and random-knob opponent in PPO rollouts, PPO validation and the oracle played v61.1.
  - Fixed: a shield-only controller never plays a profile of its own (`GroupCtl::shield_only`). Fixed-profile games are now bit-identical with and without the shield, on the laptop and on ARM.
  - `gate.py` now gates the candidate under the shield, as deployed.
  - The oracle batches made under the bug were deleted, and PPO's `best` was reset (VAL_VERSION 2).
  - Iterations 1-324 trained against a weaker mix than intended. Training continues from iteration 324.
- **Ops lessons.** `sync_up.sh` is now atomic (stage + rsync): a gate once read a half-synced `routes.json`. Rebuild binaries before syncing a task spec that uses new flags. A manual `queue.py retry` restores the automatic-retry budget.
- **26 Sep ~09:00-10:15 IST: anchor to 19/31 → v63 taken → CPU tuned.**
  - **Anchor prior.** Against v62.1, only fixed profiles 19 and 31 hold (0.505); every other profile scores about 0.04. So PPO now anchors its KL leash to a fixed prior instead of the near-uniform BC. Validation vs v62.1 went 0.40 → 0.511. Band gate for snapshot 360: 30 losses below 2500 (v62.1: 32), 91.8% at 2500+ (v62.1: 91.5%), paired +3/-0.
  - **v63 port.** v63 (main repo, v3 profile 71: p19 + V92 predict + deep terminal planner + tsell; +179/-29 vs v62.1) was copied into `crates/agent`: `v92.rs`, `data/v92_lib.bin`, and the new knobs/tail/v9/mod.
    - Verified bank-identical to the v63 build on 24 games (profile 71, and p19_v92 on 47), and on ARM vs the laptop. Profile 0 is still v61.1.
    - New action table `configs/profiles/rl3.json`: the v2 ids 0-31 + 32 p0_v92, 33 p19_v92, 34 p19_v92ext, **35 = v63**.
    - New PPO run `ppo-gru64-f2-p36-from-init-*`, warm-started from the 32-action checkpoint (new rows copied from 0/19). Anchor prior 35:0.5, 19:0.1, 31:0.1, 34:0.1.
    - The reference is now v63 in validation (fixed:35), the band gate and the candidate step. The panel gate always includes v63 and checks `beats_v63`.
    - Q12 (BC fine-tune) was removed: the anchor is a prior now, not a BC teacher.
  - **CPU.** Measured per 4k-game iteration: the PPO update takes ~9 s and the oracle steps ~2 s; the games dominate (v63's 512-sim terminal planner is expensive). Changes:
    - pipelined rollouts (iteration i+1's games play during update i, with the pre-update weights);
    - rollouts on all 62 threads;
    - 8,192 games per iteration;
    - oracle on 24 threads with 256-game batches.

    Box utilisation went from 57-61% to 83% average, with a steady 100% after warm-up. Per-phase timings are now in `metrics.jsonl` (`sec_rollout_wait`, `sec_update`, `sec_oracle`).
- **10:00 IST, 26 Sep: mask bug fixed (iteration 29 of the p36 run).**
  - The allowed-profile mask sat in one u32 slot of `.traj`, so rl3 ids 32-35 (35 = v63) read back as banned in the PPO update: ratio 0, no gradient, and the prior's 0.5 on v63 was renormalised away. 71% of daily choices were ids 32-35.
  - The oracle skipped candidates >= 32. Its targets were also near-uniform: softmax(20·Φ) over Φ gaps of ~0.006 gave a best-profile probability of ~0.10. They included masked profiles, which is why `oracle_ce` read 2e7.
  - Fix: the mask now takes two slots (W = N+5). Oracle targets are limited to allowed profiles and scaled to each decision's own spread (beta 8); the row-weight floor drops from 0.05 to 0.002.
  - Verified on the laptop: only 9 and 11 are masked, and every sampled action is allowed.
  - Validation vs v63 now logs W/D/L. The earlier 0.502 was 34 W / 736 D / 30 L, a near-mirror, not an edge.
  - New metrics `mass`/`played` per profile. Dashboard: `ops/dash/build.py`, polled every 15 min.
- **10:50 IST, 26 Sep: top-player tapes.**
  - Before this, only 33 of the 3,000 training tapes came from the current top 40. The set was capped at 3 per player and 600 per band, from 10-17 Sep only.
  - Added `data/tapes/train/top`: every game played at 2700+ at the time, 15 Aug-17 Sep, no per-player cap. That is 1,348 games from 311 players, 19 of them with 10+ games (up to 93).
  - Seeds already in train or band are excluded, so the gate's worlds stay unseen.
  - Built with `band_tapes.py --top --min-rating 2700`. The per-game rating is used because a team's older, low-rated submissions are not top play.
  - Checks: `--verify` is exact on 1,348/1,348. v63 (guarded) wins 1,337 and loses 11.
  - PPO (30% tapes) and the oracle (70% tapes) load it through the recursive `--tapes data/tapes/train` (4,348 tapes). Rebuilding `train` deletes `train/top`, so rebuild it afterwards.
- **11:40 IST, 26 Sep: "Do all" (operator).**
  - **Which days matter.** The oracle branched all 30 days on 96 games. Days 0-5 and 7-11 never change the game: the shops have not unlocked and the opening plays itself, as the operator noted. D6 matters on 66% of decisions and D12 on 30%. D15-D29 matter on 50-96%, and W/L flips sit at D13-D27.
    - PPO: `--policy-days 6,10,12-29`. The pg, entropy and KL terms train only on those days; the value head still sees all days.
    - Oracle: `--days 6,12,14-28` (17 days, all live) on 40 threads. Before, half its branches (days 0-3) were inert.
  - **Hard tapes.** v63 was scored on all 4,348 training tapes: 63 losses and 732 wins by < $3,000. These 795 go to `data/tapes/hard` and weigh 3x (`train/hard_x2` symlinks), so about 40% of tape games in PPO and the oracle are hard.
  - **Oracle regret metric.** `oracle_regret` (sampled), `oracle_regret_greedy` and `oracle_regret_v63`, in win-probability points against the oracle's best branched profile. This is in-sample on the label window. First reading (iteration 93): greedy policy 0.23, always-v63 0.53.
  - **Tournament reference is now v63** (`KRL_GATE_REF`, default 35). Before, it was v61.1, which every candidate beat, so a PASS meant nothing.
  - **Release packaging.** `build_submission.ps1 -Policy` defaults to `configs/profiles/rl3.json` and ships `configs/shield/v1.json` as `shield.json`. `main_template.py` passes `--shield`. Before this, a policy build shipped the v1 profile table and no shield.
  - **Research (running).** Mining top-player behaviour into new profiles: `.local/analysis/`.
- **11:47 IST, 26 Sep: first tournament vs v63, and validation v3.**
  - All four checkpoints FAIL against v63 on the 9-opponent panel (60 seeds each):
    - best (= i090): 43 better / 56 worse, p 0.23;
    - i070 and i080: 33-34 better / 73-76 worse, p < 0.001, weakest vs aggr_deep_afr (30) and ad_rsa12_l24 (19).
  - A policy level with v63 head-to-head was worse than v63 against the rest of the panel, and the head-to-head-only validation could not see it.
  - `VAL_VERSION` 3: best and rollback follow `val_combo` = (h2h vs v63 - 0.5) + panel delta + hard-tape delta.
    - The panel is the 8 non-v63 tournament opponents, 150 seeds each from 700000.
    - The hard tapes are 800 games from 800000.
    - Both deltas are paired against v63 (profile 35) on the same seeds; the reference is cached per run.
  - Best and drops reset at this change.
- **12:07 IST, 26 Sep: v3 validation crash fixed.**
  - The regret block's local `best` shadowed the best-validation score and was checkpointed as a tensor, so the first v3 validation crashed twice at iteration 100.
  - Fix: renamed to `obest`, and a non-numeric checkpoint `best` now resets.
  - First v3 reading (iteration 100):
    - h2h 352 W / 58 D / 390 L;
    - panel 42 better / 97 worse vs v63 (delta -0.045);
    - hard tapes 7 / 3 (+0.005);
    - combo -0.063.
- **12:22 IST, 26 Sep: oracle labels limited to material decisions.**
  - The late-day oracle produced 3.6x more labels, but their median gain was 0.0004 Phi (~$3), with a median of 5 candidates within 0.001 of the best.
  - Training toward those soft ties spread the policy (entropy 2.0 -> 2.5, v63 share 47% -> 33%, train win 0.855 -> 0.80) against a KL leash raised 11x.
  - Fix: `--oracle-min-gain 0.01` keeps 19% of rows, and `--oracle-steps 8` (was 16).
- **14:00 IST, 26 Sep: fixed v63 variants verified on the box, not the laptop (operator).**
  - New queue job Q21 (gate lane, one-off) runs `aws/verify_variants.sh` on table `configs/profiles/cand_v63var.json` (rl3 + ids 36-52), writing to `data/verify/`:
    - a 600-game head-to-head for 49;
    - the 8-opponent panel paired against v63;
    - the band gate for 51, 47 and 46.
  - The laptop runs only release candidates.
  - Head-to-heads already done on the laptop (600 games vs v63, seed 96000000): 47 won 511, 46 won 500, 51 won 466.
- **14:20 IST, 26 Sep: late top-player tapes follow the delta pull.**
  - New job Q22 (`python/refresh_top_tapes.py`, cpu lane) runs after each Q01 round:
    - takes every game played at 2700+ since 18 Sep, excluding the gate set's games;
    - keeps only tapes that replay exactly;
    - installs them as `data/tapes/train/top_late`;
    - puts v63's losses and wins by < $3,000 in `data/tapes/hard_late`, linked twice into `train/hard_x2` (3x weight).
  - The validation hard set `data/tapes/hard` is unchanged, so the val_combo stays comparable.
  - PPO starts a fresh rollout each iteration, so it picks up the new pool without a restart.
  - Caveat: these games share dates, and so players, with the band gate's window. The gate's own games are excluded, but the gate is less held-out than before.
- **14:45 IST, 26 Sep: option A (top-player labels) running as Q23; offline RL and MCTS kept as options (operator).**
  - Q23:
    - `league-rand` on rl3 (20k games, per-day random profiles) -> `data/leagues/rl3`;
    - `python/learn/idm.py`, an inverse model P(profile on day d | own dayobs d, d+1) -> soft labels on 2700+ corpus seats (`data/train/top_idm.npz`);
    - a fit report (`data/idm/report.json`) with held-out accuracy, confidence on top vs low players, and an out-of-distribution z-score.
  - Offline RL (IQL/AWR on the corpus with the inverse-model labels) is the follow-on if the labels fit.
  - MCTS / play-time search: the time budget is actTimeout 1 s per turn plus a 60 s overage bank per game (`kaggriculture.json`); v63 already searches in the end game (planner 512/2/8 + sale search).
  - A daily profile search at play time must use a rival model (the v92 route forecast), not the true future the oracle uses. Its value has to be measured offline with the model's errors before any build.
- **14:55 IST, 26 Sep: option A closed; play-time search (Q24) and offline RL (Q25) queued.**
  - Q23 (inverse model): profiles cannot be told apart from a day's own behaviour.
    - Held-out league top-1 was 0.043, against a chance rate of 0.028.
    - Real players are far outside our league's states: mean |z| 10.5 for top players and 7.4 for players below 2300, against 0.6 for the league.
    - Top and low players get the same label mix, so imitation through our profiles, and offline RL on those labels, are closed.
  - Q24 `search-eval` (new binary) runs on the 988 band tapes with three learners, each paired against v63 on the real game:
    - ref: v63 every day;
    - model: key-day profile search against a predicted rival (real prefix, then the 2 nearest pool games by requested sales);
    - truth: the same search against the real future.
    - The reference is checked against tapeplay `--pa 35` (identical margins).
  - Q25 `python/learn/offq.py` is offline RL on the oracle's counterfactual table. It fits a linear Q head over the frozen best PPO trunk on the centred Phi of the branched candidates, and exports pi + lambda*q as one policy. Lambda is picked by held-out regret, then checked by the validation composite vs v63 and the band gate.
    - Smoke run (30 files): held-out regret 0.389 -> 0.337 at lambda 10, against 0.468 for always-v63.
- **15:10 IST, 26 Sep: the ladder ranks our agents opposite to our local gates.**
  - Kaggle scores: v61.1 2605 (120 games), v62.1 2532 (189), v62 2520, v63 2470 (85). Each newer agent passed a local gate against its predecessor, and each is lower on the ladder.
  - Q26 `python/ladder_pull.py` pulls every replay of v63, v62.1, v62 and v61.1 into tapes (`data/ladder/<sub>/`). There was never a v62.2 submission.
  - Q27 `python/ladder_analyze.py` then produces:
    - results by opponent band;
    - sim fidelity on our real games (engine verify, plus the agent's own rl3 profile reproducing its real result);
    - counterfactuals for v61.1/v62/v62.1/v63/PPO best on the recorded opponent streams;
    - loss anatomy.
  - The question it answers: which of our lineage wins the games v63 actually lost, and whether our sim reproduces those games at all.
- **15:40 IST, 26 Sep: correction to the "ladder reversal".** On the replays downloaded so far, v63 has the BEST win rate of our agents.
  - Win rate: v63 74% (84 games), v62.1 68% (190), v62 54% (99 so far).
  - Against players rated below 2100, where most of the games are: v63 40-13, v62.1 87-33, v62 24-23.
  - v63's lower rating is most likely convergence: it started at 600 at 02:48Z. It is not evidence of a weaker agent.
  - Every agent loses 25-50% to "below 2100" players, and nearly always by < $3,000. Low-rated teams are often strong public agents, so the rating band is a weak proxy for strength.
- **16:05 IST, 26 Sep: results of the ladder analysis (Q27), play-time search (Q24) and the PPO check on real games.**
  - Fidelity: the engine replays 548 of 549 of our ladder games exactly. Our rl3 profile of each agent reproduces its real margins to the dollar: v63 82/84, v62.1 186/190, v62 156/157, v61.1 113/118. Our sim IS the ladder for our own play.
  - Counterfactuals on the recorded opponents: v63 beats the agent that actually played on every other agent's games (v62.1's +9/-2, v62's +18/-5, v61.1's +20/-1). The local ranking v63 > v62.1 > v62 > v61.1 holds on real ladder opponents; the rating order was convergence.
  - All 22 of v63's losses are against the COPY group (same opening as us): v63 goes 42-22 vs COPY, 6-0 vs DIFFERENT, 14-0 vs PARTIAL. v62.1: COPY 102-41.
    - Only 2 of the 22 losses were ahead at the start of the last day, so the endgame is not where they are lost.
  - Play-time search on the 988 band tapes (Q24):
    - predicted rival: +0/-0 (it never switched);
    - true rival: +4/-0.
    - Profile search adds almost nothing there, because v63 already wins 95% of those tapes.
  - PPO best (i170) vs v63 on all 549 real ladder games, paired: 14 better / 7 worse (sign test p 0.19), 408 vs 401 wins. It is the first RL result pointing the right way on real opponents; the evidence is not yet significant.
  - Q28: exact search on the ladder games themselves, to find which day and profile flips each real loss.
- **16:15 IST, 26 Sep: operator "do all 1-5".**
  - (1) Q29 `python/ladder_split.py` splits our 549 ladder games by episode parity.
    - Even: 282 games in `data/tapes/train/ladder`, with the 200 COPY games at 5x via `ladder_copy_x` links, read by PPO and the oracle.
    - Odd: 267 games in `data/tapes/ladder_val` (193 COPY).
    - Q26 re-pulls every 6 h; Q27 and Q29 follow it.
  - (2) PPO validation v4: + `tape_eval` over `ladder_val`, paired vs profile 35, weighted 2x in val_combo. PPO restarted at 16:12 IST from iteration 207; best and drops reset. The i170 weights are kept as `best_v3_i170.bin`.
  - (5) i170 on the held-out half: 192 vs 192 wins, +6/-6 (COPY +4/-4). The earlier +14/-7 over all 549 came from the other half. No real gain yet.
  - (3) Q28 search on the ladder games is running.
  - (4) Q30 `python/copy_screen.py`: 28 race-knob variants of v63 are screened on the training-half COPY games, the top 3 confirmed on the held-out half, and the best sent to the band gate. The table is generated at `data/gates/copy_cands.json`.
- **16:25 IST, 26 Sep: copy-race screen (Q30) result.**
  - lead32 = v63 with ev_h/dp_h/mp_h 32 (instead of 24), profile id 36 in `data/gates/copy_cands.json`.
    - Training-half COPY: +15/-2 (p 0.002).
    - Held-out ladder half: +12/-2 (p 0.013); held-out COPY +10/-1 (p 0.012).
    - Band gate: +2/-5 (p 0.45); 29 losses below 2500 vs 27.
  - Neutral: race clone / mirror / escalated depth, glut margins, AFR, cxd2, v92_h 72. Harmful: cxd_model 1 (-37), v92_h 24, lead 16.
  - Next: lead32 only when the day-0 group is COPY (the shield's group signal), v63 otherwise.

- 26 Sep 16:40 IST — Q31 (python/copy_group.py): lead32 / lead40 / lead32+rsa16 only against COPY (or COPY+PARTIAL) from step 25, v63 otherwise, via the new `tapeplay --group D,P,C` (the agent's existing GroupCtl) and `band_gate.py --cand-group`. Paired vs v63 on the held-out ladder half, winners to the band gate. First v4 validation (i210): combo 0.0094, ladder +4/-3. Q22 done: 112 late top tapes, 91 hard x2.
- 26 Sep 16:55 IST — Q31 result: lead32 only vs COPY (group 35,35,36): held-out ladder +10/-1 (p 0.012), zero change vs PARTIAL/DIFFERENT; band gate (991 tapes, refreshed 16:24 IST) +3/-2, 32 vs 34 losses below 2500, 2500+ 0.918 vs 0.921. COPY+PARTIAL +11/-2 ladder, +4/-3 band. lead40 and lead32+rsa16 worse on band. v64 candidate = v63 + GroupCtl [35,35,36]; RC build awaits operator approval.
- 26 Sep 17:10 IST — Q28: search on 549 ladder games: ref 401W, predicted-rival search 402W (+1/-0), true-rival 434W (+33/-0). Search value is real but only with a good rival model. New PPO best i220 (combo 0.0235, ladder +6/-3, panel 3/16).
- 26 Sep 19:05 IST — **RC v64rl_lead32copy built (operator: "Do it").** `scripts/build_submission.ps1 -Name v64rl_lead32copy -Profile 35 -Group 35,35,36 -Profiles configs/profiles/v64rl.json` (new `-Group` / template `GROUP`, passed as agent-stdio `--group`). Table v64rl = rl3 + 36 lead32, md5-identical to the Q31 table's ids 0-36. Binary 06c377de52c4. Official-engine check vs the LIVE v63 tarball (seeds 3-5, both seats): vs v9_cem (DIFFERENT) identical to v63 6/6; vs the v63 mirror (COPY) v63 ties 3/3, v64rl wins 2 (+164, +46) and ties 1; vs v61.1 both win, v64rl by more (+259 vs +44, +976 vs +904). 0 fallback turns, worst turn 84 ms. NOT uploaded or submitted: dataset upload + private notebook + submit need operator approval.
- **26 Sep 20:09 IST — SUBMITTED (operator-approved) from the private kernel `kaggriculture-private-submission-trackp`:**
  - v63.1_rl = 56581792 (kernel v11): v63 + lead32 vs COPY (`-Profile 35 -Group 35,35,36`, table v64rl), tar sha c2bac255, binary 06c377de52c4.
  - v63.2_rl = 56581889 (kernel v12): PPO i280 (`snapshots/i000280.bin`, sha de00b7f4) + rl3 + shield v1, tar sha 5487f0bc. Pre-submit: band gate +6/-6 (34 low losses, same as v63); 19:44 IST tournament FAIL 0.837 vs v63 0.852 (+12/-22, p 0.12).
  - Laptop official checks: both 18/18 DONE, 0 fallbacks, worst 94 / 86 ms. Kaggle notebook validation episodes: DONE/DONE, 719/719 bridge turns, 0 fallbacks, worst 127 / 132 ms, VALIDATION PASS. Kaggle validation episodes COMPLETE for both.
  - Payload: new PRIVATE dataset `debmalya84/kaggriculture-rl-payload` (flat `v63_{1,2}_rl.submission.tar.gz.bin`), not the shared bandit payload. `build_private_kernel.py` gained `--slug` / `--payload` and a stricter validation cell.
  - Active pair is now v63.1_rl + v63.2_rl; v63 and v62.1 retire. Q26 ladder pull now includes both new subs.
- **26 Sep 21:40 IST — operator GO for the top-50 plan (https://claude.ai/artifact/S8X7nQTTTQuscoFw1CBjNL).** Operator rules: decompile the top teams' strategy and compare with ours (inherit only what beats ours, measured by running our live agent in their exact situations = shadow mode + exact-engine branching); most top teams are per-day / per-move planners or RL; no Kaggle notebook for data (box/laptop fetch directly).
  - F1 done (Q35 `python/restore_pool.py`, follows Q20/Q22): train/top 1,355, hard_x2 2,567 links (hard x3, hard_late x2), train/ladder 315 + 876 COPY links. Root cause: Q20 (band_tapes rmtree of data/tapes/train) at 16:24 IST.
  - F2 partial: `band_gate.py --ref-profiles/--ref-group` (v63.1_rl = v64rl.json, 35, group 35,35,36).
  - Data: corpus has GM through 22 Sep and only ~600 official games/day after, so this week's top 50 have a median of 11 games in 7 days. Q36 `python/top50/fetch.py`: leaderboard API -> EpisodeService crawl (ListEpisodes by submission; token auth; returns both agents' submission/team ids and ratings; box uses ~/.kaggle/access_token as Bearer) -> last 7 days, <= 80 games per team, replays (~32 MB) converted to tapes and deleted.
  - Tracer `trace` bin (exact 20/20) for per-turn state; `python/top50/extract.py`/`features.py` written earlier (extract selection superseded by fetch.py).
- 26 Sep 22:50 IST — Sales shell pipeline (crates/agent/src/shell.rs + `branch` bin + python/top50/{branch_all,train_shell,eval_shell,dagger}.py; tapeplay --shell/--record; agent-stdio --shell). Labels: 58,680 older top-player decisions; top player better than our shadow agent on 16,854, ours on 12,852. r0 applied every turn +7/-748 (one-turn labels compound) -> direction limit + game-level tau + opponent-group gating. r1 (after DAgger r1) sell-less tau 0.8: held-out ladder +33/-10 (p 0.0006) but band gate +20/-29 (40 vs 32 losses < 2500): helps vs the ladder we meet (copies), hurts the wider field. Selection now on train-half ladder + 600 training-band tapes; DAgger pool excludes our ladder games.
  - GM dataset: full copy at D:/gm_dataset (26 GB, v 26 Sep 01:38Z) on the LAPTOP only; python/top50/gm_extract.py (laptop) -> data/top50/gm_local, engine 1.32.7 only (ladder engine since 16 Aug); scp to box; Q47 (labels) and Q50 (analysis) wait for READY. Top-100 in GM (1.32.7 era): 7,620 player-games, 97 teams.
  - Game 113764426 (v63.x lost to Lam Dang, rank 2,049, 1,753): world YARN_STORE|YARN_STORE; opponent bought sheep in a block on day 6-10 (8 by d6, 16 by d10) vs ours 4/10; our day-10 "lead" was unspent cash; equal wool collected (~413 each). Economy lesson: commit to sheep when both shops want wool.
- 26 Sep 23:25 IST — Faster tapeplay/branch (from ../docs/history/fast-tournaments-2026-09-26.md, copied from rustengine/v62): v92 `forecast_fast` (bitsets) + `Obs::from_state` (no JSON round trip; KAGG_OBS_JSON=1 restores). Verified bit-identical: tapeplay on 60 band tapes md5 a1cbfdf6 before/after, branch labels md5 8d0b45a3 (KAGG_OBS_CHECK clean); 25.0 s -> 11.9 s (2.1x). Also: tapeplay --force-route / --route-table, agent-stdio --route-table, band_gate --cand-args; Q52-53 decompiled (imitation) shells, Q54 world route-table refit.
- 27 Sep 00:00 IST — Identity: `crates/agent/src/disguise.rs` (tapeplay/agent-stdio `--disguise`): zero-quantity SELLs of products we hold none of, appended at the end of the market list on a per-game pattern. Verified: banks identical on 60 games (md5 a1cbfdf6 both ways), stream hashes differ in 60/60 games at turns 24/136/719. For the next release. Opening divergence screen Q56 (`--opening-route K`, router.opening). tapeplay `--takeover T` for twin games (Q55).
- 27 Sep 00:40 IST — **First shell no worse than v63.1_rl on real players.** Hard DAgger (dagger.py --hard: only games v63.1_rl loses or wins by < $3k; 1,046 games): r5 all sell-less 0.8 ladder +22/-10, band +7/-9; r5 COPY-only 0.8 ladder +15/-5 (p 0.04), band +5/-6 (34 vs 32 < 2500). r6 (chosen COPY-only sell-less tau 0.9): held-out ladder +10/-4 (235 vs 229 wins), band +6/-4, 32 losses < 2500 (= v63.1_rl), 2500+ 92.4% vs 91.8%, no_worse_than_ref TRUE.
  - Candidates built (not submitted): v63.3_rl = v63.1_rl + stream disguise (official engine 18/18 identical to v63.1_rl, 0 fallbacks); v63.4_rl = v63.1_rl + disguise + shell r6 (shell.json sha 2340ab95), binary 38cf8d44.
  - Grafted top-team openings (Q57): 7/12 collapse, 4/12 identical; route refit +0/-0; imitation shells neutral.
- **27 Sep 04:15 IST — PPO passes the tournament; best candidate = v63.4_rl.** Tournament round 3 (22:13Z): PPO best (i630, sha aa3f7174) PASS vs v63, 0.869 vs 0.852, +14/-4 (p 0.031), the first PASS. vs v63.1_rl: band gate +13/-7, 27 vs 32 losses below 2500, 2500+ 92.1% vs 91.8%; held-out ladder 283 vs 291 (weaker vs copies). + shell r6 (COPY-only sell-less 0.9): ladder 291 = 291 (COPY +13/-10), band +13/-7, 27 losses < 2500. Built v63.4_rl = PPO i630 + shield v1 + rl3 + shell r6 + disguise (binary 38cf8d44); official engine: better vs copies (v63 mirror +1,051/+137/+124 vs v63.1_rl's +164/0/+46), identical vs v9_cem, 0 fallbacks, worst 86 ms. Not submitted.
- 27 Sep 07:40 IST — Ratings + losses. BT fit (Elo 400 scale; opponents' leaderboard score as their rating; EpisodeService rate-limited): v63.2_rl (PPO i280) 108 games 95-13 vs median-1774 opponents -> 2346 (95% 2226-2477, rank ~572); main v63.1 bandit 74 games 43-31 -> 2422 (~416); v63.1_rl 92 games 61-31 -> 2466 (~319); v63 112 games 77-35 -> 2507 (~238). (The corpus-calibrated logistic scale 890 is too flat to rank with.) Loss anatomy (python/loss_tapes.py, Q61 after each Q26): 292 losses of our subs; 205 vs COPY, 240 within $3k, 225 last led in days 25-29 (end-game race); PPO's 13: 10 final-day COPY losses under $1k, 2 big DIFFERENT (Lam Dang wool, Humanitis strawberry), 1 vs our own sub. Losses now 5x in PPO's pool (train/losses_x, 1,460 links, restored by Q35). Conversion gate (python/loss_gate.py, Q62 every 2 h): of 293 real losses, v63.1_rl wins 81, v63.3_rl 102 (+23/-2), PPO best 71, PPO best + shell 88, v63.4_rl 90 (+25/-16).
- 27 Sep 08:10 IST — Tuning round (Q63-Q65): end-game layer knobs flat (terminal start/sims/passes/props, terminal-sell timing: +0/-0..-1 on 293 real losses); v92 top3 / window 8 convert more losses (+15/-9, +14/-9) but band worse (35 vs 32 losses < 2500); PPO i630-i780 + shell r6 all 27-28 losses < 2500, 86-91 losses converted; bigger shell big1 (3x128): on v63.1_rl ladder +14/-3 (p 0.013), band +5/-2 (32); on PPO i630/i690/i720: 92/93/90 converted, 27 < 2500, ladder 289 vs 291. Plateau: v63.4_rl (i630 + r6) stays the candidate.

## 2026-09-27 — reactive shell v2 (design docs/rshell-v2.md)
Learned, config-driven sale controller after the chain; labels from exact closed-loop replays on the
64-world seed bank; settings screen (switch OFF only if no worse everywhere and better somewhere); CMA-ES
54 dims over 64 worlds per generation; held-out per-world test; DAgger. Queue Q80-Q88.

## 2026-09-27 — PPO2 (items 1-6, operator-approved)
New run beside Q16 (control): `ppo.py --root weights/ppo2` (Q70) and its own oracle (Q71, data/oracle2).
1 oracle restarted on the current policy, days 23-29 incl. the hour-13 slots, closed-loop mirror/v63/rand games;
2 opponent mix `--mix` (mirror = the learner's own policy) + `--seed-bank` (uniform over the 64 realized worlds),
tapes 15% (losses_x5 kept), lineage snapshots (i300-i790) seeded into snapshots/; 3 KL anchor + warm start = i790
(`--anchor-net`, `--init-from X.bin`), lr 1.5e-4 halving every 400 iterations (`--lr-half-life`); 4 sigma 2000,
phi-mix 0.3; 5 `--rshell/--chain-off/--knob-over` passed to every learner rollout/validation (off until the shell
passes); 6 mid-day decisions (`--mid-days 25-29`): chronological slots (crates/dayobs slot_of), day feature d+0.5
at hour 13, profile applied at once (Base::apply_profile_now), MID1 trailer in the export so every consumer agrees.
