# agent v22 — `agents/v22_bandit.py`

Built 2026-08-09. v19's base route (`90525850_s0`) plus the first adaptive
layer to survive paired validation: in-game opponent-program identification
feeding a bandit over trained counter-schedule arms.

## Architecture

- **Base**: v19's full route (field, purchases, default sell schedule) and its
  complete safety stack (weed repair, shed clamp, sell-first, price-impact
  ranking, terminal liquidation).
- **Identifier**: the opponent's cumulative sales reconstructed exactly from
  the shared market inventory (town consumption transcribed from the
  interpreter; only $1-floor sales are invisible), matched each turn against
  36 distinct top-100 programs clustered from 90 fit-window routes.
- **Bandit**: commits to a counter-arm after two consecutive decisive reads
  (never before day 9), abandons it after 24 turns of non-confirmation.
  Two arms ship — `raj` (Raj Aryan, rank 3) and `seb` (Seb rank 1 +
  Schindler) — each a retiming of the base's own sells from turn 192 on,
  trained by (1+λ) evolutionary search against tapes of the opponents that
  beat v19 live, full-episode objective, crowd guards.

## Trained but deliberately not shipped

| layer | training result | why cut |
|---|---|---|
| `tt` arm (our own lineage) | fitness −4,458 → +97,328 | does not generalize: half the ladder looks TT-like mid-game; costs more vs lookalikes (−$1.4–2.1k) than it earns vs the real family (+$200) |
| `tao` arm | −17,344 → −6,379 | reduces training loss, no out-of-sample effect |
| threat-first ordering | ~$0 in v21 | fires on indecisive matches; −$1.8k vs out-of-library programs |
| feed-buy pull | — | byte-identical results; no effect |

## Certification (416 paired games vs v19, identical seeds, both seats)

Roster = 8-opponent hard panel + all 18 tapes of opponents that beat
v19/twin on the live ladder.

| | v19 | v22 |
|---|---:|---:|
| wins | 123/208 | **124/208** |
| mean per-opponent margin | +6,636 | **+6,739** |
| regressions | — | **zero** |
| David Schindler15 | 3/8, −2,700 | **4/8, −2,475** |
| Seb panel tape | 2/8, −225 | 2/8, **+79** |
| fmind | +4,524 | **+5,154** |

Gate: self-play DONE 145,684/145,684; latency 0.18 ms mean / 9.05 ms worst;
59,431 bytes; contract tests pass.

## Diagnosis it was built from

All 293 of our live ladder games, every loss labeled by nearest program family:
v19's losses are 62% to fresher editions of its own lineage, then tao wu11
and Raj Aryan. See BUILD_JOURNAL 2026-08-09 and the notebook's section 3c.

## Tools added

`src/kaggriculture/train/train_arms.py` (arm trainer), `src/kaggriculture/agentbuild/v22_agent.py` (assembler),
`.local/v22/` (diagnosis, loss tapes, validation, EDA data).
