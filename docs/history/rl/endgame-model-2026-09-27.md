# Learned endgame controller (rlV2 workstream D), 27 Sep 2026

Operator (27 Sep): "endgame also has to be a model or included in PPO". PPO3 carries endgame *option rows*; this is
the separate learned endgame model, trained on exact counterfactual labels.

## What it is

`crates/agent/src/endg.rs`, enabled with `--endg weights/endg/<ver>/endg.json` (agent-stdio, selfplay, tapeplay,
band gate: every runner that goes through `runner::apply_extras`). Off = byte-identical agent.

- **Decision**: once per game, at step `from` (649 = day 27 hour 1, right after the day's profile switch).
- **Proposals** (`configs/endg/proposals.json`, K = 24): patches over the endgame knobs, each applied on top of
  whatever profile PPO picks on every later step, from the proposal's own start (`from`, default 673 = day 28).
  Proposal 0 = the agent as-is. The set covers the terminal closure planner (off), the final-day sale advance
  (tsell: from 672 / 684 / 696, windows, off), and the market race of the last days (RACE clone horizons fast /
  slow, from day 27 or 28; AFR anti-front-run; RSA look; lead-sell fraction; V9 race horizon; ADV; lead-sell
  windows off; v29 pressure) and two combinations.
- **Inputs** (NF = 140): the reactive shell's 35 GLOBAL inputs + 10 per-item inputs x 9 items (+ presence flag) of
  the previous turn, + 6 knobs of the active profile. When the agent has no shell v2, an observe-mode shell
  (inputs only; verified byte-identical) is installed to compute them.
- **Two models** (small ReLU MLPs, JSON, run in Rust):
  - *proposal*: x -> K logits (prior over which proposals are worth considering in this state);
  - *objective*: x -> 2K (win logit per proposal, margin delta vs proposal 0 in $2,000 units).
  - Rule (`mode: rerank`): the top `top_m` proposals by prior + proposal 0; argmax of
    u = sigmoid(win) + lam * tanh(dm); kept only when it beats proposal 0 by `gate`. `replace` = argmax over all K.

## Labels (exact, closed loop)

`crates/runner/src/bin/endg-label.rs`: each game (w64 bank train split, seed x seat x opponent) is played once per
proposal with that proposal forced (`--endg-force K`). The agents are deterministic and the prefix to step 649 is
identical, so each replay is an exact counterfactual in which the opponent keeps reacting. Opponents: mirror
(v63.6_rl), v63.5_rl, v63, v62.1, v62, v61.1, v63.1_rl, random clone-lineage knobs.

## Terminal planner decision log

`KRL_TERM_LOG=<file>` (env; default off) appends one JSON line per planner call (`ev: plan`: step, seat, money,
config, baseline value, every proposal with actor / offset / heuristic score / stops / route length / simulated
value / dominates / better, accepted) and per skip at the start step (`ev: skip`, why = shadow | baseline).
Verified: 24 games v63.6_rl vs v63 bank-identical with the log on, with `--endg` in observe mode and forced to 0.

## Finding: `term_start != 712` switches the planner off

The planner's shadow requires the chassis's final 9-SELL liquidation markets on every shadow step, which the tape
only emits from step 712. Any other start fails the shadow, so the planner never runs: in the first 381 labelled
games `term_start` 700 / 704 / 708 / 716 were identical to `term_on: false` (planner worth ~$18/game; without it
the result is worse in 56 of 381 games and better in none). PPO3's rl4 option rows `term700_s1024` / `term704_s256` are therefore "planner off".

## Label headroom (7,680 games, 24 proposals)

Base win 0.784, oracle (best proposal per game) 0.864. Almost all of the headroom is against near-copies of the
agent: mirror (v63.6_rl) and v63.5_rl (base ~0.50, mostly draws -> oracle ~0.79) and a little against v63.1_rl
(0.43 -> 0.52). Against v63 / v62.1 / v62 / v61.1 / random clones NO proposal changes a single result. The RACE,
AFR and V9-race patches never change anything from day 27/28 on (identical to proposal 0).

## Model v1 (`weights/endg/v1/endg.json`, sha 93b1f196f5e8)

Trained on both label batches (Q120 + Q123), up/down objective, hidden 32, early-stopped. Rule tuned on 20% held-out
seeds, score = better - 3 x worse with the FIT part non-negative: gate 0.2, top_m 24 (the prior did not help
over considering every proposal; the proposal net is exported and used only when top_m < K). Validation:
+66/-0 (switch rate 8%, all flips in copy games). Picks: base 6,905 of 7,680; lead_full 411, rsa_look9 124,
tsell684 74, leadsells_off 52, others < 35.

Rust vs Python parity: 64 decisions, 0 pick mismatches, max utility difference 7.5e-7.

## Paired test vs v63.6_rl as-is (held-out bank seeds)

| test | v1 (gate 0.2) | v1, gate 0.05 |
|---|---|---|
| lineage closed loop, FIT (v63, v62.1, v62, v61.1, v63.1_rl, rand) x 64 worlds x 3 seeds x 2 seats | **+0/-0** | +0/-6 (p 0.03) |
| mirror (v63.6_rl itself) | **+102/-2** | +134/-16 |
| v63.5_rl (the live submission) | **+70/-0** (wins 127 vs 57 of 384) | not run |
| public-25 loss tapes (611, open loop) | +0/-0 | +0/-0 |
| band gate (991 real tapes): losses below 2500 | 27 (ref 27), vs v63.1_rl +17/-8 (ref +17/-8) | 27 |
| worst turn, 32 games, native box build | 8.8 ms (ref 8.8 ms) | 8.8 ms |

Verdict: v1 is neutral on every non-copy measure (lineage, public-25 losses, band: identical results) and strongly
positive against copies of our own agent (+102/-2 vs a clone of v63.6_rl, +70/-0 vs v63.5_rl). The low-gate
variant starts to lose lineage games and is rejected.

Enable: `--endg weights/endg/v1/endg.json` on the v63.6_rl args (the tarball needs the file and the flag added
to main.py's command line, like `--shell`).
