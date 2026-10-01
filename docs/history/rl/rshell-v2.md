# Reactive shell v2 — design, feature map, implementation (2026-09-27)

The v61.1 tape agent's ~65 rule layers stay in the chain (crates/agent/src/layers, a line-for-line port).
Shell v2 is ONE learned, config-driven sale controller after the chain, built for the PPO-driven agent:
PPO picks the day's knob profile; the chain proposes the turn's action; shell v2 decides, per product,
how much of the projected shed to sell now and in which slot order. Code: `crates/agent/src/rshell.rs`.

## 1. Where it runs
* Play time: inside the Rust agent binary (`agent-stdio`, the submission tarball), plain Rust matmuls, no
  Python. ~45k weights; the network and lineage table ship as JSON next to `rshell.json`. Measured worst
  turn with the shell on: ~38 ms (budget 1 s).
* Training: PyTorch on the box (`python/rshell/train.py`); labels from the Rust `branch2` bin on the box.

## 2. Rules vs learned
| v61.1 part | shell v2 | how |
|---|---|---|
| chassis repairs (hand_align, weed_repair, sell_lead, front_run, R36 debts, budget/room guards, clamp_sells, dead_stock, terminal liquidation) | rule, unchanged | `chassis.rs`; sync safety |
| economy projects (V219, V231, V233, R51 tour + warehouse, R85/R95/R97, CA, HD2/CS/SR, Y, courier, carrot, fert, opening, pipe, CH, WL, e402) | rule, whole-game switch | `--chain-off NAME` (`knobs::ECON_STAGES`), verified by the screen |
| sensing (layout similarity, clone gate, step-1 mirror, group @25, cluster @144, race lost, pre-emption, R44 probe) | model INPUTS | `RMem` + `ChainSig` |
| sell timing (R36/R37, RACE, V9 race, OR2, V44Y, CXD, v92, AFR, EV/DP/MP/FX/MPX/BD/WB3, ADV/RSA, tsell, overflow) | rule proposal with tunable knobs + their signals as inputs; the shell has the final say on quantity and order | `knobs.rs` (new constants), `--knob-over`, `--chain-off NAME` (`knobs::MARKET_STAGES`) |
| terminal planner (712+) and step-718 rescue | rule | shell window `to` <= 717 (tunable) |
| disguise | rule, last | `disguise.rs` |

## 3. Inputs (94 per item = 50 item + 35 global + 9 one-hot)
Global (`GLOBAL`): day, hour, hours_left, hour23, to_term, money_gap, our_cash_24, rival_cash_24, group one-hot
+ known, cluster_m (day-5 still on our squares >= 90%), clone_gate, poseq_6, poseq_24, similarity, mirror,
lineage (lin_sim, lin_same = rival on OUR route, lin_streak, lin_known, lin_fit), race_h, r37_h, race_lost,
afr_24, v92_on, today's profile knobs (race_clone, afr_on, rsa_look), room, carried, overflow (hour 23), profile id.

Per item (`ITEMF`), 9 items = WHEAT CARROT TOMATO STRAWBERRY MELON EGG MILK WOOL FERTILIZER:
stock (projected shed after this turn's unit actions), raw shed, in hands, share; market inv, px, px/base,
px/mean24, dpx, dinv, glut; shop_item, town_rate (24-step draw), to_draw; rival sales recovered from
public data (Δinventory + town draw − our sales): last step, last 24, ticks, since; rival_tiles/rival_ready
(visible production); preempt_24 (rival sold while we held with a sale planned); forecasts: v92 (this
turn), lineage route's tape 1/4/8/24; our plan 1/3/8/24 and steps to it; chain's qty, slot, class, #sells;
R36 debts; price calculator rev_0..4, best_wait, pf_1/4/8/24, impact; fertilizer need and surplus.

## 4. Price calculator (requirement 4)
Per item per turn: inventory simulated H (tunable, default 48) steps with the rival forecast (mix `fc_w`
of copy = our own tape, v92, lineage tape, recent rate) and the exact town draw; engine price curve
(`market::price_i`, cumulative sums so every query is O(1)). For each fraction: revenue of selling it now
(unit by unit) + the rest at the best later step; nothing past 718 counts. Feeds the net and the prior
`beta_price * (rev_c − max) / (stock * base)`.

## 5. Model
```
x (94) → norm → enc 94→128→128 (ReLU, shared by the 9 items)
       → [e_i, mean_j e_j, max_j e_j] (384) → head 384→128→11
       → 5 class logits | priority | 5 predicted margins
logit_c = model_w·net_c + bias_c + stay·[c = chain's] + beta_price·price_adv_c + urge·(2 frac_c − 1)
urge    = Σ urge_w · {clone, copy, lineage, preempt, rival_sold_1, fc_lineage, fc_v92, glut, late, money_behind, room_tight, partial}
act     if best ≠ chain's, p ≥ tau + group_tau, direction allowed (group_mode), glut floor, margin gate
order   reorder: our SELLs permuted among their own slots by priority (+ prio_urge·urge); front: urge ≥ front → ahead of all orders
```
Hard limits: only SELL quantities/order; clamp to projected shed; fertilizer never below committed need
(R51 undelivered + V219 dedicated + tape PICKUPs) + `fert_reserve`; 10-order cap; off at ≥ `to`.

## 6. The five protections
| requirement | inputs | levers | trained / scored on |
|---|---|---|---|
| clones | clone gate, poseq, mirror, group, cluster, similarity; copy forecast | sell ahead, slot 0, per-group tau/mode | closed-loop labels vs MIRROR (i790+big1) |
| same lineage | lineage table (41 routes × 30 days, `configs/lineage/v61.1.json`, `lineage-table` bin) + sale-timing fit over the last 240 steps → the rival's route and its tape as forecast | sell ahead of / wait out its dumps | closed loop vs v63, v62.1, v62, v61.1, v63.1_rl, rand-knob clones |
| front-running | recovered rival sales, preempt_24, v92/lineage forecasts | pull forward, front | AFR stays as a rule; shell beyond it |
| optimal price | price calculator | spread over turns, wait out gluts | beta_price, fc_w, horizon by CMA-ES |
| fertilizer | FERTILIZER item, need, surplus | sell surplus | hard floor; R85/e410 stay |

## 7. Pipeline (ops/tasks.json Q80–Q87)
* Seed bank: `data/worlds/w64_bank.json` (64 worlds = first-two-shop pairs, 131–185 train seeds and 3 held-out
  per world; built in the main repo, verified 192/192 on our engine).
* Q80 closed-loop labels (`branch2 --selfplay`): 64 worlds × 16 seeds, 7 opponents, 24 decisions × 5
  fractions, each an exact replay from step 0 with both agents reacting. Q81 open-loop labels on real tapes.
* Q82 train. Q83 settings screen (`python/rshell/screen.py`): every switch off alone; OFF only if no worse on
  the closed-loop panel and every tape set and better on one; the combination re-checked on fresh seeds.
* Q84 CMA-ES (`python/rshell/cmaes.py`, own CMA implementation): 54 dims, λ 16, 40 generations, all 64
  worlds per generation (rotating seeds), fitness = paired (better − worse) vs i790+big1.
* Q85 held-out test (`python/rshell/test.py`): held-out seeds per world, held-out ladder, losses split B,
  band gate, latency. Q86 DAgger round (`python/rshell/dagger.py`). Q87 corrected band reruns.

## 8. Verified so far
* With the shell off (`{"on": false}`) and an empty knob-over, tapeplay (293 loss tapes, v63.1_rl and PPO
  i790+big1) and selfplay (64 seeds) are byte-identical to the previous binaries.
* Fixed: `band_gate.py` dropped `--cand-args` whenever `--cand` was given, so every "PPO + shell" band
  number before 2026-09-27 was PPO WITHOUT the shell (Q87 reruns them).
