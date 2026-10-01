# The Rust agent (v63.x)

Short reference. The full module-by-module description is [agent-architecture.md](agent-architecture.md).

Source: `crates/agent`. Binary: `agent-stdio` (one observation JSON per line on stdin, one action JSON per line on
stdout). Configuration: one `agent.json` (`agent-stdio --config agent.json`; `--dump-knobs` prints every manager,
stage and knob).

## Turn pipeline (`base.rs`, `Base::act`)
1. **Chassis entry** (`chassis.rs`): the router picks a route for the observed world, the route tape gives the
   turn's actions, and the chassis repair guards make them legal.
2. **Race signal**: optional `lead_price` / `lead_signal` computation (which items are actually in a sale race).
3. **Dispatcher** (`dispatch.rs`): a specialist may override the policy / endgame model for the rival group.
4. **Layer chain** (`layers/`): pre phases then post phases of the 65 chain stages (`layers::CUTS`), each owned by a
   manager; stages can be switched off per manager (`stages_off`) and tuned by knobs.
5. **Sale stack**: reactive shell (`rshell.rs`) → learned sales shell (`shell.rs`) → sale manager
   (`managers/sale.rs`, batch / tranche / front-run) → **cash floor** (restores route sells the shells cut while cash
   is short early).
6. **Rival stack**: top-player market overlay (`tpp_mkt`, off by default) → pre-emption of the rival's forecast
   sales (`preempt.rs`) → game-theoretic liquidation (`gt.rs`) → premium sell / forced buys.
7. **Final check** (`managers/market_guard.rs`): legality and caps on every order; then the stream disguise.

## Chassis (`chassis.rs`)
Route tapes + router + repair guards. Guards (on by router settings):

| Guard | What it repairs |
|---|---|
| `hand_align` | keeps per-hand actions aligned with the hands actually hired |
| `weed_repair` | clears weeds the route's plan runs into |
| `budget_guard` | drops buys the bank cannot cover |
| `room_guard` | keeps shed deposits within capacity |
| `clamp_sells` | clamps SELL quantities to what is in the shed (never cancels a route sale) |
| `land_repair` | re-queues builds/plants aimed at still-LOCKED land and buys the missing land when affordable |
| `animal_guard` | (off) limits animal buys while animals sit unplaced in the shed |

The router (`router.rs`, `router.json`) maps the observed world to a route at `select_step` (step 144):
`shop_routes_new/old` per world, optional `cluster_routes` overrides (`world#cluster`), defaults. Route ids index
`routes.json` in the chassis base folder.

## Managers (`managers/`)
Each manager owns a set of chain stages (`STAGES`) and knobs (`KNOBS`); only the owner may set them.

| Manager | Owns |
|---|---|
| `market_guard` | sell_lead (one-step-early sale + suppression of the duplicate), R36 window, racepx glut gate, front_run, dead_stock, terminal_liquidation, final_check |
| `sale` | sale-timing chain stages (experiment, order, r36, r37, release, v9_race, racepx, racegate, or2, wb3, fx, dp, mp, ...), `rshell`, `shell`, `batch`, `cash_floor` |
| `endgame` | endgame chain stages, learned endgame model (`endg`), game-theoretic liquidation (`gt`) |
| `rival` | pre-emption (`preempt`, `preempt_over`), group knob overlays (`group_knobs`, per rival group DIFFERENT/PARTIAL/COPY) |
| `clone` | clone/mirror race knobs, stream disguise |
| `economy` | economy stages (v219, v231, v233, herd, fert, opening, ca, hd2, cs, y, pipe, ...) |

## `agent.json`
Paths are relative to the file. Every section is optional; a manager with `"on": false` is removed whole.

```json
{
  "base": "base",
  "controller": {"profiles": "profiles.json", "policy": "policy.bin", "shield": "shield.json",
                 "knob_over": "knobs.json", "dispatch": "dispatch/dispatch.json"},
  "managers": {
    "market_guard": {"on": true, "settings": {"sell_lead": true, "r36": true, "racepx": true},
                     "knobs": {"racepx_margin": -10}},
    "sale":    {"on": true, "stages_off": ["r127", "sm"], "rshell": "rshell/rshell.json", "shell": "shell_open.json",
                "batch": {"on": true, "items": [], "front": true},
                "knobs": {"wb3_mode": 2},
                "cash_floor": {"money": 1000, "until": 288}},
    "endgame": {"on": true, "endg": "endg.json", "gt": "gt.json"},
    "rival":   {"on": true, "preempt": "preempt.json", "group_knobs": "group_knobs_rpx0.json", "group_knobs_for": [2],
                "preempt_over": {"to": 711, "p_min": 0.2, "gate_w": 0.5, "floor": 0.3, "look": 2}},
    "clone":   {"on": true, "disguise": true},
    "economy": {"on": true, "stages_off": ["r95", "pipe"], "knobs": {"hd2_on": false, "cs_on": false, "y_on": false}}
  }
}
```
This is the v63.17 configuration. `managers/mod.rs` documents every key.

## Fixes that shaped v63.14–v63.17
- **land_repair** (v63.14): a missed BUY_LAND after a cash dip made every later build/plant target LOCKED land and
  fail silently, stranding animals; the guard buys the land and retries.
- **wb3_mode** (v63.16): `wb3` moved our wheat buys to the front of the queue unconditionally in BRUNCH worlds,
  starving cash-tight routes. Mode 2 = only on the rival-wheat signal and only if cash covers the route's other buys.
- **cash_floor** (v63.17): the sales shell deleted early route wheat sales while cash was ~$10; hires then failed,
  crops died and the farm never recovered. While money < $1,000 before step 288, route sells cut by the shells are
  restored.
- **routes**: per-world route screens with held-out validation (see [workflows.md](workflows.md)).
