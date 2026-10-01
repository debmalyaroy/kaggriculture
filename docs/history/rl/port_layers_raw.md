## v61.1 (agents/v61.1_bandit.py): agents/v61.1_bandit.py
7583 lines, 69 layer captures, 2 exec() payloads

| # | line | layer | wraps | wrapper | knobs | header |
|---|---|---|---|---|---|---|
| 1 | 992 | _SHOP | `agent` | agent |  |  |
| 2 | 1238 | _V219 | `agent` | agent | FERTILIZE=True | V219: a finite late tomato investment with dedicated, observed workers. |
| 3 | 1476 | _EXPERIMENT | `agent` | agent |  |  |
| 4 | 1508 | _ORDER | `agent` | agent |  |  |
| 5 | 1538 | _V231 | `agent` | agent | CAP=4 | Apache-2.0; later cattle transfer from prvsiyan, Moon (2026-09-10). Bounded livestock substitution; confirm owned animals before redirecting |
| 6 | 1652 | _R36_SALE | `agent` | agent |  | EXP-167, adapted from Dmitrii Gluzdov's Two Coins, One Sheep (Apache-2.0). Reserve only physically available stock after the final parent wo |
| 7 | 1878 | _R37 | `agent` | agent | PRICE_FLOOR=1, HINGE_GAIN=8.0, ADAPTIVE=True, QUOTE=True | EXP-168: adapted from lucifer19 / Harvest Nocturne, Apache-2.0. All rivalry features use public occupied tiles; no private rival inventory. |
| 8 | 1924 | _RELEASE | `agent` | agent | ERRORS=0 | Export guard: normal decisions stay identical to the frozen screened policy. |
| 9 | 1950 | _V233 | `agent` | agent |  | Adapted from prvsiyan / The Soil Remembers Rain, Apache-2.0. V233: bounded, financed six-sheep SE discovery investment. |
| 10 | 2123 | _R46_SHADOW | `_shadow_terminal` | _shadow_terminal |  |  |
| 11 | 2144 | _R51_INPUT | `agent` | agent | MAX_WORKERS=2 | EXP182: finite-harvest wheat/carrot input planner; original adaptation. |
| 12 | 2300 | _R51_WAREHOUSE | `agent` | agent |  | EXP182: project the final hour's actual worker actions before automatic deposit. |
| 13 | 2396 | _R53_LABOR | `agent` | agent |  |  |
| 14 | 2502 | _R70 | `agent` | agent |  |  |
| 15 | 2541 | _R85 | `agent` | agent | FEED=True, FERT=True |  |
| 16 | 2696 | _R95 | `agent` | agent |  | EXP226: retain physical grain for two complete days before trimming a buy. |
| 17 | 2764 | _R97 | `agent` | agent |  | EXP231: protect inputs using funded current orders without unassigned cash padding from observed physical resources. |
| 18 | 3008 | _V9_COURIER | `agent` | agent |  |  |
| 19 | 3120 | _V9_CARROT | `agent` | agent |  |  |
| 20 | 3224 | _V9_HERD | `agent` | agent |  |  |
| 21 | 3316 | _V9_FERT | `agent` | agent |  |  |
| 22 | 3373 | _V9_OPENING | `agent` | agent |  |  |
| 23 | 3511 | _V9_RACE | `agent` | agent |  |  |
| 24 | 3616 | _V9_RACEPX | `agent` | agent |  |  |
| 25 | 3714 | _V9_RACEGATE | `agent` | agent |  |  |
| 26 | 4250 | _CA | `agent` | agent | FROM=10, TO=28, DROP=0.0, BUFFER=8, FEED_DAYS=1, CASH=800 |  |
| 27 | 4452 | _OR2 | `agent` | agent | CAP=24, SN_K=0, SN_H=24, SLOT_H=6, SLOT_MARGIN=12.0 |  |
| 28 | 4691 | _CH | `agent` | agent | SHED=100, SELL=True |  |
| 29 | 4807 | _SR | `agent` | agent | MARGIN=8 |  |
| 30 | 5152 | _HD2 | `agent` | agent | FROM=192, TO=360, RATIO=1.3, MIN_GAIN=600.0, LOOKBACK=3, CARE=0.8 |  |
| 31 | 5292 | _CS | `agent` | agent | FROM=144, TO=192, RATIO=1.3, MIN_GAIN=600.0, SHOP_RULE='nomilk' |  |
| 32 | 5440 | _RACE | `agent` | agent | HORIZON_CLONE=9, HORIZON_ESCALATED=24, HORIZON_MIRROR=24 | EXP283 adaptive arm: clone-gated sale pre-emption with drop-time race escalation. Original mechanism by Ahmed Berat Ozer's project. Live top |
| 33 | 5449 | _RACE | `_r36_reserve` | _r36_reserve | HORIZON_CLONE=9, HORIZON_ESCALATED=24, HORIZON_MIRROR=24 |  |
| 34 | 5555 | _R127 | `agent` | agent |  |  |
| 35 | 5653 | _PG | `[v for v in list(globals().values()) if ` | agent_v44y_preguard |  | ---- v44y pre-guard: quote at hour 21,22 what the hour-23 day-end storage guard (EXP-154) would dump ---- The guard sells shed stock by pric |
| 36 | 5695 | _V44Y | `[v for v in list(globals().values()) if ` | v44y_lockstep_agent | REORDER_GATE=False | ---- v44y: exact lockstep best-response SELL ordering against a detected clone ---- |
| 37 | 5845 | _Y | `[v for v in list(globals().values()) if ` | _y_agent_shopherd | MIN_DAY=1 | ==== v44y shop-aware herd wrapper (appended after the public V44 file; host entry captured first) ==== |
| 38 | 5993 | _E334 | `_y_agent_shopherd` | _e334_agent |  |  |
| 39 | 6035 | _E335 | `_e334_compact` | _e334_compact |  | EXP335: empty buyable-product sales are holes too, when no purchases exist. |
| 40 | 6073 | _V11 | `[v for v in list(globals().values()) if ` | agent |  | v11 entry normalisation: Kaggle's loader takes the last callable in the module namespace.  Rebind it to a fresh `agent` so the packaged name |
| 41 | 6095 | _V13V | `_v219_request` | _v219_request |  |  |
| 42 | 6122 | _V13V | `agent` | agent |  |  |
| 43 | 6141 | _E343_WL | `agent` | agent | STEPS=8 |  |
| 44 | 6201 | _ADV | `agent` | agent | LOOK=3, FROM=216, TO=718, PROTECT=True, FRONT=False, BOOK=False | AXIS 2.1: Ready-Stock Sale Advancing & Frontloading (_adv_apply, _adv_frontload) |
| 45 | 6307 | _T62A | `agent` | _t62a_agent |  | AXIS 6.2A: Terminal 7-Turn Zero-Waste Liquidation (T_TERMINAL_CLEAR) |
| 46 | 6406 | _PIPE | `cha20_entry_agent` | _pipe_agent | MODE='EarlyCycle' |  |
| 47 | 6482 | _MA | `cha20_entry_agent` | cha20_entry_agent |  |  |
| 48 | 6512 | _MA | `cha20_entry_agent` | - |  |  |
| 49 | 6518 | _WB3 | `cha20_entry_agent` | cha20_entry_agent |  |  |
| 50 | 6544 | _WB3 | `cha20_entry_agent` | - |  |  |
| 51 | 6703 | _FX | `cha20_entry_agent` | cha20_entry_agent | FLOW_WIN=8, FLOW_MIN=999, LEAD_H=12, QUOTE_WIN=12, MIN_STEP=96 |  |
| 52 | 6747 | _FX | `cha20_entry_agent` | - | FLOW_WIN=8, FLOW_MIN=999, LEAD_H=12, QUOTE_WIN=12, MIN_STEP=96 |  |
| 53 | 6805 | _DP | `cha20_entry_agent` | cha20_entry_agent | H=8 |  |
| 54 | 6823 | _DP | `cha20_entry_agent` | - | H=8 |  |
| 55 | 6878 | _MP | `cha20_entry_agent` | cha20_entry_agent | H=8 |  |
| 56 | 6896 | _MP | `cha20_entry_agent` | - | H=8 |  |
| 57 | 6956 | _BD | `cha20_entry_agent` | cha20_entry_agent | ITEM='WHEAT', MIN=8, WIN=12, DEADLINE=6, CAP=64 |  |
| 58 | 6974 | _BD | `cha20_entry_agent` | - | ITEM='WHEAT', MIN=8, WIN=12, DEADLINE=6, CAP=64 |  |
| 59 | 7060 | _MPX | `cha20_entry_agent` | cha20_entry_agent |  |  |
| 60 | 7078 | _MPX | `cha20_entry_agent` | - |  |  |
| 61 | 7153 | _SM | `cha20_entry_agent` | cha20_entry_agent |  |  |
| 62 | 7170 | _SM | `cha20_entry_agent` | - |  |  |
| 63 | 7175 | _CXD | `cha20_entry_agent` | cxd_agent | BUDGET=800, FROM=0 |  |
| 64 | 7262 | _E410 | `cha20_entry_agent` | e410_agent |  |  |
| 65 | 7305 | _E402 | `cha20_entry_agent` | e402_agent |  | ==== F6: EXP402 late seed cap (pipe18 `e402_agent`) ==== |
| 66 | 7368 | _MG | `cha20_entry_agent` | mg_agent |  |  |
| 67 | 7416 | _IG | `cha20_entry_agent` | ig_agent |  |  |
| 68 | 7503 | _RSA | `agent` | agent | LOOK=5, FROM=144, TO=718, MIN_FRAC=0.5 | RSA: route sale advance (win-plan Stage 1, 2026-09-24). The clone field's decisive late-game edge is selling a premium product a few turns b |
| 69 | 7512 | _RSA | `{"STRAWBERRY": 120, "WOOL": 200, "EGG": ` | - | LOOK=5, FROM=144, TO=718, MIN_FRAC=0.5 |  |