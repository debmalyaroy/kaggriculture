# Simulation diff — our bot vs the top of the ladder

Agent: `agent_v3_20260804_160019.py` · 1 episodes · 240 turns replayed


## A. Same board, different move

**Overall farmer-op agreement: 24.6%**


| phase | agreement | turns |
|---|---:|---:|
| early (d0-9) | 24.6% | 240 |

### Biggest divergences

| they did | we would | turns | share |
|---|---|---:|---:|
| `PLANT` | `WATER` | 15 | 6.2% |
| `WEST` | `WATER` | 13 | 5.4% |
| `WATER` | `WEST` | 11 | 4.6% |
| `NORTH` | `WATER` | 10 | 4.2% |
| `WATER` | `EAST` | 9 | 3.8% |
| `WATER` | `HARVEST` | 8 | 3.3% |
| `NORTH` | `PLANT` | 7 | 2.9% |
| `WATER` | `NORTH` | 7 | 2.9% |
| `CARE` | `COLLECT_FERTILIZER` | 6 | 2.5% |
| `FEED` | `NORTH` | 6 | 2.5% |
| `COLLECT_FERTILIZER` | `PICKUP` | 5 | 2.1% |
| `PICKUP` | `FEED` | 5 | 2.1% |
| `NORTH` | `FEED` | 5 | 2.1% |
| `WEST` | `PLANT` | 4 | 1.7% |
| `PASS` | `CARE` | 4 | 1.7% |
| `FEED` | `WEST` | 3 | 1.2% |
| `EAST` | `SOUTH` | 3 | 1.2% |
| `HARVEST` | `PLANT` | 3 | 1.2% |

### Net action budget — ours minus theirs

| op | net turns |
|---|---:|
| `NORTH` | -3 |
| `WEST` | -3 |
| `SOUTH` | +3 |
| `BUILD_PASTURE` | -1 |
| `PASS` | -1 |
| `PLANT` | +1 |
| `DIG` | +1 |
| `PICKUP` | +1 |
| `EAST` | +1 |
| `DROP` | +1 |

Net movement: **-2** turns (we walk less than they do).


### Market orders — ours minus theirs

| order | net |
|---|---:|
| `SELL` | +167 |
| `BUY_SEED` | -13 |
| `HIRE` | -4 |
| `BUY_LAND` | +3 |
| `BUY_ANIMAL` | -1 |
| `BUY_PRODUCT` | +0 |

## B. Aggregate strategy

| metric | top-ladder median |
|---|---:|
| final bank | 3,850 |
| peak herd | 2 |
| peak crop tiles | 35 |
| peak hands/day | 5 |
| first animal day | 0 |

| asset | their median peak tiles |
|---|---:|
| WHEAT | 35 |
| CARROT | 0 |
| TOMATO | 0 |
| STRAWBERRY | 0 |
| MELON | 0 |
| GOOSE | 0 |
| COW | 0 |
| SHEEP | 2 |

## How to read this

- States come from *their* trajectory, so our errors never compound — but we are being asked about boards we would never have built. A single disagreement means nothing; a **systematic skew** is the signal.

- A large positive net on movement, or an order type they issue constantly and we never do, is a concrete policy gap worth testing.

- Everything here is a hypothesis. It has to clear three independent seed sets in `src/kaggriculture/measure/evaluate.py` before it is believed.
