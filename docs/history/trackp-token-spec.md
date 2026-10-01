# Trackp token & action spec (T0.1 / T0.2) — canonical on-disk + runtime schema

Grounded on real 1.32.7 replays from `D:\gm_dataset` (probed 2026-09-20).
This is the contract shared by: the Python corpus extractor, the Rust obs
encoder (`obstoken.rs`), BC training, RL, and the baked deploy policy. Any
change bumps `TOKEN_LAYOUT_VERSION` and must be mirrored on all sides
(gate-locked by `tests/test_obstoken_parity.py`).

`TOKEN_LAYOUT_VERSION = 1`

## Board & constants (from configuration)
- board 10×10 (100 tiles); 4 quadrants of 5×5; ownership via `unlocked_quadrants`.
- turnsPerDay 24, episodeSteps 720, shedCapacity, maxMarketOrdersPerTurn 10.
- CROPS(5) = CARROT, MELON, STRAWBERRY, TOMATO, WHEAT
- ANIMALS(3) = COW, GOOSE, SHEEP
- PRODUCTS(9) = CARROT, EGG, FERTILIZER, MELON, MILK, STRAWBERRY, TOMATO, WHEAT, WOOL
- SHOPS(8) = BAKERY, BRUNCH_SPOT, FARMERS_MARKET, ICE_CREAM_SHOP, PET_CAFE,
  PIZZA_SHOP, SMOOTHIE_SHOP, YARN_STORE
- MAX_HANDS = 12 (movers = 1 farmer + ≤12 hands = ≤13)

## Observation → tokens (≤~120 tokens/turn)
Per-seat obs keys: `day, hour, step, player, farms[2], private(mine), market, town`.
`private.inventories[i]` = items carried by mover i (0=farmer, then hands).

Token types (each token = type-embedding + projected numeric fields +
categorical id lookups):

1. **TILE tokens** — our owned, non-empty tiles (kind ∈ {PLANT, PASTURE, COOP,
   WEED}). Fields: `x, y, kind, crop_id, animal_id, planted_day_rel,
   placed_day_rel, watered_today, cared_today, fed_today, yield_units,
   consecutive_unwatered, consecutive_unfed, fertilizer_available,
   fertilized_rel, max_lifespan_rel`. Empty owned tiles → one EMPTY token per
   (or a count); LOCKED/unowned → omitted.
2. **OPP-TILE summary tokens** — opponent's observable tiles aggregated to a few
   tokens (per-kind counts + mean yield/age) to keep the set small.
3. **SHED token** — 12-item inventory vector.
4. **MARKET tokens** — 9 (one per product): `price, inventory`. The shared,
   observable market — the source of reactivity.
5. **HAND tokens** — one per hand (≤12): `x, y` + carried-inventory summary
   (`private.inventories[i]`). Farmer is its own token (`farmer x,y` + carried).
6. **GAME-STATE token** — `day, hour, step, money, opp_money, n_hands,
   land_owned(=|unlocked_quadrants|), hires_today, unlocked_shops(8-multi-hot)`.
7. **RETURN-TO-GO token** — `rtg` (final score for this seat; 1.0=win, 0.5=draw,
   0.0=loss) + `rating/3000`. Condition-on-win at inference.

No positional embedding; spatial meaning lives in each tile/hand token's (x,y).

## Action (composite, per turn)
`{farmer:[op], hands:[[op]×k], market:[[order]×≤10]}`. Extras beyond real
hands / 10 market orders are silent no-ops (so hand-action length is not
authoritative; **`farm.hands` positions are**).

- **Mover ops (farmer + each hand)** — vocab:
  `PASS, NORTH, SOUTH, EAST, WEST, PLANT<crop>, WATER, HARVEST, FEED, CARE,
   PLACE<animal>, PICKUP<item,n>, DROP, DIG, FERTILIZE, COLLECT_FERTILIZER,
   BUILD_PASTURE, BUILD_COOP`. Args: PLANT→crop, PLACE→animal, PICKUP→(item,n).
- **Market orders** — vocab:
  `BUY_SEED<crop,n>, BUY_PRODUCT<item,n>, BUY_ANIMAL<animal,n>, SELL<item,n>,
   HIRE, BUY_LAND`. Ordered by priority = decode order; cap 10.

**Decode (runtime):** encoder runs once → pointer decoder emits, autoregressively
over the encoded tokens: farmer op → each hand op (index order, conditioned) →
market loop (order or SUBMIT, cap 10). Each sub-step legality-masked by the
shared Rust rules. **v1 fallback:** factorized single-shot heads + deterministic
conflict-resolver by hand index.

**Legality mask** — produced by the shared rule set (Python for corpus, Rust for
RL/deploy): a mover can only do what its tile/inventory/position allows; a market
order only if affordable/valid. Chosen expert action must always be legal
(corpus invariant: masked-rate of the label = 0).

## On-disk corpus format (Parquet + zstd, sharded)
Columns per decision (packed-binary blobs, dictionary-encoded categoricals):
```
episode_id int64 · seat int8 · day int16 · step int16 · split int8
rtg float16 · rating float16 · world_family int16 · n_tokens int16
tokens     binary   # packed int8/int16: per token [type, fields…]
action     binary   # packed composite action (farmer + hands + market)
legal_mask binary   # packed bitset, variable per decision
```
- shards ~200–500 MB (`ParquetWriter`, flush-and-clear), row-groups ~50–100k,
  **grouped by episode** within a shard; zstd level ~9–12.
- split = `"val" if (int(sha1(episode_id)[:8],16) % 10000)/10000 < 0.15 else
  "train"` (deterministic, leak-free — reuses `data/bc_corpus.split_of`).
- engine filter: keep episodes with `engine_version == 1.32.7`
  (join `episode_features.csv`; the replay top-level `module_version` agrees).
- rows: winner-seat + high-rated loser-seat (rating ≥ ~2100), return-conditioned.

## Companion files (built + uploaded with the shards)
- `manifest.json` — schema, TOKEN_LAYOUT_VERSION, dtypes, pyarrow version, shard
  list (rows + per-split counts), build params (engine, rating filter, date,
  extractor git hash), vocab ref.
- `index.parquet` — one lightweight row/decision: `global_id, shard_id,
  row_in_shard, episode_id, seat, day, split, rtg, rating, world_family` (no
  token blob) — filtered sampling, shuffle order, Phase-4b family selection,
  class balance, reproducibility.
- `vocab.json` — frozen dictionaries: crop/animal/product/shop/mover-verb/
  market-verb/kind/world-family → id.
- `splits.parquet` — episode_id → split.
- `world_families.parquet` — episode_id → day-6 shop signature → family_id.
- `stats.json` — per-split counts, action-class balance, token-length + sell-qty
  + rating distributions.
- `norm.json` — continuous-feature mean/std from **TRAIN only** (no val leak).

RL uses none of this (self-play data is live from the Rust binary-obs bridge);
the format is a BC-only concern. Optional local `parquet→memmap`/LMDB bake for
true O(1) random access if BC training profiles as decode-bound.
