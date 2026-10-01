# Episode index: dedup + ratings/ranks per game (2026-09-25)

`python python/corpus_index.py` builds `data/slim/s1/index/episodes.parquet` (+ `summary.txt`)
from every filed part in `data/slim/s1/source={official,gm}`. It reads metadata columns only
and never the `slim` payload; ~20 s for 183k episodes. `scripts/backfill_watch.ps1` reruns it
after every FILED run.

## Dedup
- **Key = `episode_id`.** Seeds are NOT unique: 4 seeds are shared between unrelated daily
  and GM games.
- An episode filed more than once keeps the row from the newest run (the RUNID prefix of the
  part file name). `n_copies` counts how many copies there were.
- As of 2026-09-25: 183,464 episodes, 0 duplicates.
  - The official daily datasets (27,225) and GM (156,239) are **disjoint**.
  - They are different populations: daily is an elite sample (median rating ~3000); GM is the
    broad ladder (median ~2000).
  - `episode_type` flags GM's 465 validation self-play episodes.

## Ratings and ranks (per seat s = 0, 1)

| column | meaning |
|---|---|
| `rating_post_s` | GM: the seat's rating after the game. Daily: the manifest min/max rating, assigned to seats by matching each team's GM rating on the same date (`seat_resolved`). |
| `rating_pre_s` | GM: the same submission's `rating_post` from its previous GM game. This is the strength at game time. GM holds only a sample of each submission's games, so "previous" can be several ladder games back. |
| `rating_s` | `rating_pre_s`, falling back to `rating_post_s`. |
| `band_s` | `<2100`, `2100-2300`, `2300-2500`, `2500-2700` or `2700+` (the crown-panel bands). |
| `rank_s`, `rank_pct_s` | That day's leaderboard position of the submission: its latest rating that day, ranked among every submission seen playing that day (GM). |

Games by band pair are in `summary.txt`. `<2100` vs `<2100` dominates (103k); `2700+` vs `2700+`
has 19k; the cross-band mid-ladder pairs have 0.6k–11k each.

## Worlds
- Every row has `seed` (= replay `info.seed`, the engine seed).
- The `slim` record keeps both action streams and town-shop unlock changes (`tw`).
- Seed + both tapes rebuild the game exactly: `crates/runner/src/bin/tapeplay.rs` reproduced
  111/111 real v61.1 ladder games to the dollar.
- The realized world (shop sequence) depends on the seed AND both action streams. Derive it
  from `tw`, not from the seed alone.
