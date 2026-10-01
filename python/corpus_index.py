"""Deduplicated, rank-annotated episode index over the slim corpus (data/slim/s1).

    python python/corpus_index.py            # -> data/slim/s1/index/episodes.parquet (+ summary)

One row per EPISODE_ID (the only unique key: seeds are reused across unrelated games). When an
episode was filed more than once (re-extractions, overlapping runs), the row from the newest run
(RUNID = the yyyymmddThhmmZ prefix in the part file name) wins; `n_copies` records how many there were.
Metadata columns only -- the heavy `slim` payload is never read; `file` points at the part holding it.

Ratings / ranks per seat (s = 0, 1):
  rating_post_s  GM: the seat's rating after this game (rating_after_s).
  rating_pre_s   GM: the same submission's rating_post from its PREVIOUS game (by end_time);
                 null for its first game. The best estimate of strength at game time.
  rating_s       rating_pre_s, else rating_post_s, else (daily) the seat-resolved manifest rating.
  band_s         <2100 | 2100-2300 | 2300-2500 | 2500-2700 | 2700+ (the crown-panel bands).
  rank_s         that day's leaderboard position of the seat's submission: rank of its latest
                 rating that day among every submission that played that day (1 = best);
  rank_pct_s     rank_s / submissions active that day.
Daily (official) episodes carry only the manifest's min/max rating with no seat; each seat is
resolved by matching its team to that team's GM rating on the same date (`seat_resolved`).
"""
import collections
import glob
import os
import re

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "slim", "s1")
OUT = os.path.join(ROOT, "index")
COLS = ["episode_id", "end_date", "end_time", "source", "seed", "engine_version", "episode_type", "n_steps",
        "team_name_0", "team_name_1", "submission_id_0", "submission_id_1", "rating_after_0", "rating_after_1",
        "rating_min", "rating_max", "bank_0", "bank_1"]
BANDS = [(2100, "<2100"), (2300, "2100-2300"), (2500, "2300-2500"), (2700, "2500-2700"), (1e9, "2700+")]
RUN_RE = re.compile(r"part-(\d{8}T\d{4}Z)-")


def band(r):
    if r is None:
        return None
    for hi, name in BANDS:
        if r < hi:
            return name
    return None


def load_rows():
    rows = []
    for src in ("official", "gm"):
        files = sorted(glob.glob(os.path.join(ROOT, f"source={src}", "*", "*.parquet")))
        for f in files:
            m = RUN_RE.search(os.path.basename(f))
            run = m.group(1) if m else ""
            pf = pq.ParquetFile(f)             # no hive inference: `source` is also a stored column
            cols = [c for c in COLS if c in pf.schema_arrow.names]
            t = pf.read(columns=cols)
            d = t.to_pydict()
            for i in range(t.num_rows):
                r = {c: (d[c][i] if c in d else None) for c in COLS}
                r["source"] = src
                r["file"] = os.path.relpath(f, ROOT).replace("\\", "/")
                r["run"] = run
                rows.append(r)
    return rows


def main():
    rows = load_rows()
    n_in = len(rows)
    # ---- dedup on episode_id: newest run wins ----
    best, copies = {}, collections.Counter()
    for r in rows:
        e = r["episode_id"]
        copies[e] += 1
        if e not in best or r["run"] > best[e]["run"]:
            best[e] = r
    eps = list(best.values())
    for r in eps:
        r["n_copies"] = copies[r["episode_id"]]

    # ---- GM: per-submission rating timeline -> rating_pre ----
    timeline = collections.defaultdict(list)          # sub -> [(end_time, episode_id, seat, post)]
    for r in eps:
        if r["source"] != "gm":
            continue
        for s in (0, 1):
            sub, post = r[f"submission_id_{s}"], r[f"rating_after_{s}"]
            if sub is not None and post is not None:
                timeline[sub].append((r["end_time"] or "", r["episode_id"], s, post))
    pre = {}
    for sub, evs in timeline.items():
        evs.sort()
        prev = None
        for et, e, s, post in evs:
            pre[(e, s)] = prev
            prev = post
    # team -> {date: last GM rating that day} (for daily seat resolution)
    team_day = collections.defaultdict(dict)
    for r in sorted((r for r in eps if r["source"] == "gm"), key=lambda r: r["end_time"] or ""):
        for s in (0, 1):
            tm, post = r[f"team_name_{s}"], r[f"rating_after_{s}"]
            if tm and post is not None:
                team_day[tm][r["end_date"]] = post

    for r in eps:
        r["seat_resolved"] = r["source"] == "gm"
        for s in (0, 1):
            r[f"rating_post_{s}"] = r[f"rating_after_{s}"]
            r[f"rating_pre_{s}"] = pre.get((r["episode_id"], s))
        if r["source"] == "official" and r["rating_min"] is not None:
            lo, hi = r["rating_min"], r["rating_max"]
            g = [team_day.get(r[f"team_name_{s}"] or "", {}).get(r["end_date"]) for s in (0, 1)]
            if g[0] is not None or g[1] is not None:
                # assign (lo, hi) to seats so the GM ratings of that day fit best
                cost = lambda a, b: sum(abs(x - y) for x, y in ((g[0], a), (g[1], b)) if x is not None)
                a, b = (lo, hi) if cost(lo, hi) <= cost(hi, lo) else (hi, lo)
                r["rating_post_0"], r["rating_post_1"] = a, b
                r["seat_resolved"] = True
        for s in (0, 1):
            r[f"rating_{s}"] = r[f"rating_pre_{s}"] if r[f"rating_pre_{s}"] is not None else r[f"rating_post_{s}"]
            r[f"band_{s}"] = band(r[f"rating_{s}"])
        b0, b1 = r["bank_0"], r["bank_1"]
        r["winner"] = None if b0 is None or b1 is None else (0 if b0 > b1 else 1 if b1 > b0 else -1)

    # ---- daily leaderboard rank of each submission (latest rating that day) ----
    day_sub = collections.defaultdict(dict)           # date -> sub -> (end_time, rating)
    for r in eps:
        if r["source"] != "gm":
            continue
        for s in (0, 1):
            sub, post = r[f"submission_id_{s}"], r[f"rating_post_{s}"]
            if sub is None or post is None:
                continue
            cur = day_sub[r["end_date"]].get(sub)
            if cur is None or (r["end_time"] or "") > cur[0]:
                day_sub[r["end_date"]][sub] = (r["end_time"] or "", post)
    day_rank = {}
    for d, subs in day_sub.items():
        order = sorted(subs.items(), key=lambda kv: -kv[1][1])
        n = len(order)
        for i, (sub, _) in enumerate(order):
            day_rank[(d, sub)] = (i + 1, (i + 1) / n)
    for r in eps:
        for s in (0, 1):
            rk = day_rank.get((r["end_date"], r[f"submission_id_{s}"]))
            r[f"rank_{s}"], r[f"rank_pct_{s}"] = (rk if rk else (None, None))

    keep = ["episode_id", "source", "file", "run", "n_copies", "end_date", "end_time", "seed", "engine_version",
            "episode_type", "n_steps", "team_name_0", "team_name_1", "submission_id_0", "submission_id_1",
            "bank_0", "bank_1", "winner", "rating_pre_0", "rating_pre_1", "rating_post_0", "rating_post_1",
            "rating_0", "rating_1", "band_0", "band_1", "rank_0", "rank_1", "rank_pct_0", "rank_pct_1",
            "rating_min", "rating_max", "seat_resolved"]
    eps.sort(key=lambda r: (r["end_date"] or "", r["end_time"] or "", r["episode_id"]))
    table = pa.Table.from_pylist([{k: r.get(k) for k in keep} for r in eps])
    os.makedirs(OUT, exist_ok=True)
    pq.write_table(table, os.path.join(OUT, "episodes.parquet"), compression="zstd")

    # ---- summary ----
    val = sum(1 for r in eps if (r["episode_type"] or "").endswith("VALIDATION"))
    dup = sum(1 for r in eps if r["n_copies"] > 1)
    lines = [f"rows read {n_in}; unique episodes {len(eps)}; episodes filed more than once {dup}; validation {val}",
             f"by source: {dict(collections.Counter(r['source'] for r in eps))}",
             f"seat-resolved ratings: {sum(r['seat_resolved'] for r in eps)} / {len(eps)}; "
             f"with rating_pre both seats: {sum(r['rating_pre_0'] is not None and r['rating_pre_1'] is not None for r in eps)}",
             "games by band pair (rating at game time, lower band first):"]
    pairs = collections.Counter()
    for r in eps:
        if r["band_0"] and r["band_1"]:
            pairs[tuple(sorted((r["band_0"], r["band_1"]), key=lambda b: [x[1] for x in BANDS].index(b)))] += 1
    for (a, b), n in sorted(pairs.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {a:>10} vs {b:<10} {n}")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8").write(txt + "\n")
    print(txt)
    print(f"-> {os.path.join(OUT, 'episodes.parquet')}")


if __name__ == "__main__":
    main()
