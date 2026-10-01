"""Who buys the 4th quadrant (SE, $4,000) and does it pay? (operator 29 Sep)

    python python/se_land_study.py [--files 400] [--workers 2] [--out data/opp/se_land.json]

Slim corpus: per (episode, seat) -- quadrants owned at the end (steps[t].f[seat].unlocked_quadrants), the day the 4th
was bought, hires per day (HIRE orders), final bank, the rival's bank, rating after the game, and what stands on
the SE quadrant (x >= 5, y >= 5) in the daily tile snapshots (steps[t].tl at hour 1): crops by type, animals,
structures, empty/weeds -- averaged over the days after the purchase.
Compared: 4-quadrant player-games vs 3-quadrant ones, overall and within rating bands.
"""
import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q_RE = re.compile(r'"unlocked_quadrants":\[([^\]]*)\]')


def band(r):
    return "2500+" if r >= 2500 else "2200-2500" if r >= 2200 else "<2200" if r > 0 else "?"


def one(path):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    rows = []
    for g in range(pf.metadata.num_row_groups):
        for r in pf.read_row_group(g, columns=["rating_after_0", "rating_after_1", "bank_0", "bank_1", "team_name_0", "team_name_1", "slim"]).to_pylist():
            txt = r["slim"] or ""
            chunks = txt.split('{"a":[')[1:]
            if len(chunks) < 700:
                continue
            q4 = [None, None]
            hires = [[0] * 30, [0] * 30]
            nq_end = [1, 1]
            for t, ch in enumerate(chunks):
                qs = Q_RE.findall(ch[:4000])
                for s in (0, 1):
                    if s < len(qs):
                        n = qs[s].count('"')// 2
                        nq_end[s] = n
                        if n >= 4 and q4[s] is None:
                            q4[s] = t
                j = ch.find('],"m":')
                a = ch[:j] if j > 0 else ""
                if '"HIRE"' in a and t > 0:
                    try:
                        pair = json.loads("[" + a + "]")
                    except Exception:  # noqa: BLE001
                        pair = []
                    for s in (0, 1):
                        if s < len(pair) and isinstance(pair[s], dict):
                            hires[s][min((t - 1) // 24, 29)] += sum(1 for o in pair[s].get("market") or [] if o and o[0] == "HIRE")
            for s in (0, 1):
                se = Counter()
                ndays = 0
                if q4[s] is not None:
                    for d in range(q4[s] // 24 + 1, 29):
                        t = d * 24 + 1
                        if t >= len(chunks):
                            break
                        ch = chunks[t]
                        k = ch.find(',"tl":[')
                        if k < 0:
                            continue
                        try:
                            tl = json.loads(ch[k + 6: ch.rfind("]}") + 1] if ch.rfind("]}") > k else "[]")
                            tiles = tl[s]
                        except Exception:  # noqa: BLE001
                            continue
                        ndays += 1
                        for y in range(5, 10):
                            for x in range(5, 10):
                                c = tiles[y][x] if y < len(tiles) and x < len(tiles[y]) else None
                                if c is None:
                                    se["empty"] += 1
                                elif isinstance(c, str):
                                    se[c.lower()] += 1
                                elif isinstance(c, dict):
                                    kind = c.get("kind", "?")
                                    if kind == "PLANT":
                                        se["crop:" + str(c.get("crop"))] += 1
                                    elif c.get("animal"):
                                        se[f"{kind.lower()}:{c.get('animal')}"] += 1
                                    else:
                                        se[str(kind).lower()] += 1
                rt = r[f"rating_after_{s}"] or 0
                b, ob = r[f"bank_{s}"] or 0, r[f"bank_{1 - s}"] or 0
                rows.append({"team": str(r[f"team_name_{s}"]), "rating": rt, "bank": b, "obank": ob, "win": 1.0 if b > ob else 0.5 if b == ob else 0.0,
                             "quads": nq_end[s], "se_day": (q4[s] // 24) if q4[s] is not None else None, "hires": hires[s],
                             "se_use": {k: v / max(1, ndays) for k, v in se.items()}})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=400)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(RL, "data", "opp", "se_land.json"))
    a = ap.parse_args()
    fs = sorted(glob.glob(os.path.join(RL, "data", "slim", "s1", "source=gm", "date=*", "*.parquet")))
    fs = fs[:: max(1, len(fs) // a.files)][: a.files]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for rr in ex.map(one, fs):
            rows += rr
    by = defaultdict(list)
    for r in rows:
        by[(r["quads"] >= 4, band(r["rating"]))].append(r)
    rep = {"player_games": len(rows)}
    print(f"[se] {len(rows)} player-games")
    for four in (True, False):
        for b in ("2500+", "2200-2500", "<2200"):
            g = by[(four, b)]
            if not g:
                continue
            n = len(g)
            key = f"{'4 quads' if four else '<=3 quads'} {b}"
            rep[key] = {"n": n, "win": sum(x["win"] for x in g) / n, "bank": sum(x["bank"] for x in g) / n, "margin": sum(x["bank"] - x["obank"] for x in g) / n,
                        "hires_per_day_d12_28": sum(sum(x["hires"][12:28]) / 16 for x in g) / n}
            print(f"  {key:22s} n {n:6d}  win {rep[key]['win']:.3f}  bank ${rep[key]['bank']:8.0f}  margin ${rep[key]['margin']:+7.0f}  hires/day d12-28 {rep[key]['hires_per_day_d12_28']:.1f}")
    four = [r for r in rows if r["quads"] >= 4]
    if four:
        days = Counter(r["se_day"] for r in four)
        use = Counter()
        for r in four:
            for k, v in r["se_use"].items():
                use[k] += v / len(four)
        teams = Counter(r["team"] for r in four).most_common(15)
        rep["se_buy_day"] = sorted(days.items())
        rep["se_use_tiles_per_day"] = dict(use.most_common())
        rep["top_teams"] = teams
        # what the WINNING 4-quad players do on SE vs the losing ones
        for lab, grp in (("winners", [r for r in four if r["win"] == 1]), ("losers", [r for r in four if r["win"] == 0])):
            u = Counter()
            for r in grp:
                for k, v in r["se_use"].items():
                    u[k] += v / max(1, len(grp))
            rep[f"se_use_{lab}"] = dict(u.most_common(10))
        print("  4th quadrant bought on day:", sorted(days.items()))
        print("  SE use (tiles/day, of 25):", {k: round(v, 1) for k, v in use.most_common(10)})
        print("  winners use:", {k: round(v, 1) for k, v in rep["se_use_winners"].items()})
        print("  losers use :", {k: round(v, 1) for k, v in rep["se_use_losers"].items()})
        print("  teams:", teams[:10])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
