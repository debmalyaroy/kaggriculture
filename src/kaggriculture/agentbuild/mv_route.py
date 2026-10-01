"""Majority-vote route reconstruction (the boatlee / V16-RC5 method).

A single recorded tape carries its episode's accidents: weed-repair detours,
shop buys shifted by the opponent's empty-tile count (weeds and the shop draw
share one RNG stream), one market's particular glut. Voting each step's
channels across SEVERAL episodes of the same team keeps the schedule and
discards the accidents -- V16-RC5's base agreed 99.91% of market steps across
its three source replays, and the voted route beat every single-tape build.

Only worth doing for STABLE teams (fixed scripts). For adaptive teams the
episodes genuinely differ and the vote produces a chimera -- the same reason
their single tapes already fail the gauntlet (measured again 2026-08-27:
Crop Dusta 0.240, Ryo 0.104). The agreement report printed per run is the
tell: a stable team votes >90% majority strength; an adaptive one does not.

    python src/mv_route.py --team Naru041104 --min-routes 3
    python src/mv_route.py --team Naru041104 --ids 100811941_s0 100830258_s0 ...

Registers the voted tape in the route store as `mv_<team>_<today>` so every
downstream tool (`routes.py --build`, `v22_agent --base`, the tuner) can use
it like any mined route.
"""
from kaggriculture.paths import ROOT
import argparse
import collections
import datetime as dt
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

import kaggriculture.data.routes as routes  # noqa: E402


def norm(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"))


def vote_channel(values):
    """(winner, votes, total) for one step's one channel."""
    counter = collections.Counter(norm(v) for v in values)
    top, n = counter.most_common(1)[0]
    return json.loads(top), n, len(values)


def build_vote(tapes):
    length = max(len(t) for t in tapes)
    voted, stats = [], {"unanimous": 0, "majority": 0, "split": 0}
    strengths = []
    for step in range(length):
        turns = [t[step] for t in tapes if step < len(t)
                 and isinstance(t[step], dict)]
        if not turns:
            voted.append({"farmer": ["PASS"], "hands": [], "market": []})
            continue
        out, weakest = {}, 1.0
        for chan in ("farmer", "hands", "market"):
            win, n, total = vote_channel([t.get(chan) for t in turns])
            out[chan] = win
            weakest = min(weakest, n / total)
        voted.append(out)
        strengths.append(weakest)
        if weakest == 1.0:
            stats["unanimous"] += 1
        elif weakest > 0.5:
            stats["majority"] += 1
        else:
            stats["split"] += 1
    stats["mean_strength"] = sum(strengths) / max(1, len(strengths))
    return voted, stats


def _agree(a, b, upto=480):
    """Fraction of steps where two tapes take identical turns."""
    n = min(len(a), len(b), upto)
    if n == 0:
        return 0.0
    same = sum(1 for i in range(n) if norm(a[i]) == norm(b[i]))
    return same / n


def cluster_tapes(tapes, thresh=0.80):
    """Greedy single-link clustering by step agreement; returns the index
    list of the LARGEST cluster. Kills the vote's mode collapse: a 60/40
    strategy split votes as a chimera, but within-cluster it is pure."""
    n = len(tapes)
    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            sim[i][j] = sim[j][i] = _agree(tapes[i], tapes[j])
    unassigned = set(range(n))
    clusters = []
    while unassigned:
        seed = max(unassigned,
                   key=lambda i: sum(sim[i][j] for j in unassigned))
        group = [seed]
        unassigned.discard(seed)
        added = True
        while added:
            added = False
            for j in list(unassigned):
                if max(sim[j][g] for g in group) >= thresh:
                    group.append(j)
                    unassigned.discard(j)
                    added = True
        clusters.append(group)
    clusters.sort(key=len, reverse=True)
    return clusters, sim


def medoid_of(group, sim):
    return max(group, key=lambda i: sum(sim[i][j] for j in group if j != i))


def stitch_days(tapes, group, switch_pen=0.5, tpd=24):
    """Day-stitched composite (V3): pick ONE source's entire day per day --
    units respawn at the shed each dawn, so day boundaries are free switch
    points and every emitted sequence is one a real game actually played.
    DP over (day, source): score = within-cluster popularity of the
    source's day minus a switch penalty for changing source."""
    days = max(len(t) for t in tapes) // tpd + 1
    G = list(group)

    def day_key(gi, d):
        t = tapes[gi]
        seg = [norm(t[s]) if s < len(t) else "-"
               for s in range(d * tpd, (d + 1) * tpd)]
        return "\n".join(seg)

    pop = {}                              # (d, gi) -> popularity
    for d in range(days):
        keys = {gi: day_key(gi, d) for gi in G}
        counts = collections.Counter(keys.values())
        for gi in G:
            pop[(d, gi)] = counts[keys[gi]] / len(G)
    best = {gi: (pop[(0, gi)], None) for gi in G}
    for d in range(1, days):
        nxt = {}
        for gi in G:
            cand = max(G, key=lambda pg: best[pg][0]
                       - (switch_pen if pg != gi else 0.0))
            nxt[gi] = (best[cand][0]
                       - (switch_pen if cand != gi else 0.0)
                       + pop[(d, gi)], (d - 1, cand))
        # store back-pointers per day
        for gi in G:
            best[gi] = nxt[gi]
        for gi in G:
            pop[(d, gi, "bp")] = nxt[gi][1]
    end = max(G, key=lambda gi: best[gi][0])
    path = [end]
    for d in range(days - 1, 0, -1):
        prev = pop[(d, path[-1], "bp")][1]
        path.append(prev)
    path.reverse()
    out = []
    switches = 0
    for d, gi in enumerate(path):
        t = tapes[gi]
        for s in range(d * tpd, (d + 1) * tpd):
            out.append(t[s] if s < len(t) and isinstance(t[s], dict)
                       else {"farmer": ["PASS"], "hands": [], "market": []})
        if d and path[d] != path[d - 1]:
            switches += 1
    return out[:max(len(t) for t in tapes)], path, switches


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--team", required=True)
    ap.add_argument("--ids", nargs="*",
                    help="explicit route ids; default = every route of the "
                         "team on the vendored engine")
    ap.add_argument("--min-routes", type=int, default=3)
    ap.add_argument("--engine", default=None,
                    help="engine filter (default: models/engine_version.json)")
    ap.add_argument("--mode", choices=("vote", "cluster", "medoid", "stitch"),
                    default="vote",
                    help="vote = legacy per-turn majority over ALL sources; "
                         "cluster = vote within the largest strategy "
                         "cluster; medoid = the single most-representative "
                         "game of that cluster; stitch = day-stitched DP "
                         "composite (coherent days, free dawn switches)")
    ap.add_argument("--thresh", type=float, default=0.80,
                    help="cluster agreement threshold")
    args = ap.parse_args()

    engine = args.engine
    if engine is None:
        try:
            engine = json.load(open(os.path.join(
                ROOT, "models", "engine_version.json"),
                encoding="utf-8")).get("engine")
        except (OSError, ValueError):
            engine = None

    idx = routes.load_index()
    if args.ids:
        recs = [idx["routes"][i] for i in args.ids if i in idx["routes"]]
    else:
        recs = [r for r in idx["routes"].values()
                if r.get("team") == args.team
                and (engine is None or r.get("engine") == engine)
                and r.get("source") not in ("majority-vote",
                                            "trackp-factory")]
    if len(recs) < args.min_routes:
        raise SystemExit(f"only {len(recs)} route(s) for {args.team!r} on "
                         f"engine {engine} -- need {args.min_routes}")

    tapes, used = [], []
    for r in sorted(recs, key=lambda r: -(r.get("bank") or 0)):
        try:
            tapes.append(routes.load_route(r["id"]))
            used.append(r)
        except OSError:
            print(f"  (no stored tape for {r['id']} -- skipped)")
    if len(tapes) < args.min_routes:
        raise SystemExit(f"only {len(tapes)} loadable tape(s) -- "
                         f"need {args.min_routes}")

    prefix = {"vote": "mv", "cluster": "mvc", "medoid": "mvm",
              "stitch": "mvs"}[args.mode]
    if args.mode == "vote":
        voted, stats = build_vote(tapes)
    else:
        clusters, sim = cluster_tapes(tapes, args.thresh)
        group = clusters[0]
        print(f"{len(clusters)} strategy cluster(s); largest has "
              f"{len(group)}/{len(tapes)} games "
              f"(sizes: {[len(c) for c in clusters[:6]]})")
        if args.mode == "medoid":
            mi = medoid_of(group, sim)
            voted = tapes[mi]
            stats = {"unanimous": 0, "majority": 0, "split": 0,
                     "mean_strength": 1.0, "medoid": used[mi]["id"],
                     "cluster_size": len(group)}
            print(f"medoid exemplar: {used[mi]['id']} "
                  f"(bank {used[mi].get('bank', 0):,.0f})")
        elif args.mode == "stitch":
            voted, path, switches = stitch_days(tapes, group)
            stats = {"unanimous": 0, "majority": 0, "split": 0,
                     "mean_strength": 1.0, "cluster_size": len(group),
                     "stitch_sources": sorted({used[g]["id"]
                                               for g in path}),
                     "switches": switches}
            print(f"day-stitched from {len(set(path))} source game(s), "
                  f"{switches} switch(es)")
        else:
            voted, stats = build_vote([tapes[i] for i in group])
            stats["cluster_size"] = len(group)
    total = stats["unanimous"] + stats["majority"] + stats["split"]
    print(f"{args.team}: {args.mode} over {len(tapes)} tapes, {total} steps")
    print(f"  unanimous {stats['unanimous']} | majority {stats['majority']} "
          f"| split {stats['split']} | mean weakest-channel strength "
          f"{stats['mean_strength']:.3f}")
    if args.mode == "vote" and stats["mean_strength"] < 0.6:
        print("  WARNING: weak agreement -- this team is probably adaptive; "
              "a voted chimera usually fails the gauntlet. Registering "
              "anyway; let the panel veto it.")

    # Content-hashed id: re-running the constructor must NEVER silently
    # repoint an existing id at different actions (2026-09-01: a shipped
    # agent's recorded base id was overwritten by a later rebuild).
    import hashlib
    chash = hashlib.sha1(json.dumps(voted, separators=(",", ":"),
                                    sort_keys=True).encode()).hexdigest()[:6]
    rid = (f"{prefix}_{args.team.replace(' ', '_')}_"
           f"{dt.date.today().isoformat()}_{chash}")
    routes.save_route(rid, voted)
    best = used[0]
    idx = routes.load_index()
    idx["routes"][rid] = {
        "id": rid, "team": args.team,
        "episode": "+".join(str(r.get("episode")) for r in used[:6]),
        "seat": best.get("seat", 0),
        "bank": sum(float(r.get("bank") or 0) for r in used) / len(used),
        "opp_bank": sum(float(r.get("opp_bank") or 0) for r in used) / len(used),
        "date": dt.date.today().isoformat(), "window": "fit",
        "won": any(r.get("won") for r in used), "engine": engine,
        "source": "majority-vote", "obs": False, "trace": False,
        "seed": None,
        "mv_stats": stats, "mv_sources": [r["id"] for r in used],
    }
    routes.save_index(idx)
    print(f"registered route {rid} ({len(used)} sources: "
          f"{', '.join(r['id'] for r in used)})")
    print(f"next: python src/routes.py --build {rid} --out .local/candidates/"
          f"mv_{args.team.replace(' ', '_')}.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
