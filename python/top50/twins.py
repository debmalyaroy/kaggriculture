"""Twin games: top players who played exactly our opening, and what they did better from there (queue Q55).

    python python/top50/twins.py [--threads 24]

Hashes follow the GM dataset's stream_hashes.csv byte for byte (sha256 over the seat's actions, canonical JSON,
NUL between turns, replay steps 1..N = tape actions 0..N-1, first 16 hex), so "twin" means: identical actions
through turn N.
  1. our streams: every ladder game of ours (data/ladder/*/tapes), hashed at turns 136 and 200
  2. top streams: this week's top-50 games (data/top50/fetch) + the GM top-100 games (data/top50/gm_local)
  3. twins = top games whose stream through turn N equals one of ours (N = 200 first, else 136)
  4. TAKEOVER: v63.1_rl replays the top player's own actions up to turn N (following along, so its state is
     built from the same history), then plays their seat against the recorded rival (guarded) to the end:
     the same position, their continuation vs ours
  5. where their continuation won and ours did not: what they bought / sold / hired in the next 48 turns
     compared with what we did (our takeover games are recorded)
Writes data/top50/twins/report.json and twins.tsv.
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIN = os.environ.get("KRL_BIN") or os.path.join(RL, "target-dev", "release")
TAPEPLAY = os.path.join(BIN, "tapeplay" + (".exe" if os.name == "nt" else ""))
V64 = os.path.join(RL, "configs", "profiles", "v64rl.json")
OUT = os.path.join(RL, "data", "top50", "twins")
CUTS = (24, 48, 100, 136, 200)


def hashes(tape, seat):
    h, out = hashlib.sha256(), {}
    for t, pair in enumerate(tape["actions"], 1):
        a = pair[seat] if len(pair) > seat and isinstance(pair[seat], dict) else {}
        h.update(json.dumps(a, sort_keys=True, separators=(",", ":")).encode())
        h.update(b"\0")
        if t in CUTS:
            out[t] = h.hexdigest()[:16]
    return out


def ops(a):
    """market decisions of one turn as counters: BUY_ANIMAL:COW, BUY_SEED:WHEAT (units), SELL:WOOL (units), HIRE, BUY_LAND."""
    c = Counter()
    for m in (a or {}).get("market") or []:
        if not m:
            continue
        k = m[0]
        if k in ("BUY_ANIMAL", "BUY_SEED", "BUY_PRODUCT", "SELL") and len(m) >= 3:
            c[f"{k}:{m[1]}"] += int(m[2] or 0)
        elif k in ("HIRE", "BUY_LAND"):
            c[k] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=24)
    a = ap.parse_args()
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(os.path.join(OUT, "tapes"))
    ours = {n: set() for n in CUTS}
    for f in glob.glob(os.path.join(RL, "data", "ladder", "*", "tapes", "*.json")):
        t = json.load(open(f, encoding="utf-8"))
        for n, v in hashes(t, t["seat"]).items():
            ours[n].add(v)
    print(f"[twins] our streams: " + ", ".join(f"turn {n}: {len(ours[n])}" for n in CUTS), flush=True)
    twins = []
    for d in ("fetch", "gm_local"):
        for f in glob.glob(os.path.join(RL, "data", "top50", d, "tapes", "*", "*.json")):
            t = json.load(open(f, encoding="utf-8"))
            h = hashes(t, t["seat"])
            n = next((n for n in sorted(CUTS, reverse=True) if h.get(n) in ours[n]), None)
            if n:
                twins.append((f, n, t.get("team"), d))
    seen, uniq = set(), []
    for x in twins:
        k = os.path.basename(x[0])
        if k not in seen:
            seen.add(k)
            uniq.append(x)
    twins = uniq
    print(f"[twins] {len(twins)} top-player games are twins of ours (" + ", ".join(f"through {n}: {sum(x[1] == n for x in twins)}" for n in CUTS) + ")", flush=True)
    rec = os.path.join(OUT, "ours")
    os.makedirs(rec)
    rows = []
    for n in CUTS:
        fs = [x for x in twins if x[1] == n]
        if not fs:
            continue
        shards = [fs[i::a.threads] for i in range(a.threads) if fs[i::a.threads]]

        def one(sh):
            d = tempfile.mkdtemp(prefix="twin-")
            try:
                for f, *_ in sh:
                    os.symlink(f, os.path.join(d, os.path.basename(f)))
                r = subprocess.run([TAPEPLAY, "--tapes", d, "--profiles", V64, "--pa", "35", "--group", "35,35,36", "--guarded",
                                    "--takeover", str(n), "--record", rec], cwd=RL, capture_output=True, text=True)
                return r.stdout
            finally:
                shutil.rmtree(d, ignore_errors=True)

        meta = {os.path.basename(f)[:-5]: (team, src) for f, _, team, src in fs}
        with ThreadPoolExecutor(len(shards)) as ex:
            for out in ex.map(one, shards):
                for ln in out.splitlines():
                    x = ln.split("\t")
                    if len(x) < 7:
                        continue
                    tid = x[0]
                    rus, rthem, us, them = map(float, x[2:6])
                    real = 1.0 if rus > rthem else 0.0 if rus < rthem else 0.5
                    ourr = 1.0 if us > them else 0.0 if us < them else 0.5
                    rows.append({"id": tid, "cut": n, "team": meta.get(tid, ("?", "?"))[0], "src": meta.get(tid, ("?", "?"))[1],
                                 "their_result": real, "our_result": ourr, "their_margin": rus - rthem, "our_margin": us - them,
                                 "agree_before_cut": int(x[6]) / n})
    with open(os.path.join(OUT, "twins.tsv"), "w", encoding="utf-8", newline="\n") as fh:
        if rows:
            fh.write("\t".join(rows[0]) + "\n")
            for r in rows:
                fh.write("\t".join(str(v) for v in r.values()) + "\n")
    # what the top player did differently in the 48 turns after the cut, where they won and we did not
    diff_win, diff_all, n_better = Counter(), Counter(), 0
    for r in rows:
        f = next((x[0] for x in twins if os.path.basename(x[0])[:-5] == r["id"]), None)
        g = os.path.join(rec, f"{r['id']}__r.json")
        if not f or not os.path.exists(g):
            continue
        theirs, mine = json.load(open(f, encoding="utf-8")), json.load(open(g, encoding="utf-8"))
        s = theirs["seat"]
        dt = Counter()
        for t in range(r["cut"], min(r["cut"] + 48, len(theirs["actions"]), len(mine["actions"]))):
            dt.update(ops(theirs["actions"][t][s]))
            dt.subtract(ops(mine["actions"][t][s]))
        diff_all.update(dt)
        if r["their_result"] > r["our_result"]:
            n_better += 1
            diff_win.update(dt)
    b = sum(r["their_result"] > r["our_result"] for r in rows)
    w = sum(r["their_result"] < r["our_result"] for r in rows)
    by_team = {}
    for r in rows:
        x = by_team.setdefault(r["team"], {"games": 0, "their_better": 0, "ours_better": 0})
        x["games"] += 1
        x["their_better"] += r["their_result"] > r["our_result"]
        x["ours_better"] += r["their_result"] < r["our_result"]
    rep = {"twins": len(rows), "by_cut": {n: sum(r["cut"] == n for r in rows) for n in CUTS},
           "their_wins": sum(r["their_result"] == 1 for r in rows), "our_wins": sum(r["our_result"] == 1 for r in rows),
           "their_better": b, "ours_better": w,
           "agree_before_cut_mean": round(sum(r["agree_before_cut"] for r in rows) / max(1, len(rows)), 3),
           "by_team": dict(sorted(by_team.items(), key=lambda kv: -kv[1]["games"])),
           "their_minus_ours_next48_where_they_won": dict(sorted(((k, v) for k, v in diff_win.items() if abs(v) >= 3), key=lambda kv: -abs(kv[1]))[:30]),
           "their_minus_ours_next48_all": dict(sorted(((k, v) for k, v in diff_all.items() if abs(v) >= 3), key=lambda kv: -abs(kv[1]))[:30])}
    json.dump(rep, open(os.path.join(OUT, "report.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({k: rep[k] for k in ("twins", "by_cut", "their_wins", "our_wins", "their_better", "ours_better", "agree_before_cut_mean")}))
    print("where they won and we did not, their minus our orders in the next 48 turns:", rep["their_minus_ours_next48_where_they_won"])


if __name__ == "__main__":
    main()
