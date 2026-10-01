"""Analyse our own ladder games (data/ladder/<sub>/ from python/ladder_pull.py; queue Q27).

    python python/ladder_analyze.py [--subs 56567216,56538921,56538836,56524454] [--threads 16]

1. Results per agent by opponent band (W/L/D), losses below 2500, close losses (< $3,000).
2. Fidelity: tapeplay --verify on every tape (the engine reproduces the recorded banks), and the agent's
   own rl3 profile replayed against the recorded opponent stream (guarded): if our sim of that agent
   reproduces the real result, counterfactuals on these games mean something.
3. Counterfactuals on the recorded opponent streams: v61.1 (0), v62 (13), v62.1 (19), v63 (35) and any
   --policy weights, each on every game of every agent; paired against the agent that actually played.
   Caveat: the opponent is open-loop (it cannot react to a different us), so this is exact for the agent
   that played and an estimate for the others.
4. Loss anatomy: per lost game, the money gap at the end of each day (recorded play), the day the gap
   last turned against us, the final margin, and our vs their sales in the last 5 days per product.
Writes data/ladder_analysis.json (outside data/ladder, which holds only tapes).
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.environ.get("KRL_BIN") or os.path.join(RL, "target-dev", "release")
TAPEPLAY = os.path.join(BIN, "tapeplay" + (".exe" if os.name == "nt" else ""))
RL3 = os.path.join(RL, "configs", "profiles", "rl3.json")
SHIELD = os.path.join(RL, "configs", "shield", "v1.json")
NAMES = {"56567216": ("v63", 35), "56538921": ("v62.1", 19), "56538836": ("v62", 13), "56524454": ("v61.1", 0)}
TAB = chr(9)
PRODUCTS = ["CARROT", "EGG", "FERTILIZER", "MELON", "MILK", "STRAWBERRY", "TOMATO", "WHEAT", "WOOL"]


def play(args, files, threads):
    shards = [files[i::threads] for i in range(threads) if files[i::threads]]

    def one(shard):
        d = tempfile.mkdtemp(prefix="ladder-")
        try:
            for f in shard:
                shutil.copy(f, d)
            r = subprocess.run([TAPEPLAY, "--tapes", d, *args], cwd=RL, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"tapeplay rc {r.returncode}: {r.stderr[-400:]}")
            return {ln.split("\t")[0]: [float(x) for x in ln.split("\t")[2:6]] for ln in r.stdout.splitlines() if ln.count("\t") >= 5}
        finally:
            shutil.rmtree(d, ignore_errors=True)

    out = {}
    with ThreadPoolExecutor(len(shards)) as ex:
        for res in ex.map(one, shards):
            out.update(res)
    return out


def res(m):
    return "W" if m > 0 else "L" if m < 0 else "D"


def sales(t, seat, lo, hi):
    """Requested SELL units per product over steps lo..hi from a tape's recorded actions."""
    out = dict.fromkeys(PRODUCTS, 0)
    for a in t["actions"][lo:hi]:
        x = a[seat] if a and len(a) > seat else None
        for o in (x or {}).get("market") or []:
            if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL" and o[1] in out:
                try:
                    out[o[1]] += int(o[2])
                except (TypeError, ValueError):
                    pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", default="56567216,56538921,56538836,56524454")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--policy", action="append", default=[], help="NAME=weights.bin, played under the shield")
    a = ap.parse_args()
    rep = {"agents": {}}
    all_files = {}
    for sub in a.subs.split(","):
        files = sorted(glob.glob(os.path.join(RL, "data", "ladder", sub, "tapes", "*.json")))
        if not files:
            continue
        name, k = NAMES.get(sub, (sub, None))
        tapes = {os.path.basename(f)[:-5]: json.load(open(f, encoding="utf-8")) for f in files}
        all_files[sub] = (files, tapes, k)
        rows = []
        for tid, t in tapes.items():
            s = t["seat"]
            rw = t.get("rewards") or [0, 0]
            m = float(rw[s] or 0) - float(rw[1 - s] or 0)
            rows.append((tid, t.get("band") or "unknown", m, t.get("opp_team"), t.get("opp_rating")))
        by = {}
        for _, b, m, _, _ in rows:
            x = by.setdefault(b, {"W": 0, "L": 0, "D": 0})
            x[res(m)] += 1
        low = [r for r in rows if r[1] in ("lt2100", "2100-2300", "2300-2500") and r[2] < 0]
        rep["agents"][name] = {"sub": sub, "games": len(rows), "W": sum(r[2] > 0 for r in rows), "L": sum(r[2] < 0 for r in rows),
                               "by_band": by, "losses_below_2500": len(low),
                               "close_losses": sum(1 for r in rows if -3000 < r[2] < 0),
                               "losses": sorted([{"id": r[0], "band": r[1], "margin": r[2], "opp": r[3], "opp_rating": r[4]} for r in rows if r[2] < 0], key=lambda x: x["margin"])}
        print(f"[ladder] {name}: {len(rows)} games {rep['agents'][name]['W']}W/{rep['agents'][name]['L']}L; by band {by}; "
              f"losses below 2500: {len(low)}; close losses: {rep['agents'][name]['close_losses']}", flush=True)

    # 2 + 3: fidelity and counterfactuals, every game of every agent
    files = [f for v in all_files.values() for f in v[0]]
    # the verify replay also dumps both seats' money per step and the dayobs vectors (opponent-group signal)
    tmp = tempfile.mkdtemp(prefix="ladder-an-")
    ver, money, grp = {}, {}, {}
    try:
        shards = [files[i::a.threads] for i in range(a.threads) if files[i::a.threads]]

        def one(ix):
            d = os.path.join(tmp, f"s{ix}")
            os.makedirs(d)
            for f in shards[ix]:
                shutil.copy(f, d)
            r = subprocess.run([TAPEPLAY, "--tapes", d, "--verify", "--money-dump", d + ".money", "--obs-dump", d + ".obs"], cwd=RL, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(r.stderr[-400:])
            v = {ln.split(TAB)[0]: [float(x) for x in ln.split(TAB)[2:6]] for ln in r.stdout.splitlines() if ln.count(TAB) >= 5}
            m, g = {}, {}
            for ln in open(d + ".money", encoding="utf-8"):
                i_, t_, u_, th_ = ln.rstrip().split(TAB)
                m.setdefault(i_, {})[int(t_)] = float(u_) - float(th_)
            seat_of = {os.path.basename(f)[:-5]: json.load(open(f, encoding="utf-8"))["seat"] for f in shards[ix]}
            for ln in open(d + ".obs", encoding="utf-8"):
                x = ln.rstrip().split(TAB)
                if int(x[2]) == 1 and int(x[1]) == seat_of.get(x[0], -1):
                    vec = [float(y) for y in x[3:]]
                    # the shield's group rule: DIFFERENT if we shared under 10% of day-0 squares, else COPY on equal cash at step 1
                    g[x[0]] = {"layout_sim": vec[87], "pos_equal_day0": vec[90], "cash_equal_step1": vec[89],
                               "group": "DIFFERENT" if vec[90] < 0.1 else ("COPY" if vec[89] >= 0.5 else "PARTIAL")}
            return v, m, g

        with ThreadPoolExecutor(len(shards)) as ex:
            for v, m, g in ex.map(one, range(len(shards))):
                ver.update(v)
                money.update(m)
                grp.update(g)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    rep["verify_exact"] = [sum(1 for v in ver.values() if v[0] == v[2] and v[1] == v[3]), len(ver)]
    print(f"[ladder] engine replay exact: {rep['verify_exact'][0]}/{rep['verify_exact'][1]}", flush=True)
    cands = {"v61.1": ["--profiles", RL3, "--guarded", "--pa", "0"], "v62": ["--profiles", RL3, "--guarded", "--pa", "13"],
             "v62.1": ["--profiles", RL3, "--guarded", "--pa", "19"], "v63": ["--profiles", RL3, "--guarded", "--pa", "35"]}
    for p in a.policy:
        n, w = p.split("=", 1)
        cands[n] = ["--profiles", RL3, "--guarded", "--policy", w] + (["--shield", SHIELD] if os.path.exists(SHIELD) else [])
    cf = {n: play(args, files, a.threads) for n, args in cands.items()}
    rep["counterfactual"] = {}
    for sub, (fs, tapes, k) in all_files.items():
        name = NAMES.get(sub, (sub,))[0]
        ids = [os.path.basename(f)[:-5] for f in fs]
        real = {i: (tapes[i]["rewards"][tapes[i]["seat"]] or 0) - (tapes[i]["rewards"][1 - tapes[i]["seat"]] or 0) for i in ids}
        row = {}
        for n, r in cf.items():
            got = {i: r[i][2] - r[i][3] for i in ids if i in r}
            sc = lambda m: (m > 0) - (m < 0)  # noqa: E731
            better = sum(sc(got[i]) > sc(real[i]) for i in got)
            worse = sum(sc(got[i]) < sc(real[i]) for i in got)
            same_bank = sum(got[i] == real[i] for i in got)
            row[n] = {"W": sum(m > 0 for m in got.values()), "L": sum(m < 0 for m in got.values()), "vs_real": [better, worse], "same_margin": same_bank}
        rep["counterfactual"][name] = row
        print(f"[ladder] counterfactual on {name}'s {len(ids)} games (W/L, better/worse vs what really happened, identical margin): " +
              "; ".join(f"{n} {v['W']}/{v['L']} +{v['vs_real'][0]}/-{v['vs_real'][1]} same {v['same_margin']}" for n, v in row.items()), flush=True)

    # 4: opponent groups and loss anatomy (money gap per step from the exact replay)
    for sub, (fs, tapes, k) in all_files.items():
        name = NAMES.get(sub, (sub,))[0]
        ag = rep["agents"][name]
        by_group = {}
        for f in fs:
            tid = os.path.basename(f)[:-5]
            t = tapes[tid]
            m = (t["rewards"][t["seat"]] or 0) - (t["rewards"][1 - t["seat"]] or 0)
            g = grp.get(tid, {}).get("group", "unknown")
            x = by_group.setdefault(g, {"W": 0, "L": 0, "D": 0})
            x[res(m)] += 1
        ag["by_group"] = by_group
        anat = []
        for L in ag["losses"]:
            t = tapes[L["id"]]
            s_ = t["seat"]
            mg = money.get(L["id"], {})
            day_end = {d: mg.get(24 * d + 23) for d in range(30) if 24 * d + 23 in mg}
            last_lead = max((st for st, v in mg.items() if v > 0), default=None)
            anat.append({"id": L["id"], "margin": L["margin"], "band": L["band"], "opp": L["opp"], "group": grp.get(L["id"], {}).get("group"),
                         "layout_sim": grp.get(L["id"], {}).get("layout_sim"),
                         "gap_start_last_day": mg.get(695), "gap_step_710": mg.get(710), "gap_step_718": mg.get(718), "final": mg.get(max(mg) if mg else 0),
                         "last_step_we_led": last_lead, "led_at_day": {d: v for d, v in day_end.items() if d in (6, 12, 15, 20, 25, 28)},
                         "last_day_sales_us": {p: v for p, v in sales(t, s_, 696, 720).items() if v},
                         "last_day_sales_them": {p: v for p, v in sales(t, 1 - s_, 696, 720).items() if v}})
        ag["loss_anatomy"] = anat
        led_final = sum(1 for x in anat if (x["gap_start_last_day"] or 0) > 0)
        led_718 = sum(1 for x in anat if (x["gap_step_718"] or 0) > 0)
        ag["losses_led_into_last_day"] = led_final
        ag["losses_led_at_step_718"] = led_718
        print(f"[ladder] {name}: by opponent group {by_group}; of {len(anat)} losses, {led_final} led at the start of the last day, "
              f"{led_718} still led at step 718", flush=True)
    json.dump(rep, open(os.path.join(RL, "data", "ladder_analysis.json"), "w"), indent=1)
    print(f"[ladder] -> data/ladder_analysis.json")


if __name__ == "__main__":
    main()
