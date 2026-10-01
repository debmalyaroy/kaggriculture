"""Per-loss autopsy: WHICH STEP did we lose, and what would have changed it.

For every loss of a live submission this fetches the replay once, finds the
decisive step, and compares our economy against the opponent's and against
the measured field median. Designed to be run hourly next to
`src/live_status.py`.

    python src/loss_autopsy.py                    # losses of the active pair
    python src/loss_autopsy.py --subs 55978480
    python src/loss_autopsy.py --episodes 105100118
    python src/loss_autopsy.py --json

WHY NET WORTH, NOT BANK. Bank alone mis-times the decisive moment: a farm
that just spent 2,400 on cows looks behind and is not. Every comparison
here uses net worth = money + livestock (at purchase price, decayed by age)
+ shed stock at live market price. The decisive step is the LAST time the
net-worth lead changed hands -- after it, the loser never leads again.

Replays are ~32 MB and Kaggle's rolling 24h download cap is 2,000, tracked
in data/sameday/quota_ledger.json. This records what it pulls and refuses
to spend more than --max-pulls per run so a loss burst can never starve the
release scrape.
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import base64
import datetime as dt
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

for _s in ("stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

HERE = os.path.dirname(os.path.abspath(__file__))

CACHE = os.path.join(ROOT, ".local", "lossreplays")
REPORT = os.path.join(ROOT, ".local", "loss_autopsy.json")
LEDGER = os.path.join(ROOT, "data", "sameday", "quota_ledger.json")
QUOTA_CAP = 2000

BASE_PRICE = {"MILK": 160, "STRAWBERRY": 120, "WOOL": 200, "EGG": 50,
              "WHEAT": 25, "CARROT": 35, "TOMATO": 60, "MELON": 250,
              "FERTILIZER": 100}
ANIMAL_COST = {"COW": 400, "SHEEP": 500, "GOOSE": 300}
# Field medians per game for CURRENT top-10 teams, engine 1.32.7, measured
# 2026-09-03 over 80 sampled winning routes USING THE SAME ARITHMETIC AS
# counts() ABOVE: animals are ORDER COUNTS, sells are sums of min(qty, 100).
#
# The first version of this table was wrong and reported phantom leaks. It was
# copied from docs/plan-2800 section 1, where the numbers came from a stored
# signature that SUMS ORDER QUANTITY (COW 9, SHEEP 5, WHEAT 644). Measured in
# order-count units the top-10 median is COW 7 / SHEEP 3 -- exactly what our
# own base does -- so the tool flagged a "herd leak" that an index-wide test
# had already disproved (win rate by animals bought: 48.9% / 52.8% / 55.6%,
# inside noise). NEVER copy a median across tools without re-deriving it in
# the receiving tool's own units.
FIELD = {"COW": 7, "SHEEP": 3, "MILK": 236, "WOOL": 154, "FERTILIZER": 328,
         "STRAWBERRY": 282, "WHEAT": 500, "MELON": 72}
# A shortfall this large against the field median is called out as a leak.
LEAK_FRAC = 0.80


# ----------------------------------------------------------------- fetching

def _quota_used(now=None):
    now = now or time.time()
    try:
        ev = json.load(open(LEDGER, encoding="utf-8"))["events"]
    except (OSError, ValueError, KeyError):
        return 0
    return sum(1 for t in ev if now - t < 86400)


_MODE = {"fallback": False, "stop429": False}
FALLBACK_CAP = 60          # web-channel pulls per rolling 24h, on top of
                           # the primary cap; deliberately small and paced


def _fallback_used(now=None):
    now = now or time.time()
    try:
        ev = json.load(open(LEDGER, encoding="utf-8")).get("web_events") or []
    except (OSError, ValueError):
        return 0
    return sum(1 for t in ev if now - t < 86400)


def _fallback_record(n=1):
    for _ in range(6):
        try:
            try:
                d = json.load(open(LEDGER, encoding="utf-8"))
            except (OSError, ValueError):
                d = {"events": []}
            d.setdefault("web_events", []).extend([time.time()] * n)
            json.dump(d, open(LEDGER, "w", encoding="utf-8"))
            return
        except PermissionError:
            time.sleep(0.5)


def _quota_record(n=1):
    try:
        d = json.load(open(LEDGER, encoding="utf-8"))
    except (OSError, ValueError):
        d = {"events": []}
    d["events"].extend([time.time()] * n)
    json.dump(d, open(LEDGER, "w", encoding="utf-8"))


def replay(eid, allow_fetch=True):
    """Cached replay JSON, or None when it is not on disk and may not be pulled."""
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"{eid}.json")
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    if not allow_fetch:
        return None
    url = f"https://www.kaggleusercontent.com/episodes/{eid}.json"
    if _MODE["fallback"]:
        time.sleep(8)                      # paced: the fallback is a guest
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            body = r.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            _MODE["stop429"] = True        # Kaggle said no: stop the channel
        raise
    open(p, "wb").write(body)
    (_fallback_record if _MODE["fallback"] else _quota_record)()
    return json.loads(body)


# ------------------------------------------------------------------ metrics

def _tiles(farm):
    for row in farm.get("tiles") or []:
        for c in (row if isinstance(row, list) else []):
            if isinstance(c, dict):
                yield c


def net_worth(obs, seat):
    """money + livestock (cost, linearly decayed by age) + shed at live price."""
    farm = obs["farms"][seat]
    total = float(farm.get("money") or 0)
    for c in _tiles(farm):
        a = c.get("animal")
        if isinstance(a, dict):
            cost = ANIMAL_COST.get(str(a.get("type")), 400)
            age, life = float(a.get("age") or 0), 30.0
            total += cost * max(0.0, 1.0 - age / life)
    prices = (obs.get("market") or {}).get("prices") or {}
    shed = (farm.get("shed") or {}) if isinstance(farm.get("shed"), dict) else {}
    for item, qty in shed.items():
        if isinstance(qty, (int, float)):
            total += qty * float(prices.get(item, BASE_PRICE.get(item, 0)))
    return total


def counts(steps, seat):
    """BUY_ANIMAL and SELL totals for a seat over the episode."""
    bought, sold = {}, {}
    for i in range(len(steps)):
        act = steps[i][seat].get("action") or {}
        for o in act.get("market") or []:
            if not (isinstance(o, list) and o):
                continue
            if o[0] == "BUY_ANIMAL":
                bought[o[1]] = bought.get(o[1], 0) + 1
            elif o[0] == "SELL":
                try:
                    sold[o[1]] = sold.get(o[1], 0) + min(float(o[2] or 0), 100)
                except (TypeError, ValueError):
                    pass
    return bought, sold


def regime(steps):
    """Mean MILK/STRAWBERRY/EGG/WOOL price ratio at d3-5, d9-12, d20-24.

    The 4-product statistic separates worlds far better than the 2-product
    one (Cohen d 1.33 vs 0.82 at d9-12); see the calibration note at the top
    of src/band_panel.py.
    """
    keys = ("MILK", "STRAWBERRY", "EGG", "WOOL")
    out = {}
    for label, (d0, d1) in (("d3-5", (3, 5)), ("d9-12", (9, 12)),
                            ("d20-24", (20, 24))):
        vals = []
        for day in range(d0, d1 + 1):
            i = min(day * 24 + 12, len(steps) - 1)
            px = (steps[i][0]["observation"].get("market") or {}).get("prices") or {}
            if px:
                vals.append(statistics.mean(px[k] / BASE_PRICE[k]
                                            for k in keys if k in px))
        out[label] = round(statistics.mean(vals), 3) if vals else None
    return out


def autopsy(eid, allow_fetch=True):
    rep = replay(eid, allow_fetch)
    if rep is None:
        return {"episode": eid, "skipped": "not cached and quota exhausted"}
    steps = rep["steps"]
    names = (rep.get("info") or {}).get("TeamNames") or []
    me = names.index("Debmalya") if "Debmalya" in names else 0
    rewards = rep.get("rewards") or [0, 0]

    # Net-worth curve, sampled per day.
    curve = []
    for day in range(30):
        i = min(day * 24 + 23, len(steps) - 1)
        obs = steps[i][0]["observation"]
        mine, theirs = net_worth(obs, me), net_worth(obs, 1 - me)
        farm_m, farm_t = obs["farms"][me], obs["farms"][1 - me]
        cnt = lambda f, k: sum(1 for c in _tiles(f) if c.get(k))   # noqa: E731
        curve.append({"day": day, "mine": mine, "theirs": theirs,
                      "gap": mine - theirs,
                      "my_animals": cnt(farm_m, "animal"),
                      "their_animals": cnt(farm_t, "animal"),
                      "my_money": farm_m.get("money"),
                      "their_money": farm_t.get("money")})

    # Decisive day: the last day the net-worth lead changed hands.
    decisive, prev = None, None
    for row in curve:
        sign = (row["gap"] > 0) - (row["gap"] < 0)
        if prev is not None and sign != 0 and sign != prev:
            decisive = row["day"]
        if sign != 0:
            prev = sign
    # Biggest single-day swings against us, after the decisive day.
    swings = []
    for a, b in zip(curve, curve[1:]):
        swings.append({"day": b["day"], "delta": b["gap"] - a["gap"]})
    worst = sorted(swings, key=lambda s: s["delta"])[:3]

    my_buy, my_sell = counts(steps, me)
    tt_buy, tt_sell = counts(steps, 1 - me)
    leaks = []
    for k, med in FIELD.items():
        mine = my_buy.get(k) if k in ANIMAL_COST else my_sell.get(k, 0)
        if mine is not None and med and mine < med * LEAK_FRAC:
            leaks.append({"metric": k, "ours": mine, "field_median": med,
                          "theirs": (tt_buy.get(k) if k in ANIMAL_COST
                                     else tt_sell.get(k, 0))})
    # world + tail attribution (v46.1 world-tail router, 2026-09-06):
    # the shop pair names the world; if the shipped router carries a tail
    # for it, the loss happened ON that tail (the swap is deterministic).
    shops = []
    try:
        i150 = min(150, len(steps) - 1)
        shops = ((steps[i150][0]["observation"].get("town") or {})
                 .get("unlocked_shops") or [])[:2]
    except Exception:                                          # noqa: BLE001
        pass
    world = "|".join(shops) if len(shops) == 2 else None
    tail = None
    try:
        wt = json.load(open(os.path.join(ROOT, "models", "trackp",
                                         "world_tails_meta.json"),
                            encoding="utf-8"))
        if world in wt:
            tail = wt[world]["team"]
    except Exception:                                          # noqa: BLE001
        pass
    return {"episode": eid, "names": names, "we_are_seat": me,
            "our_bank": rewards[me], "their_bank": rewards[1 - me],
            "margin": rewards[me] - rewards[1 - me],
            "world": world, "tail_donor": tail,
            "regime": regime(steps), "decisive_day": decisive,
            "worst_swings": worst, "curve": curve,
            "bought": {"ours": my_buy, "theirs": tt_buy},
            "sold": {"ours": my_sell, "theirs": tt_sell},
            "leaks_vs_field": leaks}


# ------------------------------------------------------------------ driving

def losses_of(sub_id, limit=None):
    import kaggriculture.pipeline.live_status as LS
    rows = LS.episodes(sub_id)
    out = [r for r in rows if r["my"] < r["op"]]
    return out[:limit] if limit else out


def episode_ids(sub_id):
    """Episode ids paired with results (the summary API omits ids)."""
    import kaggriculture.pipeline.live_status as LS
    req = urllib.request.Request(
        LS.ENDPOINT, data=json.dumps({"submissionId": int(sub_id)}).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": LS._auth_header()})
    with urllib.request.urlopen(req, timeout=90) as r:
        payload = json.loads(r.read())
    out = []
    for e in payload.get("episodes") or []:
        ags = e.get("agents") or []
        me = next((a for a in ags if str(a.get("submissionId")) == str(sub_id)), None)
        op = next((a for a in ags if str(a.get("submissionId")) != str(sub_id)), None)
        if not me or not op or me.get("reward") is None or op.get("reward") is None:
            continue
        out.append({"episode": e.get("id"), "my": float(me["reward"]),
                    "op": float(op["reward"]), "end": e.get("endTime"),
                    "op_rating": op.get("initialScore")})
    out.sort(key=lambda r: str(r["end"] or ""), reverse=True)
    return out


def render(reports):
    L = []
    for a in reports:
        if a.get("skipped"):
            L.append(f"ep{a['episode']}: {a['skipped']}")
            continue
        L.append("=" * 68)
        L.append(f"ep{a['episode']}  vs {a['names'][1 - a['we_are_seat']]}  "
                 f"{a['our_bank']:,.0f} vs {a['their_bank']:,.0f}  "
                 f"({a['margin']:+,.0f})")
        r = a["regime"]
        L.append(f"  price regime d3-5 {r['d3-5']}  d9-12 {r['d9-12']}  "
                 f"d20-24 {r['d20-24']}"
                 + ("   LOW-PRICE WORLD" if (r["d20-24"] or 1) < 0.85 else ""))
        L.append(f"  decisive day (last lead change): {a['decisive_day']}")
        L.append("  worst single-day swings against us: "
                 + ", ".join(f"d{s['day']} {s['delta']:+,.0f}"
                             for s in a["worst_swings"]))
        L.append("  net worth / animals by day:")
        for row in a["curve"]:
            if row["day"] % 4 and row["day"] != 29:
                continue
            L.append(f"    d{row['day']:<3} {row['mine']:>9,.0f} vs "
                     f"{row['theirs']:>9,.0f}  gap {row['gap']:>+9,.0f}   "
                     f"animals {row['my_animals']:>2} vs {row['their_animals']:>2}")
        if a["leaks_vs_field"]:
            L.append("  LEAKS vs field median:")
            for k in a["leaks_vs_field"]:
                L.append(f"    {k['metric']:<11} ours {k['ours']:>6,.0f}  "
                         f"field {k['field_median']:>6,.0f}  "
                         f"opponent {k['theirs']:>6,.0f}")
    return "\n".join(L) or "no losses to analyse"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subs", nargs="*", default=None)
    ap.add_argument("--episodes", nargs="*", type=int, default=None)
    ap.add_argument("--max-pulls", type=int, default=4,
                    help="replays this run may download (protects the scrape)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-fallback", action="store_true",
                    help="disable the web-channel fallback budget")
    args = ap.parse_args()

    if args.episodes:
        targets = list(args.episodes)
    else:
        import kaggriculture.pipeline.live_status as LS
        subs = args.subs or [r for r, _ in LS.active_pair()]
        targets = []
        for s in subs:
            targets += [e["episode"] for e in episode_ids(s) if e["my"] < e["op"]]

    headroom = max(0, QUOTA_CAP - _quota_used())
    budget = min(args.max_pulls, headroom)
    # STANDING ORDER (operator, recorded 2026-09-05): quota exhaustion must
    # NOT stall analysis -- switch channel instead of waiting. When the
    # primary budget is spent, continue on a paced FALLBACK budget (separate
    # ledger key, 8s spacing, hard daily cap). A 429 from Kaggle stops the
    # fallback immediately -- the next channel after that is the browser
    # session (claude-in-chrome), not more retries.
    fb_used = _fallback_used()
    fb_budget = 0
    if budget == 0 and not args.no_fallback:
        fb_budget = max(0, min(args.max_pulls, FALLBACK_CAP - fb_used))
        if fb_budget:
            print(f"quota spent -> web-channel fallback: {fb_budget} pulls "
                  f"(paced 8s, {fb_used}/{FALLBACK_CAP} used today)",
                  flush=True)
    reports, pulled = [], 0
    for eid in targets:
        cached = os.path.exists(os.path.join(CACHE, f"{eid}.json"))
        _MODE["fallback"] = (not cached) and pulled >= budget
        if _MODE["stop429"]:
            reports.append({"episode": eid,
                            "skipped": "429 from Kaggle: channel stopped; "
                                       "next channel is the browser session"})
            continue
        if not cached and pulled >= budget + fb_budget:
            reports.append({"episode": eid,
                            "skipped": f"download budget spent "
                                       f"({budget} this run, {headroom} in quota)"})
            continue
        try:
            reports.append(autopsy(eid, allow_fetch=True))
            pulled += 0 if cached else 1
        except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
            reports.append({"episode": eid, "skipped": f"{type(exc).__name__}: {exc}"})

    if args.json:
        print(json.dumps(reports, indent=1, default=str))
    else:
        print(render(reports))
    json.dump({"at": dt.datetime.now().isoformat(), "reports": reports},
              open(REPORT, "w", encoding="utf-8"), indent=1, default=str)


if __name__ == "__main__":
    main()
