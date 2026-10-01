"""UPSET GAUNTLET: every world where a LOWER-rated opponent beat us.

Operator sustain criterion (2026-09-05): 100% below 2000, 90% in 2000-2500,
70%+ above, and NEVER lose downward. The upsets are the rating killers --
11 of v44.1's first 25 losses. This band is built from exactly those games,
so the objective is BINARY per cell: the incumbent lost it to a weaker
opponent; a candidate must FLIP it without collapsing the opponent's tape.

Cells come from .local/hardband/upset_cells.json (live_status upset scan).
Each cell is certified like the hard band: the RECORDED game (both tracks
forced) must reproduce the recorded banks to the dollar on the Rust engine,
or the cell is dropped -- an uncertified tape flatters candidates.

CREDITED flip rule (same spirit as counterfactual.py): a flip counts only
if the opponent still banks >= 60% of their recorded bank; beating a
desynced corpse is not a win.

    python .local/hardband/upset_gauntlet.py --certify
    python .local/hardband/upset_gauntlet.py <agent.py>
"""
from kaggriculture.paths import ROOT
import argparse
import json
import os
import sys

HB = os.path.join(ROOT, ".local", "hardband")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                              # noqa: BLE001
    pass
import kaggriculture.engine.serve_match as SM  # noqa: E402

CELLS = os.path.join(HB, "upset_cells.json")
CERT = os.path.join(HB, "upset_certified.json")
CACHE = os.path.join(ROOT, ".local", "lossreplays")
COLLAPSE = 0.60


def load_cell(c):
    p = os.path.join(CACHE, f"{c['episode']}.json")
    if not os.path.exists(p):
        return None
    rep = json.load(open(p, encoding="utf-8"))
    seed = (rep.get("info") or {}).get("seed")
    if seed is None:
        return None
    steps = rep.get("steps") or []
    # derive OUR seat from the recorded banks -- the ListEpisodes agents
    # array is NOT seat-ordered (the 0/36 certification failure of
    # 2026-09-05 was exactly this)
    try:
        farms = steps[-1][0]["observation"]["farms"]
        m0 = float(farms[0].get("money") or 0)
        seat = 0 if abs(m0 - float(c["my"])) < 1 else 1
        if seat == 1 and abs(float(farms[1].get("money") or 0)
                             - float(c["my"])) >= 1:
            return None                    # neither seat matches: stale row
    except Exception:                                          # noqa: BLE001
        return None
    # replay step i stores the action that LED TO state i -- the action
    # chosen AT state i is at step i+1 (verified 2026-09-05: route
    # purchases at step 144 appear at replay index 145)
    ours = [((st[seat] or {}).get("action") or {}) for st in steps[1:]]
    theirs = [((st[1 - seat] or {}).get("action") or {}) for st in steps[1:]]
    return seed, ours, theirs, seat


def play_tapes(srv, seed, acts0, acts1):
    js = srv.cmd(f"RESET {int(seed)}")
    n = min(len(acts0), len(acts1))
    for i in range(n):
        if js.get("done") or int(js.get("step") or 0) >= 719:
            break
        la = SM.action_to_line(acts0[i])
        lb = SM.action_to_line(acts1[i])
        js = srv.cmd(f"STEP2 {la}\x1e{lb}")
        if "error" in js:
            raise RuntimeError(js["error"])
    return [float(f.get("money") or 0) for f in js["farms"]]


def play_agent(srv, agent, opp_acts, seed, seat):
    js = srv.cmd(f"RESET {int(seed)}")
    while not js.get("done") and int(js.get("step") or 0) < 719:
        i = int(js.get("step") or 0)
        mine = SM.action_to_line(agent(SM.obs_for(seat, js)))
        theirs = SM.action_to_line(opp_acts[min(i, len(opp_acts) - 1)])
        la, lb = (mine, theirs) if seat == 0 else (theirs, mine)
        js = srv.cmd(f"STEP2 {la}\x1e{lb}")
        if "error" in js:
            raise RuntimeError(js["error"])
    b = [float(f.get("money") or 0) for f in js["farms"]]
    return b[seat], b[1 - seat]


def cmd_certify():
    cells = json.load(open(CELLS, encoding="utf-8"))
    srv = SM.Serve()
    ok, out = 0, []
    try:
        for c in cells:
            r = load_cell(c)
            if r is None:
                print(f"  ep{c['episode']}: replay missing", flush=True)
                continue
            seed, ours, theirs, seat = r
            a0, a1 = (ours, theirs) if seat == 0 else (theirs, ours)
            banks = play_tapes(srv, seed, a0, a1)
            mine, opp = banks[seat], banks[1 - seat]
            exact = abs(mine - c["my"]) < 1 and abs(opp - c["op"]) < 1
            print(f"  ep{c['episode']} opp {c['op_rating']:.0f} "
                  f"{'EXACT' if exact else 'OFF'} "
                  f"({mine:,.0f}/{c['my']:,.0f} vs {opp:,.0f}/{c['op']:,.0f})",
                  flush=True)
            if exact:
                ok += 1
                out.append(c)
    finally:
        srv.close()
    json.dump(out, open(CERT, "w", encoding="utf-8"), indent=1)
    print(f"\ncertified {ok}/{len(cells)} upset cells -> {CERT}")
    return 0


def cmd_score(agent_path, offset=0, limit=0):
    cells = json.load(open(CERT, encoding="utf-8"))
    if offset:
        cells = cells[offset:]
    if limit:
        cells = cells[:limit]
    srv = SM.Serve()
    flips = held = 0
    rows = []
    try:
        for c in cells:
            seed, ours, theirs, seat = load_cell(c)
            agent = SM.load_agent(os.path.abspath(agent_path))
            mine, opp = play_agent(srv, agent, theirs, seed, seat)
            credited = mine > opp and opp >= COLLAPSE * c["op"]
            flips += credited
            held += mine > opp and not credited
            rows.append({"ep": c["episode"], "mine": mine, "opp": opp,
                         "credited": bool(credited),
                         "won": bool(mine > opp)})
            print(f"  ep{c['episode']} opp {c['op_rating']:.0f}  "
                  f"{'FLIP' if credited else ('uncred-win' if mine > opp else 'still lost')} "
                  f"{mine:>9,.0f} vs {opp:>9,.0f} "
                  f"(recorded loss {c['my']:,.0f} vs {c['op']:,.0f})",
                  flush=True)
    finally:
        srv.close()
    out = os.path.join(HB, f"gaunt_{os.path.basename(agent_path)}.json")
    if (offset or limit) and os.path.exists(out):
        prior = json.load(open(out, encoding="utf-8"))
        seen = {r["ep"] for r in rows}
        rows = [r for r in prior if r["ep"] not in seen] + rows
    json.dump(rows, open(out, "w"), indent=1)
    fl = sum(r["credited"] for r in rows)
    hd = sum(r["won"] and not r["credited"] for r in rows)
    print(f"\n{os.path.basename(agent_path)}: CREDITED FLIPS {fl}/{len(rows)}"
          f" (+{hd} uncredited) <- every cell was a real upset loss")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("agent", nargs="?")
    ap.add_argument("--certify", action="store_true")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.certify:
        return cmd_certify()
    if not a.agent:
        ap.error("pass an agent to score, or --certify")
    return cmd_score(a.agent, a.offset, a.limit)


if __name__ == "__main__":
    raise SystemExit(main())
