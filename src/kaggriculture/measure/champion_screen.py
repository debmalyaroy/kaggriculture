"""Champion screen: measure the TOP LADDER TEAMS' own routes against a
panel of top agents, and against our incumbent (docs/pair-improvement-plan).

The whole project has crowned our own medoid-mined tapes against our recent
LOSS tapes -- and our agents then underperform on the ladder because the
loss-tape panel is not the top of the field. We now hold hundreds of the
current top-12 teams' routes in the index. This tool:

  1. PANEL = the best recent WINNING route of each top-N leaderboard team
     (the strongest opponents there are), used as referees.
  2. CANDIDATES = our incumbent + each top team's best recent winning route.
  3. Score every candidate vs the panel on the Rust batch engine, both seats,
     common seeds; a candidate's own team is excluded from its own panel.

Output: models/factory/champion_screen.json, ranked by field-relative score.
A candidate that beats the incumbent here is a ladder-improvement candidate
even if it loses to the incumbent HEAD-TO-HEAD (near-mirror dynamics do not
predict ladder rating; field performance does -- proven 2026-08-20).

    python src/champion_screen.py --top 8 --seeds 12
"""
from kaggriculture.paths import ROOT
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

# Current top-of-leaderboard teams (kaggle competitions leaderboard, 08-20).
TOP_TEAMS = ["Ryo Hasegawa", "tetsuya", "Arman Tuganbaev", "Crop Dusta",
             "カワシギ", "Izzoudine Mohamed KANTA", "Subramanya N",
             "Xiaowenhao404", "peikopon", "ReCurSiON", "Efe Can Celiksoy",
             "Thomas Tschinkel"]


def best_recent_win(idx, team, min_bank=0.0):
    """That team's best-bank WINNING route from its most recent play date."""
    rs = [r for r in idx["routes"].values()
          if r.get("team") == team and r.get("won")
          and float(r.get("bank", 0)) >= min_bank]
    if not rs:
        return None
    newest = max(r.get("date", "") for r in rs)
    recent = [r for r in rs if r.get("date", "") == newest] or rs
    return max(recent, key=lambda r: float(r.get("bank", 0)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=40000)
    ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args()

    import kaggriculture.data.routes as R
    import kaggriculture.pipeline.sell_search as SS
    import kaggriculture.measure.win_metric as WM
    idx = R.load_index()

    inc_id, inc_src = SS.default_base()
    print("incumbent base: %s (from %s)" % (inc_id, inc_src))

    # Champions: best recent winning route per top team.
    champs = []
    for t in TOP_TEAMS[:args.top]:
        rec = best_recent_win(idx, t)
        if rec:
            champs.append(rec)
            print("  champion %-24s %s  bank %9.0f  date %s" % (
                t, rec["id"], float(rec.get("bank", 0)), rec.get("date")))
        else:
            print("  champion %-24s (no winning route in index)" % t)
    if len(champs) < 3:
        raise SystemExit("too few champion routes to screen")

    os.makedirs(SS.WORK, exist_ok=True)
    # Panel tapes = all champions (referees). Candidates = incumbent + champs.
    panel_tapes, panel_teams = [], []
    for i, rec in enumerate(champs):
        panel_tapes.append(SS.write_tape(R.load_route(rec["id"]),
                                         os.path.join(SS.WORK,
                                                      "champ_p%d.tape" % i)))
        panel_teams.append(rec.get("team"))
    seeds = list(range(args.seed0, args.seed0 + args.seeds))

    cand_recs = [{"id": inc_id, "team": "__incumbent__"}] + champs
    cand_tapes = [SS.write_tape(R.load_route(c["id"]),
                                os.path.join(SS.WORK, "champ_c%d.tape" % i))
                  for i, c in enumerate(cand_recs)]

    # Score each candidate vs the panel EXCLUDING its own team's referee.
    evals = SS.batch_eval(cand_tapes, panel_tapes, seeds, args.threads)
    inc_ev = evals[0]

    def net_score(ev, own_team):
        cells = {k: v for k, v in ev["cells"].items()
                 if panel_teams[k[0]] != own_team}
        return (sum(cells.values()) / len(cells)) if cells else 0.0, cells

    inc_score, inc_cells = net_score(inc_ev, "__incumbent__")
    print("\nincumbent vs champion panel: %.4f (%d cells)" % (
        inc_score, len(inc_cells)))

    rows = []
    for i, c in enumerate(cand_recs[1:], start=1):
        ev = evals[i]
        sc, cells = net_score(ev, c.get("team"))
        common = sorted(set(cells) & set(inc_cells))
        t = WM.paired_test([cells[k] for k in common],
                           [inc_cells[k] for k in common])
        better = t["score_diff"] > 0 and t["significant"]
        rows.append({"id": c["id"], "team": c.get("team"),
                     "field_score": sc, "vs_incumbent_diff": t["score_diff"],
                     "p": t["p_value"], "beats_incumbent": better})
        print("  %-24s %s  field %.4f  vs-inc %+.4f  p=%.4f%s" % (
            c.get("team"), c["id"], sc, t["score_diff"], t["p_value"],
            "  ** BEATS INCUMBENT **" if better else ""))

    rows.sort(key=lambda r: -r["field_score"])
    winner = rows[0] if rows and rows[0]["field_score"] > inc_score else None
    print("\nBEST field score: %s (%s) %.4f vs incumbent %.4f" % (
        rows[0]["team"], rows[0]["id"], rows[0]["field_score"], inc_score)
        if rows else "none")
    print("VERDICT: %s" % (
        "champion %s beats the incumbent on the field panel -> build it"
        % winner["id"] if winner and winner["beats_incumbent"]
        else "no champion sign-beats the incumbent on the field panel"))

    json.dump({"when": __import__("datetime").datetime.now()
               .isoformat(timespec="seconds"),
               "incumbent": {"id": inc_id, "field_score": inc_score},
               "panel_teams": panel_teams, "rows": rows,
               "winner": winner},
              open(os.path.join(ROOT, "models", "factory",
                                "champion_screen.json"), "w",
                   encoding="utf-8"), indent=1)
    print("wrote models/factory/champion_screen.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
