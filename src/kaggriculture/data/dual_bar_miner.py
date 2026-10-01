"""Dual-bar base miner: find tapes strong vs ELITES *and* the mid-field.

2026-09-02 lesson: convergence needs BOTH bars — elite panel >= ~0.75
(the ceiling) and gauntlet ~1.000 (pass-through while K is hot). This
stages over all indexed top-50 wins:
  S1: 8 elite cells (4 spread tapes x 2 seats)      — keep top ~40
  S2: 24 elite cells (12 tapes x 2 seats)           — keep top ~12
  S3: 24 mid-field cells (12 gauntlet opps x 2 seats) — keep ~1.000 only
  S4: report survivors; chassis + full 3-seed panels run manually.

    python src/dual_bar_miner.py
"""
from kaggriculture.paths import ROOT
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

WORK = os.path.join(ROOT, ".local", "dual_miner")
SEED = 901


def main():
    import kaggriculture.data.routes as R
    import kaggriculture.train.train_arms as TA
    import kaggriculture.engine.serve_match as SM
    import kaggriculture.engine.serve_gate as serve_gate
    os.makedirs(WORK, exist_ok=True)
    srv = SM.Serve()
    man = json.load(open(os.path.join(ROOT, ".local", "elite_panel",
                    "manifest.json"), encoding="utf-8"))
    panel_all = [t["tape"] for t in man["teams"]]
    elite4 = panel_all[:: max(1, len(panel_all) // 4)][:4]
    elite12 = panel_all[:: max(1, len(panel_all) // 12)][:12]
    gaunt = [p for p in serve_gate.OPPS[:12] if os.path.exists(p)]

    idx = R.load_index()["routes"]
    top_teams = {t["team"].lower() for t in man["teams"]}

    def epnum(m):
        try:
            return int(str(m.get("episode") or "0").split("+")[0])
        except ValueError:
            return 0

    cands = sorted(
        (rid for rid, m in idx.items()
         if m.get("won") and m.get("engine") == "1.32.7"
         and any(tt in str(m.get("team", "")).lower()
                 or str(m.get("team", "")).lower() in tt
                 for tt in top_teams)),
        key=lambda rid: -epnum(idx[rid]))
    print(f"candidates: {len(cands)}", flush=True)

    def render(rid):
        p = os.path.join(WORK, f"{rid}.py")
        if not os.path.exists(p):
            TA.render(R.load_route(rid), p, f"m_{rid}")
        return p

    def cells(path, opps):
        w = 0
        for tape in opps:
            for seat in (0, 1):
                try:
                    a, b = SM.load_agent(path), SM.load_agent(tape)
                    ab, tb = (SM.run_match(a, b, SEED, srv) if seat == 0
                              else SM.run_match(b, a, SEED, srv)[::-1])
                    w += ab > tb
                except Exception:                              # noqa: BLE001
                    return -1
        return w

    s1 = []
    for i, rid in enumerate(cands):
        try:
            p = render(rid)
        except Exception:                                      # noqa: BLE001
            continue
        w = cells(p, elite4)
        s1.append((w, rid))
        if i % 50 == 0:
            print(f"S1 {i}/{len(cands)} best={max(s1)[0] if s1 else 0}",
                  flush=True)
    s1.sort(reverse=True)
    keep1 = [rid for w, rid in s1[:40] if w >= 5]
    print(f"S1 done: kept {len(keep1)} (>=5/8): "
          f"{[(w, r) for w, r in s1[:10]]}", flush=True)

    s2 = []
    for rid in keep1:
        w = cells(os.path.join(WORK, f"{rid}.py"), elite12)
        s2.append((w, rid))
        print(f"S2 {rid}: {w}/24", flush=True)
    s2.sort(reverse=True)
    keep2 = [rid for w, rid in s2[:12]]

    # S3 = the MID BAND, not the reactive gauntlet.
    #
    # 2026-09-03: the gauntlet is SATURATED -- it returned 96-0 for the
    # candidate AND 96-0 for the incumbent, i.e. it discriminates nothing and
    # this stage was blind. That blindness is what shipped v42.0, which the
    # band panel later showed FAILING the mid band at 62-18 (0.775) while
    # passing every gauntlet cell; the mid-field concession is exactly what
    # stalls the burst before the elite band (the wool-agent failure mode,
    # docs/history/gate-calibration-2026-09-03.md). v43's base scores 1.000 there.
    # So the second bar is now measured on tapes from teams rated 2000-2500,
    # which is where the discrimination actually lives.
    #
    # 2026-09-03 (instrument repair): screen on the SELECTION half only. This
    # stage picks candidates; if it screens on the whole band then
    # `band_panel --score` has no cells left that the search never saw, and
    # the survivor's panel number is the selection read back -- v44_single won
    # 77-11 on the panel that chose it and lost 84 of 86 discordant held-out
    # cells. The complement is `band_panel.panel_tapes("mid", "holdout")`.
    midband = []
    man_path = os.path.join(ROOT, ".local", "band_panel", "mid",
                            "manifest.json")
    if os.path.exists(man_path):
        import kaggriculture.measure.band_panel as BP
        midband = BP.panel_tapes("mid", split="selection", limit=12)
        if not midband:                                   # legacy manifest
            mid_man = json.load(open(man_path, encoding="utf-8"))
            midband = [t["tape"] for t in mid_man.get("teams", [])
                       if os.path.exists(t["tape"])][:12]
    if not midband:
        print("S3: no mid-band panel (build it with "
              "`python src/band_panel.py --build`) -- FALLING BACK to the "
              "saturated gauntlet; treat the second bar as UNMEASURED",
              flush=True)
        midband = gaunt
        bar_name = "gauntlet(saturated)"
    else:
        bar_name = "midband"

    out = []
    for rid in keep2:
        w = cells(os.path.join(WORK, f"{rid}.py"), midband)
        n = 2 * len(midband)
        elite = [w2 for w2, r2 in s2 if r2 == rid][0]
        out.append({"rid": rid, "elite24w": elite,
                    "second_bar": bar_name, "second_bar_cells": f"{w}/{n}",
                    "second_bar_ratio": round(w / n, 3) if n else None,
                    # Ship spec: elite >= ~0.75 AND the second bar ~1.000.
                    "dual_bar_pass": elite >= 18 and n and w / n >= 0.95})
        print(f"S3 {rid}: elite {elite}/24  {bar_name} {w}/{n}"
              f"{'  DUAL-BAR PASS' if out[-1]['dual_bar_pass'] else ''}",
              flush=True)
    json.dump(out, open(os.path.join(WORK, "survivors.json"), "w"), indent=1)
    print("MINER DONE", flush=True)


if __name__ == "__main__":
    main()
