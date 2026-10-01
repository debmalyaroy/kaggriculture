"""Paired test of the learned endgame controller against v63.6_rl as-is (same agent, no --endg).

    KRL_V2_BIN=target-endg/release python python/endg/test.py --endg weights/endg/v1/endg.json [--threads 12] [--band]

  lineage   64 worlds x 3 held-out bank seeds x FIT opponents (v63, v62.1, v62, v61.1, v63.1_rl, rand) x 2 seats,
            closed loop (python/rshell/panel.py); the mirror (v63.6 itself) reported separately
  public    every public-25 loss tape of v63.5_rl (data/rshell/public25/rca/tapes), open loop
  band      (--band) the 991-tape real-player gate (python/band_gate.py), candidate and reference
Writes data/endg/test_<name>.json.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(RL, "python", "rshell"))
import panel as P  # noqa: E402

POL = os.path.join(P.RUN, "snapshots", "i000910.bin")
TAIL = f"--shield {P.SHIELD} --shell {P.BIG1} --chain-off r127,sm,r95"
V636 = f"--profiles {P.RL3} --policy {POL} {TAIL}"


def band(extra, name, threads):
    r = subprocess.run([sys.executable, os.path.join(RL, "python", "band_gate.py"), "--cand", POL, "--cand-args",
                        f"--shell {P.BIG1} --chain-off r127,sm,r95 {extra}".strip(), "--name", name, "--ref-profiles", P.V64,
                        "--ref-profile", "35", "--ref-group", "35,35,36", "--threads", str(threads)],
                       cwd=RL, capture_output=True, text=True, env={**os.environ, "KRL_BIN": P.BIN})
    line = (r.stdout.strip().splitlines() or [""])[-1]
    m = re.search(r"losses below 2500 = (\d+); win rate 2500\+ = ([0-9.]+).*?paired vs ref \+(\d+)/-(\d+)", line)
    return {"losses_below_2500": int(m.group(1)) if m else None, "rate_2500plus": float(m.group(2)) if m else None,
            "paired_vs_v631": [int(m.group(3)), int(m.group(4))] if m else None, "line": line[-300:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endg", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--band", action="store_true")
    a = ap.parse_args()
    name = a.name or os.path.basename(os.path.dirname(os.path.abspath(a.endg)))
    cand = f"{V636} --endg {os.path.abspath(a.endg)}"
    rep = {"endg": a.endg, "ref": V636, "cand": cand}
    seeds = P.bank_seeds("heldout", 3, 0)
    opps = dict(P.FIT)
    opps["mirror"] = V636
    rc, _ = P.closed_loop(V636, seeds, a.threads, opponents=opps)
    cc, _ = P.closed_loop(cand, seeds, a.threads, opponents=opps)
    fit = lambda d: {k: v for k, v in d.items() if k[0] != "mirror"}  # noqa: E731
    rep["lineage"] = P.compare(fit(cc), fit(rc))
    rep["mirror"] = P.compare({k: v for k, v in cc.items() if k[0] == "mirror"}, {k: v for k, v in rc.items() if k[0] == "mirror"})
    rep["lineage_by_opp"] = {o: P.compare({k: v for k, v in cc.items() if k[0] == o}, {k: v for k, v in rc.items() if k[0] == o}) for o in opps}
    print(f"[endg-test {name}] lineage (FIT, held-out): +{rep['lineage']['better']}/-{rep['lineage']['worse']} p {rep['lineage']['p']} | "
          f"mirror +{rep['mirror']['better']}/-{rep['mirror']['worse']}", flush=True)
    losses = sorted(glob.glob(os.path.join(RL, "data", "rshell", "public25", "rca", "tapes", "*.json")))
    ro = P.open_loop(V636, {"l": losses}, a.threads)["l"]
    co = P.open_loop(cand, {"l": losses}, a.threads)["l"]
    rep["public_losses"] = P.compare(co, ro)
    print(f"[endg-test {name}] public-25 loss tapes: +{rep['public_losses']['better']}/-{rep['public_losses']['worse']} p {rep['public_losses']['p']}", flush=True)
    if a.band:
        rep["band"] = {"cand": band(f"--endg {os.path.abspath(a.endg)}", f"endg-{name}", a.threads), "ref": band("", "endg-ref-v636", a.threads)}
        print(f"[endg-test {name}] band: cand {rep['band']['cand']['losses_below_2500']} <2500 {rep['band']['cand']['paired_vs_v631']} | "
              f"ref {rep['band']['ref']['losses_below_2500']} {rep['band']['ref']['paired_vs_v631']}", flush=True)
    for tag, agent in (("cand", cand), ("ref", V636)):
        r = subprocess.run([P.SELFPLAY, "--seeds", ",".join(str(s) for _, s in seeds[:32]), "--threads", "4", "--a-args", agent, "--b-args", P.OPPONENTS["v63"]],
                           cwd=RL, capture_output=True, text=True)
        worst = max((float(x.split("	")[3]) for x in r.stdout.splitlines() if x.count("	") >= 4), default=None)
        rep[f"worst_turn_ms_{tag}"] = worst / 1000 if worst else None
    print(f"[endg-test {name}] worst turn (32 games, native box build): cand {rep['worst_turn_ms_cand']} ms, ref {rep['worst_turn_ms_ref']} ms", flush=True)
    os.makedirs(os.path.join(RL, "data", "endg"), exist_ok=True)
    json.dump(rep, open(os.path.join(RL, "data", "endg", f"test_{name}.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
