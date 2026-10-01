"""Per-world tail TOURNAMENT on manufactured seeds (Rust engine).

Operator direction (2026-09-06): don't pick a world's tail by the bank it
happened to record in one ladder game — make the candidates PLAY for the
seat. For every world with donors:

  candidates = every whole-prefix-exact donor tail we hold for the world,
               spliced onto OUR opening, PLUS the bare base continuation
               (tail = base route 145..719) as the null candidate;
  panel      = the strongest other donors' original full tapes (same clone
               opening, so the games diverge exactly at the routing turn);
  seeds      = that world's rows in data/worlds/seed_bank.json.

Scored by sell_search.batch_eval: W/L per the ladder's currency, both
seats, draws 0.5. The winner takes the world in world_tails.json; if the
NULL candidate wins, the world carries no tail (evidence-based drop).
Open-loop caveat applies (this is a pre-ranker); the band/gauntlet remain
the ship judge.

    python src/trackp/harness/tail_tournament.py
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import base64
import json
import os
import re
import sys
import zlib

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                              # noqa: BLE001
    pass
import kaggriculture.pipeline.sell_search as SS  # noqa: E402
from kaggriculture.trackp import routes_io as R                              # noqa: E402

WORK = os.path.join(ROOT, ".local", "tail_tournament")
SPLIT = 145
N_SEEDS = 8
MIN_EDGE = 0.05     # winner must beat the incumbent tail's score by this


def our_route():
    src = open(os.path.join(ROOT, "agents", "v45.0_bandit.py"),
               encoding="utf-8").read()
    m = re.search(r'_ROUTE = json\.loads\(zlib\.decompress\(base64\.'
                  r'b85decode\("([^"]+)"\)\)', src)
    return json.loads(zlib.decompress(
        base64.b85decode(m.group(1))).decode("utf-8"))


def main():
    os.makedirs(WORK, exist_ok=True)
    route = our_route()
    tails = json.load(open(os.path.join(ROOT, ".local", "candidates",
                                        "world_tails.json"), encoding="utf-8"))
    hits = json.load(open(os.path.join(ROOT, ".local", "candidates",
                                       "clone_winner_hits.json"),
                          encoding="utf-8"))
    bank = json.load(open(os.path.join(ROOT, "data", "worlds",
                                       "seed_bank.json"), encoding="utf-8"))
    prune = set(json.load(open(os.path.join(ROOT, ".local", "candidates",
                                            "tail_prune.json"),
                               encoding="utf-8")))
    by_world = {}
    for h in hits:
        w = h.get("world")
        if w:
            by_world.setdefault(w, []).append(h)

    changed = dropped = kept = 0
    for w in sorted(set(list(tails) + list(by_world))):
        seeds = (bank.get(w) or [])[:N_SEEDS]
        if len(seeds) < 4:
            continue
        # candidates: incumbent tail + up to 4 alt donors + NULL (base)
        cands = []          # (label, tail_actions or None)
        if w in tails:
            cands.append(("incumbent:" + tails[w]["team"], tails[w]["tail"]))
        alts = sorted(by_world.get(w, []), key=lambda h: -h["bank"])[:4]
        seen = {tails[w]["episode"]} if w in tails else set()
        for h in alts:
            if h["episode"] in seen:
                continue
            try:
                acts = R.load_route(h["id"])
            except Exception:                                  # noqa: BLE001
                continue
            cands.append((f"alt:{h['team'][:16]}:{h['id']}", acts[SPLIT:719]))
            seen.add(h["episode"])
        cands.append(("null:base", None))
        if len(cands) < 3:
            kept += 1
            continue
        cpaths = []
        for i, (label, tail) in enumerate(cands):
            acts = route[:SPLIT] + (tail if tail is not None
                                    else route[SPLIT:719])
            cpaths.append(SS.write_tape(
                acts, os.path.join(WORK, f"c{i}.tape")))
        # panel: the two strongest donors' ORIGINAL tapes PLUS every
        # candidate tape. The 2026-09-06 BT check showed tails picked only
        # vs donor tapes can lose the MIRROR war (32% of the ladder is our
        # own family), so fitness must include tail-vs-tail games;
        # self-pairings tie at 0.5 and penalize all candidates equally.
        panel = list(cpaths)
        for j, h in enumerate(alts[:2]):
            try:
                panel.append(SS.write_tape(
                    R.load_route(h["id"]),
                    os.path.join(WORK, f"p{j}.tape")))
            except Exception:                                  # noqa: BLE001
                pass
        if len(panel) <= len(cpaths):
            kept += 1
            continue
        res = SS.batch_eval(cpaths, panel, seeds, 6)
        order = sorted(range(len(cands)),
                       key=lambda i: -res[i]["score"])
        win = order[0]
        inc = 0 if w in tails else None
        line = "  ".join(f"{cands[i][0]}={res[i]['score']:.3f}"
                         for i in order[:3])
        verdict = "keep"
        if cands[win][0].startswith("null"):
            if w in tails and res[win]["score"] > res[inc]["score"] + MIN_EDGE:
                del tails[w]
                verdict = "DROP tail"
                dropped += 1
        elif (inc is None
              or (win != inc
                  and res[win]["score"] > res[inc]["score"] + MIN_EDGE)):
            label = cands[win][0]
            rid = label.split(":")[-1]
            h = next(x for x in alts if x["id"] == rid)
            tails[w] = {"bank": h["bank"], "opp_bank": h.get("opp_bank", 0),
                        "episode": h["episode"], "seat": h["seat"],
                        "team": h["team"], "tail": cands[win][1]}
            prune.discard(w)
            verdict = "SWITCH -> " + h["team"][:16]
            changed += 1
        else:
            kept += 1
        print(f"{w:36s} {verdict:22s} {line}", flush=True)

    json.dump(tails, open(os.path.join(ROOT, ".local", "candidates",
                                       "world_tails.json"), "w",
                          encoding="utf-8"))
    json.dump(sorted(prune), open(os.path.join(ROOT, ".local", "candidates",
                                               "tail_prune.json"), "w"))
    print(f"\nSWITCHED {changed}, DROPPED {dropped}, kept {kept}; "
          f"{len(tails)} worlds carry tails")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
