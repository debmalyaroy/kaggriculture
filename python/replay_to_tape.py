"""Ladder replay -> compact tape for `tapeplay` (crates/runner/src/bin/tapeplay.rs).

    python python/replay_to_tape.py <replay.json[.gz]>... --out DIR [--seat-from games.json]

Tape JSON: {"id", "seed" (info.seed = the engine seed), "seat" (our seat, or 0 if unknown),
"rewards": [b0, b1], "actions": [[a0, a1], ...]} where actions[t] is the pair applied at step t
(= replay steps[t+1][*].action; a missing/invalid action is null -> PASS)."""
import argparse
import gzip
import json
import os


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def convert(rp, seat=0):
    steps = rp["steps"]
    acts = []
    for t in range(len(steps) - 1):
        pair = []
        for s in (0, 1):
            a = steps[t + 1][s].get("action")
            pair.append(a if isinstance(a, dict) else None)
        acts.append(pair)
    return {"id": rp.get("info", {}).get("EpisodeId") or rp.get("id"), "seed": rp["info"]["seed"], "seat": seat,
            "rewards": rp.get("rewards"), "actions": acts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("replays", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seat-from", help="ourladder games.json (id -> our seat)")
    a = ap.parse_args()
    seats = {}
    if a.seat_from:
        seats = {str(g["id"]): g["seat"] for g in json.load(open(a.seat_from))}
    os.makedirs(a.out, exist_ok=True)
    n = 0
    for p in a.replays:
        eid = os.path.basename(p).split(".")[0]
        if seats and eid not in seats:
            continue
        rp = load(p)
        if not (rp.get("info") or {}).get("seed"):
            continue
        with open(os.path.join(a.out, f"{eid}.json"), "w") as fh:
            json.dump(convert(rp, seats.get(eid, 0)), fh, separators=(",", ":"))
        n += 1
    print(f"wrote {n} tapes to {a.out}")


if __name__ == "__main__":
    main()
