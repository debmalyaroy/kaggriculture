"""Mine REAL frontier tapes from cached leaderboard loss replays.

Each loss replay (.local/lossreplays/{ep}.json, kaggle-env format) holds the
OPPONENT's full 719-step action stream -- the actual winning play on the current
ladder. This extracts each winner's tape (serialised to our "farmer\\thands\\tmarket"
format), tagged with their realised world (first two unlocked shops) and final
bank, ranked by bank. These are real, current frontier tapes -- not stale clones.

Output: .local/frontier_mined/<bank>_<ep>_<world>.tape  + a ranked index.
Usage: python src/mine_frontier_tapes.py [--min-bank 100000] [--world SUBSTR]
"""
from kaggriculture.paths import ROOT
import argparse, glob, json, os, sys

CACHE = os.path.join(ROOT, ".local", "lossreplays")
TOP = os.path.join(ROOT, "data", "episodes", "top")
OUT = os.path.join(ROOT, ".local", "frontier_mined")


def op_str(o):
    return " ".join(str(t) for t in o) if isinstance(o, list) else str(o)


def row(action):
    f = action.get("farmer") or ["PASS"]
    farmer = op_str(f)
    hands = ";".join(op_str(h) for h in (action.get("hands") or []))
    market = ";".join(op_str(o) for o in (action.get("market") or []))
    return f"{farmer}\t{hands}\t{market}"


def realized_world(steps, seat):
    for s in steps:
        obs = s[seat].get("observation") or {}
        sh = ((obs.get("town") or {}).get("unlocked_shops")) or []
        if len(sh) >= 2:
            return f"{sh[0]}|{sh[1]}"
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-bank", type=float, default=90000)
    ap.add_argument("--world", default=None)
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--src", default="loss", choices=["loss","top"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows = []
    base = CACHE if a.src=="loss" else TOP
    files = sorted(glob.glob(os.path.join(base, "**", "*.json"), recursive=True))[: a.limit]
    print(f"scanning {len(files)} replays...", flush=True)
    for fp in files:
        ep = os.path.basename(fp)[:-5]
        try:
            r = json.load(open(fp))
            steps = r["steps"]; rw = r.get("rewards") or [0, 0]
        except Exception:
            continue
        if not steps or len(steps) < 700:
            continue
        win = 0 if rw[0] >= rw[1] else 1  # the winner's seat
        bank = rw[win]
        if bank < a.min_bank:
            continue
        world = realized_world(steps, win)
        if a.world and a.world not in world:
            continue
        # serialise the winner's action tape
        tape = [row(s[win].get("action") or {}) for s in steps]
        name = f"{int(bank):07d}_{ep}_{world.replace('|','-')}.tape"
        with open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(tape) + "\n")
        rows.append((bank, ep, world, name))
    rows.sort(reverse=True)
    idx = os.path.join(OUT, "index.tsv")
    with open(idx, "w", encoding="utf-8", newline="\n") as fh:
        for bank, ep, world, name in rows:
            fh.write(f"{bank:.0f}\t{ep}\t{world}\t{name}\n")
    print(f"mined {len(rows)} frontier tapes -> {OUT}")
    print("top 15 by bank:")
    for bank, ep, world, name in rows[:15]:
        print(f"  {bank:8.0f}  {world:30}  ep{ep}")
    # world distribution
    from collections import Counter
    wc = Counter(w for _, _, w, _ in rows)
    print("\nrealized-world coverage (top):", dict(wc.most_common(10)))


if __name__ == "__main__":
    main()
