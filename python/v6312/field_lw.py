"""Field check on v63.11's real losses + an equal random sample of its wins (80-agent gate games, python opponents).

    python python/v6312/field_lw.py CAND_MAIN OUT.jsonl [--workers 6] [--wins 363]"""
import argparse, json, os, random, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(RL, "python", "rshell"))
import public25 as P

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cand"); ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--wins", type=int, default=363)
    a = ap.parse_args()
    base = json.load(open(os.path.join(RL, ".local/v6312/a0_v6311.json")))
    L = [r for r in base if r["gap"] < 0]; W = [r for r in base if r["gap"] >= 0]
    random.Random(6312).shuffle(W)
    pick = L + W[:a.wins]
    done = set()
    if os.path.exists(a.out):
        done = {(r["opp"], r["seed"], r["seat"]) for r in map(json.loads, open(a.out))}
    cand = os.path.abspath(a.cand)
    tasks = [(cand, r["opp"], os.path.join(RL, ".local/field80/field", r["opp"] + ".py"), r["world"], r["seed"], r["seat"]) for r in pick if (r["opp"], r["seed"], r["seat"]) not in done]
    print(len(tasks), "games", flush=True)
    with open(a.out, "a") as f, ProcessPoolExecutor(a.workers) as ex:
        for fu in as_completed([ex.submit(P.play, t) for t in tasks]):
            f.write(json.dumps(fu.result()) + "\n"); f.flush()
