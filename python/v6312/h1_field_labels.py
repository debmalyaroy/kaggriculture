"""H1: closed-loop endgame labels against the PUBLIC agents (python opponents; each proposal is a full replay because a
python agent cannot be copied mid-game). Label agent = c4 without its dispatcher, endgame forced to proposal K.

    python python/v6312/h1_field_labels.py --out data/v6312/h1_labels.jsonl [--workers 12] [--props 0,2,3,4,8,12,20]

Games: the opponents in .local/v6312/h1_opps.json x every world's first TRAIN seed x seat 0 (held-out seeds stay for the
gate). Row per game: opp, world, seed, seat, x (endgame inputs at the decision, from KRL_ENDG_LOG), gaps {K: gap}.
Resumable: finished (game, K) runs are kept in OUT.runs.jsonl.
"""
import argparse, json, os, shutil, sys, tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(RL, "python", "rshell"))
import public25 as P  # noqa: E402

TMP = os.path.join(RL, ".local", "v6312", "h1_tmp")


def cand_dir(k):
    d = os.path.join(RL, ".local", "v6312", f"cand_h1_k{k}")
    if not os.path.exists(os.path.join(d, "main.py")):
        shutil.copytree(os.path.join(RL, ".local", "v6312", "cand_c4"), d, dirs_exist_ok=True)
        shutil.rmtree(os.path.join(d, "dispatch"), ignore_errors=True)
        props = json.load(open(os.path.join(RL, "configs", "endg", "proposals.json")))
        props["mode"] = f"force:{k}"
        json.dump(props, open(os.path.join(d, "endg.json"), "w"), indent=1)
    return os.path.join(d, "main.py")


def run(task):
    cand, opp, world, seed, seat, k = task
    fd, path = tempfile.mkstemp(prefix="endg_", suffix=".jsonl", dir=TMP)
    os.close(fd)
    os.remove(path)
    os.environ["KRL_ENDG_LOG"] = path
    row = P.play((cand, opp, os.path.join(RL, ".local", "field80", "field", opp + ".py"), world, seed, seat))
    x = None
    try:
        for l in open(path):
            d = json.loads(l)
            if d.get("player") == seat:
                x = d["x"]
        os.remove(path)
    except OSError:
        pass
    return {"opp": opp, "world": world, "seed": seed, "seat": seat, "k": k, "gap": row.get("gap"), "x": x, "error": row.get("error")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--props", default="0,2,3,4,8,12,20")
    a = ap.parse_args()
    os.makedirs(TMP, exist_ok=True)
    ks = [int(x) for x in a.props.split(",")]
    cands = {k: cand_dir(k) for k in ks}
    bank = json.load(open(os.path.join(RL, "data", "worlds", "w64_bank.json")))["train"]
    opps = json.load(open(os.path.join(RL, ".local", "v6312", "h1_opps.json")))
    runs = a.out + ".runs.jsonl"
    done = set()
    if os.path.exists(runs):
        for l in open(runs):
            r = json.loads(l)
            done.add((r["opp"], r["seed"], r["seat"], r["k"]))
    tasks = [(cands[k], o, w, ss[0], 0, k) for o in opps for w, ss in sorted(bank.items()) for k in ks if (o, ss[0], 0, k) not in done]
    print(len(tasks), "runs to do", flush=True)
    with open(runs, "a") as f, ProcessPoolExecutor(a.workers) as ex:
        for i, fu in enumerate(as_completed([ex.submit(run, t) for t in tasks])):
            f.write(json.dumps(fu.result()) + "\n")
            f.flush()
            if i % 200 == 0:
                print(i, "runs", flush=True)
    # assemble one row per game with every proposal's gap
    G = {}
    for l in open(runs):
        r = json.loads(l)
        g = G.setdefault((r["opp"], r["seed"], r["seat"]), {"opp": r["opp"], "world": r["world"], "seed": r["seed"], "seat": r["seat"], "x": None, "gaps": {}})
        if r.get("gap") is not None:
            g["gaps"][str(r["k"])] = r["gap"]
        if r["k"] == 0 and r.get("x"):
            g["x"] = r["x"]
    with open(a.out, "w") as f:
        for g in G.values():
            if g["x"] and len(g["gaps"]) == len(ks):
                f.write(json.dumps(g) + "\n")
    print("games labelled", sum(1 for g in G.values() if g["x"] and len(g["gaps"]) == len(ks)))


if __name__ == "__main__":
    main()
