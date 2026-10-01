"""H4: assemble endgame labels from the lineage (T2, endg-label rows with 24 gaps), the public field (H1 jsonl) and the
top tapes (H2 tsv, endg-label row format) onto ONE proposal subset, and write the matching proposals file.

    python python/v6312/h4_assemble.py --props 0,2,3,4,8,12,20 --out-dir data/v6312/h4 \
        --t2 .local/v6312/t2_local.tsv[,more] --h1 data/v6312/h1_labels.jsonl [--h2 data/v6312/h2_labels.tsv]

Writes OUT/props.json (the subset, proposal 0 first) and OUT/labels_{t2,h1,h2}.tsv in endg-label row format:
id seat seed opp world x[NF] gap_k... (python/endg/train.py reads these).
"""
import argparse, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--props", default="0,2,3,4,8,12,20")
ap.add_argument("--out-dir", required=True)
ap.add_argument("--t2", default="")
ap.add_argument("--h1", default="")
ap.add_argument("--h2", default="")
a = ap.parse_args()
ks = [int(x) for x in a.props.split(",")]
assert ks[0] == 0
os.makedirs(a.out_dir, exist_ok=True)
P = json.load(open("configs/endg/proposals.json"))
sub = dict(P)
sub["proposals"] = [P["proposals"][k] for k in ks]
sub["mode"] = "rerank"
json.dump(sub, open(os.path.join(a.out_dir, "props.json"), "w"), indent=1)
K = len(P["proposals"])


def from_endg_rows(paths, out, tag):
    n = 0
    with open(out, "w") as f:
        for p in [x for x in paths.split(",") if x]:
            for l in open(p):
                r = l.rstrip("\n").split("\t")
                if len(r) < 5 + K:
                    continue
                g = r[-K:]
                f.write("\t".join(r[:-K] + [g[k] for k in ks]) + "\n")
                n += 1
    print(tag, n, "rows")


if a.t2:
    from_endg_rows(a.t2, os.path.join(a.out_dir, "labels_t2.tsv"), "t2")
if a.h2:
    from_endg_rows(a.h2, os.path.join(a.out_dir, "labels_h2.tsv"), "h2")
if a.h1:
    n = 0
    with open(os.path.join(a.out_dir, "labels_h1.tsv"), "w") as f:
        for l in open(a.h1):
            g = json.loads(l)
            gaps = [g["gaps"][str(k)] for k in ks]
            f.write("\t".join([f"f{g['seed']}_{g['opp']}", str(g["seat"]), str(g["seed"]), g["opp"], g["world"]]
                              + [repr(float(v)) for v in g["x"]] + [repr(float(v)) for v in gaps]) + "\n")
            n += 1
    print("h1", n, "rows")
