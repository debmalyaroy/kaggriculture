"""Gate / checkpoint tournament on Kaggle (queue Q14 / Q17): keeps thousands of validation games off the
laptop while PPO runs.

    python python/gate_kaggle.py gate [--n 100]          # the newest BC teacher vs v61.1
    python python/gate_kaggle.py tournament [--n 60]     # BC + PPO best + last 3 snapshots

1. copies each candidate's weights.bin + python/learn/gate.py into the PRIVATE dataset
   debmalya84/krl-weights (created on first use, a new version each run) and waits until it is ready;
2. pushes one private notebook per candidate (kaggle/kernels/krl-gate-<i>, sources: that dataset +
   kaggriculture-rl-bin), which runs gate.py with the Linux ppo-rollout;
3. waits, downloads each notebook's gates/*.json into data/gates, and (tournament) writes the ranking
   to data/gates/tournament_latest.json exactly as the local gate.py would.
Resumable through data/kaggle_out/gate_state.json. Exit 0 = all notebooks completed; the verdicts
are in the JSONs (a FAIL verdict is a result, not an error).
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "learn"))
import kjob  # noqa: E402

RL = kjob.RL
OUT_ROOT = os.path.join(RL, "data", "kaggle_out")
STATE = os.path.join(OUT_ROOT, "gate_state.json")
GATES = os.path.join(RL, "data", "gates")
WDS = os.path.join(RL, "kaggle", "weights_dataset")
WDS_ID = f"{kjob.OWNER}/krl-weights"
BIN_DS = f"{kjob.OWNER}/kaggriculture-rl-bin"
TEMPLATE = os.path.join(RL, "kaggle", "gate", "notebook_template.py")
WAIT_HOURS = 6


def load():
    try:
        return json.load(open(STATE, encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save(st):
    os.makedirs(OUT_ROOT, exist_ok=True)
    json.dump(st, open(STATE + ".tmp", "w", encoding="utf-8"), indent=1)
    os.replace(STATE + ".tmp", STATE)


def candidates(mode):
    import gate  # python/learn/gate.py
    if mode == "gate":
        return [gate.resolve(os.path.join(RL, "weights", "bc", "LATEST"))]
    return gate.discover()


def publish_weights(cands):
    shutil.rmtree(WDS, ignore_errors=True)
    os.makedirs(WDS)
    files = []
    for name, w in cands:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name) + ".bin"
        shutil.copy(w, os.path.join(WDS, safe))
        files.append((name, safe))
    shutil.copy(os.path.join(RL, "python", "learn", "gate.py"), os.path.join(WDS, "gate.py"))
    json.dump({"title": "krl-weights", "id": WDS_ID, "licenses": [{"name": "CC0-1.0"}]},
              open(os.path.join(WDS, "dataset-metadata.json"), "w"), indent=1)
    rc, out = kjob.kaggle(["datasets", "status", WDS_ID])
    cwd = os.getcwd()
    os.chdir(WDS)        # the CLI builds a temp path from -p; a nested relative path breaks it
    try:
        if "ready" in out or "pending" in out:
            rc, out = kjob.kaggle(["datasets", "version", "-p", ".", "-m", f"gate candidates {kjob.now()}"], timeout=3600)
        else:
            rc, out = kjob.kaggle(["datasets", "create", "-p", "."], timeout=3600)   # private by default
    finally:
        os.chdir(cwd)
    if rc != 0:
        raise RuntimeError(f"weights dataset upload failed: {out[-400:]}")
    for _ in range(90):
        rc, out = kjob.kaggle(["datasets", "status", WDS_ID])
        if "ready" in out:
            return files
        time.sleep(20)
    raise RuntimeError("weights dataset never became ready")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gate", "tournament"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    n = a.n or (100 if a.mode == "gate" else 60)
    st = load()
    if not st.get("notebooks"):
        cands = candidates(a.mode)
        if not cands:
            kjob.log("gate: no candidates")
            return
        files = publish_weights(cands)
        code0 = open(TEMPLATE, encoding="utf-8").read()
        st = {"mode": a.mode, "n": n, "notebooks": {}}
        for i, (name, fname) in enumerate(files):
            kn = f"krl-gate-{i}"
            code = (code0.replace("__NAME__", repr(name)).replace("__WFILE__", repr(fname))
                    .replace("__N__", str(n)).replace("__THREADS__", str(a.threads)))
            kjob.make_kernel(kn, [BIN_DS, WDS_ID], code)
            st["notebooks"][kn] = {"candidate": name, "phase": "planned"}
        save(st)
        kjob.log(f"gate ({a.mode}): {len(files)} candidates: " + ", ".join(f for f, _ in files))
    for kn, nb in st["notebooks"].items():
        if nb["phase"] == "planned":
            nb["pushed_utc"] = kjob.push(kn)
            nb["phase"] = "pushed"
            shutil.rmtree(os.path.join(OUT_ROOT, kn), ignore_errors=True)
            save(st)
    t0 = time.time()
    os.makedirs(GATES, exist_ok=True)
    while any(v["phase"] == "pushed" for v in st["notebooks"].values()):
        for kn, nb in st["notebooks"].items():
            if nb["phase"] != "pushed":
                continue
            s = kjob.status(kn)
            if s == "COMPLETE":
                d = os.path.join(OUT_ROOT, kn)
                kjob.download(kn, d, "gates/manifest.txt", nb["pushed_utc"])
                for f in glob.glob(os.path.join(d, "gates", "*.json")):
                    shutil.copy(f, GATES)
                    nb["result"] = os.path.basename(f)
                nb["phase"] = "done"
                shutil.rmtree(d, ignore_errors=True)
                save(st)
                r = json.load(open(os.path.join(GATES, nb["result"])))
                kjob.log(f"GATE {nb['candidate']}: {r['verdict']} score {r['cand_score']:.3f} vs v61.1 {r['ref_score']:.3f} "
                         f"(+{r['better']}/-{r['worse']}, p {r['p']:.3g}; worse families {r['worse_families'] or 'none'})")
            elif s in ("ERROR", "CANCEL_ACKNOWLEDGED", "CANCELLED"):
                raise RuntimeError(f"{kn} ended {s}; see its log on Kaggle")
        if time.time() - t0 > WAIT_HOURS * 3600:
            raise RuntimeError("gate notebooks still running after the wait limit")
        if any(v["phase"] == "pushed" for v in st["notebooks"].values()):
            time.sleep(kjob.POLL_SEC)
    results = [json.load(open(os.path.join(GATES, v["result"]))) for v in st["notebooks"].values()]
    if st["mode"] == "tournament":
        results.sort(key=lambda r: (r["verdict"] == "PASS", r["cand_score"]), reverse=True)
        summ = {"t": kjob.now(), "ranking": [{k: r[k] for k in ("candidate", "weights", "verdict", "cand_score", "ref_score", "better", "worse", "p")} for r in results]}
        json.dump(summ, open(os.path.join(GATES, f"tournament__{summ['t'].replace(':', '')}.json"), "w"), indent=1)
        json.dump(summ, open(os.path.join(GATES, "tournament_latest.json"), "w"), indent=1)
        kjob.log("TOURNAMENT ranking: " + "; ".join(f"{r['candidate']} {r['verdict']} {r['cand_score']:.3f}" for r in summ["ranking"]))
    save({})


if __name__ == "__main__":
    main()
