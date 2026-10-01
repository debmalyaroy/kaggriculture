"""Self-play + MIRROR game generator for the bandit NN heads.

Plays strong public/ref agents against each other across many worlds and streams
the SAME 65-feature vectors + labels used by extract.py, plus a CLONE label:

  * CROSS game  (agent A vs agent B)  -> clone = 0
  * MIRROR game (agent A vs agent A)  -> clone = 1   (the self-clone positive)

The clone head learns to detect, from OUR observable state, whether the opponent
is mirroring us -- in a true mirror the two farms evolve symmetrically, so the
money gap and opp/my money deltas (already in the feature vector) carry the
signal. Also emits opp/sale rows to AUGMENT the GM data with diverse matchups.

Memory-lean: flushes fixed-size numpy chunks to .local/nn/sp_chunks (clone_*),
reusing extract._feat / _sells / label thresholds so features stay bit-identical
to the live agent (parity-verified).

Run: python -m kaggriculture.bandit.nn.selfplay --seeds 12
"""
from __future__ import annotations
import argparse, os, glob, itertools, shutil
import numpy as np
from kaggriculture.paths import ROOT
import kaggriculture.measure.eval_harness as EH
import kaggriculture.bandit.nn.extract as EX

OUT = os.path.join(ROOT, ".local", "nn")
SP = os.path.join(OUT, "sp_chunks")

# strong, cleanly-loading agents (public + crown-panel refs)
ROSTER = [
    ("tschinkel",   os.path.join(ROOT, "agents", "pub_tschinkel_2945.py")),
    ("alperen",     os.path.join(ROOT, "agents", "pub_alperen_rhythm.py")),
    ("tetsutani",   os.path.join(ROOT, ".local/crown_panel/refs/2500-2700/pub_tetsutani_mirror.py")),
    ("nathanjacob", os.path.join(ROOT, ".local/crown_panel/refs/2500-2700/pub_nathanjacob_anticlone.py")),
    ("metacounter", os.path.join(ROOT, ".local/crown_panel/refs/2500-2700/kaggriculture-metacounter-r1-scored-agent.py")),
    ("bestmarket",  os.path.join(ROOT, ".local/crown_panel/refs/2500-2700/best-market-agent-high-strategy__AGENT_B64.py")),
]


def _steps_of(a0, a1, seed):
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.run([a0, a1])
    return env.steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=5000)
    ap.add_argument("--chunk-rows", type=int, default=200_000)
    a = ap.parse_args()
    roster = [(n, p) for n, p in ROSTER if os.path.exists(p)]
    agents = {n: EH._as_agent(p) for n, p in roster}
    print(f"selfplay roster: {[n for n,_ in roster]}", flush=True)
    shutil.rmtree(SP, ignore_errors=True); os.makedirs(SP, exist_ok=True)
    F = EX.NFEAT
    CH = a.chunk_rows
    cX = np.zeros((CH, F), np.float32); cy = np.zeros(CH, np.float32)          # clone
    oX = np.zeros((CH, F), np.float32); oy = np.zeros(CH, np.float32)          # opp (augment)
    sX = np.zeros((CH, F), np.float32); sY = np.zeros((CH, len(EX.PRODUCTS)), np.float32)  # sale
    n = 0; chunk = 0; total = 0; games = 0
    names = [n for n, _ in roster]

    def flush(cnt):
        nonlocal chunk
        if cnt < 1: return
        np.save(os.path.join(SP, f"clone_X_{chunk:03d}.npy"), cX[:cnt])
        np.save(os.path.join(SP, f"clone_y_{chunk:03d}.npy"), cy[:cnt])
        np.save(os.path.join(SP, f"opp_X_{chunk:03d}.npy"), oX[:cnt])
        np.save(os.path.join(SP, f"opp_y_{chunk:03d}.npy"), oy[:cnt])
        np.save(os.path.join(SP, f"sale_X_{chunk:03d}.npy"), sX[:cnt])
        np.save(os.path.join(SP, f"sale_Y_{chunk:03d}.npy"), sY[:cnt])
        print(f"  flushed sp chunk {chunk:03d} ({cnt} rows; total {total})", flush=True)
        chunk += 1

    pairings = list(itertools.combinations_with_replacement(names, 2))  # i<=j incl mirrors
    for si in range(a.seeds):
        seed = a.seed0 + si
        for i, j in pairings:
            clone = 1.0 if i == j else 0.0
            try:
                steps = _steps_of(agents[i], agents[j], seed)
            except Exception as e:
                print(f"  skip {i} vs {j} seed{seed}: {e}", flush=True); continue
            games += 1
            T = len(steps)
            # observe from seat 0's perspective; opponent = seat 1
            W = 0
            for t in range(T):
                obs = (steps[t][W] or {}).get("observation") or {}
                if not obs.get("market"):
                    continue
                feat, shed = EX._feat(steps, t, W)
                sells = EX._sells((steps[t][W] or {}).get("action"))
                sale_lab = [1.0 if (sells.get(p, 0) >= EX.SALE_MIN or
                                    sells.get(p, 0) >= EX.SALE_FRAC * max(float(shed.get(p, 0)), 1.0))
                            else 0.0 for p in EX.PRODUCTS]
                dump = 0
                for k in range(1, EX.OPP_HORIZON + 1):
                    if t + k >= T: break
                    os_ = EX._sells((steps[t + k][1 - W] or {}).get("action"))
                    if sum(os_.get(p, 0) for p in EX.PREMIUM) >= EX.DUMP_UNITS:
                        dump = 1; break
                cX[n] = feat; cy[n] = clone
                oX[n] = feat; oy[n] = dump
                sX[n] = feat; sY[n] = sale_lab
                n += 1; total += 1
                if n >= CH:
                    flush(n); n = 0
        print(f"  seed {seed} done: games={games} rows={total}", flush=True)
    flush(n)
    import json
    json.dump({"nfeat": F, "games": games, "rows": total, "chunks": chunk,
               "roster": names, "chunks_dir": SP},
              open(os.path.join(OUT, "sp_meta.json"), "w"), indent=1)
    print(f"\nCOMPLETE: {total} rows in {chunk} chunks from {games} games -> {SP}", flush=True)


if __name__ == "__main__":
    main()
