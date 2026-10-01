"""Play our bandit vs the CURRENT top public field on the fast Rust serve path.

Field = the freshly-downloaded score-topping clone family (.local/pubtop/*.py:
v53-v56, farmer-john x2, clone-race) + the curated killers (tschinkel, nathanjacob,
tetsutani, k0006, alperen). Each opponent is played over N distinct shop-worlds,
BOTH seats (shared-market games are not seat-symmetric), on ServeEnv (0.6-5s/game,
exact-bank vs official). Reports per-opponent W-L and flags EVERY loss with
seed/seat/gap so "did we lose any" is answered exactly.

Parallel across opponents (one worker per opponent: own bandit binary + own kagg
process, cleaned up on exit). Run:
  NN_WORKERS=5 python -m kaggriculture.bandit.gate.pubtop_tournament --seeds 10 --config configs/bandit_config.json
"""
import os, json, argparse, subprocess, time
import multiprocessing as mp
from kaggriculture.paths import ROOT
from kaggriculture.bandit.gate import harness as H
from kaggriculture.bandit.gate import loss_forensics as LF

PUBTOP = os.path.join(ROOT, ".local", "pubtop")


def field():
    """(name, path) for every top public opponent."""
    out = []
    fdir = os.environ.get("FIELD_DIR")
    fdir = os.path.join(ROOT, fdir) if fdir else PUBTOP
    for f in sorted(os.listdir(fdir)):
        if f.endswith(".py"):
            out.append((f[:-3], os.path.join(fdir, f)))
    if os.environ.get("FIELD_NO_KILLERS") != "1":
        for kn, kp in LF.KILLERS:                  # curated killers
            if os.path.exists(kp):
                out.append((f"kill_{kn}", kp))
    return out


def _run_opp(task):
    name, path, cfg, ws = task
    safe = name.replace("/", "_")
    if isinstance(cfg, str):                       # pyagent mode: cfg is our .py path
        ours = LF.load_pyagent(cfg)
    else:
        ours = H.build_agent(f"tourn_{safe}", cfg) # this worker's own bandit binary
    games = []
    try:
        for wn, seed in ws:
            for seat in (0, 1):
                try:
                    _, _, us, them = LF.capture_game(ours, path, seed, seat)
                    games.append((wn, seed, seat, us, them))
                except Exception as exc:
                    games.append((wn, seed, seat, None, str(exc)[:60]))
    finally:
        try:
            subprocess.run(["powershell", "-NoProfile", "-Command",
                            f"Get-Process kagg -ErrorAction SilentlyContinue | "
                            f"Where-Object {{ $_.Path -like '*tourn_{safe}_wingate*' }} | Stop-Process -Force"],
                           capture_output=True)
        except Exception:
            pass
    wins = sum(1 for g in games if g[3] is not None and g[3] > g[4])
    ties = sum(1 for g in games if g[3] is not None and g[3] == g[4])
    losses = [g for g in games if g[3] is not None and g[3] < g[4]]
    errs = [g for g in games if g[3] is None]
    gaps = [g[3] - g[4] for g in games if g[3] is not None]
    mg = sum(gaps) / len(gaps) if gaps else 0.0
    return name, wins, ties, losses, errs, mg, len(games)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--config", default="configs/bandit_config.json")
    ap.add_argument("--pyagent", default=None, help="play this .py file as OURS instead of the mbandit binary")
    ap.add_argument("--label", default="shipped")
    a = ap.parse_args()
    workers = int(os.environ.get("NN_WORKERS", "5"))
    if a.pyagent:
        cfg = os.path.join(ROOT, a.pyagent) if not os.path.isabs(a.pyagent) else a.pyagent
        a.label = a.label if a.label != "shipped" else os.path.basename(a.pyagent)
    else:
        cfg = json.load(open(os.path.join(ROOT, a.config)))
    ws = H.world_seeds(a.seeds)
    fld = field()
    print(f"PUBTOP TOURNAMENT  config={a.label}  opponents={len(fld)}  worlds={len(ws)}  seats=2  "
          f"games={len(fld)*len(ws)*2}  workers={workers}", flush=True)
    print(f"worlds: {[w for w,_ in ws]}", flush=True)
    tasks = [(n, p, cfg, ws) for n, p in fld]
    t0 = time.time()
    rows = []
    with mp.Pool(workers, maxtasksperchild=1) as pool:
        for name, wins, ties, losses, errs, mg, n in pool.imap_unordered(_run_opp, tasks):
            rows.append((name, wins, ties, losses, errs, mg, n))
            tag = ("CLEAN" if not losses else f"LOST {len(losses)}") + (f" +{ties}tie" if ties else "")
            print(f"  {name:24s} W{wins:2d} T{ties:2d} L{len(losses):2d} /{n:2d}  meanGap={mg:+9.0f}  {tag}"
                  + (f"  ERR={len(errs)}" if errs else ""), flush=True)
    print(f"\n=== VERDICT ({time.time()-t0:.0f}s) ===", flush=True)
    total_games = sum(r[6] for r in rows)
    total_wins = sum(r[1] for r in rows)
    total_ties = sum(r[2] for r in rows)
    all_losses = [(r[0], l) for r in rows for l in r[3]]
    score = total_wins + 0.5 * total_ties
    print(f"overall: {total_wins}W {total_ties}T {len(all_losses)}L / {total_games}  "
          f"expected-score={score:.1f}/{total_games} ({100*score/total_games:.1f}%)", flush=True)
    if not all_losses:
        print("NO LOSSES vs the entire top public field (ties allowed).", flush=True)
    else:
        print(f"LOST {len(all_losses)} games across {len(set(n for n,_ in all_losses))} opponents:", flush=True)
        for name, l in sorted(all_losses, key=lambda x: x[1][3] - x[1][4]):
            wn, seed, seat, us, them = l
            print(f"    {name:24s} world={wn:22s} seed={seed} seat={seat}  us={us:.0f} them={them:.0f} gap={us-them:+.0f}", flush=True)


if __name__ == "__main__":
    main()
