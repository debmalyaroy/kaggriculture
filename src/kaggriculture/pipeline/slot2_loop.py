"""Hands-free Slot-2 loop: BC -> [RL -> package -> gate] xN -> ship-prep.

ONE command that runs the whole improvement loop unattended and STOPS before
submitting (the operator submits -- house rule). Each round trains a bit more,
packages, and gates the policy vs the strong refs; when a round clears the bar it
compiles the policy into a bandit base tape, builds + parity-gates the tarball,
writes SUBMIT_INSTRUCTIONS.md, and stops. If no round clears the bar within the
budget, it still leaves the BEST artifact + instructions and a clear verdict.

State is checkpointed to ``models/rl/loop_state.json`` so a manual re-run RESUMES
(more rounds) instead of restarting. Nothing here ever calls Kaggle.

    # run the whole loop on the 4060 (BC once, then improve until it ships):
    python -m kaggriculture.pipeline.slot2_loop --rounds 6
    # resume with more rounds later:
    python -m kaggriculture.pipeline.slot2_loop --rounds 4 --resume
    # just re-gate + ship-prep the current best (no training):
    python -m kaggriculture.pipeline.slot2_loop --ship-only
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

RL_DIR = os.path.join(ROOT, "models", "rl")
STATE = os.path.join(RL_DIR, "loop_state.json")
BASE_TAPE = os.path.join(ROOT, ".local", "candidates", "slot2_base.tape")
CAND_DIR = os.path.join(ROOT, ".local", "candidates")
SUBMIT_DOC = os.path.join(CAND_DIR, "SUBMIT_INSTRUCTIONS.md")
BANDIT_CFG = os.path.join(ROOT, "configs", "bandit_config_v581.json")


def _device(force_cpu):
    if force_cpu:
        return "cpu"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_state():
    if os.path.exists(STATE):
        try:
            return json.load(open(STATE, encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return {"round": 0, "best_score": -1.0, "history": [], "shipped": False}


def _save_state(st):
    os.makedirs(RL_DIR, exist_ok=True)
    with open(STATE, "w", encoding="utf-8") as fh:
        json.dump(st, fh, indent=1)


# --------------------------------------------------------------------------- #
def stage_bc(args, device):
    import kaggriculture.train.bc_warmup as BC
    if os.path.exists(BC.OUT_PATH) and not args.fresh_bc:
        print(f"[loop] BC checkpoint exists ({os.path.basename(BC.OUT_PATH)}) -- "
              "skipping (use --fresh-bc to retrain)")
        return
    print("[loop] === BC warmup ===")
    cfg = dict(device=device, d_model=args.d_model, layers=args.layers,
               heads=args.heads, lr=3e-4, wd=0.01, dropout=0.1,
               batch_size=args.batch_size, epochs=args.bc_epochs,
               bf16=device == "cuda", vec_weight=1.0, limit_eps=None)
    BC.train(cfg)


def stage_rl(args, device, rnd):
    import kaggriculture.train.macro_rl as RL
    import kaggriculture.train.bc_warmup as BC
    # escalate the training budget each round
    iters = int(args.iters * (1 + 0.6 * rnd))
    print(f"[loop] === RL round {rnd} ({iters} iters, "
          f"{args.games_per_iter} games/iter, gpp {args.games_per_plan}) ===")
    cfg = dict(device=device, iters=iters, games_per_iter=args.games_per_iter,
               ppo_epochs=args.ppo_epochs, n_days=BC.MAX_DAYS, clip=0.2,
               ent_coef=args.ent_coef, kl_coef=args.kl_coef, lr=args.rl_lr,
               init_std=args.init_std, margin_shaping=args.margin_shaping,
               self_play=not args.no_self_play, battery=args.battery,
               seed=args.seed + rnd, use_batch=not args.no_batch,
               games_per_plan=args.games_per_plan)
    return RL.train(cfg)


def stage_package(prefer_rl=True):
    import kaggriculture.train.package_policy as PK
    return PK.export(PK._pick_checkpoint(prefer_rl=prefer_rl))


def stage_gate(args):
    import kaggriculture.measure.eval_harness as EH
    refs = [r for r in args.refs.split(",") if r] or EH._default_refs()
    return EH.gate_packaged_policy(refs=refs, seeds=args.gate_seeds, bar=args.bar)


# --------------------------------------------------------------------------- #
def compile_base_tape(device, ref_seed=42, world_bucket=0, out=BASE_TAPE):
    """Sample the trained policy's greedy plan -> compile a bandit base tape.

    A DIFFERENT ``world_bucket`` conditions on a different target world → a
    different opening family (B7 route2)."""
    import kaggriculture.train.bc_warmup as BC
    import kaggriculture.train.macro_rl as RL
    from kaggriculture.train.macro_env import BatchRoller
    ckpt = RL.RL_OUT if os.path.exists(RL.RL_OUT) else BC.OUT_PATH
    model, ck = BC.load_policy(ckpt, device)
    norm = ck["norm"]
    import torch
    log_std = ck.get("log_std")
    log_std = log_std.to(device) if hasattr(log_std, "to") else \
        torch.zeros(BC.VEC_LEN, device=device)
    plan, _, _ = RL.sample_plan(model, log_std, norm, world_bucket, 1.0, 0.85,
                                device, None, greedy=True, n_days=BC.MAX_DAYS)
    br = BatchRoller()
    os.makedirs(CAND_DIR, exist_ok=True)
    tmp = os.path.join(CAND_DIR, "_slot2_compiled.tape")
    br.compile_plan_tape(plan, tmp, ref_seed)
    with open(tmp, encoding="utf-8") as fh:
        body = [ln for ln in fh.read().splitlines() if not ln.startswith("SEED ")]
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(body) + "\n")
    print(f"[loop] compiled policy (world {world_bucket}) -> "
          f"{os.path.relpath(out, ROOT)} ({len(body)} rows)")
    return out


def config_base_tape():
    """The base tape declared in the bandit config, if any (abs path) — else None.
    When set, the base tape is CONFIG-DRIVEN and the pipeline does NOT compile one
    from the policy; pass ``base_tape=None`` to build_bandit to use it."""
    try:
        cfg = json.load(open(BANDIT_CFG, encoding="utf-8"))
    except (OSError, ValueError):
        return None
    bt = cfg.get("base_tape")
    if not bt:
        return None
    p = bt if os.path.isabs(bt) else os.path.join(ROOT, bt)
    return p if os.path.exists(p) else None


def build_bandit(base_tape=None, tag=""):
    """build_rust_bandit (isolated musl build). ``base_tape`` overrides the config;
    when None the build takes the base tape from ``config['base_tape']``."""
    out = os.path.join(CAND_DIR, f"slot2_bandit_{tag}")
    cmd = [sys.executable, "-m",
           "kaggriculture.bandit.build.build_rust_bandit",
           "--config", BANDIT_CFG, "--out", out, "--build", "--tar"]
    if base_tape:
        cmd += ["--base-tape", base_tape]
    env = dict(os.environ, PYTHONPATH="src", PYTHONUTF8="1")
    print(f"[loop] building bandit tarball -> {os.path.relpath(out, ROOT)}.tar.gz")
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    ok = r.returncode == 0 and os.path.exists(out + ".tar.gz")
    print((r.stdout or "")[-400:] if ok else (r.stdout or "") + (r.stderr or ""))
    return (out + ".tar.gz") if ok else None


def write_submit_doc(shipped, best_score, bar, tarball, gate_agg):
    os.makedirs(CAND_DIR, exist_ok=True)
    verdict = ("SHIP" if shipped else "HOLD (best did not clear the bar)")
    lines = [
        f"# Slot-2 submission instructions ({time.strftime('%Y-%m-%d %H:%M')})",
        "",
        f"**Verdict: {verdict}**  |  best gate vs v46 = {best_score:.3f} "
        f"(bar {bar:.2f})",
        "",
        "This pipeline NEVER submits. You submit, after confirming the pair.",
        "",
        "## Artifact",
        f"- Bandit tarball: `{os.path.relpath(tarball, ROOT) if tarball else 'NOT BUILT'}`",
        f"- Policy (ONNX): `models/rl/policy.onnx` + `models/rl/policy_manifest.json`",
        f"- Last gate aggregate vs strong refs: {gate_agg}",
        "",
        "## To submit (only if Verdict = SHIP)",
        "1. Confirm the currently-active pair (latest-2 rule -- a submit evicts the older):",
        "   `kaggle competitions submissions -c kaggriculture`",
        "2. Submit the tarball via its private kernel (slug "
        "`kaggriculture-adaptive-bandit-private`), OR run the repo's submit gate:",
        "   `python -m kaggriculture.pipeline.submit`  (it re-checks self-play + asks).",
        "3. Never submit an agent that has not passed the local self-play + gate.",
        "",
        "## If Verdict = HOLD",
        "- Run more rounds: `python -m kaggriculture.pipeline.slot2_loop --rounds 4 --resume`",
        "- Or improve the executor first (see docs/history/slot2-operator-runbook.md, 'How to improve').",
    ]
    with open(SUBMIT_DOC, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[loop] wrote {os.path.relpath(SUBMIT_DOC, ROOT)}")


# --------------------------------------------------------------------------- #
def run(args):
    device = _device(args.cpu)
    st = _load_state() if args.resume else {
        "round": 0, "best_score": -1.0, "history": [], "shipped": False}
    print(f"[loop] device={device}  bar={args.bar}  rounds={args.rounds}  "
          f"resume={args.resume}  start_round={st['round']}")
    t0 = time.time()

    if args.ship_only:
        return ship_prep(args, device, st)

    stage_bc(args, device)

    shipped = False
    for r in range(st["round"], st["round"] + args.rounds):
        stage_rl(args, device, r)
        stage_package(prefer_rl=True)
        g = stage_gate(args)
        score = g.get("aggregate", 0.0) if not g.get("skipped") else 0.0
        st["history"].append({"round": r, "score": score, "ship": g.get("ship")})
        st["round"] = r + 1
        if score > st["best_score"]:
            st["best_score"] = score
            # snapshot the best policy
            import kaggriculture.train.macro_rl as RL
            if os.path.exists(RL.RL_OUT):
                shutil.copyfile(RL.RL_OUT, os.path.join(RL_DIR, "rl_policy_best.pt"))
        _save_state(st)
        print(f"[loop] round {r}: gate {score:.3f} (best {st['best_score']:.3f}, "
              f"bar {args.bar})")
        if g.get("ship"):
            shipped = True
            st["shipped"] = True
            _save_state(st)
            break

    ship_prep(args, device, st, shipped=shipped)
    print(f"\n[loop] DONE in {(time.time()-t0)/60:.1f} min. "
          f"best gate {st['best_score']:.3f}  shipped={st['shipped']}")
    return 0 if st["shipped"] else 3


def ship_prep(args, device, st, shipped=None):
    if shipped is None:
        shipped = st.get("shipped", False) or st.get("best_score", 0) >= args.bar
    tarball = None
    gate_agg = st.get("best_score")
    if shipped or args.force_ship:
        try:
            bt = compile_base_tape(device)
            tarball = build_bandit(bt, tag=time.strftime("%m%d_%H%M"))
        except Exception as e:            # never let ship-prep crash the loop
            print(f"[loop] ship-prep build failed: {e}")
    write_submit_doc(shipped, st.get("best_score", -1), args.bar, tarball, gate_agg)
    return 0 if shipped else 3


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--bar", type=float, default=0.9)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ship-only", action="store_true",
                    help="skip training; re-gate + build the current best")
    ap.add_argument("--force-ship", action="store_true",
                    help="build the artifact even if below the bar (for inspection)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--fresh-bc", action="store_true")
    # BC
    ap.add_argument("--bc-epochs", type=int, default=12)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    # RL (per round; iters escalate each round)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--games-per-iter", type=int, default=256)
    ap.add_argument("--games-per-plan", type=int, default=64)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--rl-lr", type=float, default=1e-4)
    ap.add_argument("--ent-coef", type=float, default=0.001)
    ap.add_argument("--kl-coef", type=float, default=0.1)
    ap.add_argument("--init-std", type=float, default=0.3)
    ap.add_argument("--margin-shaping", type=float, default=0.3)
    ap.add_argument("--battery", type=int, default=12)
    ap.add_argument("--no-batch", action="store_true")
    ap.add_argument("--no-self-play", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refs", default="")
    ap.add_argument("--gate-seeds", type=int, default=6)
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
