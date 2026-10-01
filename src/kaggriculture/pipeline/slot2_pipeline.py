"""Slot-2 pipeline orchestrator: corpus -> BC warmup -> macro-RL -> package -> gate.

One command that runs the whole BC->macro-RL predator build. It NEVER submits and
NEVER runs the heavy GPU training unless you pass the training flags -- the
default ``--smoke`` proves every stage wires together on CPU in ~2 min so the
pipeline is ready to run on the RTX 4060 (see ``docs/history/slot2-training-runbook.md``).

Stages (select with ``--stages``):
  corpus   -- assert the macro BC corpus exists (build hint if missing)
  bc       -- train the return-conditioned macro BC policy (``bc_warmup``)
  rl       -- self-play PPO fine-tune (``macro_rl``)
  package  -- export ONNX + parity gate + manifest (``package_policy``)
  gate     -- evaluate the packaged policy vs the strong refs on the faithful
              harness (``measure.eval_harness``) -- ship-or-fallback verdict

    python -m kaggriculture.pipeline.slot2_pipeline --smoke
    python -m kaggriculture.pipeline.slot2_pipeline \
        --stages bc,rl,package,gate --epochs 12 --iters 300 --games-per-iter 64
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

ALL_STAGES = ("corpus", "bc", "rl", "package", "gate")


def _device(force_cpu: bool) -> str:
    if force_cpu:
        return "cpu"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def stage_corpus() -> bool:
    import kaggriculture.train.bc_warmup as BC
    a = os.path.join(BC.MACRO_DIR, "macro.parquet")
    b = os.path.join(BC.MACRO_DIR, "macro_destbreso.parquet")
    have = os.path.exists(a) or os.path.exists(b)
    if have:
        print(f"[slot2] corpus OK ({[os.path.basename(p) for p in (a, b) if os.path.exists(p)]})")
    else:
        print("[slot2] corpus MISSING -- run: "
              "python -m kaggriculture.data.bc_corpus --limit 0 "
              "&& python -m kaggriculture.data.bc_corpus --source destbreso")
    return have


def stage_bc(args, device) -> dict:
    import kaggriculture.train.bc_warmup as BC
    cfg = dict(device=device, d_model=args.d_model, layers=args.layers,
               heads=args.heads, lr=args.lr, wd=0.01, dropout=0.1,
               batch_size=args.batch_size, epochs=args.epochs,
               bf16=not args.no_bf16, vec_weight=1.0, limit_eps=args.limit_eps)
    return BC.train(cfg)


def stage_rl(args, device) -> dict:
    import kaggriculture.train.macro_rl as RL
    import kaggriculture.train.bc_warmup as BC
    cfg = dict(device=device, iters=args.iters,
               games_per_iter=args.games_per_iter, ppo_epochs=args.ppo_epochs,
               n_days=BC.MAX_DAYS, clip=0.2, ent_coef=args.ent_coef,
               kl_coef=args.kl_coef, lr=args.rl_lr, init_std=args.init_std,
               margin_shaping=args.margin_shaping,
               self_play=not args.no_self_play, battery=args.battery,
               seed=args.seed, use_batch=not args.no_batch,
               games_per_plan=args.games_per_plan)
    return RL.train(cfg)


def stage_package(args) -> dict:
    import kaggriculture.train.package_policy as PK
    return PK.export(PK._pick_checkpoint(prefer_rl=not args.bc_only))


def stage_gate(args) -> dict:
    try:
        import kaggriculture.measure.eval_harness as EH
    except Exception as e:
        print(f"[slot2] gate stage: eval_harness unavailable ({e}); skipping")
        return dict(skipped=True)
    return EH.gate_packaged_policy(refs=args.refs.split(",") if args.refs else None,
                                   seeds=args.gate_seeds)


def run(args) -> int:
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    device = _device(args.cpu)
    print(f"[slot2] stages={stages}  device={device}  "
          f"{'SMOKE' if args.smoke else 'FULL'}")
    t0 = time.time()
    results = {}
    if "corpus" in stages:
        if not stage_corpus() and not args.smoke:
            print("[slot2] STOP: corpus missing")
            return 2
    if "bc" in stages:
        results["bc"] = stage_bc(args, device)
    if "rl" in stages:
        results["rl"] = stage_rl(args, device)
    if "package" in stages:
        results["package"] = {"parity": stage_package(args).get("parity")}
    if "gate" in stages:
        results["gate"] = stage_gate(args)
    print(f"\n[slot2] pipeline done in {time.time() - t0:.1f}s")
    for k, v in results.items():
        if isinstance(v, dict):
            v = {kk: vv for kk, vv in v.items()
                 if kk in ("best_val", "n_params", "best_winrate", "parity",
                           "verdict", "ship", "skipped")}
        print(f"  {k}: {v}")
    return 0


def _smoke() -> int:
    import kaggriculture.train.bc_warmup as BC
    import kaggriculture.train.macro_rl as RL
    import kaggriculture.train.package_policy as PK
    print("[slot2][smoke] corpus")
    stage_corpus()
    print("[slot2][smoke] bc")
    BC._smoke()
    print("[slot2][smoke] rl")
    RL._smoke()
    print("[slot2][smoke] package")
    PK._smoke()
    print("[slot2][smoke] gate")
    try:
        import kaggriculture.measure.eval_harness as EH
        EH._smoke()
    except Exception as e:
        print(f"[slot2][smoke] gate: eval_harness not present yet ({e})")
    print("[slot2][smoke] OK -- all stages wired")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stages", default=",".join(ALL_STAGES))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-bf16", action="store_true")
    # BC
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit-eps", type=int, default=None)
    # RL
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--games-per-iter", type=int, default=64)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--rl-lr", type=float, default=1e-4)
    ap.add_argument("--ent-coef", type=float, default=0.001)
    ap.add_argument("--kl-coef", type=float, default=0.1)
    ap.add_argument("--init-std", type=float, default=0.3)
    ap.add_argument("--margin-shaping", type=float, default=0.3)
    ap.add_argument("--battery", type=int, default=12)
    ap.add_argument("--no-batch", action="store_true",
                    help="disable the B4 fast Rust-batch rollout path")
    ap.add_argument("--games-per-plan", type=int, default=64,
                    help="opponents/seeds per plan (batch mode; higher=faster/game)")
    ap.add_argument("--no-self-play", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    # package / gate
    ap.add_argument("--bc-only", action="store_true",
                    help="package the BC checkpoint (skip RL)")
    ap.add_argument("--refs", default="", help="gate reference agents (csv)")
    ap.add_argument("--gate-seeds", type=int, default=6)
    args = ap.parse_args()
    if args.smoke:
        return _smoke()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
