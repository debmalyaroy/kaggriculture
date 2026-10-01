"""The DAILY pipeline that builds BOTH seats (bandit + trackp) and gates the pair.

One command, run daily (or on a schedule). It NEVER submits -- it produces the two
tarballs + a release report + SUBMIT_INSTRUCTIONS.md; the operator submits.

Stages (each is skippable via --skip):
  1. data    -- fold any new high-rated games into the BC corpus (best-effort)
  2. train   -- ONE incremental improvement: BC warm-start (once) + a few RL
                rounds, RESUMING yesterday's best (models/rl/loop_state.json)
  3. bandit  -- compile the trained policy -> base tape + rails -> kagg-bandit
                tarball; parity gate + faithful gate vs v46
  4. trackp  -- export ONNX (+ native-inference package when F1.1 lands); until
                then the trackp seat SHIP is HELD with a clear reason
  5. pair    -- gate both seats banded (crown panel) and pick the submission pair
                (bandit + trackp, else bandit + diversified route2 fallback)
  6. report  -- write release_report.md + SUBMIT_INSTRUCTIONS.md

    python -m kaggriculture.pipeline.daily_slot2                 # full daily run
    python -m kaggriculture.pipeline.daily_slot2 --rl-rounds 2   # lighter train step
    python -m kaggriculture.pipeline.daily_slot2 --skip train    # rebuild+gate only
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import json
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

CAND = os.path.join(ROOT, ".local", "candidates")
RELEASE = os.path.join(CAND, "release_report.md")


def _device(force_cpu):
    if force_cpu:
        return "cpu"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# --------------------------------------------------------------------------- #
def stage_data(args):
    """Best-effort corpus refresh; never fatal (CDN 429 is expected)."""
    if "data" in args.skip:
        print("[daily] data: skipped"); return
    print("[daily] === data: corpus refresh (best-effort) ===")
    # The download supervisor + bc_corpus own this; here we only note freshness.
    try:
        import kaggriculture.train.bc_warmup as BC
        a = os.path.join(BC.MACRO_DIR, "macro.parquet")
        b = os.path.join(BC.MACRO_DIR, "macro_destbreso.parquet")
        n = sum(os.path.getsize(p) for p in (a, b) if os.path.exists(p))
        print(f"[daily] corpus on disk: {n/1e6:.0f} MB "
              f"({[os.path.basename(p) for p in (a, b) if os.path.exists(p)]})")
    except Exception as e:
        print(f"[daily] data note: {e}")


def stage_train(args, device):
    """One incremental improvement, resuming the running best."""
    if "train" in args.skip:
        print("[daily] train: skipped"); return None
    print(f"[daily] === train: {args.rl_rounds} RL round(s), resume ===")
    import kaggriculture.pipeline.slot2_loop as LOOP

    class A:  # reuse the loop's stage fns with a lightweight arg bag
        pass
    a = A()
    a.__dict__.update(dict(
        cpu=device == "cpu", fresh_bc=args.fresh_bc, d_model=args.d_model,
        layers=args.layers, heads=args.heads, batch_size=args.batch_size,
        bc_epochs=args.bc_epochs, iters=args.iters,
        games_per_iter=args.games_per_iter, games_per_plan=args.games_per_plan,
        ppo_epochs=args.ppo_epochs, rl_lr=args.rl_lr, ent_coef=args.ent_coef,
        kl_coef=args.kl_coef, init_std=args.init_std,
        margin_shaping=args.margin_shaping, battery=args.battery,
        no_batch=False, no_self_play=args.no_self_play, seed=args.seed,
        refs=args.refs, gate_seeds=args.gate_seeds, bar=args.bar,
        rounds=args.rl_rounds, resume=True, ship_only=False, force_ship=False))
    LOOP.stage_bc(a, device)
    st = LOOP._load_state()
    best = st.get("best_score", -1.0)
    for r in range(st["round"], st["round"] + args.rl_rounds):
        LOOP.stage_rl(a, device, r)
        LOOP.stage_package(prefer_rl=True)
        g = LOOP.stage_gate(a)
        score = g.get("aggregate", 0.0) if not g.get("skipped") else 0.0
        st["history"].append({"round": r, "score": score, "ship": g.get("ship")})
        st["round"] = r + 1
        if score > best:
            best = score
            st["best_score"] = best
            import shutil
            import kaggriculture.train.macro_rl as RL
            if os.path.exists(RL.RL_OUT):
                shutil.copyfile(RL.RL_OUT,
                                os.path.join(ROOT, "models", "rl", "rl_policy_best.pt"))
        LOOP._save_state(st)
        print(f"[daily] round {r}: gate {score:.3f} (best {best:.3f})")
    return best


def stage_bandit(args, device):
    """Compile the policy -> bandit tarball + gate."""
    if "bandit" in args.skip:
        print("[daily] bandit: skipped"); return None
    print("[daily] === bandit seat: compile -> build -> gate ===")
    import kaggriculture.pipeline.slot2_loop as LOOP
    try:
        # BASE TAPE IS CONFIG-DRIVEN: if the bandit config declares a base_tape,
        # use it (a curated/robust economy); only compile one from the policy
        # when the config leaves it unset.
        cfg_bt = LOOP.config_base_tape()
        if cfg_bt:
            print(f"[daily] base tape from config: {os.path.relpath(cfg_bt, ROOT)}")
            bt = None                      # build_rust_bandit reads config['base_tape']
        else:
            bt = LOOP.compile_base_tape(device, world_bucket=0)
        tar = LOOP.build_bandit(bt, tag="daily_" + time.strftime("%m%d"))
    except Exception as e:
        print(f"[daily] bandit build failed: {e}")
        return None
    # B7: a route2 variant conditioned on a DIFFERENT world -> distinct opening
    route2 = None
    if not args.no_route2:
        try:
            bt2 = LOOP.compile_base_tape(device, world_bucket=8,
                                         out=os.path.join(LOOP.CAND_DIR,
                                                          "slot2_base_route2.tape"))
            route2 = LOOP.build_bandit(bt2, tag="route2_" + time.strftime("%m%d"))
        except Exception as e:
            print(f"[daily] route2 build failed: {e}")
    # faithful competitive gate vs v46 (best-effort)
    agg = None
    try:
        import kaggriculture.measure.eval_harness as EH
        stage = os.path.join(ROOT, ".local", "scratch", "bandit_parity_stage",
                             "main.py")
        refs = EH._default_refs()
        if os.path.exists(stage) and refs:
            ss = EH.seat_swapped(stage, refs, seeds=args.gate_seeds)
            agg = ss["_aggregate"]["score"]
            print(f"[daily] bandit gate vs v46: {agg:.3f}")
    except Exception as e:
        print(f"[daily] bandit gate note: {e}")
    return {"tarball": tar, "gate": agg, "route2": route2}


def stage_trackp(args, device):
    """Export the trackp ONNX seat; native-inference package = F1.1 (pending)."""
    if "trackp" in args.skip:
        print("[daily] trackp: skipped"); return None
    print("[daily] === trackp seat: ONNX export (native inference = F1.1) ===")
    import kaggriculture.train.package_policy as PK
    try:
        m = PK.export(PK._pick_checkpoint(prefer_rl=True))
    except Exception as e:
        print(f"[daily] trackp export failed: {e}")
        return None
    # R2: native inference is READY via the pure-numpy forward (native_infer),
    # validated numpy==torch — no Rust ONNX crate needed for macro (~30 calls/game).
    try:
        import kaggriculture.train.native_infer as NI
        parity = NI.validate()["pass_"]
    except Exception:
        parity = m.get("parity", {}).get("pass_")
    return {"onnx": os.path.join("models", "rl", "policy.onnx"),
            "parity": parity, "ship_ready": bool(parity),
            "native_inference": "numpy (validated ==torch)",
            "guardrails": {}}


def _windows_bandit_gate_stage(out_dir):
    """Stage a WINDOWS-runnable copy of the built bandit so its ECONOMY can be
    gated on this box (the shipped tarball stays musl/Linux for Kaggle). Swaps the
    musl `kagg` for an isolated Windows `kagg.exe` (never touches rustengine/kagg.exe)."""
    import shutil
    win = os.path.join(ROOT, ".local", "scratch", "bandit_target", "release", "kagg.exe")
    if not os.path.exists(win):
        subprocess.run(["cargo", "build", "--release", "--bin", "kagg",
                        "--manifest-path", "rustengine/Cargo.toml"], cwd=ROOT,
                       env=dict(os.environ, CARGO_TARGET_DIR=os.path.join(
                           ".local", "scratch", "bandit_target")),
                       capture_output=True, text=True)
    if not os.path.exists(win) or not os.path.isdir(out_dir):
        return None
    stage = out_dir + "_wingate"
    shutil.rmtree(stage, ignore_errors=True)
    shutil.copytree(out_dir, stage)
    for nm in ("kagg", "kagg.exe"):
        p = os.path.join(stage, nm)
        if os.path.exists(p):
            os.remove(p)
    shutil.copyfile(win, os.path.join(stage, "kagg.exe"))
    return os.path.join(stage, "main.py")


def _full_gate(agent, harness, args):
    """Run the FULL banded release gate on a harness agent (never crashes)."""
    try:
        import kaggriculture.measure.eval_harness as EH
        return EH.full_gate(agent, seeds=args.gate_seeds, bar=args.bar,
                            harness=harness)
    except Exception as e:
        print(f"[daily] {harness} full_gate note: {e}")
        return {"harness": harness, "ship": False, "aggregate": None,
                "bands": {}, "verdict": "error"}


def stage_release(args, best, bandit, trackp, device):
    """Version → full gate both harnesses → LOCK gated configs → submit checklist."""
    import json
    import kaggriculture.pipeline.release as REL
    version = REL.next_version()
    seats = {}

    # --- BANDIT: gate the built agent, lock its config on SHIP ---
    if bandit and bandit.get("tarball"):
        out_dir = bandit["tarball"][:-len(".tar.gz")]
        cfg_path = os.path.join(out_dir, "config.json")
        # gate on a WINDOWS-staged copy (the tarball's musl binary can't run here)
        main_py = _windows_bandit_gate_stage(out_dir)
        g = _full_gate(main_py, "bandit", args) if main_py and os.path.exists(main_py) \
            else {"ship": False, "aggregate": bandit.get("gate")}
        locked = None
        if g.get("ship"):
            cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
            locked = REL.lock_config("bandit", version, cfg, bandit["tarball"], g)
        seats["bandit"] = {"ship": g.get("ship"), "artifact": bandit["tarball"],
                           "gate": g.get("aggregate"), "locked": locked,
                           "bands": g.get("bands")}
    else:
        seats["bandit"] = {"ship": False, "artifact": None}

    # --- TRACK-P: gate the live-NN agent (R2 numpy), lock on SHIP ---
    if trackp and trackp.get("ship_ready"):
        try:
            import kaggriculture.measure.eval_harness as EH
            onnx = os.path.join(ROOT, "models", "rl", "policy.onnx")
            man = os.path.join(ROOT, "models", "rl", "policy_manifest.json")
            agent = EH.build_policy_agent(onnx, man)
            g = _full_gate(agent, "trackp", args)
        except Exception as e:
            print(f"[daily] trackp gate note: {e}")
            g = {"ship": False, "aggregate": None, "bands": {}}
        locked = None
        if g.get("ship"):
            import kaggriculture.train.native_infer as NI
            bundle = NI.save_bundle()          # R3: pack the actual WEIGHTS (npz)
            tcfg = {"weights_npz": os.path.relpath(bundle, ROOT),  # the packable weights
                    "runtime": "native_infer.numpy_forward + SeasonCbsController",
                    "guardrails": trackp.get("guardrails", {})}
            locked = REL.lock_config("trackp", version, tcfg, bundle, g)
        seats["trackp"] = {"ship": g.get("ship"), "artifact": trackp.get("onnx"),
                           "gate": g.get("aggregate"), "locked": locked,
                           "bands": g.get("bands")}
    else:
        seats["trackp"] = {"ship": False, "artifact": trackp.get("onnx") if trackp else None}

    doc = REL.submit_checklist(version, seats)
    print(f"[daily] release {version}: bandit_ship={seats['bandit'].get('ship')} "
          f"trackp_ship={seats['trackp'].get('ship')} -> {os.path.relpath(doc, ROOT)}")
    return version, seats


def run(args):
    device = _device(args.cpu)
    import kaggriculture.pipeline.release as REL
    print(f"[daily] device={device}  skip={args.skip}  bar={args.bar}  "
          f"next_version={REL.peek_version()}")
    t0 = time.time()
    stage_data(args)
    best = stage_train(args, device)
    bandit = stage_bandit(args, device)
    trackp = stage_trackp(args, device)
    version, seats = stage_release(args, best, bandit, trackp, device)
    print(f"\n[daily] DONE {version} in {(time.time()-t0)/60:.1f} min. "
          f"seats={ {k: v.get('ship') for k, v in seats.items()} }")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["data", "train", "bandit", "trackp"])
    ap.add_argument("--bar", type=float, default=0.9)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--fresh-bc", action="store_true")
    ap.add_argument("--no-route2", action="store_true",
                    help="skip the B7 route2 (diversified second bandit)")
    # train (one incremental step/day)
    ap.add_argument("--rl-rounds", type=int, default=3)
    ap.add_argument("--bc-epochs", type=int, default=12)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
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
    ap.add_argument("--no-self-play", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refs", default="")
    ap.add_argument("--gate-seeds", type=int, default=6)
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
