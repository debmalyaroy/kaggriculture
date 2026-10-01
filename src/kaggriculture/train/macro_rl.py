"""E3.1-E3.5 -- self-play MACRO RL fine-tune (PPO) over the BC-warmed policy.

Fine-tunes ``models/rl/bc_policy.pt`` against the REAL engine reward
(``macro_env``). The policy is a continuous Gaussian over the per-day 26-dim
``MacroAction`` vector (mean = the transformer's ``vec_head``, a learned diagonal
log-std held here). A plan is sampled AUTOREGRESSIVELY -- day t's action updates
the cumulative-economy features day t+1 sees -- exactly matching how the corpus
was built, so BC and RL share one state representation.

Ingredients (master plan E3.1-E3.5):
  * PPO clipped surrogate with an EMA return baseline (a learned critic is a
    documented hook; the return is episodic/terminal so GAE collapses to
    return-minus-baseline broadcast over the 30 days).
  * Opponent pool sampling: real destbreso/top-100 tapes (``macro_env`` pool)
    + FROZEN-SELF (a snapshot of the policy playing greedily), so training is
    genuinely self-play against a non-stationary league.
  * TEACHER-KL: an anchor to the frozen BC prior's mean keeps the policy from
    collapsing / forgetting the expert warmup.
  * GATED PROMOTION: a new checkpoint is saved ONLY if it beats the last
    promoted one on a frozen battery -- non-forgetting by construction.
  * Return-conditioned: rollouts condition on WINNING (rtg=1) so the policy is
    asked for winning plans.

This is a CPU-smoke-runnable scaffold; real training runs on the RTX 4060
(``docs/history/slot2-training-runbook.md``). Reward fidelity comes from the engine, so
the loop is honest even while the greedy executor (B2.0) is the micro layer --
swapping in PC-TAPF (B2) only raises the ceiling, not the interface.

    python -m kaggriculture.train.macro_rl --smoke          # 1 tiny iter, CPU
    python -m kaggriculture.train.macro_rl --iters 200 --games-per-iter 64
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import json
import math
import os
import random
import sys
import time
from typing import List, Optional, Tuple

import kaggriculture.train.macro_actions as MA
import kaggriculture.train.bc_warmup as BC
from kaggriculture.train.worlds import WorldSampler
from kaggriculture.train.macro_env import (
    MacroEnv, BatchRoller, OpponentPool, PlanController, replay_agent,
    cells_to_batch_tape)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

RL_OUT = os.path.join(ROOT, "models", "rl", "rl_policy.pt")
LEAGUE_DIR = os.path.join(ROOT, "models", "rl", "league")
VEC_LEN = MA.VECTOR_LEN
N_CLASS = len(MA.MACRO_CLASSES)


def _load_league():
    """T1: the persistent self-play league = pre-compiled self tapes on disk."""
    import glob
    out = []
    for p in sorted(glob.glob(os.path.join(LEAGUE_DIR, "*.tape"))):
        out.append({"kind": "league_tape", "team": os.path.basename(p),
                    "path": p, "rating": 2400, "weight": 1.0})
    return out


# --------------------------------------------------------------------------- #
# Autoregressive plan sampling from the policy.
# --------------------------------------------------------------------------- #
def _day_feature_row(day, world_bucket, rtg, rating, cum, norm):
    """Build a single [IN_DIM] feature row matching bc_warmup._features."""
    import numpy as np
    x = np.zeros(BC.IN_DIM, np.float32)
    x[0] = day / float(BC.MAX_DAYS)
    db = min(day // (BC.MAX_DAYS // BC.DAY_BUCKETS + 1), BC.DAY_BUCKETS - 1)
    x[1 + db] = 1.0
    x[1 + BC.DAY_BUCKETS + (world_bucket % BC.WORLD_BUCKETS)] = 1.0
    base = 1 + BC.DAY_BUCKETS + BC.WORLD_BUCKETS
    x[base] = rtg
    x[base + 1] = rating
    cum_mu = np.asarray(norm["cum_mu"], np.float32)
    cum_sd = np.asarray(norm["cum_sd"], np.float32)
    x[base + 2:base + 2 + VEC_LEN] = (cum - cum_mu) / cum_sd
    return x


def sample_plan(model, log_std, norm, world_bucket, rtg, rating,
                device, rng_t, greedy=False, n_days=BC.MAX_DAYS):
    """Autoregressively sample a plan. Returns (plan, feats[T,IN], acts[T,VEC]).

    ``feats``/``acts`` are the normalised inputs and sampled normalised action
    vectors kept for the PPO update; ``plan`` is the decoded MacroAction list.
    """
    import numpy as np
    import torch
    tgt_mu = np.asarray(norm["tgt_mu"], np.float32)
    tgt_sd = np.asarray(norm["tgt_sd"], np.float32)
    cum = np.zeros(VEC_LEN, np.float32)
    feats = np.zeros((n_days, BC.IN_DIM), np.float32)
    acts = np.zeros((n_days, VEC_LEN), np.float32)
    plan: List[MA.MacroAction] = []
    std = torch.exp(log_std)
    for day in range(n_days):
        x = _day_feature_row(day, world_bucket, rtg, rating, cum, norm)
        feats[day] = x
        xt = torch.from_numpy(feats[:day + 1][None]).to(device)  # causal seq
        with torch.no_grad():
            _, vpred = model(xt)
            mean = vpred[0, day]                       # normalised action mean
            if greedy:
                a = mean
            else:
                a = mean + std * torch.randn_like(mean)
        an = a.cpu().numpy().astype(np.float32)
        acts[day] = an
        # denormalise -> raw vector -> MacroAction (buckets clamp themselves)
        raw = np.clip(an * tgt_sd + tgt_mu, 0, None)
        ma = MA.MacroAction.from_vector(raw.tolist())
        plan.append(ma)
        cum = cum + raw
    MA.sanitize_plan(plan)                    # P1: coherent + bootstrap-viable
    return plan, feats, acts


def gaussian_logp(mean, log_std, acts):
    """Sum-over-dims diagonal-Gaussian log-prob, per day. mean/acts: [T,VEC]."""
    import torch
    var = torch.exp(2 * log_std)
    lp = -0.5 * (((acts - mean) ** 2) / var + 2 * log_std + math.log(2 * math.pi))
    return lp.sum(dim=-1)                     # [T]


# --------------------------------------------------------------------------- #
# Frozen-self opponent (self-play league member).
# --------------------------------------------------------------------------- #
def frozen_self_cells(model, log_std, norm, seed, device):
    """Compile the policy's greedy plan into opponent action cells."""
    plan, _, _ = sample_plan(model, log_std, norm, world_bucket=0, rtg=1.0,
                             rating=0.85, device=device, rng_t=None, greedy=True)
    ctrl = PlanController(plan)
    # dry-run against PASS to realise a legal cell list
    env = MacroEnv().rollout(plan, seed=seed, opp_cells=None, our_seat=0)
    return env["tape_cells"]


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(cfg: dict) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F

    t0 = time.time()
    device = cfg["device"]
    # load BC warm start
    if not os.path.exists(BC.OUT_PATH):
        raise FileNotFoundError(
            f"{BC.OUT_PATH} missing -- run kaggriculture.train.bc_warmup first")
    model, ck = BC.load_policy(BC.OUT_PATH, device)
    norm = ck["norm"]
    model.train()
    # frozen BC prior (teacher) -- same weights, never updated
    prior, _ = BC.load_policy(BC.OUT_PATH, device)
    prior.eval()
    for p in prior.parameters():
        p.requires_grad_(False)

    log_std = torch.nn.Parameter(
        torch.full((VEC_LEN,), math.log(cfg["init_std"]), device=device))
    # A5: learned critic (value baseline) — MLP over the mean-pooled day features.
    critic = None
    if cfg.get("critic", True):
        critic = torch.nn.Sequential(
            torch.nn.Linear(BC.IN_DIM, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 1)).to(device)
    params = list(model.parameters()) + [log_std] + \
        (list(critic.parameters()) if critic else [])
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)

    pool = OpponentPool()
    env = MacroEnv(pool=pool, margin_shaping=cfg["margin_shaping"])
    rng = random.Random(cfg["seed"])
    use_batch = cfg.get("use_batch", False)
    gpp = cfg.get("games_per_plan", 16)
    br = BatchRoller(pool=pool) if use_batch else None
    league = _load_league() if use_batch else []      # T1: growing self-play league
    league_frac = cfg.get("league_frac", 0.3)
    worlds = WorldSampler()                            # T2/T3: world generator
    os.makedirs(LEAGUE_DIR, exist_ok=True)

    def snapshot_league(it):
        """Compile the current policy's greedy plan → a new league opponent tape."""
        if not use_batch:
            return
        try:
            plan, _, _ = sample_plan(model, log_std.detach(), norm, 0, 1.0, 0.85,
                                     device, None, greedy=True, n_days=cfg["n_days"])
            path = os.path.join(LEAGUE_DIR, f"league_{it:04d}.tape")
            br.compile_plan_tape(plan, path)
            league.append({"kind": "league_tape", "team": f"self_{it}",
                           "path": path, "rating": 2400, "weight": 1.0})
        except Exception as e:
            print(f"[rl] league snapshot failed: {e}")

    ema = 0.5                                   # EMA baseline of the return
    promoted_winrate = -1.0
    hist = []
    clip = cfg["clip"]

    def rollout_batch_fast(n_plans):
        """Fast path: sample N plans, BOUNDED-parallel compile them, then play ALL
        their games in ONE kagg batch call. Reward = mean over each plan's
        opponents. Compile parallelism is hard-capped (memory-safe)."""
        import numpy as np
        import torch
        plans, metas, per_plan_games = [], [], []
        for _ in range(n_plans):
            wb = worlds.sample_bucket(rng)            # T2: condition on a target world
            plan, feats, acts = sample_plan(
                model, log_std.detach(), norm, wb, rtg=1.0, rating=0.85,
                device=device, rng_t=None, n_days=cfg["n_days"])
            plans.append(plan)
            metas.append((feats, acts))
            games = []
            for _g in range(gpp):
                if league and rng.random() < league_frac:
                    entry = rng.choice(league)      # T1: train vs a past self
                else:
                    entry = pool.sample(rng)
                    if entry.get("kind") in (None, "pass", "frozen_self", "agent"):
                        entry = {"kind": "pass"}
                games.append((entry, rng.randint(1, 2**31 - 1)))
            per_plan_games.append(games)
        # bounded-parallel compile + one batch for all plans
        tapes = br.parallel_compile(plans, n_workers=cfg.get("compile_workers", 3))
        allres = br.rollout_many(tapes, per_plan_games, our_seat=0)
        ms = cfg["margin_shaping"]
        batch = []
        for (feats, acts), res in zip(metas, allres):
            if not res:
                continue
            rew = float(np.mean([r["win"] + ms * (r["our_bank"] - r["opp_bank"])
                                 / 1e5 for r in res]))
            win = float(np.mean([r["win"] for r in res]))
            xt = torch.from_numpy(feats[None]).to(device)
            with torch.no_grad():
                _, vmean = model(xt)
                old_lp = gaussian_logp(vmean[0], log_std.detach(),
                                       torch.from_numpy(acts).to(device))
            batch.append(dict(feats=feats, acts=acts, reward=rew, win=win,
                              old_lp=old_lp.cpu().numpy(),
                              bank=np.mean([r["our_bank"] for r in res]),
                              opp=np.mean([r["opp_bank"] for r in res])))
        return batch

    def rollout_batch(n_games):
        if use_batch:
            return rollout_batch_fast(max(1, n_games // gpp))
        batch = []
        for _ in range(n_games):
            entry = pool.sample(rng)
            seed = rng.randint(1, 2**31 - 1)
            rating = float(entry.get("rating") or 0) / 3000.0
            wb = 0
            # opponent cells: real tape, frozen-self, or PASS
            kind = entry.get("kind")
            if kind == "frozen_self" and cfg["self_play"]:
                opp_cells = frozen_self_cells(model, log_std.detach(),
                                              norm, seed, device)
            else:
                opp_cells = pool.cells_for(entry)
            plan, feats, acts = sample_plan(
                model, log_std.detach(), norm, wb, rtg=1.0,
                rating=max(rating, 0.7), device=device, rng_t=None,
                n_days=cfg["n_days"])
            r = env.rollout(plan, seed=seed, opp_cells=opp_cells, our_seat=0)
            # old logp under the sampling policy
            xt = torch.from_numpy(feats[None]).to(device)
            with torch.no_grad():
                _, vmean = model(xt)
                old_lp = gaussian_logp(vmean[0], log_std.detach(),
                                       torch.from_numpy(acts).to(device))
            batch.append(dict(feats=feats, acts=acts, reward=r["reward"],
                              win=r["win"], old_lp=old_lp.cpu().numpy(),
                              bank=r["our_bank"], opp=r["opp_bank"]))
        return batch

    for it in range(cfg["iters"]):
        batch = rollout_batch(cfg["games_per_iter"])
        rewards = np.array([b["reward"] for b in batch], np.float32)
        wins = np.mean([b["win"] for b in batch])
        ema = 0.9 * ema + 0.1 * float(rewards.mean())
        # A5: baseline = learned critic (mean-pooled features) if enabled, else EMA
        if critic is not None:
            feat_pool = torch.from_numpy(
                np.stack([b["feats"].mean(axis=0) for b in batch])).to(device)
            with torch.no_grad():
                values = critic(feat_pool).squeeze(-1).cpu().numpy()
            adv = rewards - values
        else:
            adv = rewards - ema
        if adv.std() > 1e-6:
            adv = adv / (adv.std() + 1e-6)

        # PPO epochs
        for _ep in range(cfg["ppo_epochs"]):
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for bi, b in enumerate(batch):
                xt = torch.from_numpy(b["feats"][None]).to(device)
                _, vmean = model(xt)
                acts = torch.from_numpy(b["acts"]).to(device)
                new_lp = gaussian_logp(vmean[0], log_std, acts)
                old_lp = torch.from_numpy(b["old_lp"]).to(device)
                ratio = torch.exp(new_lp - old_lp)
                A = float(adv[bi])
                surr = torch.min(ratio * A,
                                 torch.clamp(ratio, 1 - clip, 1 + clip) * A)
                pg = -surr.mean()
                # entropy of a diagonal Gaussian
                ent = (0.5 + 0.5 * math.log(2 * math.pi) + log_std).sum()
                # teacher-KL: anchor mean to the frozen BC prior
                with torch.no_grad():
                    _, pmean = prior(xt)
                kl = F.mse_loss(vmean[0], pmean[0])
                loss = pg - cfg["ent_coef"] * ent + cfg["kl_coef"] * kl
                if critic is not None:       # A5: value regression to the return
                    fp = torch.from_numpy(b["feats"].mean(axis=0)[None]).to(device)
                    v = critic(fp).squeeze()
                    loss = loss + 0.5 * (v - float(b["reward"])) ** 2
                loss = loss / len(batch)
                loss.backward()
                total += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

        # gated promotion vs a frozen battery
        wr = _battery_winrate(model, log_std.detach(), norm, pool, env,
                              device, cfg["battery"], rng, br=br)
        promoted = wr > promoted_winrate
        if promoted:
            promoted_winrate = wr
            _save(model, log_std, ck, norm, it, wr)
            snapshot_league(it)                       # T1: archive this self

        std_mean = float(torch.exp(log_std).mean().detach())
        hist.append(dict(it=it, train_win=float(wins), battery_win=wr,
                         ema=ema, std=std_mean, promoted=promoted))
        print(f"[rl] iter {it:3d}  train_win {wins:.3f}  battery {wr:.3f}"
              f"  ema {ema:.3f}  std {std_mean:.3f}"
              f"  {'PROMOTED' if promoted else ''}")

    dt = time.time() - t0
    print(f"[rl] done {dt:.1f}s  best battery win {promoted_winrate:.3f} -> {RL_OUT}")
    return dict(best_winrate=promoted_winrate, iters=cfg["iters"], seconds=dt)


def _battery_winrate(model, log_std, norm, pool, env, device, n, rng, br=None):
    """Frozen-battery win rate. S4: uses the fast batch path when available
    (one greedy-plan compile, played vs the n battery opponents in one batch)."""
    import numpy as np
    ops = [e for e in pool.entries if e.get("kind") == "destbreso_tape"][:n]
    if not ops:
        ops = pool.entries[:n]
    plan, _, _ = sample_plan(model, log_std, norm, 0, 1.0, 0.85,
                             device, None, greedy=True, n_days=BC.MAX_DAYS)
    if br is not None:                        # S4 fast path
        games = [(e, int(e.get("seed", 42)) & 0x7fffffff or 42) for e in ops]
        res = br.rollout_batch(plan, games, our_seat=0)
        return float(np.mean([r["win"] for r in res])) if res else 0.0
    wins = []
    for e in ops:
        cells = pool.cells_for(e)
        seed = int(e.get("seed", 42)) & 0x7fffffff or 42
        r = env.rollout(plan, seed=seed, opp_cells=cells, our_seat=0)
        wins.append(r["win"])
    return float(np.mean(wins)) if wins else 0.0


def _save(model, log_std, ck, norm, it, wr):
    import torch
    os.makedirs(os.path.dirname(RL_OUT), exist_ok=True)
    torch.save(dict(config=ck["config"], state_dict=model.state_dict(),
                    log_std=log_std.detach().cpu(), norm=norm,
                    schema=ck["schema"], promoted_iter=it, battery_win=wr,
                    warm_start=BC.OUT_PATH), RL_OUT)


def _smoke() -> int:
    # ensure a BC checkpoint exists (train a tiny one if absent)
    if not os.path.exists(BC.OUT_PATH):
        BC._smoke()
    cfg = dict(device="cpu", iters=1, games_per_iter=3, ppo_epochs=1,
               n_days=BC.MAX_DAYS, clip=0.2, ent_coef=0.0, kl_coef=0.1,
               lr=1e-4, init_std=0.3, margin_shaping=0.5, self_play=False,
               battery=2, seed=0)
    res = train(cfg)
    assert os.path.exists(RL_OUT), "rl checkpoint not written"
    print(f"[rl][smoke] OK  battery_win={res['best_winrate']:.2f}  ckpt={RL_OUT}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--games-per-iter", type=int, default=64)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.001)
    ap.add_argument("--kl-coef", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--init-std", type=float, default=0.3)
    ap.add_argument("--margin-shaping", type=float, default=0.3)
    ap.add_argument("--battery", type=int, default=12)
    ap.add_argument("--batch", action="store_true",
                    help="B4 fast path: compile-once + Rust kagg batch (~10ms/game)")
    ap.add_argument("--games-per-plan", type=int, default=16,
                    help="opponents/seeds per plan in batch mode (amortises compile)")
    ap.add_argument("--no-self-play", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        return _smoke()
    device = "cpu"
    if not args.cpu:
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
        except Exception:
            pass
    cfg = dict(device=device, iters=args.iters,
               games_per_iter=args.games_per_iter, ppo_epochs=args.ppo_epochs,
               n_days=BC.MAX_DAYS, clip=args.clip, ent_coef=args.ent_coef,
               kl_coef=args.kl_coef, lr=args.lr, init_std=args.init_std,
               margin_shaping=args.margin_shaping, self_play=not args.no_self_play,
               battery=args.battery, seed=args.seed,
               use_batch=args.batch, games_per_plan=args.games_per_plan)
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
