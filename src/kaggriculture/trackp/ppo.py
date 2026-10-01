"""P3.3 -- PPO + PFSP league self-play over macro decisions.

One iteration:
  1. export current actor -> pure-python weights (the SAME code path that
     ships, so training and inference can never drift)
  2. rollouts on `kagg serve`: opponents drawn 40/30/20/10 from
     anchors / past selves / current mirror / exploiters (PFSP within kind)
  3. GAE over each episode's ~30 macro decisions; reward = win +
     lambda(t) * clipped gap, lambda ANNEALED across iterations, never bank
  4. clipped PPO update (per-head categorical, summed logprobs)
  5. snapshot to the league on a cadence; CHECKPOINT SELECTION BY THE FROZEN
     ANCHOR SET ONLY (greedy eval) -- self-play score is never the yardstick

Exploiter mode (P3.4): same machinery, opponent fixed to the frozen best
policy; the exploiter's final win rate vs it IS the exploitability metric.

Run with the GPU env:
  .../envs/llm/python.exe src/trackp/ppo.py --iters 20 [--init-iql]
  .../envs/llm/python.exe src/trackp/ppo.py --exploiter --iters 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from kaggriculture.trackp import common, league, macro, rollouts  # noqa: E402

GAMMA = 0.995
LAM = 0.95
CLIP = 0.2
ENT = 0.01
HID = 256
LAMBDA0, LAMBDA1 = 0.30, 0.10  # gap shaping anneal


def _actor_critic():
    import torch.nn as nn

    def mlp(out):
        return nn.Sequential(nn.Linear(macro.FEAT_DIM, HID), nn.Tanh(),
                             nn.Linear(HID, HID), nn.Tanh(),
                             nn.Linear(HID, out))
    return mlp(macro.N_LOGITS), mlp(1)


def _export_weights(actor) -> dict:
    sd = actor.state_dict()
    return {"w1": sd["0.weight"].tolist(), "b1": sd["0.bias"].tolist(),
            "w2": sd["2.weight"].tolist(), "b2": sd["2.bias"].tolist(),
            "wo": sd["4.weight"].tolist(), "bo": sd["4.bias"].tolist(),
            "heads": [[n, k] for n, k in macro.HEADS]}


def _head_logprob_entropy(logits, bins):
    import torch
    lp = torch.zeros(logits.shape[0], device=logits.device)
    ent = torch.zeros(logits.shape[0], device=logits.device)
    off = 0
    for hi, (_, k) in enumerate(macro.HEADS):
        lg = torch.log_softmax(logits[:, off:off + k], dim=1)
        lp = lp + lg.gather(1, bins[:, hi:hi + 1]).squeeze(1)
        pr = lg.exp()
        ent = ent - (pr * lg).sum(dim=1)
        off += k
    return lp, ent


def _anchor_eval(runner, W, anchors, n=12):
    """Greedy (argmax) score vs the frozen anchor set -- THE yardstick."""
    picks = anchors[:n]
    jobs = [{"seed": a["seed"], "me": {"l1": W, "sample_temp": 1e-6,
                                       "record": False},
             "opp": {"kind": "tape", "tape": a["tape"]}} for a in picks]
    res = runner.run(jobs)
    ok = [r for r in res if "banks" in r]
    if not ok:
        return -1.0, 0.0
    score = sum(1.0 if r["banks"][0] > r["banks"][1] else
                (0.5 if r["banks"][0] == r["banks"][1] else 0.0)
                for r in ok) / len(ok)
    mgap = sum(r["banks"][0] - r["banks"][1] for r in ok) / len(ok)
    return score, mgap


def train(iters=20, episodes_per_iter=48, jobs=8, lr=3e-4, epochs=4,
          minibatch=512, init_iql=False, exploiter=False, seed=7,
          snapshot_every=4, temp=1.0):
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    actor, critic = _actor_critic()
    if init_iql:
        ck = torch.load(os.path.join(common.MODELS, "iql_actor.pt"),
                        map_location="cpu", weights_only=False)
        actor.load_state_dict(ck["actor"])
        print("warm-started from IQL", flush=True)
    actor.to(dev)
    critic.to(dev)
    opt = torch.optim.Adam([*actor.parameters(), *critic.parameters()],
                           lr=lr)

    lg = league.League(seed=seed)
    anchors = league.load_anchors()
    if not anchors:
        raise SystemExit("no anchors -- run league.py --build-anchors")
    runner = rollouts.Runner(jobs)
    tag = "exploiter" if exploiter else "ppo"
    log_path = os.path.join(common.MODELS, f"{tag}_log.jsonl")
    best = {"anchor_score": -1.0, "iter": -1}

    frozen_best = None
    if exploiter:
        bp = os.path.join(common.MODELS, "l1_weights_best.json")
        if not os.path.exists(bp):
            raise SystemExit("exploiter mode needs l1_weights_best.json")
        with open(bp, encoding="utf-8") as fh:
            frozen_best = json.load(fh)

    rng = np.random.default_rng(seed)
    try:
        for it in range(iters):
            t0 = time.time()
            lam = LAMBDA0 + (LAMBDA1 - LAMBDA0) * (it / max(1, iters - 1))
            W = _export_weights(actor)
            jobs_list, refs = [], []
            for e in range(episodes_per_iter):
                if exploiter:
                    opp = {"kind": "policy", "policy": {"l1": frozen_best}}
                    member = None
                    seed_e = int(rng.integers(1, 2 ** 31))
                else:
                    member = lg.sample()
                    if member["kind"] == "anchor":
                        a = member["ref"]
                        opp = {"kind": "tape", "tape": a["tape"]}
                        seed_e = a["seed"]
                    elif member["kind"] == "current":
                        opp = {"kind": "policy", "policy": {"l1": W}}
                        seed_e = int(rng.integers(1, 2 ** 31))
                    else:
                        with open(member["ref"], encoding="utf-8") as fh:
                            ow = json.load(fh)
                        opp = {"kind": "policy", "policy": {"l1": ow}}
                        seed_e = int(rng.integers(1, 2 ** 31))
                jobs_list.append({"seed": seed_e,
                                  "me": {"l1": W, "record": True,
                                         "sample_temp": temp,
                                         "rng_seed": int(
                                             rng.integers(1, 2 ** 31))},
                                  "opp": opp})
                refs.append(member)
            res = runner.run(jobs_list)

            F, BNS, LP, ADV, RET = [], [], [], [], []
            n_ok = 0
            wr = 0.0
            for member, r in zip(refs, res):
                if "banks" not in r or not r.get("decisions"):
                    continue
                n_ok += 1
                gap = r["banks"][0] - r["banks"][1]
                win = 1.0 if gap > 0 else (0.5 if gap == 0 else 0.0)
                wr += win
                rew = win + lam * max(-1.0, min(1.0, gap / macro.GAP_CLIP))
                lg.record(member, win)
                decs = r["decisions"]
                feats = [d["f"] for d in decs]
                with torch.no_grad():
                    v = critic(torch.tensor(
                        feats, dtype=torch.float32, device=dev)
                    ).squeeze(-1).cpu().numpy()
                T = len(decs)
                adv = np.zeros(T)
                gae = 0.0
                for t in reversed(range(T)):
                    r_t = rew if t == T - 1 else 0.0
                    v_next = 0.0 if t == T - 1 else v[t + 1]
                    delta = r_t + GAMMA * v_next - v[t]
                    gae = delta + GAMMA * LAM * gae
                    adv[t] = gae
                ret = adv + v
                for t, d in enumerate(decs):
                    F.append(d["f"])
                    BNS.append(d["bins"])
                    LP.append(d["logp"])
                    ADV.append(adv[t])
                    RET.append(ret[t])
            if not F:
                print(f"iter {it}: no data", flush=True)
                continue
            Ft = torch.tensor(F, dtype=torch.float32, device=dev)
            Bt = torch.tensor(BNS, dtype=torch.long, device=dev)
            LPt = torch.tensor(LP, dtype=torch.float32, device=dev)
            At = torch.tensor(ADV, dtype=torch.float32, device=dev)
            Rt = torch.tensor(RET, dtype=torch.float32, device=dev)
            At = (At - At.mean()) / (At.std() + 1e-6)

            n = len(F)
            idx = np.arange(n)
            for _ in range(epochs):
                rng.shuffle(idx)
                for s in range(0, n, minibatch):
                    mb = torch.tensor(idx[s:s + minibatch], device=dev)
                    logits = actor(Ft[mb])
                    lp, ent = _head_logprob_entropy(logits, Bt[mb])
                    ratio = torch.exp(lp - LPt[mb])
                    surr = torch.min(
                        ratio * At[mb],
                        torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * At[mb])
                    v = critic(Ft[mb]).squeeze(-1)
                    loss = (-surr.mean() + 0.5 * ((v - Rt[mb]) ** 2).mean()
                            - ENT * ent.mean())
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [*actor.parameters(), *critic.parameters()], 1.0)
                    opt.step()

            a_score, a_gap = _anchor_eval(runner, _export_weights(actor),
                                          anchors)
            row = {"iter": it, "episodes": n_ok, "decisions": n,
                   "train_winrate": round(wr / max(1, n_ok), 3),
                   "anchor_score": round(a_score, 3),
                   "anchor_gap": round(a_gap, 0), "lambda": round(lam, 3),
                   "secs": round(time.time() - t0, 1)}
            print(json.dumps(row), flush=True)
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            if a_score > best["anchor_score"] or (
                    a_score == best["anchor_score"] and it > best["iter"]):
                best = {"anchor_score": a_score, "iter": it,
                        "anchor_gap": a_gap}
                torch.save({"actor": actor.state_dict(), "meta": row},
                           os.path.join(common.MODELS, f"{tag}_best.pt"))
                with open(os.path.join(
                        common.MODELS,
                        "l1_weights_best.json" if not exploiter
                        else "exploiter_weights.json"),
                        "w", encoding="utf-8") as fh:
                    json.dump(_export_weights(actor), fh)
            if not exploiter and it and it % snapshot_every == 0:
                snap = os.path.join(league.SNAPSHOTS, f"snap_{it:03d}.json")
                with open(snap, "w", encoding="utf-8") as fh:
                    json.dump(_export_weights(actor), fh)
    finally:
        runner.close()
    if exploiter:
        # the exploitability metric: final training win rate vs frozen best
        edge = best.get("anchor_score")  # anchors still reported for context
        expl = os.path.join(league.EXPLOITERS,
                            f"exploiter_{int(time.time())}.json")
        with open(expl, "w", encoding="utf-8") as fh:
            json.dump(_export_weights(actor), fh)
        with open(os.path.join(common.MODELS, "exploiter_report.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"train_winrate_vs_frozen": row["train_winrate"],
                       "anchor_score": edge, "weights": expl}, fh, indent=1)
    print(json.dumps({"best": best}), flush=True)
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--episodes", type=int, default=48)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--init-iql", action="store_true")
    ap.add_argument("--exploiter", action="store_true")
    a = ap.parse_args()
    train(iters=a.iters, episodes_per_iter=a.episodes, jobs=a.jobs,
          init_iql=a.init_iql, exploiter=a.exploiter)
