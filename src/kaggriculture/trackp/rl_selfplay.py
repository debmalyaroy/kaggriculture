"""Trackp RL self-play (PPO) -- warm-start from BC, improve by self-play.

Actor-critic over N parallel `kagg serve` envs (batched NN inference), sparse
terminal reward (win margin), GAE, PPO-clip, entropy bonus, and a teacher-KL to
the frozen BC policy (anti-collapse). Opponent pool = PASS + gate killers + past
selves (diversity sampling). The composite action is a product of factorized
categorical heads (farmer/12 hands/10 market x verb+arg+qty); logprob = sum of
per-head logprobs. Reactive rails (reactive_rails) are applied to OUR action so
train == deploy. Resumable + self-stop time budget (2-3 min checkpoints).

    python -m kaggriculture.trackp.rl_selfplay --resume policy_bc.pt --steps 500000000
    python -m kaggriculture.trackp.rl_selfplay --smoke
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse, glob, json, os, random, subprocess, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import kaggriculture.data.trackp_corpus as TC
import kaggriculture.trackp.bc_train as B
import kaggriculture.trackp.policy_agent as PA
import kaggriculture.trackp.reactive_rails as RR
from kaggriculture.engine.serve_match import (Serve, KAGG, obs_for, _call,
                                              action_to_line, EPISODE_STEPS)
from kaggriculture.paths import ROOT as _ROOT


class VecServe:
    """One `kagg vecserve` process holding N games, stepped in ONE stdio round-trip
    (Rust steps all N -> the N-IPC-per-game-step tax collapses to 1). This is the
    core of the fast RL: the env stepping runs in Rust, only the tiny obs JSON
    crosses to Python for the batched NN forward."""
    def __init__(self, n):
        self.n = n
        self.p = subprocess.Popen([KAGG, "vecserve"], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True,
                                  encoding="utf-8", cwd=_ROOT)

    def reset(self, seed0):
        self.p.stdin.write(f"VRESET {self.n} {seed0}\n"); self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())

    def step(self, pairs):
        payload = "\x1f".join(f"{la}\x1e{lb}" for la, lb in pairs)
        self.p.stdin.write(f"VSTEP {payload}\n"); self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())

    def close(self):
        try:
            self.p.stdin.write("QUIT\n"); self.p.stdin.flush()
        except OSError:
            pass
        try:
            self.p.wait(timeout=10)
        except Exception:
            self.p.kill()

# head layout (name, count, cardinality)
HEADS = ([("fverb", 1, B.N_MOVER), ("farg", 1, B.N_ARG),
          ("hverb", B.MAX_HANDS, B.N_MOVER), ("harg", B.MAX_HANDS, B.N_ARG),
          ("mverb", B.MAX_MARKET, B.N_MARKET), ("marg", B.MAX_MARKET, B.N_ARG),
          ("mqty", B.MAX_MARKET, B.QTY_CAP + 1)])

# flat layout for CACHING the frozen-teacher logits: the teacher never changes, so
# we forward it ONCE during rollout and reuse the logits across both PPO epochs
# instead of re-forwarding it inside every minibatch (removes ~1/4 of PPO's cost).
_TFLAT_SPEC = []            # (name, slots, card, offset)
_o = 0
for _n, _s, _c in HEADS:
    _TFLAT_SPEC.append((_n, _s, _c, _o)); _o += _s * _c
_TFLAT_W = _o


def _flatten_heads(out, bs):
    """Head dict -> (bs, _TFLAT_W). out[name] is (bs, card) or (bs, slots, card)."""
    return torch.cat([out[nm].reshape(bs, sl * cd) for nm, sl, cd, _ in _TFLAT_SPEC], 1)


def _unflatten_heads(flat, bs):
    """(bs, _TFLAT_W) -> head dict, mirroring the model's forward output shapes."""
    out = {}
    for nm, sl, cd, off in _TFLAT_SPEC:
        seg = flat[:, off:off + sl * cd]
        out[nm] = seg.reshape(bs, cd) if sl == 1 else seg.reshape(bs, sl, cd)
    return out


class ActorCritic(nn.Module):
    def __init__(self, dims=None):
        super().__init__()
        self.policy = B.build_model(**(dims or {}))       # dims from ckpt (else 12M default)
        d = self.policy.cls.shape[-1]
        self.value = nn.Linear(d, 1)

    def forward(self, toks, mask, rtg):
        out = self.policy(toks, mask, rtg)
        out["value"] = self.value(out["h"]).squeeze(-1)
        return out


def _flat_logits(out):
    """[(B, slots, card)] flattened to a list of (B, card) per head-slot."""
    cols = []
    for name, slots, _card in HEADS:
        t = out[name]
        if slots == 1:
            cols.append((name, 0, t))
        else:
            for s in range(slots):
                cols.append((name, s, t[:, s]))
    return cols


def sample_and_logp(out, sample=True):
    """Sample every head-slot; return idx tensor (B, H), summed logprob (B,), entropy (B,)."""
    cols = _flat_logits(out)
    idxs, logps, ents = [], [], []
    for _name, _s, logit in cols:
        lp = F.log_softmax(logit, -1)
        p = lp.exp()
        if sample:
            idx = torch.multinomial(p, 1).squeeze(-1)
        else:
            idx = logit.argmax(-1)
        idxs.append(idx)
        logps.append(lp.gather(-1, idx.unsqueeze(-1)).squeeze(-1))
        ents.append(-(p * lp).sum(-1))
    return (torch.stack(idxs, 1), torch.stack(logps, 1).sum(1), torch.stack(ents, 1).sum(1))


def logp_of(out, idxs):
    """Recompute summed logprob + entropy of stored idxs (B,H) under current out."""
    cols = _flat_logits(out)
    logps, ents = [], []
    for j, (_name, _s, logit) in enumerate(cols):
        lp = F.log_softmax(logit, -1)
        p = lp.exp()
        logps.append(lp.gather(-1, idxs[:, j].unsqueeze(-1)).squeeze(-1))
        ents.append(-(p * lp).sum(-1))
    return torch.stack(logps, 1).sum(1), torch.stack(ents, 1).sum(1)


_MV, _KV = PA.MOVER_VERBS, PA.MARKET_VERBS
# idx_row (56) layout, matching _flat_logits(HEADS):
#   0 fverb | 1 farg | 2..13 hverb*12 | 14..25 harg*12 |
#   26..35 mverb*10 | 36..45 marg*10 | 46..55 mqty*10
_O_HV, _O_HA = 2, 2 + B.MAX_HANDS
_O_MV, _O_MA, _O_MQ = 2 + 2 * B.MAX_HANDS, 2 + 2 * B.MAX_HANDS + B.MAX_MARKET, \
    2 + 2 * B.MAX_HANDS + 2 * B.MAX_MARKET


def idxs_to_action(idx_row, n_hands):
    """Build the action DIRECTLY from the sampled index row -- no throwaway -inf
    tensors, no argmax round-trip. ``n_hands`` = the seat's hand count (from the
    engine), which is all the obs was ever used for here."""
    fv = _MV[int(idx_row[0])]
    fa = PA._mover_arg_name(fv, int(idx_row[1]))
    farmer = [fv] + ([fa] if fa else [])
    nh = min(int(n_hands), B.MAX_HANDS)
    hands = []
    for i in range(nh):
        hv = _MV[int(idx_row[_O_HV + i])]
        ha = PA._mover_arg_name(hv, int(idx_row[_O_HA + i]))
        hands.append([hv] + ([ha] if ha else []))
    market = []
    for i in range(B.MAX_MARKET):
        mv = _KV[int(idx_row[_O_MV + i])]
        if mv == "PASS":
            continue
        ma = PA._market_arg_name(mv, int(idx_row[_O_MA + i]))
        qty = int(idx_row[_O_MQ + i])
        market.append([mv] + ([ma, qty] if ma else []))
    return {"farmer": farmer, "hands": hands, "market": market}


def encode(obs, seat, nm, ns):
    toks = TC.encode_tokens(obs, seat)
    n = min(toks.shape[0], TC.MAX_TOKENS)
    x = np.zeros((TC.MAX_TOKENS, TC.TOK_W), np.float32)
    m = np.zeros((TC.MAX_TOKENS,), np.bool_)
    x[:n] = (toks[:n].astype(np.float32) - nm) / ns
    m[:n] = True
    return x, m


def opponent_pool(device, n_top=16):
    """LIVE opponents = the TOP-SCORED reactive ladder agents (highest rating first),
    plus PASS as a weak anchor for the no-loss-<2500 profile, plus every crown-panel
    ref across all bands for spectrum. Returns [(agent_fn, rating)] so the reward
    weights losses by opponent strength (never lose <2500; hold 70-90% >=2500).
    Training vs only strong opponents would drop games to weak ones -- fatal in a
    BT/Elo final -- so PASS + mid-band refs stay in."""
    import kaggriculture.measure.eval_harness as EH
    pool = [(lambda o: {}, 1200.0)]              # PASS = a weak-end anchor
    seen = set()
    # 1) top-scored LIVE reactive agents (what actually wins the tournament)
    try:
        import kaggriculture.data.selfplay_corpus as SC
        for _name, path, rating in SC.top_reactive_agents(n_top):
            if path in seen:
                continue
            try:
                pool.append((EH._as_agent(path), float(rating))); seen.add(path)
            except Exception:
                pass
    except Exception:
        pass
    # 2) extra spectrum: every playable crown-panel ref (all bands)
    try:
        import kaggriculture.measure.tournament_gate as TG
        for _name, path, rating in TG._refs():
            if path in seen:
                continue
            try:
                pool.append((EH._as_agent(path), float(rating))); seen.add(path)
            except Exception:
                pass
    except Exception:
        pass
    if len(pool) <= 1:                           # fallback: at least the killers
        try:
            import kaggriculture.bandit.gate.harness as H
            for _n, p in H.killers():
                try:
                    pool.append((EH._as_agent(p), 2700.0))
                except Exception:
                    pass
        except Exception:
            pass
    return pool


def rollout(model, envs, opps, nm, ns, device, rtg, rails, gamma=0.997, lam=0.95,
            weak_thresh=2500.0, weak_loss=3.0):
    """Play all envs to terminal in lockstep; return a flat batch of transitions.
    Terminal reward is win/draw/loss, but a LOSS to an opponent rated below
    ``weak_thresh`` costs ``weak_loss`` (vs 1.0 to a strong one) -- pushing the
    tournament profile: never lose to <2500, hold 70-90% >=2500."""
    n = len(envs)
    states = [e.cmd(f"RESET {random.randint(1, 10**7)}") for e in envs]
    seats = [random.randint(0, 1) for _ in range(n)]
    picks = [random.choice(opps) for _ in range(n)]
    opp_ag = [p[0] for p in picks]
    opp_rat = [p[1] for p in picks]
    buf = {k: [] for k in ("tok", "mask", "idx", "logp", "val", "rew", "env", "t")}
    ep_ret = []
    step = 0
    while step < EPISODE_STEPS - 1:
        xs = np.zeros((n, TC.MAX_TOKENS, TC.TOK_W), np.float32)
        ms = np.zeros((n, TC.MAX_TOKENS), np.bool_)
        obs_list = []
        for i, js in enumerate(states):
            o = obs_for(seats[i], js)
            obs_list.append(o)
            xs[i], ms[i] = encode(o, seats[i], nm, ns)
        tx = torch.from_numpy(xs).to(device)
        tm = torch.from_numpy(ms).to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            out = model(tx, tm, rtg.expand(n, 1))
            idx, logp, _ent = sample_and_logp(out, sample=True)
            val = out["value"]
        idx_cpu = idx.cpu().numpy(); logp_cpu = logp.cpu().numpy(); val_cpu = val.cpu().numpy()
        lines = []
        for i, js in enumerate(states):
            _fm = (obs_list[i].get("farms") or [{}])
            _nh = len((_fm[seats[i]] if seats[i] < len(_fm) else {}).get("hands") or [])
            act = idxs_to_action(idx_cpu[i], _nh)
            act = RR.apply_rails(act, obs_list[i], seats[i], rails)
            opp_act = _call(opp_ag[i], obs_for(1 - seats[i], js))
            la = action_to_line(act if seats[i] == 0 else opp_act)
            lb = action_to_line(opp_act if seats[i] == 0 else act)
            lines.append(f"STEP2 {la}\x1e{lb}")
        # PIPELINED IPC: write all N step commands, THEN read all N replies -- the N
        # kagg processes step CONCURRENTLY across cores instead of one-at-a-time.
        for e, ln in zip(envs, lines):
            e.p.stdin.write(ln + "\n")
        for e in envs:
            e.p.stdin.flush()
        newstates = [json.loads(e.p.stdout.readline()) for e in envs]
        for i in range(len(envs)):
            buf["tok"].append(xs[i]); buf["mask"].append(ms[i])
            buf["idx"].append(idx_cpu[i]); buf["logp"].append(float(logp_cpu[i]))
            buf["val"].append(float(val_cpu[i])); buf["rew"].append(0.0)
            buf["env"].append(i); buf["t"].append(step)
            states[i] = newstates[i]
        step += 1
    # terminal reward = normalized bank margin for our seat
    for i, js in enumerate(states):
        banks = [float(f.get("money") or 0) for f in js["farms"]]
        mine, opp = banks[seats[i]], banks[1 - seats[i]]
        if mine > opp:
            r = 1.0
        elif mine == opp:
            r = 0.0
        else:
            r = -weak_loss if opp_rat[i] < weak_thresh else -1.0    # weak-loss penalty
        ep_ret.append(r)
        # place terminal reward on the last transition of this env
        for k in range(len(buf["rew"]) - 1, -1, -1):
            if buf["env"][k] == i:
                buf["rew"][k] = r
                break
    # GAE per env
    adv = [0.0] * len(buf["rew"])
    ret = [0.0] * len(buf["rew"])
    for i in range(n):
        idxs = [k for k in range(len(buf["env"])) if buf["env"][k] == i]
        gae = 0.0
        nextv = 0.0
        for k in reversed(idxs):
            delta = buf["rew"][k] + gamma * nextv - buf["val"][k]
            gae = delta + gamma * lam * gae
            adv[k] = gae
            ret[k] = gae + buf["val"][k]
            nextv = buf["val"][k]
    return buf, adv, ret, float(np.mean(ep_ret))


class League:
    """PFSP-lite opponent league: a pool of past-policy snapshots (CPU state_dicts).
    ``sample`` returns a snapshot to load into the frozen opponent model, or None =
    MIRROR (opponent = the current learner). Snapshots are added periodically during
    training so the learner keeps facing progressively stronger past selves."""
    def __init__(self, cap=10, mirror_frac=0.35):
        self.snaps, self.cap, self.mirror_frac = [], cap, mirror_frac

    def add(self, model):
        sd = {k: v.detach().cpu().clone() for k, v in model.policy.state_dict().items()}
        self.snaps.append(sd)
        if len(self.snaps) > self.cap:
            self.snaps.pop(random.randrange(len(self.snaps) - 1))   # keep newest, thin the rest

    def sample(self):
        if not self.snaps or random.random() < self.mirror_frac:
            return None
        return random.choice(self.snaps)


def _toks_from_flat(flat, nm, ns):
    """Parse vecserve's 'ntok v v ...' flat token string -> normalized (x, mask)."""
    parts = flat.split()
    x = np.zeros((TC.MAX_TOKENS, TC.TOK_W), np.float32)
    m = np.zeros((TC.MAX_TOKENS,), np.bool_)
    ntok = int(parts[0]) if parts else 0
    if ntok > 0:
        arr = np.array(parts[1:1 + ntok * TC.TOK_W], np.int32).reshape(ntok, TC.TOK_W)
        k = min(ntok, TC.MAX_TOKENS)
        x[:k] = (arr[:k].astype(np.float32) - nm) / ns
        m[:k] = True
    return x, m


def _parse_batch(flats, nm, ns, x, m):
    """Vectorized parse of n 'ntok v v...' flat strings into preallocated
    x[n,MAXT,W] / m[n,MAXT]. Uses np.fromstring (C-level) instead of Python
    str.split()+list->array per env, and normalizes the whole batch at once."""
    W, MT = TC.TOK_W, TC.MAX_TOKENS
    x.fill(0.0); m.fill(False)
    for i, flat in enumerate(flats):
        a = np.fromstring(flat, dtype=np.float32, sep=" ")   # parse in C, not Python
        if a.size:
            ntok = int(a[0]); k = ntok if ntok < MT else MT
            if k:
                x[i, :k] = a[1:1 + k * W].reshape(k, W)
                m[i, :k] = True
    x -= nm; x /= ns                 # broadcast-normalize the full [n,MAXT,W] batch
    x *= m[:, :, None]               # re-zero padding (matches per-token normalization)


def rollout_sp(model, opp_model, vs, nm, ns, device, rtg, rails, league,
               gamma=0.997, lam=0.95, teacher=None):
    """FAST self-play rollout. Learner (current policy) plays an NN opponent (a
    league snapshot, or a mirror of itself) -- BOTH seats are neural nets, so each
    game-step is TWO batched GPU forwards (learner N, opponent N) instead of 24
    Python opponent calls, and the N games are stepped with PIPELINED IPC (all
    written, then all read -> concurrent across cores). Trains on the LEARNER's
    transitions only. Reward = win/draw/loss for the learner's seat (the tournament
    <2500/>=2500 profile is enforced by the gate, not per self-play game)."""
    n = vs.n
    snap = league.sample()
    mirror = snap is None
    if not mirror:
        opp_model.policy.load_state_dict(snap)
    opp_model.eval()
    opp_net = model if mirror else opp_model
    states = vs.reset(random.randint(1, 10**7))
    lseat = [random.randint(0, 1) for _ in range(n)]          # learner's seat per env
    buf = {k: [] for k in ("tok", "mask", "idx", "logp", "val", "rew", "env", "t")}
    if teacher is not None:
        buf["tlog"] = []                     # cached frozen-teacher logits (fp16)
    # opponent buffers are never stored -> reuse in place; learner buffers ARE
    # stored in buf, so they're freshly allocated each step (no view aliasing).
    ox = np.zeros((n, TC.MAX_TOKENS, TC.TOK_W), np.float32); om = np.zeros((n, TC.MAX_TOKENS), np.bool_)
    step = 0
    while step < EPISODE_STEPS - 1:
        lx = np.empty((n, TC.MAX_TOKENS, TC.TOK_W), np.float32); lm = np.empty((n, TC.MAX_TOKENS), np.bool_)
        lflats = [None] * n; oflats = [None] * n; nhl = [0] * n; nho = [0] * n
        for i, g in enumerate(states):                 # cheap dict lookups only
            ls = "0" if lseat[i] == 0 else "1"; osd = "1" if lseat[i] == 0 else "0"
            lflats[i] = g["t" + ls]; oflats[i] = g["t" + osd]
            nhl[i] = g["nh" + ls]; nho[i] = g["nh" + osd]
        _parse_batch(lflats, nm, ns, lx, lm)           # vectorized C-level parse
        _parse_batch(oflats, nm, ns, ox, om)
        lx_t = torch.from_numpy(lx).to(device); lm_t = torch.from_numpy(lm).to(device)
        # fp16 autocast for the (inference-only) rollout forwards -- master weights
        # stay fp32; halves the forward cost with no effect on sampled actions.
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            lout = model(lx_t, lm_t, rtg.expand(n, 1))
            lidx, llogp, _ = sample_and_logp(lout, sample=True)
            lval = lout["value"]
            oout = opp_net(torch.from_numpy(ox).to(device), torch.from_numpy(om).to(device), rtg.expand(n, 1))
            oidx, _, _ = sample_and_logp(oout, sample=True)
            if teacher is not None:                       # cache teacher logits for PPO KL
                tout = teacher(lx_t, lm_t, rtg.expand(n, 1))
                tflat_c = _flatten_heads(tout, n).to("cpu", torch.float16).numpy()
        lidx_c = lidx.cpu().numpy(); llogp_c = llogp.cpu().numpy(); lval_c = lval.cpu().numpy()
        oidx_c = oidx.cpu().numpy()
        lidx_l = lidx_c.tolist(); oidx_l = oidx_c.tolist()   # native ints -> fast decode
        pairs = []
        for i in range(n):                        # rails are deploy-only now (need obs)
            la_a = idxs_to_action(lidx_l[i], nhl[i])
            oa_a = idxs_to_action(oidx_l[i], nho[i])
            la = action_to_line(la_a if lseat[i] == 0 else oa_a)
            lb = action_to_line(oa_a if lseat[i] == 0 else la_a)
            pairs.append((la, lb))
        newstates = vs.step(pairs)                 # ONE Rust IPC steps all N games
        for i in range(n):
            buf["tok"].append(lx[i]); buf["mask"].append(lm[i])
            buf["idx"].append(lidx_c[i]); buf["logp"].append(float(llogp_c[i]))
            buf["val"].append(float(lval_c[i])); buf["rew"].append(0.0)
            buf["env"].append(i); buf["t"].append(step)
            if teacher is not None:
                buf["tlog"].append(tflat_c[i])
            states[i] = newstates[i]
        step += 1
    ep_ret = []
    for i, g in enumerate(states):
        banks = [g["m0"], g["m1"]]
        mine, opp = banks[lseat[i]], banks[1 - lseat[i]]
        r = 1.0 if mine > opp else (0.0 if mine == opp else -1.0)
        ep_ret.append(r)
        for k in range(len(buf["rew"]) - 1, -1, -1):
            if buf["env"][k] == i:
                buf["rew"][k] = r; break
    adv = [0.0] * len(buf["rew"]); ret = [0.0] * len(buf["rew"])
    for i in range(n):
        idxs = [k for k in range(len(buf["env"])) if buf["env"][k] == i]
        gae = 0.0; nextv = 0.0
        for k in reversed(idxs):
            delta = buf["rew"][k] + gamma * nextv - buf["val"][k]
            gae = delta + gamma * lam * gae
            adv[k] = gae; ret[k] = gae + buf["val"][k]; nextv = buf["val"][k]
    return buf, adv, ret, float(np.mean(ep_ret))


def ppo_update(model, teacher, opt, scaler, buf, adv, ret, device, rtg,
               epochs=2, mb=512, clip=0.2, vf=0.5, ent_c=0.01, kl_c=0.5):
    # mb bounded by O(seq^2*mb) attention memory; 512 fits the 6M model on a 3090
    # and halves minibatch iterations vs 256 (faster PPO). Drop to 256 for a 12M model.
    tok = torch.tensor(np.array(buf["tok"]), device=device)
    mask = torch.tensor(np.array(buf["mask"]), device=device)
    idx = torch.tensor(np.array(buf["idx"]), device=device, dtype=torch.long)
    oldlp = torch.tensor(buf["logp"], device=device)
    adv_t = torch.tensor(adv, device=device); ret_t = torch.tensor(ret, device=device)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-6)
    # cached frozen-teacher logits (rollout computed them once) -> no teacher forward here
    tlog = torch.tensor(np.array(buf["tlog"]), device=device) if buf.get("tlog") else None
    N = tok.shape[0]
    last = 0.0
    for _ in range(epochs):
        perm = torch.randperm(N, device=device)
        for s in range(0, N, mb):
            b = perm[s:s + mb]
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                out = model(tok[b], mask[b], rtg.expand(len(b), 1))
                lp, ent = logp_of(out, idx[b])
                ratio = (lp - oldlp[b]).exp()
                s1 = ratio * adv_t[b]
                s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t[b]
                pg = -torch.min(s1, s2).mean()
                vloss = F.mse_loss(out["value"], ret_t[b])
                if tlog is not None:                       # reuse cached teacher logits
                    tout = _unflatten_heads(tlog[b].float(), len(b))
                else:
                    with torch.no_grad():
                        tout = teacher(tok[b], mask[b], rtg.expand(len(b), 1))
                kl = _teacher_kl(out, tout)
                loss = pg + vf * vloss - ent_c * ent.mean() + kl_c * kl
            opt.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            last = float(loss.detach())
    return last


def _teacher_kl(out, tout):
    kl = 0.0
    for name, _slots, _card in HEADS:
        lp = F.log_softmax(out[name], -1)
        tlp = F.log_softmax(tout[name], -1)
        kl = kl + (tlp.exp() * (tlp - lp)).sum(-1).mean()
    return kl


@torch.no_grad()
def _ema_update(teacher, model, decay):
    """Move the teacher a little toward the current policy (target-network / EMA):
    teacher <- decay*teacher + (1-decay)*model. Anchors the KL to a SMOOTHED self,
    which recovers the consolidation a 2nd PPO epoch gives, cheaply."""
    for tp, mp in zip(teacher.policy.parameters(), model.policy.parameters()):
        tp.mul_(decay).add_(mp.detach(), alpha=1.0 - decay)
    for tb, mb in zip(teacher.policy.buffers(), model.policy.buffers()):
        tb.copy_(mb)


def save(path, model, opt, scaler, gstep, done, nm, ns, cfg=None):
    tmp = path + ".tmp"
    torch.save({"state": model.policy.state_dict(), "value": model.value.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict() if scaler else None,
                "global_step": int(gstep), "done": bool(done), "layout_version": 2,
                "norm_mean": nm.tolist(), "norm_std": ns.tolist(), "rl": True,
                "config": cfg or {}}, tmp)          # carry model dims so serve rebuilds it
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", default=None, help="BC or RL checkpoint to start/continue from")
    ap.add_argument("--out", default="policy_rl.pt")
    ap.add_argument("--steps", type=int, default=500_000_000, help="env-step target")
    ap.add_argument("--envs", type=int, default=64,
                    help="self-play envs; rollout is batch-bound (256 best on a 24GB card, 64 on a laptop)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ckpt-secs", type=float, default=150.0, help="checkpoint cadence (<=3 min loss)")
    ap.add_argument("--max-hours", type=float, default=10.5)
    ap.add_argument("--rails", default=None, help="JSON file of reactive rail config")
    ap.add_argument("--weak-thresh", type=float, default=2500.0,
                    help="opponents below this rating are 'weak' -- losing to them is penalized more")
    ap.add_argument("--weak-loss", type=float, default=3.0,
                    help="reward for LOSING to a <weak-thresh opponent (vs -1 to a strong one)")
    ap.add_argument("--legacy-opp", action="store_true",
                    help="use the slow Python-opponent rollout instead of fast NN self-play")
    ap.add_argument("--league-cap", type=int, default=10)
    ap.add_argument("--league-every", type=int, default=30,
                    help="rollouts between adding a past-self snapshot to the league")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-compile", action="store_true",
                    help="disable torch.compile on the RL forward/backward (debug only)")
    # --- sample-efficiency levers (esp. when running --ppo-epochs 1 for speed) ---
    ap.add_argument("--ppo-epochs", type=int, default=2,
                    help="PPO gradient passes per rollout buffer (1=fast/less sample-eff, 2+=more)")
    ap.add_argument("--kl-c", type=float, default=0.5,
                    help="teacher-KL coefficient; raise (e.g. 1.0) under --ppo-epochs 1 for stability")
    ap.add_argument("--teacher-mode", choices=["bc", "ema", "refresh"], default="bc",
                    help="bc=frozen BC anchor (imitation prior); ema=slow copy of SELF "
                         "(consolidation/target-net); refresh=hard-copy self periodically")
    ap.add_argument("--teacher-ckpt", default=None,
                    help="load the teacher-KL anchor from THIS checkpoint (e.g. an improved "
                         "local BC), independent of the --resume policy. Must share model dims.")
    ap.add_argument("--teacher-ema-decay", type=float, default=0.999,
                    help="EMA decay for --teacher-mode ema (higher=slower teacher)")
    ap.add_argument("--teacher-refresh-every", type=int, default=200,
                    help="rollouts between teacher hard-refresh for --teacher-mode refresh")
    a = ap.parse_args()
    import json
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = (dev == "cuda")
    rails = json.load(open(a.rails)) if a.rails and os.path.exists(a.rails) else []
    nm, ns = np.zeros(TC.TOK_W, np.float32), np.ones(TC.TOK_W, np.float32)
    gstep = 0
    rp = a.resume or _find_ckpt(a.out)
    ck = None
    dims = {}
    if rp and os.path.exists(rp):
        ck = torch.load(rp, map_location=dev, weights_only=False)
        _cfg = ck.get("config") or {}
        dims = {k: _cfg[k] for k in ("d_model", "layers", "heads", "ff") if k in _cfg}
    model = ActorCritic(dims).to(dev)                 # match the warm-start ckpt's dims
    teacher = ActorCritic(dims).to(dev)
    if ck is not None:
        model.policy.load_state_dict(ck["state"])
        if ck.get("value"):
            model.value.load_state_dict(ck["value"])
        nm = np.array(ck.get("norm_mean", nm), np.float32)
        ns = np.maximum(np.array(ck.get("norm_std", ns), np.float32), 1e-3)
        gstep = int(ck.get("global_step", 0)) if ck.get("rl") else 0
        print(f"[rl] warm-start from {rp} (rl={ck.get('rl', False)}) dims={dims} gstep={gstep}", flush=True)
    if a.teacher_ckpt and os.path.exists(a.teacher_ckpt):
        tck = torch.load(a.teacher_ckpt, map_location=dev, weights_only=False)
        try:
            teacher.policy.load_state_dict(tck["state"])          # refreshed teacher (e.g. improved BC)
            print(f"[rl] teacher <- {a.teacher_ckpt} (independent of policy warm-start)", flush=True)
        except Exception as e:
            print(f"[rl] teacher-ckpt dims mismatch ({e}); falling back to policy copy", flush=True)
            teacher.policy.load_state_dict(model.policy.state_dict())
    else:
        teacher.policy.load_state_dict(model.policy.state_dict())  # frozen BC teacher = the warm-start
    for p in teacher.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    scaler = torch.amp.GradScaler("cuda") if amp else None
    rtg = torch.tensor([[1.0]], device=dev)
    n_envs = 2 if a.smoke else a.envs
    use_sp = not a.legacy_opp
    envs = None; vs = None; opp_model = None; league = None; opps = None
    if use_sp:                                   # FAST path: NN self-play over vecserve
        vs = VecServe(n_envs)
        opp_model = ActorCritic(dims).to(dev)
        for p in opp_model.parameters():
            p.requires_grad_(False)
        league = League(cap=a.league_cap)
        print(f"[rl] SELF-PLAY (Rust vecserve) dev={dev} envs={n_envs} "
              f"target={a.steps:,} ppo_epochs={a.ppo_epochs} kl_c={a.kl_c} "
              f"teacher={a.teacher_mode}", flush=True)
    else:                                        # legacy: Python-opponent rollout
        envs = [Serve() for _ in range(n_envs)]
        opps = opponent_pool(dev)
        print(f"[rl] legacy-opp dev={dev} envs={n_envs} opps={len(opps)} "
              f"target={a.steps:,}", flush=True)
    # torch.compile the forward/backward -- the transformer forward is ~90% of
    # rollout and PPO is ~50% of the iteration, so compiling the learner/opponent/
    # teacher (SDPA path instead of the slow nested-tensor kernel) is the single
    # lever over the whole RL. Raw modules stay for save()/league snapshots.
    compile_on = (dev == "cuda") and not a.no_compile
    _c = (lambda m: torch.compile(m)) if compile_on else (lambda m: m)
    fmodel = _c(model)
    fteacher = _c(teacher)
    fopp = _c(opp_model) if opp_model is not None else None
    print(f"[rl] torch.compile={'on' if compile_on else 'off'} "
          f"(first rollout+PPO pay one-time compile cost)", flush=True)
    from kaggriculture.trackp import trainmetrics as TM
    t_start = time.time(); t_ckpt = time.time(); n_roll = 0
    _m_gstep = gstep; _m_time = time.time()
    try:
        while gstep < a.steps:
            if use_sp:
                buf, adv, ret, winr = rollout_sp(fmodel, fopp, vs, nm, ns, dev, rtg,
                                                 rails, league, teacher=fteacher)
            else:
                buf, adv, ret, winr = rollout(fmodel, envs, opps, nm, ns, dev, rtg, rails,
                                              weak_thresh=a.weak_thresh, weak_loss=a.weak_loss)
            loss = ppo_update(fmodel, fteacher, opt, scaler, buf, adv, ret, dev, rtg,
                              epochs=1 if a.smoke else a.ppo_epochs, kl_c=a.kl_c)
            gstep += len(buf["rew"]); n_roll += 1
            # teacher consolidation: ema=slow copy of self, refresh=periodic hard-copy.
            # (bc mode leaves the frozen BC anchor untouched.)
            if a.teacher_mode == "ema":
                _ema_update(teacher, model, a.teacher_ema_decay)
            elif a.teacher_mode == "refresh" and n_roll % max(1, a.teacher_refresh_every) == 0:
                teacher.policy.load_state_dict(model.policy.state_dict())
            print(f"[rl] gstep={gstep:,} meanR={winr:+.2f} loss={loss:.3f} "
                  f"{(time.time()-t_start)/3600:.2f}h", flush=True)
            if use_sp and n_roll % max(1, a.league_every) == 0:
                league.add(model)                # snapshot a past self into the league
            if time.time() - t_ckpt >= a.ckpt_secs:
                save(a.out, model, opt, scaler, gstep, False, nm, ns, dims)
                _dt = max(time.time() - _m_time, 1e-6)
                TM.append(a.out, "rl", gstep=int(gstep), meanR=round(float(winr), 4),
                          winrate=round((float(winr) + 1.0) / 2.0, 4),
                          loss=round(float(loss), 4),
                          eps=int((gstep - _m_gstep) / _dt))
                _m_gstep = gstep; _m_time = time.time(); t_ckpt = time.time()
            if not a.smoke and (time.time() - t_start) / 3600.0 >= a.max_hours:
                save(a.out, model, opt, scaler, gstep, False, nm, ns, dims)
                print("[rl] TIME BUDGET hit -- checkpoint saved, exit to commit.", flush=True)
                break
            if a.smoke:
                break
    finally:
        if vs is not None:
            vs.close()
        for e in (envs or []):
            e.close()
    save(a.out, model, opt, scaler, gstep, gstep >= a.steps, nm, ns, dims)
    print(f"[rl] saved {a.out} @ gstep {gstep:,}", flush=True)


def _find_ckpt(out):
    if os.path.exists(out):
        return out
    c = sorted(glob.glob("/kaggle/input/**/policy_*.pt", recursive=True))
    return c[0] if c else None


if __name__ == "__main__":
    main()
