"""Wrap a trained BC/RL checkpoint as a Kaggle ``agent(obs)`` -- the shared
inference path used by the tournament gate, RL rollouts, and deploy.

Flow each turn: obs -> TC.encode_tokens -> transformer -> factorized heads ->
verb-aware decode -> config-driven reactive rails (reactive_rails.apply_rails)
-> sanitize (legal, no desync). ``sample=False`` = argmax (gate/deploy);
``sample=True`` = categorical (RL exploration).
"""
from __future__ import annotations
import numpy as np
import torch

import kaggriculture.data.trackp_corpus as TC
import kaggriculture.trackp.bc_train as B
import kaggriculture.trackp.reactive_rails as RR

MOVER_VERBS, MARKET_VERBS = TC.MOVER_VERBS, TC.MARKET_VERBS
CROPS, ANIMALS, PRODUCTS = TC.CROPS, TC.ANIMALS, TC.PRODUCTS


def _mover_arg_name(verb, aid):
    if aid <= 0:
        return None
    tbl = {"PLANT": CROPS, "PLACE": ANIMALS, "PICKUP": PRODUCTS}.get(verb)
    return tbl[aid - 1] if tbl and aid - 1 < len(tbl) else None


def _market_arg_name(verb, aid):
    if aid <= 0:
        return None
    tbl = {"BUY_SEED": CROPS, "BUY_ANIMAL": ANIMALS,
           "BUY_PRODUCT": PRODUCTS, "SELL": PRODUCTS}.get(verb)
    return tbl[aid - 1] if tbl and aid - 1 < len(tbl) else None


def load_policy(ckpt_path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck.get("config") or {}
    dims = {k: cfg[k] for k in ("d_model", "layers", "heads", "ff") if k in cfg}
    model = B.build_model(**dims).to(device)          # dims from ckpt (else 12M default)
    model.load_state_dict(ck["state"])
    model.eval()
    nm = np.array(ck.get("norm_mean", np.zeros(TC.TOK_W)), np.float32)
    ns = np.maximum(np.array(ck.get("norm_std", np.ones(TC.TOK_W)), np.float32), 1e-3)
    return model, nm, ns, device


def _pick(logits, sample):
    if sample:
        p = torch.softmax(logits, -1)
        return int(torch.multinomial(p, 1).item())
    return int(torch.argmax(logits, -1).item())


def decode_action(out, obs, seat, sample=False):
    """Factorized head logits -> action dict (verb-aware)."""
    def g(k):
        return out[k][0]
    fv = MOVER_VERBS[_pick(g("fverb"), sample)]
    fa = _mover_arg_name(fv, _pick(g("farg"), sample))
    farmer = [fv] + ([fa] if fa else [])
    me = (obs.get("farms") or [{}])[seat] if seat < len(obs.get("farms") or []) else {}
    n_hands = len(me.get("hands") or [])
    hands = []
    for i in range(min(n_hands, TC.MAX_HANDS)):     # model has MAX_HANDS slots
        hv = MOVER_VERBS[_pick(out["hverb"][0, i], sample)]
        ha = _mover_arg_name(hv, _pick(out["harg"][0, i], sample))
        hands.append([hv] + ([ha] if ha else []))
    market = []
    for i in range(TC.MAX_MARKET):
        mv = MARKET_VERBS[_pick(out["mverb"][0, i], sample)]
        if mv == "PASS":
            continue
        ma = _market_arg_name(mv, _pick(out["marg"][0, i], sample))
        qty = _pick(out["mqty"][0, i], sample)
        order = [mv] + ([ma] if ma else []) + ([qty] if ma else [])
        market.append(order)
    return {"farmer": farmer, "hands": hands, "market": market}


def encode_batch(obs, seat, nm, ns, device):
    toks = TC.encode_tokens(obs, seat)
    n = min(toks.shape[0], TC.MAX_TOKENS)
    x = np.zeros((1, TC.MAX_TOKENS, TC.TOK_W), np.float32)
    m = np.zeros((1, TC.MAX_TOKENS), np.bool_)
    x[0, :n] = (toks[:n].astype(np.float32) - nm) / ns
    m[0, :n] = True
    return (torch.from_numpy(x).to(device), torch.from_numpy(m).to(device))


def make_agent(ckpt_path, rail_config=None, rtg=1.0, sample=False, device=None):
    """Return agent(obs[, config]) -> action dict, with reactive rails applied."""
    model, nm, ns, device = load_policy(ckpt_path, device)
    rtg_t = torch.tensor([[float(rtg)]], device=device)

    # NB: no_grad as a context manager (not a decorator) so agent.__code__
    # keeps co_argcount==2 -- serve_match._call truncates args by co_argcount.
    def agent(obs, config=None):
        seat = int((obs.get("player") if isinstance(obs, dict) else 0) or 0)
        with torch.no_grad():
            x, m = encode_batch(obs, seat, nm, ns, device)
            out = model(x, m, rtg_t)
            act = decode_action(out, obs, seat, sample=sample)
        return RR.apply_rails(act, obs, seat, rail_config)

    agent.model = model
    return agent
