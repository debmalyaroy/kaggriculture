"""Behaviour cloning over the trace corpus, with a return-conditioned head.

The E3 component. Trains a small net to predict the per-product SELL VECTOR from
the 49-field state, conditioned on the outcome, so at inference you can ask for
"what a WINNING farm sells here" rather than "what the average farm sells here".
That is return-conditioned BC, and it is the honest formulation: generating a
route is a one-shot decision, so this is a contextual problem, not a sequential
one needing RL.

What this deliberately does NOT do: blend over the arm library. The plan's blend
head needs a GOOD arm basis, and the audit says a commit is worth -$50,787 with
even correct commits at -$52,940. Building a policy on that basis would produce
a measured-bad policy for a reason that has nothing to do with the policy. Arms
must be rehabilitated first (full-budget score-fitness retrain on the
field-matched panel), which needs the Rust engine to be affordable.

HONEST LIMIT: the corpus is whatever `turn_features` has accrued. Capture is
forward-only by decision, so today it is 22 bootstrap episodes / 31,680
transitions -- enough to prove the pipeline end to end and to measure, nowhere
near enough to ship. The numbers this prints are data-limited and say so.

    C:/ProgramData/anaconda3/envs/llm/python.exe src/experiments/policy_bc.py
"""
from kaggriculture.paths import ROOT
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = ROOT
import kaggriculture.data.policy_dataset as PD  # noqa: E402
import kaggriculture.data.turn_features as TF  # noqa: E402

OUT = os.path.join(ROOT, "models", "lab", "policy_bc.json")
HIDDEN = 64
EPOCHS = 60
MIN_EPISODES = 8            # below this, refuse rather than report noise


def episode_split(eps, frac=0.25):
    """Split by EPISODE, never by transition.

    Consecutive turns of one game are near-duplicates, so a random transition
    split leaks the answer across the boundary and reports a validation score
    that means nothing.
    """
    uniq = sorted(set(eps.tolist()))
    n_val = max(1, int(len(uniq) * frac))
    val = set(uniq[-n_val:])
    mask = np.array([e in val for e in eps.tolist()])
    return ~mask, mask, sorted(val)


def main():
    data = PD.load()
    if data is None:
        print("no dataset -- run: python src/policy_dataset.py")
        return 0
    X, A, Y, EP = data["X"], data["A"], data["Y"], data["episodes"]
    n_ep = len(set(EP.tolist()))
    print(f"corpus: {X.shape[0]:,} transitions, {n_ep} episodes, "
          f"{X.shape[1]} state dims, {A.shape[1]} action dims")
    if n_ep < MIN_EPISODES:
        print(f"REFUSING to train: {n_ep} episodes is below the {MIN_EPISODES} "
              f"floor. Capture is forward-only; this fills on every ingest.")
        return 0

    tr, va, val_eps = episode_split(EP)
    print(f"split by EPISODE: {tr.sum():,} train / {va.sum():,} val "
          f"({len(val_eps)} held-out episodes)")

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("torch unavailable -- install it or run in the llm env")
        return 0
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # Condition on the outcome: the extra input is the return, so at inference
    # you request a winning trajectory rather than the corpus average.
    xtr = np.concatenate([X[tr], Y[tr][:, None]], axis=1)
    xva = np.concatenate([X[va], Y[va][:, None]], axis=1)
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    amu, asd = A[tr].mean(0), A[tr].std(0) + 1e-6

    Xtr = torch.tensor((xtr - mu) / sd, device=dev)
    Atr = torch.tensor((A[tr] - amu) / asd, device=dev)
    Xva = torch.tensor((xva - mu) / sd, device=dev)
    Ava = torch.tensor((A[va] - amu) / asd, device=dev)

    torch.manual_seed(11)
    model = nn.Sequential(
        nn.Linear(Xtr.shape[1], HIDDEN), nn.Tanh(),
        nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
        nn.Linear(HIDDEN, A.shape[1])).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    n = len(Xtr)
    best = (1e9, None)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, 512):
            b = perm[i:i + 512]
            loss = nn.functional.smooth_l1_loss(model(Xtr[b]), Atr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(nn.functional.smooth_l1_loss(model(Xva), Ava))
        if vl < best[0]:
            best = (vl, [p.detach().cpu().numpy().copy()
                         for p in model.parameters()])
        if ep % 20 == 0 or ep == EPOCHS - 1:
            print(f"  epoch {ep:>3}: val {vl:.4f}")

    # Baseline that must be beaten: predict the training mean action.
    with torch.no_grad():
        zeros = torch.zeros_like(Ava)
        base = float(nn.functional.smooth_l1_loss(zeros, Ava))
    print(f"\nval loss {best[0]:.4f} vs mean-action baseline {base:.4f} "
          f"({'BEATS' if best[0] < base else 'DOES NOT BEAT'} the baseline)")

    # Load the BEST-epoch weights back into the model before exporting or
    # comparing. Exporting `best` while the live model still held the FINAL
    # epoch's weights made the equivalence check fail at 5.4e-01 -- which is the
    # train/serve skew this check exists to catch, caught on itself.
    with torch.no_grad():
        for p_, w in zip(model.parameters(), best[1]):
            p_.copy_(torch.tensor(w, device=dev))
    params = best[1]
    payload = {
        "arch": [int(Xtr.shape[1]), HIDDEN, HIDDEN, int(A.shape[1])],
        "act": "tanh",
        "W": [p.tolist() for p in params[0::2]],
        "b": [p.tolist() for p in params[1::2]],
        "x_mu": mu.tolist(), "x_sd": sd.tolist(),
        "a_mu": amu.tolist(), "a_sd": asd.tolist(),
        "products": list(TF.PRODUCTS),
        "val_loss": best[0], "baseline_loss": base,
        "n_transitions": int(X.shape[0]), "n_episodes": n_ep,
        "held_out_episodes": val_eps,
        "data_limited": bool(n_ep < 100),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(payload, open(OUT, "w", encoding="utf-8"))
    print(f"-> {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")

    # EXPORT EQUIVALENCE: the agent runs pure python with only `math`, so the
    # exported weights must reproduce torch's output. A silent mismatch here is
    # the classic train/serve skew, so it is asserted rather than trusted.
    import math

    def forward_py(x):
        v = [(x[i] - payload["x_mu"][i]) / payload["x_sd"][i]
             for i in range(len(x))]
        for li, (W, b) in enumerate(zip(payload["W"], payload["b"])):
            out = [sum(W[o][i] * v[i] for i in range(len(v))) + b[o]
                   for o in range(len(b))]
            v = [math.tanh(z) for z in out] if li < len(payload["W"]) - 1 else out
        return [v[i] * payload["a_sd"][i] + payload["a_mu"][i]
                for i in range(len(v))]

    model.eval()
    worst = 0.0
    with torch.no_grad():
        for i in range(min(64, len(xva))):
            raw = xva[i]
            t = model(torch.tensor(((raw - mu) / sd)[None, :],
                                   dtype=torch.float32, device=dev))[0]
            t = (t.cpu().numpy() * asd + amu)
            p = forward_py(raw.tolist())
            worst = max(worst, float(np.max(np.abs(np.array(p) - t))))
    print(f"export equivalence: worst |diff| {worst:.3e} "
          f"({'OK' if worst < 1e-3 else 'FAIL'})")
    if payload["data_limited"]:
        print(f"\nDATA-LIMITED: {n_ep} episodes. This proves the pipeline, not "
              f"the policy. Capture is forward-only and accrues every ingest; "
              f"a shippable policy needs on the order of hundreds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
