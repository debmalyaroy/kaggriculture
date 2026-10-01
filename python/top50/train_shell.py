"""Train the sales shell on exact-engine sale labels, and compare the top teams' choices with ours (queue Q41+).

    python python/top50/train_shell.py --branch DIR [DIR ...] --out weights/shell/NAME [--epochs 30]

Input: `branch` rows (python/top50/branch_all.py): for each (game, turn, item) decision, the studied player's
class, our live agent's class (shadow), the shell features and the seat's final margin under each of
hold / sell part / sell all. Score of an outcome = win 1, draw 0.5, loss 0 (+ margin / 1e6 as a tie-break).
  target  = the class with the best score, whoever played it (inherit only what wins)
  weight  = best score - worst score (decisions where the choice cannot change the result carry ~0 weight)
Split by game (every 5th game held out). Writes NAME/shell.json (for `--shell`), NAME/report.json:
  compare     who is right more often and what it is worth: top player vs our agent, per team and per item
              (value = score of the choice; "better" = strictly higher score), and the head-room of the best
  model       held-out agreement with the best class; tau chosen on held-out games: apply the model only
              where it is >= tau confident, else keep our agent's choice; gain in score vs our agent alone
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch

NF = 20
FEAT = ["day", "hour", "hours_left", "stock", "mkt_inv", "px", "px_rel", "dpx", "dinv", "money_gap", "shop_item", "stock_share",
        "i_wheat", "i_carrot", "i_tomato", "i_strawberry", "i_melon", "i_egg", "i_milk", "i_wool"]
COLS = ["id", "seat", "step", "item", "their", "ours", "stock_n"] + FEAT + ["m_hold", "m_part", "m_all", "m_real"]


def score(m):
    return (m > 0).astype(float) + 0.5 * (m == 0) + m / 1e6


def load(dirs):
    rows, idx = [], []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.tsv"))):
            if os.path.basename(f).startswith("index_"):
                idx.append(pd.read_csv(f, sep="\t", dtype=str))
            elif os.path.getsize(f):
                rows.append(pd.read_csv(f, sep="\t", header=None, names=COLS))
    df = pd.concat(rows, ignore_index=True)
    team = pd.concat(idx).drop_duplicates("id").set_index("id").team if idx else pd.Series(dtype=str)
    df["team"] = df.id.astype(str).map(team).fillna("?")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2, help="hidden layers (the Rust shell reads any depth)")
    ap.add_argument("--target", choices=["best", "their"], default="best",
                    help="best = the exact-engine best of hold/part/all; their = imitate the studied top player's own choice "
                         "(the decompiled shell: their rules, not per-turn labels)")
    ap.add_argument("--min-team-games", type=int, default=0, help="--target their: only teams with this many labelled games")
    ap.add_argument("--min-team-win", type=float, default=0.0, help="--target their: only teams whose recorded win rate in these games is >= this")
    a = ap.parse_args()
    torch.manual_seed(0)
    df = load(a.branch)
    df = df[~df.team.isin(["us", "cand"])].reset_index(drop=True) if a.target == "their" else df  # imitate top players only
    if a.target == "their" and (a.min_team_games or a.min_team_win):
        g = df.groupby(["team", "id"]).m_real.first().reset_index()
        st = g.groupby("team").agg(n=("id", "size"), win=("m_real", lambda m: float((m > 0).mean() + 0.5 * (m == 0).mean())))
        keep = set(st[(st.n >= a.min_team_games) & (st.win >= a.min_team_win)].index)
        df = df[df.team.isin(keep)].reset_index(drop=True)
        print(f"[shell] imitating {len(keep)} teams: {sorted(keep)[:30]}", flush=True)
    S = np.stack([score(df[c].to_numpy(float)) for c in ("m_hold", "m_part", "m_all")], 1)
    best = S.argmax(1)
    spread = S.max(1) - S.min(1)
    their = df.their.to_numpy(int)
    ours = df.ours.to_numpy(int)
    s_their, s_ours, s_best = S[np.arange(len(S)), their], S[np.arange(len(S)), ours], S.max(1)
    df["d_their_ours"] = s_their - s_ours
    df["d_best_ours"] = s_best - s_ours

    def cmp(x):
        return {"decisions": int(len(x)), "their_better": int((x.d_their_ours > 1e-9).sum()), "ours_better": int((x.d_their_ours < -1e-9).sum()),
                "same": int((x.d_their_ours.abs() <= 1e-9).sum()), "their_minus_ours": round(float(x.d_their_ours.sum()), 2),
                "best_minus_ours": round(float(x.d_best_ours.sum()), 2),
                "win_flips_their_better": int((x.d_their_ours >= 0.5).sum()), "win_flips_ours_better": int((x.d_their_ours <= -0.5).sum())}

    rep = {"rows": int(len(df)), "games": int(df.id.nunique()), "all": cmp(df),
           "by_item": {it: cmp(x) for it, x in df.groupby("item")},
           "by_team": {tm: cmp(x) for tm, x in df.groupby("team") if len(x) >= 100},
           "by_day": {int(d): cmp(x) for d, x in df.groupby(df.step // 24)}}
    # ---- model
    games = sorted(df.id.unique())
    test = set(games[::5])
    te = df.id.isin(test).to_numpy()
    X = df[FEAT].to_numpy(np.float32)
    mu, sd = X[~te].mean(0), X[~te].std(0) + 1e-6
    Xn = (X - mu) / sd
    if a.target == "their":
        y, w = their.copy(), np.ones(len(their))
    else:
        y = best
        w = np.clip(spread, 0, 1.5) + 1e-3
    xt, yt, wt = map(torch.tensor, (Xn[~te], y[~te], w[~te].astype(np.float32)))
    mods, w_in = [], NF
    for _ in range(a.depth):
        mods += [torch.nn.Linear(w_in, a.hidden), torch.nn.ReLU()]
        w_in = a.hidden
    net = torch.nn.Sequential(*mods, torch.nn.Linear(w_in, 3))
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    n = len(yt)
    for ep in range(a.epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 4096):
            b = perm[i:i + 4096]
            loss = (torch.nn.functional.cross_entropy(net(xt[b]), yt[b], reduction="none") * wt[b]).sum() / wt[b].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        if ep % 5 == 4:
            print(f"[shell] epoch {ep + 1}: loss {tot / n:.4f}", flush=True)
    with torch.no_grad():
        P = torch.softmax(net(torch.tensor(Xn[te])), 1).numpy()
    St, ot = S[te], ours[te]
    pred, conf = P.argmax(1), P.max(1)
    info = spread[te] > 1e-6
    rep["model"] = {"held_out_rows": int(te.sum()), "agree_best_weighted": round(float(((pred == y[te]) * w[te]).sum() / w[te].sum()), 3),
                    "agree_best_informative": round(float((pred == y[te])[info].mean()), 3) if info.any() else None,
                    "ours_agree_best_informative": round(float((ot == y[te])[info].mean()), 3) if info.any() else None,
                    "their_agree_best_informative": round(float((their[te] == y[te])[info].mean()), 3) if info.any() else None}
    sweep = []
    base = St[np.arange(len(St)), ot].sum()
    for tau in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        use = conf >= tau
        ch = np.where(use, pred, ot)
        val = St[np.arange(len(St)), ch].sum()
        sweep.append({"tau": tau, "override_share": round(float((use & (pred != ot)).mean()), 4), "gain_vs_ours": round(float(val - base), 2),
                      "flips_gained": int(((St[np.arange(len(St)), ch] - St[np.arange(len(St)), ot]) >= 0.5).sum()),
                      "flips_lost": int(((St[np.arange(len(St)), ch] - St[np.arange(len(St)), ot]) <= -0.5).sum())})
    rep["tau_sweep"] = sweep
    tau = max(sweep, key=lambda r: r["gain_vs_ours"])["tau"]
    rep["tau"] = tau
    os.makedirs(a.out, exist_ok=True)
    layers = [m for m in net if isinstance(m, torch.nn.Linear)]
    json.dump({"version": 1, "features": FEAT, "mean": mu.tolist(), "std": sd.tolist(), "tau": tau,
               "layers": [{"w": l.weight.detach().numpy().tolist(), "b": l.bias.detach().numpy().tolist()} for l in layers]},
              open(os.path.join(a.out, "shell.json"), "w"))
    json.dump(rep, open(os.path.join(a.out, "report.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({k: rep[k] for k in ("rows", "games", "all", "model", "tau")}, indent=1))
    print(f"[shell] -> {a.out}/shell.json (tau {tau})")


if __name__ == "__main__":
    main()
