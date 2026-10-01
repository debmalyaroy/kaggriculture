"""Profile audit (queue Q05): which lever profiles help or hurt when switched in day by day.

    python python/learn/profile_audit.py   ->  data/gates/profile_audit.json + printed table

From the random-profile league (data/train/league_meta.npz): a linear probability model
score ~ share of days on each profile + opponent-kind fixed effects, profile 0 (v61.1) as the
baseline. A profile whose coefficient is below 0 with |t| > 2.5 is listed under `mask`. This is a
REPORT: masking a profile in PPO (ppo-rollout --allow) is a separate decision.
"""
import json
import os

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import profile_names  # noqa: E402

RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    m = np.load(os.path.join(RL, "data", "train", "league_meta.npz"))
    names = profile_names()
    A = len(names)
    sched, y, kind = m["sched"].astype(int), m["score"].astype(float), m["opp_kind"].astype(int)
    share = np.stack([(sched == k).mean(1) for k in range(A)], 1)
    K = int(kind.max()) + 1
    X = np.concatenate([share[:, 1:], np.eye(K)[kind]], 1)  # profile 0 = baseline, kinds as intercepts
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    s2 = res @ res / max(1, len(y) - X.shape[1])
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    rows = []
    for k in range(1, A):
        b, e = beta[k - 1], se[k - 1]
        rows.append({"id": k, "name": names[k], "coef_full_game": float(b), "se": float(e), "t": float(b / e),
                     "days_played": int((sched == k).sum())})
    rows.sort(key=lambda r: r["t"])
    mask = [r["id"] for r in rows if r["t"] < -2.5]
    out = {"games": int(len(y)), "learner_score": float(y.mean()), "baseline": "profile 0 (v61.1)",
           "note": "coef = change in win probability if the WHOLE game were played on this profile instead of v61.1",
           "profiles": rows, "mask": mask}
    os.makedirs(os.path.join(RL, "data", "gates"), exist_ok=True)
    json.dump(out, open(os.path.join(RL, "data", "gates", "profile_audit.json"), "w"), indent=1)
    print(f"[audit] {len(y)} league games; learner score {y.mean():.3f}")
    for r in rows:
        print(f"  {r['id']:>2} {r['name']:<16} coef {r['coef_full_game']:+.3f} (t {r['t']:+.1f}), days {r['days_played']}")
    print(f"[audit] mask candidates (t < -2.5): {mask or 'none'}")


if __name__ == "__main__":
    main()
