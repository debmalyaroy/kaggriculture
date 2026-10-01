"""Market-duel policy: offline Q-learning on the dump-timing race.

The only truly interactive sub-game: both sides plan a big SELL of the same
relay product; whoever lands first sells into the better price. Today's agent
adapts one parameter (the lead). This trains a policy over a discretized duel
state, against the EMPIRICAL family schedules (models/relay/family_dumps.json
or the v2 lab file), and reports regret vs the oracle best response on
HELD-OUT families it never trained on.

State  (turns_to_our_dump bucket, evidence level, observed_offset bucket)
   turns_to_our_dump: [0-2, 3-5, 6-10, 11-20, >20]
   evidence: how many of their dumps we've already seen (0, 1, 2+)
   observed_offset: their dump relative to ours from evidence so far
                    [theirs_much_earlier, ~same_or_earlier, later, unknown]
Action  advance our dump by {0, 1, 2, 4, 6, 10} turns
Reward  +1.0 win the collision (ours lands strictly first)
        -1.0 lose it (we sell into their dump)
        -0.03 * advance (distortion: selling early forfeits price accrual)
        0 if no collision within the window (plus any distortion paid)

Episode: one contested product. Their true dump time is sampled from the
family's consensus event +/- jitter drawn from its member spread; our
scheduled time starts offset around theirs. Evidence arrives as their EARLIER
dumps of other products are observed (same relative bias family-wide) --
matching how the runtime actually learns mid-game.

    python src/experiments/duel_lab.py --episodes 200000
"""
from kaggriculture.paths import ROOT
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = ROOT

DUMPS = os.path.join(ROOT, "models", "relay", "family_dumps.json")
OUT = os.path.join(ROOT, "models", "lab", "duel_q.json")

ACTIONS = (0, 1, 2, 4, 6, 10)
DIST_COST = 0.03


def bucket_state(turns_to_ours, evidence_n, offset):
    if turns_to_ours <= 2:
        t = 0
    elif turns_to_ours <= 5:
        t = 1
    elif turns_to_ours <= 10:
        t = 2
    elif turns_to_ours <= 20:
        t = 3
    else:
        t = 4
    e = min(evidence_n, 2)
    if offset is None:
        o = 3
    elif offset <= -3:
        o = 0
    elif offset <= 1:
        o = 1
    else:
        o = 2
    return (t, e, o)


class Duel:
    """One contested dump against one sampled opponent schedule."""

    def __init__(self, rng, jitter, bias):
        self.rng = rng
        # their true time relative to our current schedule
        self.bias = bias                       # family-wide timing bias
        self.theirs = bias + rng.randint(-jitter, jitter)
        self.ours = 0                          # we act when <=20 turns out
        self.horizon = 20
        self.evidence = 0
        self.observed = None

    def observe(self):
        """Earlier dumps of other products leak the family's bias."""
        if self.rng.random() < 0.45:
            self.evidence += 1
            noise = self.rng.randint(-2, 2)
            self.observed = self.bias + noise

    def play(self, advance):
        ours_final = -advance                  # advance = land earlier
        reward = -DIST_COST * advance
        if abs(self.theirs - ours_final) <= 6:  # collision window
            reward += 1.0 if ours_final < self.theirs else -1.0
        return reward


def episode(policy, rng, jitter, bias, learn=None, eps=0.0):
    d = Duel(rng, jitter, bias)
    for _ in range(rng.randint(0, 2)):
        d.observe()
    s = bucket_state(rng.randint(3, 20), d.evidence, d.observed)
    if learn is not None and rng.random() < eps:
        a_i = rng.randrange(len(ACTIONS))
    else:
        a_i = policy(s)
    r = d.play(ACTIONS[a_i])
    if learn is not None:
        learn(s, a_i, r)
    return r


def oracle_reward(rng, jitter, bias):
    """Best response knowing the true distribution: advance just past the
    worst plausible 'theirs' if a collision is plausible, else don't move."""
    d = Duel(rng, jitter, bias)
    earliest_theirs = bias - jitter
    if earliest_theirs > 6:                    # no collision possible
        return d.play(0)
    need = max(0, 1 - earliest_theirs)         # land strictly before
    adv = min(ACTIONS, key=lambda a: (a < need, a))
    if adv < need:
        adv = ACTIONS[-1]
    return d.play(adv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200000)
    ap.add_argument("--dumps", default=DUMPS)
    args = ap.parse_args()
    rng = random.Random(11)

    fams = json.load(open(args.dumps, encoding="utf-8"))["classes"]
    # family timing model: (bias, jitter) per family. The real dumps file
    # currently yields too few consensus classes to span the space (2 as of
    # 2026-08-12), so TRAIN on a synthetic grid covering the plausible range
    # and HOLD OUT both the real families and off-grid synthetic ones --
    # the policy must generalize to timing models it never saw.
    params = []
    for cls, events in sorted(fams.items(), key=lambda kv: int(kv[0])):
        iqr = 2
        if events and len(events[0]) >= 5:     # v2 schema has t_iqr
            iqr = max(1, int(sum(e[4] for e in events) / len(events)))
        bias = rng.randint(-8, 8)
        params.append((cls, bias, max(1, min(6, iqr))))
    train_p = [(f"syn{b}_{j}", b, j)
               for b in (-8, -5, -2, 0, 2, 5, 8) for j in (1, 3, 5)]
    test_p = params + [(f"synho{b}_{j}", b, j)
                       for b in (-7, -3, 1, 4, 7) for j in (2, 4)]
    print(f"{len(train_p)} train families (synthetic grid), "
          f"{len(test_p)} held-out ({len(params)} real)")

    Q = {}

    def qrow(s):
        return Q.setdefault(s, [0.0] * len(ACTIONS))

    def greedy(s):
        row = qrow(s)
        return max(range(len(ACTIONS)), key=lambda i: row[i])

    def learn(s, a_i, r, lr=0.05):
        row = qrow(s)
        row[a_i] += lr * (r - row[a_i])        # 1-step episodic

    for ep in range(args.episodes):
        cls, bias, jitter = train_p[rng.randrange(len(train_p))]
        eps = max(0.05, 1.0 - ep / (0.6 * args.episodes))
        episode(greedy, rng, jitter, bias, learn=learn, eps=eps)

    # -------- evaluation on held-out families ----------------------------
    def avg(fn, n=20000):
        tot = 0.0
        for _ in range(n):
            cls, bias, jitter = test_p[rng.randrange(len(test_p))]
            tot += fn(rng, jitter, bias)
        return tot / n

    r_policy = avg(lambda rg, j, b: episode(greedy, rg, j, b))
    r_static = avg(lambda rg, j, b: Duel(rg, j, b).play(2))   # fixed lead 2
    r_never = avg(lambda rg, j, b: Duel(rg, j, b).play(0))
    r_oracle = avg(oracle_reward)
    print(f"held-out mean reward: policy {r_policy:+.3f}  "
          f"static-lead-2 {r_static:+.3f}  never-adapt {r_never:+.3f}  "
          f"oracle {r_oracle:+.3f}")
    print(f"regret vs oracle: {r_oracle - r_policy:.3f} "
          f"(static's regret {r_oracle - r_static:.3f})")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"actions": list(ACTIONS),
               "q": {"|".join(map(str, k)): [round(v, 4) for v in row]
                     for k, row in sorted(Q.items())},
               "heldout": {"policy": r_policy, "static2": r_static,
                           "never": r_never, "oracle": r_oracle}},
              open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"-> {OUT} ({len(Q)} states)")


if __name__ == "__main__":
    main()
