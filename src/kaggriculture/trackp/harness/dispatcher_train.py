"""Train the learned tape-selector (2026-09-07).

Reads dispatcher/train_144.jsonl (features@day6 -> best continuation, from
REAL adaptive outcomes) and fits a SHALLOW decision tree (robust, matches
the 93.8%-router meta). Exports a pure-python nested-threshold tree
(inlinable, math-only) plus the candidate continuation names, so the agent
selects with zero train/serve skew and no sklearn dependency at runtime.

Cross-validated accuracy is reported; a tree that can't beat "always pick
the majority candidate" is rejected (no signal).

    C:/ProgramData/anaconda3/envs/llm/python.exe \
        src/trackp/harness/dispatcher_train.py --depth 4
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import json
import os
import sys
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                              # noqa: BLE001
    pass


def export_tree(clf):
    """sklearn tree -> nested list [feature, thr, left, right] | ['leaf', cls]."""
    t = clf.tree_

    def rec(node):
        if t.children_left[node] == t.children_right[node]:  # leaf
            cls = int(clf.classes_[int(t.value[node][0].argmax())])
            return ["leaf", cls]
        return [int(t.feature[node]), float(t.threshold[node]),
                rec(int(t.children_left[node])),
                rec(int(t.children_right[node]))]
    return rec(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=os.path.join(
        ROOT, "models", "trackp", "dispatcher", "train_144.jsonl"))
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(
        ROOT, "models", "trackp", "dispatcher", "tree_144.json"))
    a = ap.parse_args()
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score
    X, Y, banks = [], [], []
    for ln in open(a.data, encoding="utf-8"):
        r = json.loads(ln)
        X.append(r["f"])
        Y.append(int(r["y"]))
        banks.append(r.get("banks"))
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.int64)
    print(f"{len(X)} rows, {X.shape[1]} features, "
          f"{len(set(Y.tolist()))} classes used")
    maj = Counter(Y.tolist()).most_common(1)[0]
    print(f"majority class {maj[0]} = {maj[1]/len(Y):.3f} baseline")
    clf = DecisionTreeClassifier(max_depth=a.depth, min_samples_leaf=4,
                                 random_state=7)
    # cross-val (guard against a tree with no real signal)
    k = min(5, len(X) // max(2, len(set(Y.tolist()))))
    if k >= 2:
        cv = cross_val_score(clf, X, Y, cv=k)
        print(f"{k}-fold CV accuracy {cv.mean():.3f} +/- {cv.std():.3f}")
        cv_mean = cv.mean()
    else:
        cv_mean = 0.0
        print("too few rows for CV")
    clf.fit(X, Y)
    # regret-aware quality: how much bank does the tree's pick leave vs oracle?
    if banks and all(b for b in banks):
        pred = clf.predict(X)
        oracle = sum(max(b) for b in banks) / len(banks)
        got = sum(b[int(p)] for b, p in zip(banks, pred)) / len(banks)
        base0 = sum(b[0] for b in banks) / len(banks)  # always-base
        print(f"mean margin: always-base {base0:,.0f} | tree {got:,.0f} | "
              f"oracle {oracle:,.0f}")
    tree = export_tree(clf)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"tree": tree, "depth": a.depth, "cv_acc": cv_mean,
               "n": len(X)}, open(a.out, "w"))
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
