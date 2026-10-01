"""Identifier bake-off: current logistic vs scaled L-BFGS logistic vs GBDTs.

Read-only against the route index; touches NOTHING the release pipeline runs.
Walk-forward evaluation: for each of the last K distinct days d in the index,
train on rows with date < d, test on rows with date == d. Metrics per model:
accuracy, macro-F1, log-loss. Temperature scaling is fit per fold on a slice
of the train tail and applied to the test fold (proper calibration protocol).

    python src/experiments/identifier_lab.py --folds 3 --out models/lab/identifier_bakeoff.json
"""
from kaggriculture.paths import ROOT
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = ROOT

import kaggriculture.data.features as features  # noqa: E402
import kaggriculture.data.routes as R  # noqa: E402
import kaggriculture.train.train_identifier as TI  # noqa: E402


def build_rows():
    """Same dataset construction as train_identifier.build_dataset, but keeps
    the DATE of every row so walk-forward folds can be cut on real days."""
    import random
    idx = R.load_index()
    rows = sorted(idx["routes"].values(),
                  key=lambda r: (r.get("date", ""), r["id"]), reverse=True)
    feats, recs = {}, []
    for rec in rows:
        try:
            feats[rec["id"]], _ = features.route_features(R.load_route(rec["id"]))
            recs.append(rec)
        except Exception:                                          # noqa: BLE001
            continue
    reps, labels = [], {}
    for rec in recs:
        f = feats[rec["id"]]
        for fi, rep in enumerate(reps):
            if features.distance(f, rep) < TI.LINK:
                labels[rec["id"]] = fi
                break
        else:
            reps.append(f)
            labels[rec["id"]] = len(reps) - 1
    sizes = {}
    for v in labels.values():
        sizes[v] = sizes.get(v, 0) + 1
    keep = {fi for fi, n in sizes.items() if n >= TI.MIN_FAMILY}
    remap = {fi: i for i, fi in enumerate(sorted(keep))}
    novel = len(remap)
    rng = random.Random(11)
    X, y, dates = [], [], []
    for rec in recs:
        cls = remap.get(labels[rec["id"]], novel)
        acts = R.load_route(rec["id"])
        for t in TI.PREFIX_TURNS:
            base_vec = features.prefix_features(acts, t)
            for p in TI.DROPOUT:
                vec = [v * (0 if rng.random() < p else 1) for v in base_vec]
                X.append(vec + [t / 720.0])
                y.append(cls)
                dates.append(rec.get("date", ""))
    return (np.array(X, dtype=np.float64), np.array(y),
            np.array(dates), novel + 1)


def metrics(y_true, proba, k):
    from sklearn.metrics import f1_score, log_loss
    pred = proba.argmax(axis=1)
    return {
        "acc": float((pred == y_true).mean()),
        "macro_f1": float(f1_score(y_true, pred, average="macro",
                                   zero_division=0)),
        "log_loss": float(log_loss(y_true, proba, labels=list(range(k)))),
    }


def fit_temperature(proba, y, iters=60):
    """1-D search for the temperature minimizing log-loss on a val slice."""
    from sklearn.metrics import log_loss
    logit = np.log(np.clip(proba, 1e-12, None))
    k = proba.shape[1]
    best_t, best_ll = 1.0, log_loss(y, proba, labels=list(range(k)))
    for t in np.geomspace(0.25, 4.0, iters):
        z = logit / t
        p = np.exp(z - z.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        ll = log_loss(y, p, labels=list(range(k)))
        if ll < best_ll:
            best_t, best_ll = float(t), float(ll)
    return best_t


def apply_temperature(proba, t):
    z = np.log(np.clip(proba, 1e-12, None)) / t
    p = np.exp(z - z.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def current_logistic(Xtr, ytr, Xte, k):
    W, b = TI.train(Xtr, ytr, k)
    z = Xte @ W + b
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def scaled_lbfgs(Xtr, ytr, Xte, k):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0,
                             class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte)), clf.classes_


def lgbm(Xtr, ytr, Xte, k):
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=k,
                             n_estimators=400, learning_rate=0.08,
                             max_depth=6, num_leaves=48,
                             subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=1.0, n_jobs=4, verbosity=-1)
    cut = max(1, int(0.9 * len(Xtr)))
    clf.fit(Xtr[:cut], ytr[:cut],
            eval_set=[(Xtr[cut:], ytr[cut:])],
            callbacks=[lgb.early_stopping(40, verbose=False)])
    return clf.predict_proba(Xte), clf.classes_


def xgb(Xtr, ytr, Xte, k):
    from xgboost import XGBClassifier
    present = np.unique(ytr)                   # xgb wants contiguous labels
    lut = {c: i for i, c in enumerate(present)}
    ytr_c = np.array([lut[v] for v in ytr])
    clf = XGBClassifier(objective="multi:softprob",
                        num_class=len(present),
                        n_estimators=400, learning_rate=0.08, max_depth=6,
                        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                        n_jobs=4, early_stopping_rounds=40, verbosity=0)
    cut = max(1, int(0.9 * len(Xtr)))
    clf.fit(Xtr[:cut], ytr_c[:cut], eval_set=[(Xtr[cut:], ytr_c[cut:])],
            verbose=False)
    return clf.predict_proba(Xte), present


def catb(Xtr, ytr, Xte, k):
    from catboost import CatBoostClassifier
    clf = CatBoostClassifier(loss_function="MultiClass", iterations=400,
                             learning_rate=0.08, depth=6, l2_leaf_reg=3.0,
                             verbose=False, thread_count=4)
    cut = max(1, int(0.9 * len(Xtr)))
    clf.fit(Xtr[:cut], ytr[:cut], eval_set=(Xtr[cut:], ytr[cut:]),
            early_stopping_rounds=40)
    return clf.predict_proba(Xte), clf.classes_


def expand(proba, classes, k):
    """Some folds miss classes in train; expand proba to full k columns."""
    if proba.shape[1] == k:
        return proba
    full = np.full((len(proba), k), 1e-12)
    for j, c in enumerate(classes):
        full[:, int(c)] = proba[:, j]
    return full / full.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(
        ROOT, "models", "lab", "identifier_bakeoff.json"))
    args = ap.parse_args()

    X, y, dates, k = build_rows()
    days = sorted(set(d for d in dates if d))
    test_days = days[-args.folds:]
    print(f"{len(X)} rows, {k} classes, {len(days)} days; "
          f"testing on {test_days}")

    results = {}
    for day in test_days:
        te = dates == day
        tr = dates < day
        if tr.sum() < 1000 or te.sum() < 100:
            print(f"skip {day}: train {tr.sum()} test {te.sum()}")
            continue
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        # calibration slice = last 10% of train (most recent)
        ncal = max(200, int(0.1 * len(Xtr)))
        Xcal, ycal = Xtr[:ncal], ytr[:ncal]   # rows sorted newest-first
        fold = {}
        for name, fn in [("current_logistic", None),
                         ("scaled_lbfgs", scaled_lbfgs),
                         ("lightgbm", lgbm), ("xgboost", xgb),
                         ("catboost", catb)]:
            t0 = time.time()
            try:
                if name == "current_logistic":
                    proba = current_logistic(Xtr, ytr, Xte, k)
                else:
                    proba, classes = fn(Xtr, ytr, Xte, k)
                    proba = expand(np.asarray(proba), classes, k)
                m = metrics(yte, proba, k)
                # temperature fit on the calibration slice
                if name == "current_logistic":
                    pcal = current_logistic(Xtr, ytr, Xcal, k)
                else:
                    pcal, classes = fn(Xtr, ytr, Xcal, k)
                    pcal = expand(np.asarray(pcal), classes, k)
                t = fit_temperature(pcal, ycal)
                m["log_loss_calibrated"] = metrics(
                    yte, apply_temperature(proba, t), k)["log_loss"]
                m["temperature"] = t
                m["train_s"] = round(time.time() - t0, 1)
                fold[name] = m
                print(f"  {day} {name:18s} acc {m['acc']:.3f} "
                      f"macroF1 {m['macro_f1']:.3f} ll {m['log_loss']:.3f} "
                      f"ll_cal {m['log_loss_calibrated']:.3f} "
                      f"T {t:.2f} ({m['train_s']}s)", flush=True)
            except Exception as exc:                               # noqa: BLE001
                fold[name] = {"error": str(exc)[:200]}
                print(f"  {day} {name}: ERROR {str(exc)[:120]}", flush=True)
        results[day] = fold

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"rows": len(X), "classes": k, "results": results},
              open(args.out, "w", encoding="utf-8"), indent=1)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
