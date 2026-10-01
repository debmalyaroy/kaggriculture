"""Estimate the bandit's decision-theoretic commit gate from field data.

Reads the accumulated commit-audit rows and paired-validation records and
emits `models/v22/identifier/gates.json`:

    { "match_prob": p*,          # posterior needed to commit
      "e_gain_correct": ...,     # mean margin delta when a commit was right
      "e_cost_false": ...,       # mean margin cost of a wrong commit
      "n_commits": N }

The optimal threshold solves  p * E[gain] > (1-p) * E[cost]  ->
p* = E[cost] / (E[gain] + E[cost]).  Until `--min-commits` real commits
exist the tool refuses to move the gate and keeps the conservative default,
printing exactly what data is still missing -- built now, activates itself
as the audit fills.

    python -m kaggriculture.train.train_gates
"""
from kaggriculture.paths import ROOT
import argparse
import json
import os
import sys


AUDIT = os.path.join(ROOT, "models", "v22", "commit_audit_rows.jsonl")
OUT = os.path.join(ROOT, "models", "v22", "identifier", "gates.json")
DEFAULT = {"match_prob": 0.85, "e_gain_correct": None, "e_cost_false": None,
           "n_commits": 0, "status": "default (insufficient data)"}


MIN_JUDGED = 25


def arms_allowed(gates=None, path=None):
    """May arm commits ship? The ONE predicate every consumer must use.

    The original condition was `n_commits >= 25` alone -- evidence VOLUME with
    no regard for what the evidence said. That is perverse: commit_judge.py
    exists to prove arms harmful, and by recording 27 judgements it mechanically
    unlocked the feature it disproved. A rebuild on 2026-08-13 duly shipped
    arms ON and cost -2,971/game on the field-matched panel.

    Arms ship only when there are enough judged commits AND the audit says a
    commit is worth making. On the current record it is not: mean margin_delta
    is -50,787 over 27 commits and even the CORRECT ones average -52,940.

    Returns (allowed: bool, reason: str).
    """
    if gates is None:
        try:
            gates = json.load(open(path or OUT, encoding="utf-8"))
        except (OSError, ValueError):
            return False, "gates.json unreadable -- arms stay off"
    n = int(gates.get("n_commits") or 0)
    if n < MIN_JUDGED:
        return False, f"{n} judged commit(s) (need >={MIN_JUDGED})"
    if "arms_positive_value" in gates:
        if not gates.get("arms_positive_value"):
            return False, (
                f"audit says a commit is NEGATIVE-VALUE: "
                f"E[gain|correct]=${gates.get('e_gain_correct_raw', 0):,.0f} "
                f"vs E[cost|wrong]=${gates.get('e_cost_false_raw', 0):,.0f}")
        return True, f"{n} judged commits, audit favourable"
    # Older gates.json files carry only the clamped values.
    e_gain = float(gates.get("e_gain_correct") or 0.0)
    e_cost = float(gates.get("e_cost_false") or 0.0)
    if e_gain <= e_cost:
        return False, (f"audit says a commit is NEGATIVE-VALUE: "
                       f"E[gain|correct]=${e_gain:,.0f} <= "
                       f"E[cost|wrong]=${e_cost:,.0f}")
    return True, f"{n} judged commits, audit favourable"


POLICY_AUDIT = os.path.join(ROOT, "models", "lab", "policy_judgments.jsonl")
MIN_POLICY_JUDGED = 25


def policy_allowed(path=None):
    """May the learned blend policy ship? Same shape as arms_allowed().

    A learned policy is a live intervention exactly like an arm, so it gets
    the same discipline BEFORE it exists in any agent: blocked until
    counterfactual judging shows E[score with policy] > E[score without], in
    score units, over enough judged games. The predicate tests the evidence's
    SIGN, never its volume alone -- the ARM GUARD's original volume-only
    condition mechanically unlocked the feature its own audit disproved, and
    that bug class must not be recreated here.

    Evidence rows live in models/lab/policy_judgments.jsonl, one JSON object
    per judged game with at least {"score_delta": <policy minus baseline>}.
    No file, no rows, or negative value all mean NO.

    Returns (allowed: bool, reason: str).
    """
    recs = []
    try:
        with open(path or POLICY_AUDIT, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("score_delta") is not None:
                    recs.append(r)
    except OSError:
        return False, "no policy judgments on record -- policy stays off"
    # Newest-generation scoping (see _newest_generation): the gate prices the
    # policy that would ship TODAY, not every dead predecessor.
    gen, scope = _newest_generation(recs)
    if len(gen) >= MIN_POLICY_JUDGED:
        recs = gen
    else:
        scope = (f"pooled ({scope} has only {len(gen)} row(s), "
                 f"need >={MIN_POLICY_JUDGED})")
    rows_ = [float(r["score_delta"]) for r in recs]
    if len(rows_) < MIN_POLICY_JUDGED:
        return False, (f"{len(rows_)} judged game(s) "
                       f"(need >={MIN_POLICY_JUDGED}) [{scope}]")
    mean = sum(rows_) / len(rows_)
    if mean <= 0:
        return False, (f"audit says the policy is NEGATIVE-VALUE: mean "
                       f"score_delta {mean:+.4f} over {len(rows_)} games "
                       f"[{scope}]")
    return True, (f"{len(rows_)} judged games, mean score_delta "
                  f"{mean:+.4f} [{scope}]")


def _newest_generation(rows_, tag_key="judged_pair"):
    """Scope evidence to the NEWEST judged generation (2026-08-14 decision).

    The daily rehab job retrains arms / the policy and re-judges them. If the
    gate pooled every historical row, the 29 dead v23.1 commits (or the 52
    negative per-turn-policy games) would drown any rehabilitated generation
    forever -- the gate would be pricing agents that no longer exist. The
    gate's job is to price the intervention that WOULD ship today, so it
    judges the newest generation alone, PROVIDED that generation has enough
    rows on its own; otherwise it stays on the pooled (conservative) view.
    Old rows keep their value as history, never as a veto on new evidence.
    """
    tags = [r.get(tag_key) for r in rows_ if r.get(tag_key)]
    if not tags:
        return rows_, "pooled (no generation tags)"
    newest = tags[-1]
    gen = [r for r in rows_ if r.get(tag_key) == newest]
    return gen, f"generation '{newest}'"


RECALL_FILE = os.path.join(ROOT, "models", "lab", "preranker_recall.json")
MIN_RECALL = 0.75
MIN_RECALL_CANDS = 6


def preranker_allowed(path=None):
    """May the Rust pre-ranker widen the crown funnel? Sign-tested like the
    other guards: a fast pre-ranker that ranks badly makes the funnel wider
    AND worse, so speed is never the criterion -- measured recall@N against
    the official engine's closed-loop ranking is (src/rust_prerank.py
    --recall writes the evidence). No measurement means NO.

    Returns (allowed: bool, reason: str).
    """
    try:
        rec = json.load(open(path or RECALL_FILE, encoding="utf-8"))
    except (OSError, ValueError):
        return False, "no recall measurement on record -- pre-ranker gated"
    r = float(rec.get("recall_at_n") or 0.0)
    n_c = int(rec.get("candidates") or 0)
    if n_c < MIN_RECALL_CANDS:
        return False, (f"recall measured over only {n_c} candidate(s) "
                       f"(need >={MIN_RECALL_CANDS})")
    if r < MIN_RECALL:
        return False, (f"recall@{rec.get('n')} = {r:.2f} over {n_c} "
                       f"candidates (need >={MIN_RECALL}) -- the open-loop "
                       f"approximation is losing the official winners")
    return True, f"recall@{rec.get('n')} = {r:.2f} over {n_c} candidates"


def rows():
    if not os.path.exists(AUDIT):
        return []
    out = []
    with open(AUDIT, encoding="utf-8") as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except Exception:                                      # noqa: BLE001
                continue
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--min-commits", type=int, default=25,
                    help="real commits needed before the gate moves")
    args = ap.parse_args()

    data = rows()
    commits = [r for r in data if r.get("arm")]
    # Newest-generation scoping (2026-08-14, see _newest_generation): the
    # daily rehab job re-judges freshly retrained arms under a new
    # judged_pair tag. When that generation carries enough rows on its own,
    # the gate prices IT; otherwise the pooled view stands.
    gen, gen_scope = _newest_generation(commits)
    if len(gen) >= args.min_commits:
        commits = gen
        print(f"gate scoped to {gen_scope} ({len(commits)} commits)")
    else:
        print(f"gate pooled: {gen_scope} has {len(gen)} commit(s), "
              f"need >={args.min_commits}")
    correct = [r for r in commits if r.get("commit_correct")]
    wrong = [r for r in commits if r.get("commit_correct") is False]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    if len(commits) < args.min_commits or not correct or not wrong:
        need = args.min_commits - len(commits)
        DEFAULT["n_commits"] = len(commits)
        DEFAULT["status"] = (f"default: {len(commits)} commits on record "
                             f"({len(correct)} judged correct, {len(wrong)} "
                             f"wrong); need >={args.min_commits} with both "
                             f"outcomes represented ({max(0, need)} more)")
        json.dump(DEFAULT, open(OUT, "w", encoding="utf-8"), indent=1)
        print(DEFAULT["status"])
        return 0

    # PREFER WIN UNITS. The ladder pays win/draw/loss, so the value of a
    # commit is whether it changed the RESULT, not by how many dollars.
    # commit_judge records `score_delta` (score with the arm minus score
    # without) since 2026-08-13; when present it is the criterion and the
    # dollar figures are diagnostics. Older rows have only margin_delta, so
    # the dollar path stays as a documented fallback.
    scored = [r for r in commits if r.get("score_delta") is not None]
    sc_gain = [float(r["score_delta"]) for r in scored
               if r.get("commit_correct")]
    sc_cost = [-float(r["score_delta"]) for r in scored
               if r.get("commit_correct") is False]
    # BOTH OUTCOMES MUST BE REPRESENTED IN THE SCORED SUBSET, not just in the
    # full row set. Caught 2026-08-13: the first two score_delta rows were both
    # WRONG commits (each turned a win into a loss, score_delta -1.0), so
    # sc_gain was empty, e_gain_raw fell to 0, and the max(1.0, ...) clamps
    # below turned "no measured upside" into "upside 1" -- driving p_star to
    # 1/(1+1) = 0.5, the most PERMISSIVE threshold available, on evidence that
    # says committing is catastrophic. A missing-upside estimate must fail
    # conservative, never permissive.
    if scored and sc_gain and sc_cost:
        e_gain_raw = sum(sc_gain) / len(sc_gain)
        e_cost_raw = sum(sc_cost) / len(sc_cost)
        units = "score"
        print(f"gate in WIN UNITS from {len(scored)}/{len(commits)} rows "
              f"carrying score_delta")
    elif scored:
        print(f"WARNING: {len(scored)} row(s) carry score_delta but only "
              f"{'wrong' if sc_cost else 'correct'} outcomes "
              f"({len(sc_gain)} correct / {len(sc_cost)} wrong) -- a ratio "
              f"cannot be estimated from one side. Falling back to the margin "
              f"path and clamping the threshold to its most conservative "
              f"value; arms stay blocked by arms_allowed() regardless.")
        e_gain_raw = (sum(float(r.get("margin_delta") or 0) for r in correct)
                      / len(correct))
        e_cost_raw = (-sum(float(r.get("margin_delta") or 0) for r in wrong)
                      / len(wrong))
        units = "margin$ (score evidence one-sided)"
    else:
        e_gain_raw = (sum(float(r.get("margin_delta") or 0) for r in correct)
                      / len(correct))
        e_cost_raw = (-sum(float(r.get("margin_delta") or 0) for r in wrong)
                      / len(wrong))
        units = "margin$"
        print("WARNING: no row carries score_delta, so this gate is priced in "
              "MARGIN DOLLARS -- a currency the ladder does not pay. Re-run "
              "src/kaggriculture/experiments/commit_judge.py to record outcomes.")
    # The clamps below keep p_star well-defined, but they also HIDE the most
    # important case: on the 27 real judged commits e_gain_raw is -52,940 --
    # a correct commit still loses money. No threshold can make committing
    # profitable then, so the raw values and an explicit verdict are recorded
    # for consumers instead of being clamped away. `arms_allowed()` below is
    # the single predicate every consumer must use.
    e_gain = max(1.0, e_gain_raw)
    e_cost = max(1.0, e_cost_raw)
    # NO MEASURED UPSIDE => MOST CONSERVATIVE THRESHOLD. With e_gain_raw <= 0
    # the Bayes ratio is not merely uncertain, it is undefined: there is no
    # evidence that committing ever pays. Computing p_star from the clamped
    # placeholders instead yields 1/(1+1) = 0.5, i.e. the LOOSEST gate, which
    # is the opposite of what the evidence supports. Fail conservative.
    if e_gain_raw <= 0:
        p_star = 0.95
    else:
        p_star = e_cost / (e_gain + e_cost)
    p_star = min(0.95, max(0.55, p_star))
    out = {"match_prob": round(p_star, 3),
           "e_gain_correct": round(e_gain, 1),
           "e_cost_false": round(e_cost, 1),
           "e_gain_correct_raw": round(e_gain_raw, 4),
           "e_cost_false_raw": round(e_cost_raw, 4),
           "gate_units": units,
           "arms_positive_value": bool(e_gain_raw > 0
                                       and e_gain_raw > e_cost_raw),
           "n_commits": len(commits), "status": "estimated from field data"}

    # Per-family thresholds with empirical-Bayes shrinkage (2026-08-13,
    # from src/experiments/gates_v2.py; self-tested there): small-n families
    # hug the pooled threshold, well-evidenced costly families gate higher.
    # Backward-compatible extra key -- consumers that only read match_prob
    # are unaffected; the agent generator may adopt it when arms return.
    try:
        from gates_v2 import eb_thresholds
        fam = eb_thresholds(commits, default_p=p_star)
        fam.pop("_pooled", None)
        if fam:
            out["family_thresholds"] = fam
    except Exception as exc:                                       # noqa: BLE001
        print(f"per-family thresholds skipped: {exc}")

    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"gate estimated: commit at posterior >= {p_star:.3f} "
          f"(E[gain]={e_gain:,.0f}, E[cost]={e_cost:,.0f}, "
          f"n={len(commits)}, {len(out.get('family_thresholds', {}))} "
          f"per-family)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
