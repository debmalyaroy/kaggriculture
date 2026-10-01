"""Assemble v22: base route + trained counter-arms + in-game bandit.

Stage 3 of the 2026-08-09 build. The shipped file carries:

* the base route (field channel, purchases, default sell schedule);
* one trained market-channel override per opponent family ("arm"), all
  sharing the base's first 192 turns so switching is never a splice;
* the opponent-program identifier: sales reconstructed exactly from the
  shared market inventory (town consumption transcribed from the
  interpreter), matched against a library of distinct top-100 programs;
* a Thompson-style bandit over the arms: a posterior over program families
  updated every turn from the match distances, committing to an arm when
  decisive for 2 consecutive reads, reverting to the base if the opponent
  later diverges from the matched program;
* the per-turn sell optimizer (shed clamp, sell-first, price-impact ranking,
  threat-first against the matched program's forecast sells);
* a bounded feed-buy pull: scheduled wheat purchases advance when wheat
  quotes below base, debited against the schedule so totals never change.

    python -m kaggriculture.agentbuild.v22_agent --out agents/v22_bandit.py
"""
from kaggriculture.paths import ROOT
import argparse
import base64
import datetime as dt
import json
import os
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.data.routes as R  # noqa: E402
import kaggriculture.agentbuild.counter_agent as CA  # noqa: E402

ARM_DIR = os.path.join(ROOT, "models", "v22", "arms")
PREFIX_END = 192
# The step the runtime dispatches on: day 3 dawn, the first town shop unlock
# (kaggriculture.py `_end_of_day`, next_day % townShopUnlockInterval == 0).
# Everything before it is played from the base, so every branch must carry
# the base's own turns there.
BRANCH_STEP = 72
# The two runtime triggers, and the earliest step at which each observable
# actually EXISTS (a route may only diverge after the evidence that selects
# it is public -- the causality rule):
#   shop   step 72,  the day-3 town shop unlock; key = the first shop drawn.
#   drain  step 145, the day-6 dawn demand tick; key = SHEEP / COW / GOOSE,
#          from which animal product the town's market inventory is losing
#          more than the baseline 1 unit a turn.
BRANCH_TRIGGERS = {"shop": 72, "drain": 145}
BRANCH_KEYS = {
    "shop": {"BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP",
             "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE"},
    "drain": {"SHEEP", "COW", "GOOSE"},
}

# Which identified program family plays against which trained arm.
# THUNDER/lemon deliberately have NO arm: half the ladder's programs look
# TT-ish mid-game, and validation showed the tt arm loses more against the
# lookalikes (-$1.4k..-2.1k) than it gains against the real thing (+$200).
# Identification of those opponents still feeds threat-first ordering.
TEAM2ARM = {
    "tao wu11": "tao",
    "Raj Aryan": "raj",
    "Seb (allegedly)": "seb", "David Schindler15": "seb",
    # 30-series arms (2026-08-30), trained on the fresh elite panel: each
    # binds the teams whose tapes were its TARGETS. ring30 also guards
    # Borrun (same family); band30 targets Schott + Xiao.
    "A Poor Vul": "apv30",
    "ringbearer": "ring30", "Borrun": "ring30",
    "Stephen Schott": "band30", "Xiaowenhao404": "band30",
}

# v22.1 (base 91468035_s1): arms retrained on the fresh base. The frontier
# mapping re-tests the lookalike question the tt arm failed -- kept only if
# the NOVEL referees stay clean in validation.
TEAM2ARM_V221 = {
    "THUNDER THUNDER": "frontier", "lemon13418": "frontier",
    "Seb (allegedly)": "seb2", "David Schindler15": "seb2",
}


def validate_branchpack(base, bp, trigger="shop"):
    """BRANCH GUARD (2026-09-04). Refuse a branchpack the runtime cannot play.

    The runtime swaps the WHOLE route for the branch at the first shop
    unlock, so a malformed branch is not a slightly degraded schedule -- it
    is 648 turns of silent no-ops, because illegal ops never raise. What is
    checkable is checked here, where a build can still fail loudly:

    * same length as the base (720). The runtime clamps its index at
      ``len(route) - 1``, so a short tape REPEATS its final turn for the
      rest of the game instead of passing.
    * <= 10 market orders a turn: the engine drops the extras silently.
    * the pre-dispatch prefix must equal the base's. Turns 0..71 are played
      from the BASE, so a branch is a suffix grafted onto the board the base
      built; a branch carrying its own prefix was compiled for a board this
      agent never had.

    Returns the report lines. Raises SystemExit on any violation.

    It also reports how many DISTINCT routes the pack really carries.
    ``models/trackp/branchpack_kaileh*.json`` advertised "8 branches" and
    held 3 distinct routes, 6 of the 8 keys pointing at a route
    byte-identical to the base -- so the shipped v37 "branch agent" could
    only branch in 2 of 8 worlds, and the 0.800-vs-0.767 panel result that
    justified the mechanism cannot have been measuring branching.
    """
    import hashlib

    def sha(obj):
        return hashlib.sha1(json.dumps(obj, separators=(",", ":"),
                                       sort_keys=True).encode()).hexdigest()

    if not isinstance(bp, dict) or not bp:
        raise SystemExit("BRANCH GUARD: --branchpack must be a non-empty "
                         "{key: tape} object")
    step = BRANCH_TRIGGERS[trigger]
    legal = BRANCH_KEYS[trigger]
    stray = sorted(set(bp) - legal)
    if stray:
        raise SystemExit(
            f"BRANCH GUARD: {trigger!r} dispatch can never produce the "
            f"key(s) {stray} -- those branches would be dead weight in the "
            f"submission. Legal keys: {sorted(legal)}")
    base_h = sha(base)
    seen = {}
    for shop, tape in sorted(bp.items()):
        if not isinstance(tape, list) or len(tape) != len(base):
            raise SystemExit(
                f"BRANCH GUARD: branch {shop!r} has "
                f"{len(tape) if isinstance(tape, list) else '?'} turns, base "
                f"has {len(base)} -- refusing")
        for i, turn in enumerate(tape):
            if not isinstance(turn, dict):
                raise SystemExit(f"BRANCH GUARD: branch {shop!r} turn {i} is "
                                 f"{type(turn).__name__}, not a dict")
            n_ord = len(turn.get("market") or [])
            if n_ord > 10:
                raise SystemExit(
                    f"BRANCH GUARD: branch {shop!r} turn {i} queues {n_ord} "
                    f"market orders; the engine drops everything past 10 "
                    f"without warning -- refusing")
        if tape[:step] != base[:step]:
            raise SystemExit(
                f"BRANCH GUARD: branch {shop!r} does not share the base's "
                f"first {step} turns. Those turns are played from the "
                f"BASE, so a branch must be its suffix stitched onto the base "
                f"prefix, not a tape carrying its own opening")
        seen.setdefault(sha(tape), []).append(shop)
    same = seen.get(base_h, [])
    if len(seen) == 1 and same:
        raise SystemExit("BRANCH GUARD: every branch is the base -- the pack "
                         "would change nothing; refusing to ship a dispatch "
                         "that cannot dispatch")
    return [f"branchpack ({trigger} @ step {step}): {len(bp)} keys -> "
            f"{len(seen)} distinct routes; "
            f"{len(same)} key(s) identical to the base "
            f"({', '.join(same) or 'none'})"]


def market_overrides(base, arm_route):
    """The turns (>= PREFIX_END) where the arm's market differs from base.

    An arm may only RETIME SALES. Structural orders (BUY_LAND, HIRE,
    BUY_ANIMAL, BUY_SEED, BUY_PRODUCT) always come from the base: the unit
    route (farmer/hands) is the base's farm plan, and a market override
    that drops its land purchase or feed buys wrecks the economy it is
    grafted onto. Measured 2026-08-11 (v23.1 postmortem): the raw-replace
    override dropped the t=240 BUY_LAND and halved the bank in 14 of 19
    ladder losses. Overrides here are therefore arm SELLs + base
    non-SELLs; if that would exceed the 10-order cap, base structural
    orders win and the lowest-value arm SELLs are dropped.
    """
    max_orders = 10
    diff = {}
    for t in range(PREFIX_END, min(len(base), len(arm_route))):
        a = arm_route[t].get("market") or []
        b = base[t].get("market") or []
        if a == b:
            continue
        arm_sells = [o for o in a
                     if isinstance(o, list) and o and o[0] == "SELL"]
        base_rest = [o for o in b
                     if not (isinstance(o, list) and o and o[0] == "SELL")]
        keep = max(0, max_orders - len(base_rest))
        merged = arm_sells[:keep] + base_rest
        if merged != b:
            diff[str(t)] = merged
    return diff


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="90525850_s0")
    ap.add_argument("--out", default=os.path.join("agents", "v22_bandit.py"))
    ap.add_argument("--arms", nargs="*", default=["tt", "tao", "raj", "seb"])
    ap.add_argument("--duel-policy", action="store_true",
                    help="embed the learned duel policy (models/lab/"
                         "duel_q.json) instead of the static/escalated relay "
                         "lead. PANEL-GATED: never ship a --duel-policy build "
                         "that has not beaten the static build on a paired "
                         "field-matched panel.")
    ap.add_argument("--gru", action="store_true",
                    help="embed the GRU identifier (models/lab/"
                         "gru_weights.json) instead of the logistic for "
                         "in-game matching. PANEL-GATED: never ship a --gru "
                         "build that has not beaten the logistic build on "
                         "the paired-seed panel (Tier 2).")
    ap.add_argument("--gru-auto", action="store_true",
                    help="embed the GRU identifier ONLY if its class count "
                         "matches today's identifier AND its recorded "
                         "held-out accuracy beats the logistic's. Never "
                         "raises: a stale or losing GRU is skipped with a "
                         "log line. This is how the daily pipeline carries "
                         "the GRU the moment E5 measurement favours it.")
    ap.add_argument("--policy", action="store_true",
                    help="embed the deep-RL sell head (models/lab/"
                         "policy_bc.json): return-conditioned BC, bounded "
                         "per-turn sell delta. GUARDED: policy_allowed() "
                         "must be favourable, or --lab-force-policy.")
    ap.add_argument("--lab-force-policy", action="store_true",
                    help="LAB ONLY: embed the policy head even while the "
                         "POLICY GUARD says no, for counterfactual judging. "
                         "Stamps LAB BUILD -- NEVER SUBMIT.")
    ap.add_argument("--premium-lead", action="store_true",
                    help="one-turn conservation sell lead + deposit "
                         "advance on premium goods (margin-positive "
                         "8/8 cells in the tape shell; targets the "
                         "close-loss band). OFF by default")
    ap.add_argument("--no-pull", action="store_true",
                    help="build without the always-on sell-acceleration "
                         "layer (_PULL)")
    ap.add_argument("--branch-on", choices=sorted(BRANCH_TRIGGERS),
                    default="shop",
                    help="what the runtime dispatch reads: 'shop' = the "
                         "first town shop unlocked (step 72); 'drain' = the "
                         "day-6 market-inventory demand tick (step 145), "
                         "keys SHEEP/COW/GOOSE")
    ap.add_argument("--branchpack", default=None,
                    help="JSON of {{first_shop: route}} branch tapes; the "
                         "runtime swaps the base at the day-3 draw (A4)")
    ap.add_argument("--tomato-read", action="store_true",
                    help="day-16 TOMATO demand read (rank 2's third decision "
                         "point, 182/190 = 95.8%%): at step 385 read the "
                         "one-turn town TOMATO drain and, iff it is >= 3 "
                         "(two shops buying), re-crop up to --tomato-seeds "
                         "of the base's OWN later PLANT WHEAT ops to TOMATO. "
                         "Additive, not a tape swap. OFF by default and "
                         "independently gateable -- paired vs the same build "
                         "with the flag off, held-out cells, MEDIAN MARGIN "
                         "reported alongside cells.")
    ap.add_argument("--adaptive-market", action="store_true",
                    help="COMPETITOR PORT, OFF by default: boatlee V29-R1's "
                         "`_adaptive_market` liquidation controller (also "
                         "shipped verbatim by LB #2 lynnsakurai -- identical "
                         "_AM_CONFIG, all 20 constants). A NARROW extra-sell "
                         "layer: starts at step 456, touches only "
                         "STRAWBERRY/MILK/WOOL, hard-caps at 18 extra units "
                         "per item ALL GAME, gates on price/base with a "
                         "reserve ladder, and DISABLES ITSELF against a "
                         "near-mirror opponent (latched at step 289). It is "
                         "worth testing precisely because it is shaped "
                         "UNLIKE our own dead market layers (_TIMING was a "
                         "twice-measured regression, _PULL contributes "
                         "exactly nothing) -- both of ours are unconditional "
                         "always-on price reactions. Gate it ALONE: reactive "
                         "band, held-out seeds fixed after the flag set is "
                         "frozen, MEDIAN MARGIN alongside cells, mirror-"
                         "deduped n=worlds.")
    ap.add_argument("--regime-gear", action="store_true",
                    help="COMPETITOR PORT, OFF by default; IMPLIES "
                         "--adaptive-market. lynnsakurai's one genuinely new "
                         "layer on top of that controller: a "
                         "protect/convert/balanced gear re-evaluated only at "
                         "72-turn epoch boundaries from day 18 (steps "
                         "432/504/576/648), chosen from the public CASH GAP "
                         "to the opponent, the STRAWBERRY/MILK/WOOL price "
                         "index and the public capacity gap. Verified "
                         "non-degenerate on 3,000 of our own episodes (fires "
                         "in 9-31%% of worlds by checkpoint), so unlike most "
                         "candidates it can actually be sign-tested. Gate it "
                         "SEPARATELY, against --adaptive-market alone.")
    ap.add_argument("--tomato-seeds", type=int, default=17,
                    help="tiles the day-16 read re-crops (rank 2's median "
                         "dose is 17 seeds); only meaningful with "
                         "--tomato-read")
    ap.add_argument("--no-relay-m2", action="store_true",
                    help="LAB ONLY: build without per-class relay schedules, "
                         "for the layer-causality audit")
    ap.add_argument("--nn-sched", action="store_true",
                    help="embed the nearest-neighbour opponent SCHEDULE "
                         "library (models/relay/nn_lib.json, built by "
                         "src/nn_schedule.py) and let the relay race against "
                         "the RETRIEVED schedule instead of the collapsed "
                         "per-class consensus. Offline it predicts the step "
                         "of their next dump far better (WHEAT 0.287 -> "
                         "0.808 hit@+-6); that is a prediction result, not a "
                         "win result. OFF by default -- PANEL-GATED, and "
                         "gate it ALONE (band panels + paired sign test) "
                         "with the CLEAN ratio, never the raw one.")
    ap.add_argument("--nn-outrank", action="store_true",
                    help="LAYER 2 (implies --nn-sched): let the retrieved "
                         "schedule outrank the relay's ONLINE PHASE MODEL in "
                         "the race decision. Without this the schedule layer "
                         "is nearly dead code, because the phase model "
                         "shadows it from step 216 on -- and that phase model "
                         "is beaten by the unconditional median on most "
                         "products at day 13-16. Gate this SEPARATELY from "
                         "--nn-sched.")
    ap.add_argument("--lab-force-arms", action="store_true",
                    help="LAB ONLY: keep arm commits enabled even while the "
                         "ARM GUARD says no, so commit_judge can produce the "
                         "counterfactual evidence the guard needs (E4). The "
                         "emitted file is stamped LAB BUILD -- NEVER SUBMIT "
                         "and src/submit.py refuses it outright.")
    args = ap.parse_args()

    # A base may be a registered route id OR a raw tape JSON on disk (the
    # Track-P owned-economy harvest hands over a file, not a route id).
    if os.path.exists(args.base):
        base = json.load(open(args.base, encoding="utf-8"))
    else:
        base = R.load_route(args.base)
    # Validate the branchpack FIRST: it is the cheapest check in the build and
    # a bad branch is the most expensive mistake (see validate_branchpack).
    branch_routes = None
    if getattr(args, "branchpack", None):
        branch_routes = json.load(open(args.branchpack, encoding="utf-8"))
        for line in validate_branchpack(base, branch_routes,
                                        args.branch_on):
            print(line)
    arms = {}
    for name in args.arms:
        path = os.path.join(ARM_DIR, name, "best_route.json")
        if not os.path.exists(path):
            raise SystemExit(f"arm {name!r} not trained: {path} missing")
        arm_route = json.load(open(path, encoding="utf-8"))
        arms[name] = market_overrides(base, arm_route)
        print(f"arm {name}: {len(arms[name])} market turns differ from base")

    # Identifier model: trained weights + a class->arm mapping computed HERE
    # (offline, where team names may be used to find representatives); the
    # shipped runtime is entirely name-free.
    import kaggriculture.data.features as features
    idw = json.load(open(os.path.join(ROOT, "models", "v22", "identifier",
                                      "weights.json"), encoding="utf-8"))
    import kaggriculture.train.train_identifier as TI
    idx2 = R.load_index()
    # The class our own base belongs to may NEVER map to an arm: committing
    # against a mirror of ourselves is the measured mistake (v24 smoke,
    # -$683 vs our own twin).
    own_vec = features.prefix_features(base, 480) + [480 / 720.0]
    own_probs = TI.stdlib_predict(idw, own_vec)
    own_cls = max(range(len(own_probs)), key=lambda i: own_probs[i])
    print(f"own base class: {own_cls} ({own_probs[own_cls]:.2f}) -- excluded")
    class2arm = {}
    for team, arm in TEAM2ARM.items():
        recs = [r for r in idx2["routes"].values() if r.get("team") == team]
        if not recs or arm not in arms:
            continue
        rec = max(recs, key=lambda r: r.get("date", ""))
        vec = features.prefix_features(R.load_route(rec["id"]), 480) + [480 / 720.0]
        probs = TI.stdlib_predict(idw, vec)
        cls = max(range(len(probs)), key=lambda i: probs[i])
        if probs[cls] >= 0.5 and cls != len(probs) - 1 and cls != own_cls:
            class2arm[str(cls)] = arm
            safe = team.encode("ascii", "replace").decode()
            print(f"class {cls} <- {safe} ({probs[cls]:.2f}) -> arm {arm}")
    print(f"identifier: {len(idw['b'])} classes; {len(class2arm)} mapped to arms")

    def pack(obj):
        return base64.b85encode(zlib.compress(
            json.dumps(obj, separators=(",", ":")).encode("utf-8"), 9)).decode("ascii")

    # Tuned commit threshold from the gate estimator (field data). Falls back
    # to the conservative 0.85 until the commit audit has >=25 judged commits.
    match_prob = 0.85
    judged_commits = 0
    e_gain = e_cost = 0.0
    gp = os.path.join(ROOT, "models", "v22", "identifier", "gates.json")
    if os.path.exists(gp):
        try:
            g = json.load(open(gp, encoding="utf-8"))
            judged_commits = int(g.get("n_commits") or 0)
            e_gain = float(g.get("e_gain_correct") or 0.0)
            e_cost = float(g.get("e_cost_false") or 0.0)
            if g.get("match_prob"):
                match_prob = float(g["match_prob"])
                print(f"commit threshold from gates.json: {match_prob} "
                      f"({g.get('status', '')})")
        except Exception:                                          # noqa: BLE001
            pass
    # Arm commits stay OFF until the commit audit has judged real ladder
    # commits. v23/v23.1 (2026-08-11) shipped {'1': 'raj'} with 0 judged
    # commits: the identifier mapped ~half the field into class 1 and the
    # arm's market overlay wrecked the base economy (banks halved, BUY_LAND
    # dropped) -- 14 of 19 ladder losses, rating 2776 -> 1056. An arm is a
    # live intervention; it needs field evidence BEFORE it may fire.
    # Evidence VOLUME is not evidence SIGN. The original guard opened at 25
    # judged commits regardless of what those commits SAID, which is perverse:
    # commit_judge.py's whole finding was that arms destroy value, and by
    # recording 27 judgements it mechanically UNLOCKED the feature it had just
    # disproved. Caught 2026-08-13 when a rebuild silently shipped arms ON
    # (class 1 -> raj) and cost -2,971/game on the field-matched panel.
    # The audit says: mean margin_delta -50,787 over 27 commits, and even the
    # CORRECT commits average -52,940. So the gate now needs the evidence to
    # be favourable as well as present.
    import kaggriculture.train.train_gates as TG
    allowed, why = TG.arms_allowed(path=gp)
    if class2arm and not allowed:
        if getattr(args, "lab_force_arms", False):
            # E4 evidence path: the guard's own verdict can only change if
            # counterfactual judging of the RETRAINED arms exists, and that
            # judging needs a build that actually commits. This is a lab
            # instrument, not an exemption -- the file is stamped and
            # submit.py refuses it.
            print(f"LAB BUILD: ARM GUARD says no ({why}) but --lab-force-arms "
                  f"keeps {class2arm} enabled FOR COUNTERFACTUAL JUDGING ONLY")
        else:
            print(f"ARM GUARD: {why} -- shipping with arm commits DISABLED "
                  f"(class2arm was {class2arm})")
            class2arm = {}
    # M2 relay config: per-class consensus dump schedules (src/relay_config.py).
    # Missing file degrades to the clone-assumption relay, never breaks a build.
    #
    # STALENESS GUARD: the schedules are keyed by identifier CLASS ID, so a
    # file generated against a previous identifier aims them at the wrong
    # classes (2026-08-12 the retrain went 11 -> 17 classes). If the dumps
    # file predates the identifier weights, ship WITHOUT per-class schedules
    # (clone-assumption relay only) rather than with wrong ones.
    famdumps = {"classes": {}}
    fp = os.path.join(ROOT, "models", "relay", "family_dumps.json")
    wp = os.path.join(ROOT, "models", "v22", "identifier", "weights.json")
    if getattr(args, "no_relay_m2", False):
        # Layer-causality audit instrument (Zhang constant-test): build
        # without per-class relay schedules to measure what the layer earns.
        print("relay M2: DISABLED by --no-relay-m2 (causality audit build)")
        fp = None
    if fp and os.path.exists(fp):
        stale = (os.path.exists(wp)
                 and os.path.getmtime(fp) < os.path.getmtime(wp))
        if stale:
            print("RELAY GUARD: family_dumps.json is OLDER than the identifier "
                  "weights (class ids may not line up) -- embedding no "
                  "per-class schedules; run src/relay_config.py to refresh")
        else:
            try:
                famdumps = json.load(open(fp, encoding="utf-8"))
                print(f"relay M2: {len(famdumps.get('classes', {}))} class "
                      f"schedules embedded")
            except Exception:                                      # noqa: BLE001
                pass

    # NN schedule library (src/nn_schedule.py). Same staleness discipline as
    # the M2 schedules, for a different reason: these are RETRIEVAL keys over
    # the live field, and the top of the board is a freshness treadmill, so a
    # library older than a week retrieves neighbours that no longer play.
    # A missing or stale library degrades to M2, never breaks a build.
    nnlib = {}
    nnoutrank = bool(getattr(args, "nn_outrank", False))
    nnsched = bool(getattr(args, "nn_sched", False)) or nnoutrank
    if nnsched:
        np_ = os.path.join(ROOT, "models", "relay", "nn_lib.json")
        if not os.path.exists(np_):
            raise SystemExit(
                "--nn-sched needs models/relay/nn_lib.json -- run "
                "`python src/nn_schedule.py --build` first")
        age_d = (dt.datetime.now().timestamp() - os.path.getmtime(np_)) / 86400
        if age_d > 7:
            raise SystemExit(
                f"NN GUARD: nn_lib.json is {age_d:.1f} days old; the field it "
                f"retrieves against turns over in days. Rebuild it.")
        nnlib = json.load(open(np_, encoding="utf-8"))
        if nnlib.get("dim") != len(idw["W"]):
            # The library vectors and the live query must come from the same
            # feature builder. A dim mismatch means features.py changed under
            # one of them -- the identifier's train/serve skew bug's cousin.
            raise SystemExit(
                f"NN GUARD: library dim {nnlib.get('dim')} != identifier "
                f"input dim {len(idw['W'])} -- rebuild nn_lib.json against "
                f"the current features.py")
        print(f"NN schedule: {len(nnlib.get('ids') or [])} routes x "
              f"{len(nnlib.get('checkpoints') or [])} checkpoints embedded "
              f"(built {nnlib.get('built')})")

    grumodel = None
    if getattr(args, "gru", False):
        gp2 = os.path.join(ROOT, "models", "lab", "gru_weights.json")
        grumodel = json.load(open(gp2, encoding="utf-8"))
        if grumodel["k"] != len(idw["b"]):
            # class ids come from the shared clustering label space; a count
            # mismatch means the GRU was trained on a DIFFERENT day's labels
            # and would aim _FAMILY_DUMPS/_CLASS2ARM at the wrong classes --
            # the arm bug's cousin. Refuse rather than ship it.
            raise SystemExit(
                f"GRU GUARD: gru_weights.json has {grumodel['k']} classes "
                f"but the identifier has {len(idw['b'])} -- regenerate "
                f"seq_dataset + retrain the GRU on today's labels first")
        print(f"GRU identifier embedded ({grumodel['k']} classes, "
              f"hidden {grumodel['hidden']})")
    elif getattr(args, "gru_auto", False):
        # Pipeline mode: embed the GRU only when it is FRESH (class count
        # matches today's identifier) and MEASURED BETTER (its recorded
        # held-out accuracy beats the logistic's from the same day's
        # training). Anything else is a log line, never a failed build --
        # the daily release must not die because a challenger is stale.
        try:
            gp2 = os.path.join(ROOT, "models", "lab", "gru_weights.json")
            cand = json.load(open(gp2, encoding="utf-8"))
            meta = json.load(open(os.path.join(
                ROOT, "models", "v22", "identifier", "meta.json"),
                encoding="utf-8"))
            g_acc = float(cand.get("heldout_acc") or 0.0)
            l_acc = float(meta.get("heldout_acc") or 1.0)
            if cand.get("k") != meta.get("classes"):
                print(f"GRU AUTO: stale ({cand.get('k')} classes vs "
                      f"identifier {meta.get('classes')}) -- logistic ships")
            elif g_acc <= l_acc:
                print(f"GRU AUTO: measured no better ({g_acc:.3f} vs "
                      f"logistic {l_acc:.3f}) -- logistic ships")
            else:
                grumodel = cand
                print(f"GRU AUTO: embedded ({cand['k']} classes, held-out "
                      f"{g_acc:.3f} > logistic {l_acc:.3f})")
        except (OSError, ValueError, KeyError) as exc:
            print(f"GRU AUTO: unavailable ({type(exc).__name__}) -- "
                  f"logistic ships")

    duel_q = {}
    if getattr(args, "duel_policy", False):
        dp = os.path.join(ROOT, "models", "lab", "duel_q.json")
        duel_q = json.load(open(dp, encoding="utf-8"))
        acts = duel_q.get("actions") or []
        states = duel_q.get("q") or {}
        if not acts or not states:
            raise SystemExit("DUEL GUARD: duel_q.json has no actions/states")
        for k, row in states.items():
            if len(row) != len(acts):
                raise SystemExit(
                    f"DUEL GUARD: state {k} has {len(row)} values but there "
                    f"are {len(acts)} actions -- the table and the action list "
                    f"disagree, so argmax would index the wrong lead")
        ho = duel_q.get("heldout") or {}
        print(f"duel policy embedded: {len(states)} states, actions {acts}; "
              f"held-out policy {ho.get('policy')} vs static2 "
              f"{ho.get('static2')} (oracle {ho.get('oracle')})")

    # Deep-RL sell head (E3). Same discipline as arms: the guard rules unless
    # a stamped lab build explicitly bypasses it for counterfactual judging.
    policy = None
    if getattr(args, "policy", False) or getattr(args, "lab_force_policy",
                                                 False):
        p_ok, p_why = TG.policy_allowed()
        pp = os.path.join(ROOT, "models", "lab", "policy_bc.json")
        if p_ok or getattr(args, "lab_force_policy", False):
            policy = json.load(open(pp, encoding="utf-8"))
            if policy.get("baseline_loss", 0) <= policy.get("val_loss", 1):
                raise SystemExit("POLICY GUARD: exported policy does not beat "
                                 "its mean-action baseline -- refusing to "
                                 "embed it")
            print(f"policy head embedded: {policy['n_transitions']:,} "
                  f"transitions / {policy['n_episodes']} episodes, val "
                  f"{policy['val_loss']:.4f} vs baseline "
                  f"{policy['baseline_loss']:.4f}"
                  + ("" if p_ok else f"  [LAB FORCED past guard: {p_why}]"))
        else:
            print(f"POLICY GUARD: {p_why} -- policy head NOT embedded")

    src = TEMPLATE.format(
        label=os.path.basename(args.out), route_id=args.base,
        built=dt.date.today().isoformat(),
        n_arms=len(arms), n_lib=len(idw["b"]), match_prob=match_prob,
        payload=pack(base), armpack=pack(arms), idmodel=pack(idw),
        pull=(not args.no_pull),
        branchpack=(pack(branch_routes) if branch_routes else ""),
        branch_on=args.branch_on,
        branch_step=BRANCH_TRIGGERS[args.branch_on],
        class2arm=pack(class2arm), famdumps=pack(famdumps.get("classes", {})),
        grumodel=pack(grumodel),
        nnsched=nnsched, nnoutrank=nnoutrank, nnlib=pack(nnlib),
        duel_on=bool(getattr(args, "duel_policy", False)),
        plead_on=bool(getattr(args, "premium_lead", False)),
        duelpack=pack(duel_q),
        tomato_on=bool(getattr(args, "tomato_read", False)),
        tomato_seeds=int(getattr(args, "tomato_seeds", 17)),
        am_on=bool(getattr(args, "adaptive_market", False)
                   or getattr(args, "regime_gear", False)),
        am_late=bool(getattr(args, "regime_gear", False)),
        policypack=pack(policy))
    if (getattr(args, "lab_force_arms", False)
            or getattr(args, "lab_force_policy", False)):
        forced = " + ".join(
            n for n, on in (("arms", getattr(args, "lab_force_arms", False)),
                            ("policy", getattr(args, "lab_force_policy",
                                               False))) if on)
        src = src.replace('"""Kaggriculture bandit route agent --',
                          f'"""LAB BUILD -- NEVER SUBMIT ({forced} forced '
                          f'past the guard for counterfactual judging)\n\n'
                          f'Kaggriculture bandit route agent --', 1)
    out = os.path.join(ROOT, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(src)
    print(f"wrote {os.path.relpath(out, ROOT)} ({os.path.getsize(out):,} bytes)")


TEMPLATE = '''"""Kaggriculture bandit route agent -- {label}

Base route {route_id}; {n_arms} trained counter-schedule arms; in-game
program identification over {n_lib} top-100 programs with a bandit that
commits to the matching arm. Built {built} by src/kaggriculture/agentbuild/v22_agent.py.
"""
import base64
import copy
import gc
import json
import math
import zlib

_ROUTE = json.loads(zlib.decompress(base64.b85decode("{payload}")).decode("utf-8"))
# Per-first-shop branch routes (A4): swap the base for the branch whose
# source world drew the same day-3 shop. Empty pack = no branching.
_BRANCH_ON = "{branch_on}"
_BRANCH_STEP = {branch_step}
_BRANCHES_RAW = "{branchpack}"
_BRANCHES = (json.loads(zlib.decompress(base64.b85decode(
    _BRANCHES_RAW)).decode("utf-8")) if _BRANCHES_RAW else {{}})
_ARMS = json.loads(zlib.decompress(base64.b85decode("{armpack}")).decode("utf-8"))
_IDMODEL = json.loads(zlib.decompress(base64.b85decode("{idmodel}")).decode("utf-8"))
_CLASS2ARM = json.loads(zlib.decompress(base64.b85decode("{class2arm}")).decode("utf-8"))
_FAMILY_DUMPS = json.loads(zlib.decompress(base64.b85decode("{famdumps}")).decode("utf-8"))
# GRU identifier (None unless built with --gru): sequence model over the six
# live feature snapshots; cleared Tier 1 on 2026-08-13 (wins macro-F1 AND
# log-loss vs the logistic on all walk-forward folds, no thin-fold collapse).
_GRUMODEL = json.loads(zlib.decompress(base64.b85decode("{grumodel}")).decode("utf-8"))
# Deep-RL sell head (E3): return-conditioned behaviour cloning over 4.2M real
# transitions. None unless built with --policy; guarded by policy_allowed().
_POLICY = json.loads(zlib.decompress(base64.b85decode("{policypack}")).decode("utf-8"))

_MARKET_PARAMS = {{
    "WHEAT": (25, 10000, 400, "sqrt", 0.8, "log", 0.2),
    "CARROT": (35, 10000, 450, "log", 0.2, "sqrt", 0.7),
    "TOMATO": (60, 10000, 200, "linear", 0.4, "sqrt", 0.6),
    "STRAWBERRY": (120, 10000, 100, "sqrt", 0.7, "linear", 1.6),
    "MELON": (250, 10000, 300, "log", 0.2, "sq", 3.6),
    "EGG": (50, 10000, 332, "linear", 0.4, "log", 0.2),
    "MILK": (160, 10000, 122, "sqrt", 0.6, "linear", 1.6),
    "WOOL": (200, 10000, 105, "log", 0.2, "sq", 3.2),
    "FERTILIZER": (100, 10000, 200, "linear", 0.4, "linear", 0.4),
}}
_PRICE_FLOOR = 1
# Engine 1.32.7 hinge rebalance (PR #1399): flipped by
# scripts/engine_swap_1327.py when the LADDER's replays show 1.32.7.
_ENGINE_1327 = True
if _ENGINE_1327:
    _MARKET_PARAMS["CARROT"] = (35, 10000, 450, "hinge", 1.0, "sqrt", 0.7)
    _MARKET_PARAMS["TOMATO"] = (60, 10000, 200, "hinge", 0.4, "sqrt", 0.6)
    _MARKET_PARAMS["EGG"] = (50, 10000, 332, "hinge", 0.4, "log", 0.2)
_TRACKED = ("STRAWBERRY", "MELON", "MILK", "WOOL", "EGG", "TOMATO", "CARROT",
            "WHEAT", "FERTILIZER")

_SHOPS = {{
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}}
_CENTER_PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                    "EGG", "MILK", "WOOL")

# Bandit gates. Identification is cheap; acting on a wrong identification is
# not, so the posterior must be decisive twice in a row before committing,
# and a committed arm is abandoned if the opponent later stops matching.
_COMMIT_MIN_STEP = 216     # never before day 9
_COMMIT_MAX_STEP = 600     # too late to matter after day 25
_MATCH_MIN_SOLD = 60
_MATCH_PROB = {match_prob}  # posterior to commit; tuned from field data by
                            # src/kaggriculture/train/train_gates.py (default 0.85 until >=25
                            # judged commits exist)
_DIVERGE_PROB = 0.50       # committed class below this = non-confirmation
_COMMIT_STREAK = 2
_MISMATCH_LIMIT = 24       # turns of non-confirmation before the arm is dropped

# Feed-buy pull: advance a scheduled wheat purchase when wheat is cheap.
# Measured 2026-08-09: no effect anywhere (byte-identical results with it
# off). Shipped off.
_FEED_PULL = False

# Signature obfuscation: PERMANENTLY OFF -- proven a dead end, not merely
# unvalidated. Measured 2026-08-10 (src/obfuscate.py, engine-priced):
#   * SELL-timing jitter leaves the final sell basket byte-identical (proof in
#     obfuscate.jitter_invariance), so it can NEVER move self_identify, which
#     reads exactly those totals -- the acceptance test and this mechanism are
#     orthogonal by construction. The shipped +2 jitter drops our identifier
#     confidence 0.985 -> 0.964 (noise).
#   * The only transform that moves the identifier at all (+60-turn delay:
#     0.985 -> 0.675) costs -$133,377/episode and still cannot touch the
#     final-mix or farm-event channels a real competitor (the public
#     first-48-turn detector) keys on.
#   * Farm-event edits that WOULD fool that detector (swap herd, shuffle
#     seeds) require farming differently -> real cost, no identity win here.
# Identity is intrinsic to how we farm; the only working lever is basin choice,
# which the crown already exercises (we sit in the sheep-first basin: 75.5% win,
# shared with ~26 field teams, so identifiable-as-a-family, not uniquely).
_OBFUSCATE = False
_OBF_MAX_DELAY = 2

# Threat-first ordering: measured 2026-08-09 as the source of the NOVEL
# regression -- it reorders on ANY match, including wrong ones, and costs
# -$1.8k against programs outside the library while earning ~$0 against
# programs inside it (v21's result). Shipped off.
_THREAT_FIRST = False
_FEED_HORIZON = 48         # turns ahead a scheduled BUY may be advanced from
_FEED_CAP = 40             # units advanced per turn

# Near-mirror market relay v2 (2026-08-11) -- same layer as the route runtime.
# The public meta front-runs shared-market dumps in mirror games (boatlee
# 3-lead, shiv17 5-lead, both public); this is the scan-window answer. See
# tape_runtime.py for the full rationale.
_RELAY = True
_RELAY_STREAK = 12         # consecutive NEAR-matches before arming
_RELAY_START = 216
_RELAY_FRACTION = 1.0
_RELAY_MAX = 40
_RELAY_COOLDOWN = 2
_RELAY_TOL = 4
_RELAY_LEAD = 2            # fire when the first landing is this close
_RELAY_MIN_QTY = 8         # only dumps, not routine sales
_RELAY_MIN_QUOTE = 3       # never race into a crashed market
_RELAY_PRODUCTS = ("FERTILIZER",)
# M2 collision targeting: when the identifier has a sustained match AND that
# class carries a consensus dump schedule (_FAMILY_DUMPS), race only REAL
# collisions -- against THEIR timing -- and widen products to everything the
# schedule proves they dump. Unmatched opponents fall back to the structural
# clone lock above (measured +202/game).
_RELAY_HORIZON = 8         # our dumps this far ahead are collision candidates
_RELAY_M2_STREAK = 8       # consecutive same-class matches before trusting M2
_RELAY_COLLIDE = 6         # |their step - our step| that counts as a race
_RELAY_M2_PRODUCTS = ("FERTILIZER", "MELON", "WOOL", "STRAWBERRY", "MILK")
# In-game learning (2026-08-11). M2-RL: the opponent's dump events are
# directly observable as >=_RELAY_MIN_QTY jumps in the reconstructed sell
# curve, so their schedule is LEARNED during the episode (observed dump
# phases mod 24 -> predicted next dump) and outranks the static family
# average once >=2 events are seen. M3-RL: the race outcome is observable
# too (did their dump land before our fired one?); being pre-empted
# escalates the lead -- an online bandit over lead size (within-episode
# only; the container is fresh per episode, so cross-game learning stays
# in the offline arms/tournament loop).
_RELAY_OBS_MIN = 2         # observed dumps before the online model outranks M2
_RELAY_LEAD_MAX = 8        # escalation ceiling for the online lead

# NN SCHEDULE RETRIEVAL (2026-09-03) -- OFF by default, gate it alone.
#
# M2 races the opponent's dumps against a per-identifier-CLASS consensus
# schedule. The 20-class identifier has collapsed on the current field
# (1,624 / 279 / 97 members in three live classes), so that schedule is one
# global schedule for 80% of the ladder. Measured over 600 held-out-day
# routes, step of their next >=8-unit dump, hit within +-6 turns at s=360:
#
#     WHEAT  STRAW  MELON   MILK   WOOL   FERT
#     0.287  0.870  0.103  0.389  0.396  0.325   global median (ships today)
#     0.302  0.879  0.192  0.399  0.356  0.335   class-conditional (M2)
#     0.808  0.947  0.745  0.860  0.853  0.617   nearest neighbour (this)
#
# Retrieval, not classification, carries the information; same-team hit rate
# is 1.6-3.3%, so it is not memorising an opponent's own other episode. It
# holds under the observation noise the live agent has (feature dropout 0.2,
# 250-route library: 0.771 / 0.936 / 0.707 / 0.781 / 0.784 / 0.539).
#
# This changes only WHICH scheduled SELL the relay races and WHEN. It never
# touches production, purchases or the route -- unlike an ARM COMMIT, which
# is a different mechanism and stays guarded (see arm-commit-guard).
_NN_SCHED = {nnsched}
_NN_MIN_SOLD = 60          # their reconstructed sales before a query is real
_NN_MAX_DIST = 0.60        # reject the neighbour if nothing is close enough
# LAYER 2, gated separately: let the retrieved schedule OUTRANK the online
# phase model. Necessary because the schedule layer is otherwise nearly dead
# code -- the relay consults it only for products where fewer than
# _RELAY_OBS_MIN dumps have been observed, which by step 216+ is almost
# never (measured: an NN build and a class build gave identical banks on 9 of
# 10 probe cells). And the online model it would displace is BAD: same 600
# held-out routes, step of their next >=8-unit dump, hit within +-6 --
#
#   s=360 (day 16)   coverage   PHASE(ships)    NN1    global median
#     FERTILIZER       0.968       0.231       0.571       0.325
#     MELON            0.731       0.000       0.775       0.103
#     WOOL             0.484       0.014       0.806       0.396
#     MILK             0.923       0.029       0.811       0.389
#     WHEAT            0.955       0.039       0.802       0.287
#
# The phase model is beaten by the UNCONDITIONAL MEDIAN on most products in
# the day-13-16 window where the loss study says games are decided. That is
# a negative result about a shipped layer, not a reason to trust this one:
# a better OFFLINE predictor is not a win, and this ships only behind a
# paired sign test on the band panels with the CLEAN ratio.
#
# GATED AND REJECTED 2026-09-03. Band panels, seeds 501-502, both seats,
# 652 paired cells (524 clean), _NN_OUTRANK on vs off, everything else the
# same build:
#
#   cells        sub2000 0.975 / mid 0.988 / top100 0.988 -- IDENTICAL
#   discordant   0-0 in every band, p = 1.0000
#   margin       median -23/game raw, -70/game CLEAN; mean -69 / -86;
#                worse on 352 of 522 cells (2:1 against)
#
# Zero win-cell benefit and a small consistent NEGATIVE margin, against a
# bar of >= +$1,000/game (~3pp). The prediction gain is real and does not
# convert -- the same shape as the sell-timing result in
# .local/memory/market-regime-is-the-bottleneck.md, where a large oracle
# gain was worth nothing to any causal policy. Do NOT flip this on without
# fresh sign-tested evidence of the OPPOSITE sign.
_NN_OUTRANK = {nnoutrank}
_NNLIB = json.loads(zlib.decompress(base64.b85decode("{nnlib}")).decode("utf-8"))
# --- duel policy (E2, 2026-08-13) --------------------------------------------
# Offline Q-learning over the dump race (src/experiments/duel_lab.py). On
# HELD-OUT families it scored 0.141 against the shipped static lead's 0.003 and
# a never-race baseline's 0.114, with an oracle ceiling of 0.577 -- so it is the
# one place a learned policy beat the hand-written rule on data it had never
# seen. Unlike sell timing, the race is genuinely interactive (our action
# changes their payoff), which is the shape RL is actually for.
# State: (turns-to-our-dump bucket, evidence level, observed-offset bucket),
# exactly duel_lab.bucket_state. Action: the lead in turns.
# GATED like everything else: OFF until it clears a paired panel.
_DUEL_POLICY = {duel_on}
_DUEL_Q = json.loads(zlib.decompress(base64.b85decode("{duelpack}")).decode("utf-8")) if {duel_on} else {{}}

# --- market timing (v24.2, 2026-08-13) ---------------------------------------
# Replaces the old `_price_hold`, which was wrong in three ways at once and
# never ran (see docs/history/issues-and-improvements.md, 2026-08-13 night):
#   * it read farms[me]["coins"] -- the engine key is "money" -- so its cash
#     floor tripped every turn and the capture path was DEAD CODE;
#   * its trigger assumed MEAN REVERSION ("the town drains the excess"), which
#     22 real replays falsify: after a heavy sell day price is -21.8% three
#     days later, it does not come back;
#   * its horizon was 6 TURNS while the prize sits at ~72.
#
# This layer is momentum on the observed price instead, which needs no
# foresight: defer a scheduled SELL only while that product's own price is
# still climbing, and release the moment it stops. Sized from the constrained
# counterfactual over those replays (.local/feasible_timing_v241.py): the
# achievable gain saturates at ~+2,507/game (2.5% of bank) at a 72-turn
# horizon, so nothing here chases the infeasible +16,518 an unconstrained
# hold-until-best suggested -- the shed is only 100 units and the route pushes
# ~1,418 units through it per game.
# MEASURED REGRESSION -- OFF. Field-matched paired ablation 2026-08-13
# (7 opponents spanning the real 413..3,763 sell-volume range, 3 seeds, both
# seats, 84 games each): v24.2 lost on EVERY opponent, 55/84 wins (65.5%) vs
# v24.1's 67/84 (79.8%), -2,971/game. The 7/7 consistency is the result, not
# the margin.
#
# The mechanism, which the contested-dump data had already priced: deferring a
# sell FORFEITS THE DUMP RACE. Winning the race is worth +1,914/game measured
# on the ladder; this layer's whole upside is +203/game. Trading the first for
# the second is a ~10x losing trade, and 1,217 of 2,392 real collisions were
# exact same-turn TIES, which deferral converts from ties into losses. Skipping
# `racing` products is not enough -- the relay only marks what it actively
# fired on, not the far larger set of simultaneous sells.
#
# Kept in the tree (not deleted) because the code is correct and cheap, and a
# future engine rebalance that weakens the race would change the trade. Do not
# flip this to True without re-running .local/ablation_v242.py.
_TIMING = False
# Always-on market ACCELERATION (2026-09-01, operator spec: the reactive
# layer must run on top of the tape in EVERY game, reacting to opponent
# moves and market state; family guards are unrelated protections).
_PULL = {pull}
_PULL_RATIO = 1.35         # pull a future sell to NOW above this price/base
_PULL_RATIO_CONTESTED = 1.10   # lower bar for products the OPPONENT farms
_PULL_WINDOW = 36          # how far ahead a scheduled sell may be pulled
_PULL_MAX_PER_TURN = 2
_PULL_BASE = {{"CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120,
              "MELON": 250, "EGG": 50, "MILK": 160, "WOOL": 200}}
# WHEAT excluded: feed-critical, and its log-shaped price drifts over base
# normally (the 2026-09-01 pull-log showed it re-pulled every turn at 34-37
# until the shed bled dry and the animals starved).
# Measured gain per product at a 72-turn horizon: STRAWBERRY +1,022,
# MILK +1,012, WOOL +733, WHEAT +195. FERTILIZER is EXCLUDED because its price
# decays monotonically 95 -> ~10 (selling on sight is already optimal, gain
# +0) and MELON because it measured +9. Deferring either only risks the shed.
_TIMING_PRODUCTS = ("STRAWBERRY", "MILK", "WOOL", "WHEAT")
_TIMING_MAX_DEFER = 72     # turns; the constrained gain saturates here
_TIMING_END = 640          # no new deferrals after this (endgame owns the tail)
_TIMING_MAX_ITEMS = 3      # concurrent deferred products
_TIMING_WINDOW = 24        # trailing window for the trend estimate, turns
_TIMING_SAMPLE = 4         # price sampled every N turns (keeps the state tiny)
_TIMING_MIN_RISE = 0.02    # need a >=2% climb over the window to defer
_TIMING_SHED_MARGIN = 25   # spare shed units to leave for incoming DROPs
_TIMING_CASH_FLOOR = 2500  # never defer while cash-poor: BUYs must not starve
_TIMING_MAX_UNITS = 45     # cap total deferred stock (shed is 100)

_SELLABLE = ("STRAWBERRY", "MELON", "MILK", "WOOL", "EGG", "TOMATO", "CARROT",
             "WHEAT", "FERTILIZER")
_MARKET_OPS = ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND")
_UNIT_OPS = ("NORTH", "SOUTH", "EAST", "WEST", "PASS", "PICKUP", "PLACE", "DROP",
             "PLANT", "WATER", "HARVEST", "FERTILIZE", "BUILD_COOP",
             "BUILD_PASTURE", "DIG", "FEED", "CARE", "COLLECT_FERTILIZER")
_SHED_CAP = 100
_MAX_ORDERS = 10

# Ported margin mechanism (2026-08-30 loss audit: 8/10 trailing losses
# under $7k): one-turn conservation lead on premium goods vs ANY
# opponent, with a deposit advance so a same-turn-deposit schedule can
# still feed it. Borrowed quantities repay through _PLEAD_DUE at the
# exact due step. Quantity-conserving; changes when, never whether.
_PLEAD = {plead_on}
_PLEAD_ITEMS = ("WOOL", "MILK", "MELON", "STRAWBERRY")
_PLEAD_CAP = 10
_PLEAD_MIN_QUOTE = 3
_PLEAD_START = 216
_PLEAD_STOP = 640

# Day-16 TOMATO demand read (rank 2 / Crop Dusta, docs/cropdusta-adaptive-
# study-2026-09-03.md). Baseline town consumption is exactly 1 unit/turn and
# every unlocked shop listing a product adds 1, so a ONE-TURN tomato drain of
# >= 3 means two shops are buying tomato. Their rule fits 182/190 = 95.8% of
# games across 9 build-days: at step 385 they either open a tomato block or
# grow zero tomato all game.
#
# It is ADDITIVE here, never a tape swap: the decision only re-crops PLANT
# ops the base ALREADY schedules (same tile, same hand, same watering
# schedule), buys the seeds those plantings need, re-points the matching
# PICKUP at the tile it actually re-cropped, and offers the produce. With
# _TOMATO_ON false the layer returns the action untouched.
_TOMATO_ON = {tomato_on}
_TOMATO_STEP = 385         # day 16, hour 1 -- the turn the tick is readable
_TOMATO_TICK = 3           # >= 3 == two shops buying, not just the centre
_TOMATO_SEEDS = {tomato_seeds}

# --- COMPETITOR PORT: boatlee V29-R1 adaptive market (OFF by default) --------
# Ported from `.local/mined/2026-09-03-b/lynnsakurai_farming-score/
# decoded_main.py` (LB #2 ships boatlee's controller verbatim; the decoded
# _AM_CONFIG below matches all 20 of boatlee's constants). Written up in
# docs/history/competitor-intel-2026-09-03b.md section 2.
#
# WHY THIS ONE WHEN OUR OWN MARKET LAYERS ARE DEAD. _TIMING (defer a sell into
# a rising price) measured a REGRESSION twice -- deferring forfeits the dump
# race, worth +$1,914/game -- and _PULL contributes EXACTLY nothing (identical
# 467-31 cells with and without). Both of ours are unconditional, always-on
# price reactions. This one is the opposite shape and that is the whole
# hypothesis: it starts at step 456 (not earlier), touches only three items,
# caps at 18 extra units per item for the WHOLE GAME, and latches ITSELF OFF
# against a near-mirror opponent at step 289. It is a hypothesis, not a
# prediction; gate it like anything else and expect it to die.
#
# TWO DELIBERATE DEVIATIONS FROM THE SOURCE, both house rules:
#   * state lives in the PER-EPISODE state dict, never a module global. A
#     module-global _AM_STATE leaks across episodes when `kagg serve` and the
#     ladder reuse the process (this is the bug tests/test_branch_dispatch.py
#     exists for).
#   * the layer never displaces an order the tape already queued. The source
#     is already written this way (it only inserts when len(market) < 10), so
#     this costs nothing -- but it is asserted rather than assumed, because
#     dropping a BUY_SEED is a silent no-op that looks like laziness.
_AM_ON = {am_on}
_AM_LATE = {am_late}       # protect/convert gear (lynnsakurai's own addition)
_AM_ITEMS = ("STRAWBERRY", "MILK", "WOOL")
_AM_BASE_PRICE = {{"STRAWBERRY": 120.0, "MILK": 160.0, "WOOL": 200.0}}
_AM_CONFIG = {{
    "start_step": 456, "reserve_early": 12, "reserve_mid": 8,
    "reserve_late": 6, "reserve_tail": 3, "shed_soft": 72, "shed_hard": 88,
    "shed_reserve_cut": 7, "price_gate": 0.66, "pressure_trigger": 2.0,
    "pressure_gate_cut": 0.18, "capacity_trigger": 0,
    "capacity_gate_cut": 0.08, "tranche": 4, "pressure_tranche": 4,
    "shed_tranche": 4, "tail_tranche": 5, "demand_reserve": 2,
    "demand_reserve_cap": 14, "demand_gate": 0.045, "demand_gate_cap": 0.24,
    "max_extra_per_item": 18, "mirror_latch_step": 289,
    "mirror_composition_distance": 2, "mirror_money_distance": 250,
    "mirror_max_extra_per_item": 0, "mirror_urgent_max_extra_per_item": 6,
    "late_first_step": 432, "late_cash_gap": 1000,
    "late_protect_price_index": 0.78, "late_convert_price_index": 0.82,
    "late_reserve_shift": 2, "late_gate_shift": 0.04,
    "late_tranche_shift": 2, "late_epoch_turns": 72,
}}

# --- latency: keep the collector away from the embedded payloads -------------
# MEASURED 2026-08-14. Mean turn 3.4 ms and p99 ~8 ms, but ONE turn per episode
# spiked to 245-317 ms -- and with gc disabled that same worst turn fell to
# 12.2 ms. The spike moved with the seed (step 519 / 587 / 499), so it was never
# agent logic: it was a generational collection walking the large immortal
# objects this module unpacks at import (the 719-turn route, the identifier
# weights, the arm overlays, the relay schedules).
#
# `submit.py` gates at worst < 250 ms because Kaggle runs on ~1.6 vCPU, so this
# GC pause -- not the agent -- is what failed the 2026-08-14 release gate, and
# what made the check pass some days and fail others on identical logic.
#
# freeze() moves everything already allocated into a permanent generation that
# is never scanned again, which targets the cause exactly. Thresholds are then
# raised so the remaining per-turn garbage is collected in fewer, cheaper
# passes. The collector stays ENABLED: disabling it outright would leak any
# reference cycle for the whole episode, and this fix does not need that.
gc.collect()
gc.freeze()
gc.set_threshold(50000, 50, 50)
_LAST_STEP = 718
_WEED_CATCHUP = 8
_STATE = {{0: {{}}, 1: {{}}}}

# Identifier feature layout, mirrored exactly from src/kaggriculture/data/features.py:
# 9 products x 24 curve samples (every 30 turns, zero-padded future,
# normalized by the block max) + endgame share + t/720.
_ID_PRODUCTS = ("STRAWBERRY", "MELON", "MILK", "WOOL", "EGG", "TOMATO",
                "CARROT", "WHEAT", "FERTILIZER")
_ID_CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
_ID_ANIMALS = ("COW", "SHEEP", "GOOSE")
_ID_WINDOWS = ((0, 144), (144, 288), (288, 432), (432, 576), (576, 720))
_ID_SAMPLES = 24
_ID_STRIDE = 30
_NOVEL_CLASS = len(_IDMODEL["b"]) - 1


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        value = obj.get(key, default)
    else:
        value = getattr(obj, key, default)
    return default if value is None else value


def _tile_at(farm, position):
    tiles = _get(farm, "tiles", []) or []
    try:
        x, y = int(position[0]), int(position[1])
    except (TypeError, ValueError, IndexError):
        return None
    if 0 <= y < len(tiles) and 0 <= x < len(tiles[y]):
        return tiles[y][x]
    return None


def _shed_adjacent(position, size):
    half = size // 2
    try:
        spot = (int(position[0]), int(position[1]))
    except (TypeError, ValueError, IndexError):
        return False
    return spot in ((half - 1, half - 1), (half, half - 1),
                    (half - 1, half), (half, half))


def _align_hands(action, obs):
    me = int(_get(obs, "player", 0) or 0)
    farms = _get(obs, "farms", []) or []
    farm = farms[me] if me < len(farms) else {{}}
    want = len(_get(farm, "hands", []) or [])
    hands = list(action.get("hands") or [])
    while len(hands) < want:
        hands.append(["PASS"])
    action["hands"] = hands[:want]
    if not (isinstance(action.get("farmer"), list) and action["farmer"]
            and action["farmer"][0] in _UNIT_OPS):
        action["farmer"] = ["PASS"]
    action["hands"] = [h if (isinstance(h, list) and h and h[0] in _UNIT_OPS)
                       else ["PASS"] for h in action["hands"]]
    return action


def _projected_shed(obs, action):
    me = int(_get(obs, "player", 0) or 0)
    farms = _get(obs, "farms", []) or []
    farm = farms[me] if me < len(farms) else {{}}
    private = _get(obs, "private", {{}}) or {{}}
    shed = dict(_get(private, "shed", {{}}) or {{}})
    inventories = _get(private, "inventories", []) or []
    size = len(_get(farm, "tiles", []) or []) or 10
    positions = [list(_get(farm, "farmer", [0, 0]) or [0, 0])]
    positions += [list(p) for p in (_get(farm, "hands", []) or [])]
    ops = [action.get("farmer")] + list(action.get("hands") or [])
    room = max(0, _SHED_CAP - sum(int(v or 0) for v in shed.values()))
    for i, op in enumerate(ops):
        if not (isinstance(op, list) and op):
            continue
        if i >= len(positions) or not _shed_adjacent(positions[i], size):
            continue
        carried = inventories[i] if i < len(inventories) else {{}}
        if op[0] == "DROP":
            for item, count in (carried or {{}}).items():
                n = min(int(count or 0), room)
                if n <= 0:
                    continue
                shed[item] = int(shed.get(item, 0) or 0) + n
                room -= n
        elif (op[0] == "PLACE" and len(op) >= 2
              and op[1] not in ("COW", "SHEEP", "GOOSE")):
            # PLACE item [n] beside the shed is the engine's second deposit
            # path (kaggriculture.py PLACE falls through to the shed drop).
            # Missing it made this projection undercount every route that
            # deposits per-item, and _safe_market then cancelled real sales
            # (the v32 bankruptcy, 2026-08-21). Animals are placement, not
            # deposits; PICKUP is ignored because overestimating here only
            # relaxes the clamp toward the engine's own partial-fill floor.
            item = op[1]
            try:
                want = int(op[2]) if len(op) >= 3 else 1
            except (TypeError, ValueError):
                want = 1
            have = int((carried or {{}}).get(item, 0) or 0)
            n = min(max(0, want), have, room)
            if n > 0:
                shed[item] = int(shed.get(item, 0) or 0) + n
                room -= n
    return shed


def _safe_market(obs, action):
    remaining = _projected_shed(obs, action)
    out = []
    for raw in (action.get("market") or []):
        if not (isinstance(raw, list) and raw and raw[0] in _MARKET_OPS):
            continue
        order = list(raw)
        if order[0] == "SELL" and len(order) >= 3:
            item = order[1]
            try:
                want = max(0, int(order[2]))
            except (TypeError, ValueError):
                want = 0
            have = max(0, int(remaining.get(item, 0) or 0))
            qty = min(want, have)
            if qty <= 0:
                continue
            order[2] = qty
            remaining[item] = have - qty
        out.append(order)
    action["market"] = out[:_MAX_ORDERS]
    return action


def _sell_first(action):
    orders = action.get("market") or []
    sells = [o for o in orders if o and o[0] == "SELL"]
    rest = [o for o in orders if not (o and o[0] == "SELL")]
    action["market"] = (sells + rest)[:_MAX_ORDERS]
    return action


def _shape(name, value, scale_t=0.0):
    value = max(0.0, float(value))
    if name == "linear":
        return value
    if name == "sq":
        return value * value
    if name == "sqrt":
        return math.sqrt(value)
    if name == "log":
        return math.log1p(value)
    if name == "hinge":
        if not scale_t or scale_t <= 0:
            return value
        u = value / scale_t
        return u + 8.0 * max(0.0, u - 1.0) ** 2
    return value


def _quote(item, inventory):
    spec = _MARKET_PARAMS.get(item)
    if not spec:
        return 0.0
    base, equilibrium, scale, below_f, below_t, above_f, above_t = spec
    if inventory < equilibrium:
        amplitude = below_t * base / _shape(below_f, scale, scale)
        price = base + amplitude * _shape(below_f, equilibrium - inventory,
                                          scale)
    else:
        amplitude = above_t * base / _shape(above_f, scale, scale)
        price = base - amplitude * _shape(above_f, inventory - equilibrium,
                                          scale)
    return max(_PRICE_FLOOR, int(round(price)))


def _track_opponent(obs, step, state):
    """Their cumulative sales, reconstructed exactly from public state."""
    inventory = _get(_get(obs, "market", {{}}) or {{}}, "inventory", {{}}) or {{}}
    now = {{item: int(_get(inventory, item, 0) or 0) for item in _TRACKED}}
    prev = state.get("prev_inv")
    ours = state.get("our_last_sells") or {{}}
    opp = state.setdefault("opp_cum", {{item: 0 for item in _TRACKED}})
    if prev is not None:
        # 1.32.6 balance change (2026-08-07): the town centre buys once per
        # day flat -- the 2x/4x late-game schedule is gone -- and shops are
        # sampled WITH replacement, so unlocked_shops may list the same shop
        # several times and each instance consumes. Iterating the list as-is
        # counts duplicates correctly.
        consumed = {{item: 0 for item in _TRACKED}}
        prev_step = step - 1
        if prev_step % 4 == 0:
            for shop in state.get("prev_shops", ()):
                products = _SHOPS.get(shop, ())
                mult = 2 if len(products) == 1 else 1
                for item in products:
                    consumed[item] += mult
        if prev_step % 24 == 0:
            for item in _CENTER_PRODUCTS:
                consumed[item] += 1
        for item in _TRACKED:
            est = (now[item] - prev[item]) - int(ours.get(item, 0)) + consumed[item]
            if est > 0:
                opp[item] += est
                if step >= 648:
                    state["opp_late"] = int(state.get("opp_late", 0)) + est
    if step % _ID_STRIDE == 0:
        snaps = state.setdefault("id_snaps", {{}})
        for item in _ID_PRODUCTS:
            snaps.setdefault(item, []).append(opp.get(item, 0))
    state["prev_inv"] = now

    # Farm-EVENT tracking (identifier v3): the opponent's farm is public, and
    # plants/herd-adds/hires/land are observable as state diffs. Unlike sale
    # curves these cannot be obfuscated without farming differently.
    me = int(_get(obs, "player", 0) or 0)
    farms = _get(obs, "farms", []) or []
    oppfarm = farms[1 - me] if len(farms) > 1 else {{}}
    fs = state.setdefault("farm", {{
        "plant": [[0] * 5 for _ in _ID_CROPS], "herd": [0, 0, 0],
        "hires_cum": 0, "hires_series": [], "pasture": 0, "coop": 0,
        "land": [], "prev_tiles": None, "prev_hands": None,
        "prev_counts": None}})
    tiles = _get(oppfarm, "tiles", []) or []
    crop_now, animal_now = {{}}, [0, 0, 0]
    pasture = coop = 0
    tile_kinds = {{}}
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            if not isinstance(tile, dict):
                continue
            kind = tile.get("kind")
            crop = tile.get("crop")
            if crop:
                crop_now[(x, y)] = crop
            animal = tile.get("animal")
            if animal in _ID_ANIMALS:
                animal_now[_ID_ANIMALS.index(animal)] += 1
            if kind == "PASTURE":
                pasture += 1
            elif kind == "COOP":
                coop += 1
    wi = 0
    for w, (lo, hi) in enumerate(_ID_WINDOWS):
        if lo <= step < hi:
            wi = w
            break
    prev_tiles = fs["prev_tiles"]
    if prev_tiles is not None:
        for pos, crop in crop_now.items():
            if prev_tiles.get(pos) != crop and crop in _ID_CROPS:
                fs["plant"][_ID_CROPS.index(crop)][wi] += 1
    prev_counts = fs["prev_counts"]
    if prev_counts is not None:
        for i in range(3):
            if animal_now[i] > prev_counts[i]:
                fs["herd"][i] += animal_now[i] - prev_counts[i]
    hands = len(_get(oppfarm, "hands", []) or [])
    if fs["prev_hands"] is not None and hands > fs["prev_hands"]:
        fs["hires_cum"] += hands - fs["prev_hands"]
    if step % _ID_STRIDE == 0:
        fs["hires_series"].append(fs["hires_cum"])
    quads = len(_get(oppfarm, "unlocked_quadrants", []) or [])
    if quads > len(fs["land"]) + 1:
        fs["land"].extend([step] * (quads - 1 - len(fs["land"])))
    fs["pasture"], fs["coop"] = pasture, coop
    fs["prev_tiles"], fs["prev_hands"] = crop_now, hands
    fs["prev_counts"] = animal_now
    town = _get(obs, "town", {{}}) or {{}}
    state["prev_shops"] = tuple(_get(town, "unlocked_shops", ()) or ())
    return opp


def _id_features(state, step):
    """The identifier's v3 input, mirroring src/kaggriculture/data/features.py exactly:
    sale curves (reconstruction) + endgame + plantings + herd + labor +
    builds/land (farm-state diffs) + t."""
    snaps = state.get("id_snaps") or {{}}
    vals = [v for series in snaps.values() for v in series]
    top = max(vals) if vals else 0
    top = top if top > 0 else 1
    vec = []
    for item in _ID_PRODUCTS:
        series = list(snaps.get(item, ()))[:_ID_SAMPLES]
        vec.extend([v / top for v in series]
                   + [0.0] * (_ID_SAMPLES - len(series)))
    total = sum(state.get("opp_cum", {{}}).values())
    vec.append(state.get("opp_late", 0) / max(1, total))

    fs = state.get("farm") or {{}}
    def norm(v):
        m = max(max(v), 1)
        return [x / m for x in v]
    plant = [fs.get("plant", [[0] * 5] * 5)[c][w]
             for c in range(5) for w in range(5)]
    vec.extend(norm(plant))
    vec.extend(norm(list(fs.get("herd", [0, 0, 0]))))
    labor = list(fs.get("hires_series", ()))[:_ID_SAMPLES]
    labor = norm(labor) if labor else []
    vec.extend((labor + [0.0] * _ID_SAMPLES)[:_ID_SAMPLES])
    land = fs.get("land", [])
    bl = [fs.get("pasture", 0), fs.get("coop", 0), len(land),
          (land[0] if land else 720), (land[-1] if land else 720)]
    vec.extend(norm(bl))
    vec.append(min(step, 719) / 720.0)
    return vec


_GRU_STEPS = (144, 192, 240, 288, 360, 480)


def _gru_advance(state, step):
    """Advance the GRU one checkpoint at a time; cache posterior between.

    One GRU step is ~6 ms of pure python and runs at most 6 times a game
    (checkpoint crossings); the cached posterior costs nothing per turn.
    The hidden state lives in the episode state blob, so a container
    restart mid-episode simply rebuilds from the crossed checkpoints.
    """
    P = _GRUMODEL
    H = P["hidden"]
    crossed = 0
    for t in _GRU_STEPS:
        if step >= t:
            crossed += 1
    if crossed == 0:
        return None
    done = int(state.get("gru_done", 0))
    if done >= crossed and state.get("gru_p") is not None:
        return state["gru_p"]
    h = list(state.get("gru_h") or [0.0] * H)
    while done < crossed:
        x = _id_features(state, _GRU_STEPS[done])
        xn = [(v - m) / s for v, m, s in zip(x, P["mu"], P["sd"])]
        mean = sum(xn) / len(xn)
        var = sum((v - mean) ** 2 for v in xn) / len(xn)
        inv = 1.0 / math.sqrt(var + 1e-5)
        xl = [(v - mean) * inv * g + bb
              for v, g, bb in zip(xn, P["ln_g"], P["ln_b"])]
        gi = [sum(w * v for w, v in zip(P["w_ih"][j], xl)) + P["b_ih"][j]
              for j in range(3 * H)]
        gh = [sum(w * v for w, v in zip(P["w_hh"][j], h)) + P["b_hh"][j]
              for j in range(3 * H)]
        nh = []
        for j in range(H):
            r = 1.0 / (1.0 + math.exp(-(gi[j] + gh[j])))
            zz = 1.0 / (1.0 + math.exp(-(gi[H + j] + gh[H + j])))
            n = math.tanh(gi[2 * H + j] + r * gh[2 * H + j])
            nh.append((1.0 - zz) * n + zz * h[j])
        h = nh
        done += 1
    state["gru_h"], state["gru_done"] = h, done
    z = [sum(w * v for w, v in zip(P["w_out"][j], h)) + P["b_out"][j]
         for j in range(P["k"])]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    state["gru_p"] = [v / s for v in e]
    return state["gru_p"]


def _match(opp, step, state):
    """(class, posterior, decisive) from the trained identifier model."""
    total = sum(opp.values())
    if total < _MATCH_MIN_SOLD:
        return None
    if _GRUMODEL is not None:
        p = _gru_advance(state, step)
        if p is None:
            return None                    # before the first checkpoint
    else:
        x = _id_features(state, step)
        W, b = _IDMODEL["W"], _IDMODEL["b"]
        k = len(b)
        z = [b[j] + sum(x[i] * W[i][j] for i in range(len(x)))
             for j in range(k)]
        m = max(z)
        e = [2.718281828459045 ** (v - m) for v in z]
        s = sum(e)
        p = [v / s for v in e]
    cls = max(range(len(p)), key=lambda j: p[j])
    nov = (_GRUMODEL["k"] - 1) if _GRUMODEL is not None else _NOVEL_CLASS
    decisive = p[cls] >= _MATCH_PROB and cls != nov
    return cls, p[cls], decisive


def _bandit(obs, step, state):
    """Commit to a counter-arm on sustained decisive identification; abandon
    the arm if the opponent later diverges from the matched program."""
    opp = _track_opponent(obs, step, state)
    m = _match(opp, step, state)
    state["matched"] = m[0] if m else None
    arm = state.get("arm")
    if arm is not None:
        # The arm survives only while the model keeps confirming the class.
        confirmed = (m is not None and m[0] == state.get("arm_class")
                     and m[1] >= _DIVERGE_PROB)
        state["mismatch"] = 0 if confirmed else int(state.get("mismatch", 0)) + 1
        if state["mismatch"] >= _MISMATCH_LIMIT:
            state["arm"] = None          # they are not playing that program
            state["arm_dead"] = True
        return state.get("arm")
    if state.get("arm_dead") or not (_COMMIT_MIN_STEP <= step <= _COMMIT_MAX_STEP):
        return None
    if m is None:
        return None
    cls = m[0]
    if _CLASS2ARM.get(str(cls)) is None:
        return None
    # SPRT commit (2026-08-13, replaces posterior>=threshold + streak): a
    # sequential probability ratio test over the per-turn evidence bit
    # "posterior >= 0.7 for the tracked class". Cutting the bit at 0.7, not
    # 0.5, matters: an ambiguous ~0.5 posterior stream random-walks into a
    # false commit 44% of the time at 0.5 vs 2.8% at 0.7 (measured,
    # src/experiments/gates_v2.py). Commits in ~4 turns on decisive streams;
    # ambiguity DRIFTS TO ABANDON. alpha is tight because a wrong commit
    # cost v23.1 half its economy; beta is loose because a missed arm is
    # the safe failure. Tracked class switches reset the test.
    if str(cls) in (state.get("sprt_dead") or {{}}):
        return None                         # abandoned classes STAY dead --
                                            # restarts would compound alpha
    if state.get("sprt_class") != cls:
        state["sprt_class"], state["sprt_llr"] = cls, 0.0
    _nov = (_GRUMODEL["k"] - 1) if _GRUMODEL is not None else _NOVEL_CLASS
    x = 1 if (m[1] >= 0.7 and cls != _nov) else 0
    state["sprt_llr"] = float(state.get("sprt_llr", 0.0)) + (
        1.0986122886681098 if x else -1.0986122886681098)  # ln(0.75/0.25)
    if state["sprt_llr"] <= -2.2925347571405443:            # ln(0.10/0.98)
        state.setdefault("sprt_dead", {{}})[str(cls)] = True
        state["sprt_class"] = None
        return None
    if state["sprt_llr"] >= 3.8066624897703196:             # ln(0.90/0.02)
        state["arm"] = _CLASS2ARM[str(cls)]
        state["arm_class"] = cls
        state["committed_at"] = step
    return state.get("arm")



def _rt(state):
    """The route THIS EPISODE is playing: the dispatched branch, else base."""
    return state.get("route") or _ROUTE


def _feed_pull(obs, action, step, state):
    """Advance a scheduled wheat purchase while wheat quotes below base.

    Bounded: only orders the schedule already contains inside the horizon,
    debited via `buy_due` so season totals are unchanged.
    """
    if not _FEED_PULL or step + 1 >= _LAST_STEP:
        return action
    due = state.setdefault("buy_due", 0)
    market = list(action.get("market") or [])
    if due > 0:
        kept = []
        for o in market:
            if (isinstance(o, list) and len(o) >= 3 and o[0] == "BUY_PRODUCT"
                    and o[1] == "WHEAT" and state["buy_due"] > 0):
                try:
                    qty = max(0, int(o[2]))
                except (TypeError, ValueError):
                    qty = 0
                take = min(state["buy_due"], qty)
                state["buy_due"] -= take
                qty -= take
                if qty <= 0:
                    continue
                o = [o[0], o[1], qty]
            kept.append(o)
        market = kept
    inventory = _get(_get(obs, "market", {{}}) or {{}}, "inventory", {{}}) or {{}}
    have = int(_get(inventory, "WHEAT", 10000) or 0)
    if (_quote("WHEAT", have) < _MARKET_PARAMS["WHEAT"][0]
            and len(market) < _MAX_ORDERS
            and not any(isinstance(o, list) and o and o[0] == "BUY_PRODUCT"
                        and len(o) >= 2 and o[1] == "WHEAT" for o in market)):
        planned = 0
        for dt_ in range(1, _FEED_HORIZON + 1):
            idx = step + dt_
            if idx >= len(_rt(state)):
                break
            for o in (_rt(state)[idx].get("market") or []):
                if (isinstance(o, list) and len(o) >= 3 and o[0] == "BUY_PRODUCT"
                        and o[1] == "WHEAT"):
                    try:
                        planned += max(0, int(o[2]))
                    except (TypeError, ValueError):
                        pass
        qty = min(planned - state["buy_due"], _FEED_CAP)
        if qty > 0:
            market.append(["BUY_PRODUCT", "WHEAT", int(qty)])
            state["buy_due"] += int(qty)
    action["market"] = market[:_MAX_ORDERS]
    return action


def _structure_sig(farm):
    """Money-free structural signature (see tape_runtime.py: money diverges
    between near-clones within turns; the build does not)."""
    counts = {{}}
    tiles = _get(farm, "tiles", []) or []
    for row in tiles:
        for tile in row:
            if isinstance(tile, dict):
                key = tile.get("crop") or tile.get("animal") or tile.get("kind")
                if key:
                    counts[str(key)] = counts.get(str(key), 0) + 1
    return (len(_get(farm, "hands", []) or []),
            len(_get(farm, "unlocked_quadrants", []) or []),
            counts)


def _is_near_mirror(obs):
    me = int(_get(obs, "player", 0) or 0)
    farms = _get(obs, "farms", []) or []
    if len(farms) < 2:
        return False
    a, b = _structure_sig(farms[me]), _structure_sig(farms[1 - me])
    dist = 2 * abs(a[0] - b[0]) + 3 * abs(a[1] - b[1])
    for key in set(a[2]) | set(b[2]):
        dist += abs(a[2].get(key, 0) - b[2].get(key, 0))
    return dist <= _RELAY_TOL



def _net_due(ledger, step, action):
    """Net owed quantities against this turn's sells; carry any remainder
    to the next step so quantity conservation holds under ANY market
    source (arm overrides replace the market wholesale)."""
    due = ledger.pop(step, None)
    if not due:
        return
    market = []
    for order in (action.get("market") or []):
        if (isinstance(order, list) and len(order) >= 3
                and order[0] == "SELL" and order[1] in due):
            owed = due[order[1]]
            try:
                qty = max(0, int(order[2]))
            except (TypeError, ValueError):
                qty = 0
            take = min(owed, qty)
            due[order[1]] = owed - take
            qty -= take
            if qty <= 0:
                continue
            order = [order[0], order[1], qty]
        market.append(order)
    action["market"] = market
    # Remainder DROPS by design (measured: carrying it forward suppresses
    # future sells game-long, -20 discordant on the gauntlet 2026-09-01).
    # The arm-override double-sell hazard is closed at the override site:
    # ledgers borrowed against the ROUTE's plan are cleared when an
    # override replaces that plan.


def _nn_sched(state, step):
    """The retrieved opponent dump schedule, in _FAMILY_DUMPS event format.

    Queries only AT a library checkpoint, because the feature vector's last
    element is t/720 and the library was built at those exact t -- querying
    off-checkpoint would compare vectors built at different times, which is
    train/serve skew wearing a virtuous face. The answer is cached until the
    next checkpoint, so this costs one L1 pass over the library three times
    a game, not once a turn.

    Returns None (and the caller falls back to M2) when the opponent has not
    sold enough to make the query real, or when nothing in the library is
    close enough -- a far neighbour is a guess, and racing a guess distorts
    our own schedule for nothing.
    """
    if not _NN_SCHED or not _NNLIB:
        return None
    checkpoints = _NNLIB.get("checkpoints") or ()
    if step in checkpoints and state.get("nn_cp") != step:
        state["nn_cp"] = step
        state["nn_sched"] = None
        if sum((state.get("opp_cum") or {{}}).values()) >= _NN_MIN_SOLD:
            ci = list(checkpoints).index(step)
            vecs = _NNLIB["V"][ci]
            scale = float(_NNLIB.get("scale") or 1000)
            q = _id_features(state, step)
            n = min(len(q), _NNLIB.get("dim") or len(q))
            best, best_d = None, None
            for j, row in enumerate(vecs):
                d = 0.0
                for i in range(n):
                    d += abs(q[i] - row[i] / scale)
                if best_d is None or d < best_d:
                    best_d, best = d, j
            if best is not None and best_d / max(1, n) <= _NN_MAX_DIST:
                products = _NNLIB.get("products") or ()
                # _FAMILY_DUMPS event shape: [step, item, qty, support,
                # t_iqr, q_iqr]. A retrieved schedule is a single route, so
                # support is 1.0 and the spreads are 0 -- the relay's
                # "race only TIGHT consensus" filter passes it, which is the
                # intent: this is not a guess averaged over 1,600 members.
                state["nn_sched"] = [
                    [ev[0], products[ev[1]], ev[2], 1.0, 0, 0]
                    for ev in _NNLIB["S"][best]
                    if 0 <= ev[1] < len(products)]
                state["nn_dist"] = best_d / max(1, n)
                state["nn_id"] = best
    return state.get("nn_sched")


def _relay(obs, action, step, state):
    """Near-mirror market relay: pull scheduled dumps ahead of the mirror's,
    repay at the scheduled step. Quantity-conserved, slot-aware."""
    ledger = state.setdefault("relay_due", {{}})
    _net_due(ledger, step, action)

    # ---- M2-RL observation: their dump events are visible as jumps in the
    # reconstructed sell curve. Collect them every turn (even before any
    # lock), and grade the last race (M3-RL): a dump of theirs landing while
    # one of ours for the same product was still scheduled un-fired means we
    # were pre-empted -- escalate the online lead.
    cur = {{k: int(v or 0) for k, v in (state.get("opp_cum") or {{}}).items()}}
    prevc = state.get("relay_prev_cum")
    if prevc is not None:
        for item in _RELAY_M2_PRODUCTS:
            jump = cur.get(item, 0) - prevc.get(item, 0)
            if jump > 0:
                state.setdefault("opp_jumps", {{}}).setdefault(
                    item, []).append(jump)
            # ADAPTIVE THRESHOLD (2026-09-01): a dump is big RELATIVE TO
            # THIS OPPONENT -- 2x their median nonzero sell, floored at 3,
            # capped at the static cutoff. Trickle-sellers (audit: 3 of 8
            # losses showed dumps=0 at the fixed threshold) self-lower it;
            # dump-sellers keep the high bar. Deterministic, no learning.
            js = sorted((state.get("opp_jumps") or {{}}).get(item) or [])
            if len(js) >= 4:
                # median + 3*MAD (modified z-score): keys on the SPREAD of
                # their routine sells, and MAD's 50% breakdown point means
                # the dumps in the history cannot inflate the threshold.
                # med>=1 and mad>=1 make the effective range [4, cap];
                # the max(3,...) floor is defensive, not reachable.
                med = js[len(js) // 2]
                devs = sorted(abs(x - med) for x in js)
                mad = devs[len(devs) // 2]
                thr = min(_RELAY_MIN_QTY, max(3, med + 3 * max(1, mad)))
            else:
                thr = _RELAY_MIN_QTY
            if jump < thr:
                continue
            state.setdefault("opp_dumps", {{}}).setdefault(item, []).append(step)
            fired = (state.get("relay_fired") or {{}}).get(item)
            if fired is not None and 0 <= step - fired <= _RELAY_COLLIDE:
                continue                    # we fired first: race won
            upcoming = False
            for t2 in range(step, min(step + _RELAY_COLLIDE + 1, len(_rt(state)))):
                for o2 in (_rt(state)[t2].get("market") or []):
                    if (isinstance(o2, list) and len(o2) >= 3
                            and o2[0] == "SELL" and o2[1] == item):
                        try:
                            if int(o2[2]) >= _RELAY_MIN_QTY:
                                upcoming = True
                        except (TypeError, ValueError):
                            pass
            if upcoming:                    # pre-empted: learn a longer lead
                state["relay_lead_rl"] = min(
                    int(state.get("relay_lead_rl", _RELAY_LEAD)) + 2,
                    _RELAY_LEAD_MAX)
    state["relay_prev_cum"] = cur

    if not _RELAY or step < _RELAY_START or step + 1 >= len(_rt(state)):
        return action
    if step >= 640:
        return action

    # M2 lock: sustained identifier match on a class with a known schedule.
    matched = state.get("matched")
    if matched is not None and state.get("m2_cls") == matched:
        state["m2_streak"] = int(state.get("m2_streak", 0)) + 1
    else:
        state["m2_streak"], state["m2_cls"] = 1, matched
    their = None
    # NN retrieval outranks the class consensus when it is on and confident:
    # it needs no identifier commit at all (the identifier has collapsed to
    # three live classes), and it falls through to M2 whenever the query is
    # not yet real or no library route is close enough.
    if _NN_SCHED:
        their = _nn_sched(state, step)
    if their is None and matched is not None and (
            state["m2_streak"] >= _RELAY_M2_STREAK):
        their = _FAMILY_DUMPS.get(str(matched))

    # Clone lock: structural near-mirror streak (fallback for the unmatched).
    if _is_near_mirror(obs):
        state["nmirror"] = int(state.get("nmirror", 0)) + 1
    else:
        state["nmirror"] = 0
    clone_lock = state["nmirror"] >= _RELAY_STREAK

    # OBSERVED-DUMP LOCK (2026-09-01): the online model watches every
    # opponent's dump schedule from turn 0, but action was gated to family
    # identification (3 classes) or near-mirrors -- so vs the mid-field the
    # relay observed everything and did NOTHING (the 55%-zone audit). Once
    # a product has enough observed dumps, engage for THAT product against
    # ANY opponent; the phase predictor downstream already prefers the
    # online model.
    obs_products = tuple(
        k for k, v in (state.get("opp_dumps") or {{}}).items()
        if len(v) >= _RELAY_OBS_MIN + 1)  # +1: stricter bar to ACT vs any opponent than to outrank the family prior
    if their is None and not clone_lock and not obs_products:
        return action
    if step - int(state.get("relay_last", -10 ** 9)) < _RELAY_COOLDOWN:
        return action

    if their is not None:
        products = _RELAY_M2_PRODUCTS
    elif clone_lock:
        products = _RELAY_PRODUCTS
    else:
        products = obs_products
    future = []
    for t in range(step + 1, min(step + 1 + _RELAY_HORIZON, len(_rt(state)))):
        turn = _rt(state)[t]
        if not isinstance(turn, dict):
            continue
        for order in turn.get("market") or []:
            if (isinstance(order, list) and len(order) >= 3
                    and order[0] == "SELL" and order[1] in products):
                try:
                    qty = max(0, int(order[2]))
                except (TypeError, ValueError):
                    continue
                if qty < _RELAY_MIN_QTY:
                    continue
                # When does the first dump land if we do nothing? Priority:
                # the ONLINE model of this opponent (M2-RL), then the static
                # family schedule (M2), then the clone assumption.
                ev = (state.get("opp_dumps") or {{}}).get(order[1]) or []
                nn_hit = None
                if _NN_OUTRANK and their is not None:
                    for e2 in their:
                        if (len(e2) >= 3 and e2[1] == order[1]
                                and abs(e2[0] - t) <= _RELAY_COLLIDE
                                and (nn_hit is None
                                     or abs(e2[0] - t) < abs(nn_hit - t))):
                            nn_hit = e2[0]
                if _NN_OUTRANK and their is not None:
                    # Retrieval outranks the online phase model: no collision
                    # in the retrieved schedule means no race at this dump,
                    # exactly as an identified schedule already implies.
                    if nn_hit is not None:
                        first = min(nn_hit, t)
                    elif clone_lock:
                        first = t
                    else:
                        continue
                elif len(ev) >= _RELAY_OBS_MIN:
                    ph = {{}}
                    for s in ev[-6:]:
                        ph[s % 24] = ph.get(s % 24, 0) + 1
                    phase = max(ph, key=lambda k: ph[k])
                    t_pred = step + ((phase - step) % 24)
                    if abs(t_pred - t) <= _RELAY_COLLIDE:
                        first = min(t_pred, t)
                    elif clone_lock:
                        first = t
                    else:
                        continue           # observed: no race at this dump
                elif their is not None:
                    # v2 schedules may extend events with [support, t_iqr,
                    # q_iqr]; race only TIGHT consensus (members agree within
                    # 3 turns, or decisive weighted support) -- a loose event
                    # is a guess, and racing a guess distorts our schedule
                    # for nothing. Loose events defer to live observation.
                    coll = []
                    for ev in their:
                        if len(ev) < 3 or ev[1] != order[1]:
                            continue
                        if len(ev) >= 6 and ev[4] > 3 and ev[3] < 0.55:
                            continue       # loose: observe, don't race
                        if abs(ev[0] - t) <= _RELAY_COLLIDE:
                            coll.append(ev[0])
                    if coll:
                        first = min(min(coll), t)
                    elif clone_lock:
                        first = t          # identified but also mirror-close
                    else:
                        continue           # identified: no race at this dump
                else:
                    first = t              # clone assumption
                lead_rl = int(state.get("relay_lead_rl", _RELAY_LEAD))
                win = lead_rl
                if _DUEL_POLICY and _DUEL_Q:
                    # Learned lead, from the same bucketing the policy was
                    # trained on. Falls back to the static/escalated lead when
                    # the state was never visited, rather than guessing.
                    pol = _duel_lead(t - step, len(ev), (first - t) if ev else None)
                    if pol is not None:
                        win = max(win, pol)
                if len(ev) >= _RELAY_OBS_MIN and first < t:
                    # Evidence says THEIR dump lands before our scheduled
                    # one: extend the engage window to cover the observed
                    # landing instead of hoping the global lead reaches it
                    # (distilled from the duel-lab Q-policy 2026-08-12:
                    # advance to just beat the observed timing, bounded).
                    win = max(lead_rl, min(_RELAY_LEAD_MAX, (t - first) + 1))
                if first - step <= win:
                    future.append((t, order[1], qty))
    if not future:
        return action
    shed = _projected_shed(obs, action)
    existing = list(action.get("market") or [])
    already = {{o[1] for o in existing
               if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL"}}
    room = _MAX_ORDERS - len(existing)
    inventory = _get(_get(obs, "market", {{}}) or {{}}, "inventory", {{}}) or {{}}

    def _damage(item, qty):
        held = int(_get(inventory, item, 10000) or 0)
        return qty * max(0.0, _quote(item, held) - _quote(item, held + qty))

    new_orders = []
    for due_step, item, planned in sorted(
            future, key=lambda r: -_damage(r[1], r[2])):
        if room <= 0:
            break
        if item in already:
            continue
        held = int(_get(inventory, item, 0) or 0)
        if _quote(item, held) < _RELAY_MIN_QUOTE:
            continue           # market already crashed; the race pays nothing
        qty = min(int(planned * _RELAY_FRACTION), _RELAY_MAX,
                  max(0, int(shed.get(item, 0) or 0)))
        if qty <= 0:
            continue
        new_orders.append(["SELL", item, qty])
        slot = ledger.setdefault(due_step, {{}})
        slot[item] = slot.get(item, 0) + qty
        already.add(item)
        shed[item] = max(0, int(shed.get(item, 0) or 0) - qty)
        state.setdefault("relay_fired", {{}})[item] = step   # M3 race feedback
        room -= 1
    if not new_orders:
        return action
    action["market"] = (new_orders + existing)[:_MAX_ORDERS]
    state["relay_last"] = step
    return action


def _duel_lead(turns_to_ours, evidence_n, offset):
    """Lead in turns from the learned duel policy, or None if unseen.

    Bucketing is duel_lab.bucket_state verbatim -- a divergence here would look
    up a different state than the one trained, which is the same class of bug as
    aiming per-class dump schedules at the wrong class ids.
    """
    if turns_to_ours <= 2:
        tb = 0
    elif turns_to_ours <= 5:
        tb = 1
    elif turns_to_ours <= 10:
        tb = 2
    elif turns_to_ours <= 20:
        tb = 3
    else:
        tb = 4
    eb = min(int(evidence_n), 2)
    if offset is None:
        ob = 3
    elif offset <= -3:
        ob = 0
    elif offset <= 1:
        ob = 1
    else:
        ob = 2
    row = (_DUEL_Q.get("q") or {{}}).get("%d|%d|%d" % (tb, eb, ob))
    if not row:
        return None
    acts = _DUEL_Q.get("actions") or []
    if not acts or len(row) != len(acts):
        return None
    best = max(range(len(row)), key=lambda i: row[i])
    return int(acts[best])


def _impact_slots(obs, action):
    market = list(action.get("market") or [])
    rows = []
    inventory = _get(_get(obs, "market", {{}}) or {{}}, "inventory", {{}}) or {{}}
    prices = _get(_get(obs, "market", {{}}) or {{}}, "prices", {{}}) or {{}}
    for index, order in enumerate(market):
        if not (isinstance(order, list) and len(order) >= 3
                and order[0] == "SELL" and order[1] in _MARKET_PARAMS):
            continue
        item = order[1]
        try:
            qty = max(0, int(order[2]))
        except (TypeError, ValueError):
            qty = 0
        have = int(_get(inventory, item, 10000) or 0)
        now = float(_get(prices, item, _quote(item, have)) or 0)
        after = float(_quote(item, have + qty))
        rows.append((float(qty) * max(0.0, now - after), -index, list(order)))
    if len(rows) < 2:
        return action
    rows.sort(reverse=True)
    ranked = iter(row[2] for row in rows)
    action["market"] = [
        next(ranked) if (isinstance(o, list) and len(o) >= 3 and o[0] == "SELL"
                         and o[1] in _MARKET_PARAMS) else o
        for o in market
    ]
    return action


def _threat_first(action, state, step):
    # Measured -$1.8k vs out-of-library programs (2026-08-09) and the route
    # library is no longer embedded; permanently off. Kept as the record.
    who = None
    if who is None:
        return action
    threatened = set()
    if not threatened:
        return action
    market = list(action.get("market") or [])
    hot = [o for o in market if isinstance(o, list) and len(o) >= 3
           and o[0] == "SELL" and o[1] in threatened]
    if not hot:
        return action
    rest = [o for o in market if o not in hot]
    action["market"] = (hot + rest)[:_MAX_ORDERS]
    return action


def _weed_repair(obs, action, step, state):
    me = int(_get(obs, "player", 0) or 0)
    farms = _get(obs, "farms", []) or []
    farm = farms[me] if me < len(farms) else {{}}
    positions = [list(_get(farm, "farmer", [0, 0]) or [0, 0])]
    positions += [list(p) for p in (_get(farm, "hands", []) or [])]
    ops = [action.get("farmer")] + list(action.get("hands") or [])
    active = state.setdefault("weed", {{}})
    for actor in list(active):
        job = active[actor]
        age = step - job["start"]
        idx = int(actor)
        if age == 1:
            if idx < len(ops):
                ops[idx] = list(job["intended"])
        elif 2 <= age <= 1 + _WEED_CATCHUP:
            prev = _rt(state)[step - 1] if 0 < step - 1 < len(_rt(state)) else None
            if prev and idx < len(ops):
                prev_ops = [prev.get("farmer")] + list(prev.get("hands") or [])
                if idx < len(prev_ops) and isinstance(prev_ops[idx], list):
                    ops[idx] = list(prev_ops[idx])
        else:
            active.pop(actor, None)
    for i, op in enumerate(ops):
        if not (isinstance(op, list) and op):
            continue
        if op[0] not in ("PLANT", "BUILD_PASTURE", "BUILD_COOP"):
            continue
        if i >= len(positions):
            continue
        tile = _tile_at(farm, positions[i])
        if isinstance(tile, dict) and tile.get("kind") == "WEED":
            active[str(i)] = {{"start": step, "intended": list(op)}}
            ops[i] = ["DIG"]
    action["farmer"] = ops[0] if ops else ["PASS"]
    action["hands"] = ops[1:]
    return action


def _terminal_market(obs, action):
    shed = _projected_shed(obs, action)
    prices = _get(_get(obs, "market", {{}}) or {{}}, "prices", {{}}) or {{}}
    rows = []
    for index, item in enumerate(_SELLABLE):
        qty = max(0, int(shed.get(item, 0) or 0))
        if qty <= 0:
            continue
        rows.append((qty * max(1.0, float(prices.get(item, 1) or 1)), -index,
                     item, qty))
    rows.sort(reverse=True)
    action["market"] = [["SELL", item, qty] for _, _, item, qty in rows[:_MAX_ORDERS]]
    return action


def _px_trend(obs, step, state, item):
    """Fractional change in `item`'s price across the trailing window.

    Sampled every _TIMING_SAMPLE turns so the state stays a handful of floats
    per product. Returns None until a full window exists -- no window, no
    opinion, so the early game never defers on one observation.
    """
    prices = _get(_get(obs, "market", {{}}) or {{}}, "prices", {{}}) or {{}}
    hist = state.setdefault("px_hist", {{}})
    series = hist.setdefault(item, [])
    if step % _TIMING_SAMPLE == 0:
        now = float(_get(prices, item, 0) or 0)
        if now > 0:
            series.append((step, now))
            keep = (_TIMING_WINDOW // _TIMING_SAMPLE) + 1
            if len(series) > keep:
                del series[:len(series) - keep]
    if len(series) < 3 or step - series[0][0] < _TIMING_WINDOW - _TIMING_SAMPLE:
        return None
    old, new_ = series[0][1], series[-1][1]
    if old <= 0:
        return None
    return (new_ - old) / old


def _pull_sells(obs, action, step, state):
    """Always-on price/opponent-reactive sell ACCELERATION.

    Momentum-deferral (_market_timing) can only delay; nothing captured a
    live spike (wool at 242 while the tape sold milk into a 17-glut,
    2026-09-01 autopsy). This pulls a scheduled future SELL forward when
    (a) its live price clears _PULL_RATIO x base, or (b) the OPPONENT'S
    observed farm (herd/crops) says their volume for that product is
    coming -- sell into strength before they land. Quantity-conserved via
    a due-ledger (the pulled quantity is cancelled when its original turn
    arrives); slot-aware; endgame untouched."""
    if not _PULL or step < 48 or step >= 640 or step + 1 >= len(_rt(state)):
        return action
    ledger = state.setdefault("pull_due", {{}})
    _net_due(ledger, step, action)
    market = list(action.get("market") or [])
    if len(market) >= _MAX_ORDERS - 1:
        return action
    mkt = _get(obs, "market", {{}}) or {{}}
    prices = _get(mkt, "prices", {{}}) or {{}}
    private = _get(obs, "private", {{}}) or {{}}
    shed = _get(private, "shed", {{}}) or {{}}
    # contested set from the opponent's OBSERVED farm portfolio
    contested = set()
    fs = state.get("farm") or {{}}
    herd = fs.get("herd") or [0, 0, 0]
    try:
        if int(herd[1]) >= 5:
            contested.add("MILK")
        if int(herd[2]) >= 4:
            contested.add("WOOL")
        if int(herd[0]) >= 4:
            contested.add("EGG")
    except (TypeError, ValueError, IndexError):
        pass
    fired = state.get("relay_fired") or {{}}
    pulled = 0
    for item, base in _PULL_BASE.items():
        if pulled >= _PULL_MAX_PER_TURN or len(market) >= _MAX_ORDERS - 1:
            break
        if item in fired and step - int(fired[item]) <= _PULL_WINDOW:
            continue        # relay racing it RECENTLY; not a lifetime ban
        try:
            pnow = float(prices.get(item) or 0)
        except (TypeError, ValueError):
            continue
        ratio = _PULL_RATIO_CONTESTED if item in contested else _PULL_RATIO
        if pnow < ratio * base:
            continue
        if int(shed.get(item, 0) or 0) < 1:
            continue
        if any(isinstance(o, list) and len(o) >= 3 and o[0] == "SELL"
               and o[1] == item for o in market):
            continue                    # already selling it this turn
        for t in range(step + 1, min(step + 1 + _PULL_WINDOW, len(_rt(state)))):
            hit = None
            for o in (_rt(state)[t].get("market") or []):
                if (isinstance(o, list) and len(o) >= 3 and o[0] == "SELL"
                        and o[1] == item):
                    try:
                        q = max(0, int(o[2]))
                    except (TypeError, ValueError):
                        q = 0
                    if q >= 2:
                        hit = (t, q)
                    break
            if hit:
                t2, q = hit
                already = ledger.get(t2, {{}}).get(item, 0)
                if already >= q:
                    break               # this scheduled sell is already pulled
                q = min(q - already, int(shed.get(item, 0) or 0))
                if q < 2:
                    break
                market.append(["SELL", item, q])
                ledger.setdefault(t2, {{}})[item] =                     ledger.get(t2, {{}}).get(item, 0) + q
                pulled += 1
                _dbglog("PULL step=%d item=%s qty=%d price=%.0f from_t=%d"
                        % (step, item, q, pnow, t2))
                break
    if pulled:
        action["market"] = market[:_MAX_ORDERS]
    return action


def _tomato_read(obs, action, step, state):
    """Day-16 TOMATO demand read (rank 2's third decision point).

    At step 385 the one-turn drop in the town's TOMATO inventory is read.
    Baseline drain is exactly 1/turn (the town centre) and each unlocked
    shop that lists the product adds 1, so a tick >= 3 means two shops are
    buying tomato -- demand the field serves with nothing (median TOMATO
    sold across the index is 0) into a 1.32.7 hinge price.

    Deliberately NOT a tape swap. The day-6 herd branch measured WORSE for
    us because substituting 648 turns of someone else's schedule costs more
    economy than the read buys. This layer changes only:

      * PLANT WHEAT -> PLANT TOMATO, on plantings the base already
        schedules after the read, up to _TOMATO_SEEDS tiles. Same tile,
        same hand, same watering the tape already books for that tile, so
        no plant is left unwatered on its planting day.
      * the seed purchase those plantings need (one order, once).
      * PICKUP WHEAT -> PICKUP TOMATO, and ONLY on a tile this layer
        re-cropped. The tape harvests onto the tile and collects by NAME,
        so without this the re-cropped yield would sit on the ground.
      * one SELL TOMATO while the shed holds any. SELL partially fills per
        unit, so an oversized order is harmless -- it costs a queue slot.

    Off (the default) this returns the action untouched.
    """
    if not _TOMATO_ON:
        return action
    inv = _get(_get(obs, "market", {{}}) or {{}}, "inventory", {{}}) or {{}}
    tom = int(inv.get("TOMATO") or 0)
    if step == _TOMATO_STEP - 1:
        state["_tom_pre"] = tom
        return action
    if step == _TOMATO_STEP and "_tom_pre" in state:
        state["tom_tick"] = tick = state.pop("_tom_pre") - tom
        state["tom_on"] = tick >= _TOMATO_TICK
        if state["tom_on"]:
            state["tom_left"] = _TOMATO_SEEDS
            state["tom_tiles"] = []
            market = list(action.get("market") or [])
            if len(market) < _MAX_ORDERS:
                market.append(["BUY_SEED", "TOMATO", _TOMATO_SEEDS])
                action["market"] = market
    if not state.get("tom_on"):
        return action
    try:
        me = 1 if int(_get(obs, "player", 0) or 0) == 1 else 0
        farm = (_get(obs, "farms", []) or [])[me]
    except (IndexError, TypeError, ValueError):
        return action
    pos = ([list(_get(farm, "farmer", []) or [])]
           + [list(p or []) for p in (_get(farm, "hands", []) or [])])
    ops = [action.get("farmer")] + list(action.get("hands") or [])
    tiles = state.setdefault("tom_tiles", [])
    changed = False
    for i, o in enumerate(ops):
        if not (isinstance(o, list) and len(o) >= 2):
            continue
        at = pos[i] if i < len(pos) else None
        if not at:
            continue
        if (o[0] == "PLANT" and o[1] == "WHEAT"
                and state.get("tom_left", 0) > 0):
            ops[i] = ["PLANT", "TOMATO"]
            state["tom_left"] -= 1
            if at not in tiles:
                tiles.append(at)
            changed = True
        elif o[0] == "PICKUP" and o[1] == "WHEAT" and at in tiles:
            ops[i] = ["PICKUP", "TOMATO"] + list(o[2:])
            changed = True
    if changed:
        action["farmer"] = ops[0]
        action["hands"] = ops[1:]
    shed = _get(_get(obs, "private", {{}}) or {{}}, "shed", {{}}) or {{}}
    have = int(shed.get("TOMATO") or 0)
    if have > 0:
        market = list(action.get("market") or [])
        if (len(market) < _MAX_ORDERS
                and not any(isinstance(o, list) and len(o) >= 2
                            and o[0] == "SELL" and o[1] == "TOMATO"
                            for o in market)):
            market.append(["SELL", "TOMATO", have])
            action["market"] = market
    return action


def _am_count(farm, kind):
    total = 0
    for row in (_get(farm, "tiles", []) or []):
        for tile in row or []:
            if not isinstance(tile, dict):
                continue
            if (kind == "STRAWBERRY" and tile.get("kind") == "PLANT"
                    and tile.get("crop") == kind):
                total += 1
            elif kind in ("COW", "SHEEP") and tile.get("animal") == kind:
                total += 1
    return total


def _am_capacity(obs, item):
    """Opponent minus own producer count -- public: obs["farms"] carries BOTH
    farms' tiles (only `private` is per-seat), so this needs no prediction."""
    farms = list(_get(obs, "farms", []) or [])
    if len(farms) < 2:
        return 0
    seat = 1 if int(_get(obs, "player", 0) or 0) == 1 else 0
    kind = {{"STRAWBERRY": "STRAWBERRY", "MILK": "COW", "WOOL": "SHEEP"}}[item]
    return _am_count(farms[1 - seat], kind) - _am_count(farms[seat], kind)


def _am_near_mirror(obs):
    farms = list(_get(obs, "farms", []) or [])
    if len(farms) < 2:
        return False
    seat = 1 if int(_get(obs, "player", 0) or 0) == 1 else 0
    def macro(f):
        return (_am_count(f, "COW"), _am_count(f, "SHEEP"),
                _am_count(f, "STRAWBERRY"),
                len(_get(f, "hands", []) or []),
                len(_get(f, "unlocked_quadrants", []) or []),
                float(_get(f, "money", 0) or 0))
    a, b = macro(farms[seat]), macro(farms[1 - seat])
    comp = sum(abs(a[i] - b[i]) for i in range(3))
    return (comp <= int(_AM_CONFIG["mirror_composition_distance"])
            and a[3] == b[3] and a[4] == b[4]
            and abs(a[5] - b[5]) <= float(_AM_CONFIG["mirror_money_distance"]))


def _am_town_demand(shops, item, step):
    demand = 1 if item != "FERTILIZER" and step % 24 == 0 else 0
    if step % 4 != 0:
        return demand
    for shop in shops:
        products = _SHOPS.get(shop, ())
        if item in products:
            demand += 2 if len(products) == 1 else 1
    return demand


def _am_shop_units(shops, item):
    units = 0
    for shop in shops:
        products = _SHOPS.get(shop, ())
        if item in products:
            units += 2 if len(products) == 1 else 1
    return units


def _am_reserve(step, item, total_shed):
    if step < 528:
        reserve = _AM_CONFIG["reserve_early"]
    elif step < 600:
        reserve = _AM_CONFIG["reserve_mid"]
    elif step < 648:
        reserve = _AM_CONFIG["reserve_late"]
    elif step < 684:
        reserve = _AM_CONFIG["reserve_tail"]
    elif step < 704:
        reserve = 2
    else:
        reserve = 0
    if total_shed >= _AM_CONFIG["shed_hard"]:
        reserve = 0
    elif total_shed >= _AM_CONFIG["shed_soft"]:
        reserve = max(0, reserve - _AM_CONFIG["shed_reserve_cut"])
    return reserve


def _am_pickup_reserve(action, item):
    n = 0
    for op in [action.get("farmer")] + list(action.get("hands") or []):
        if (isinstance(op, (list, tuple)) and len(op) >= 2
                and op[0] == "PICKUP" and op[1] == item):
            try:
                n += max(0, int(op[2])) if len(op) >= 3 else 1
            except (TypeError, ValueError):
                n += 1
    return n


def _am_existing_sell(action, item):
    return sum(max(0, int(o[2] or 0))
               for o in (action.get("market") or [])
               if isinstance(o, list) and len(o) >= 3
               and o[0] == "SELL" and o[1] == item)


def _am_late_mode(obs, state, step, total_shed, prices):
    """lynnsakurai's protect/convert gear, re-evaluated ONLY at 72-turn epoch
    boundaries from day 18. Returns "balanced" whenever _AM_LATE is off, so
    the two flags are independently gateable."""
    if not _AM_LATE:
        return "balanced"
    first = int(_AM_CONFIG["late_first_step"])
    width = max(1, int(_AM_CONFIG["late_epoch_turns"]))
    epoch = first + ((step - first) // width) * width if step >= first else -1
    if epoch < 0 or int(state.get("am_epoch", -1)) == epoch:
        return str(state.get("am_mode", "balanced"))
    mode = "balanced"
    if (not bool(state.get("am_mirror", False))
            and total_shed < _AM_CONFIG["shed_hard"]):
        farms = list(_get(obs, "farms", []) or [])
        seat = 1 if int(_get(obs, "player", 0) or 0) == 1 else 0
        gap = 0.0
        if len(farms) >= 2:
            gap = (float(_get(farms[seat], "money", 0) or 0)
                   - float(_get(farms[1 - seat], "money", 0) or 0))
        ratios = []
        for item in _AM_ITEMS:
            v = _get(prices, item, None)
            p = _AM_BASE_PRICE[item] if v is None else max(0.0, float(v or 0))
            ratios.append(p / _AM_BASE_PRICE[item])
        price_index = sum(ratios) / len(ratios)
        cap = sum(_am_capacity(obs, i) for i in _AM_ITEMS)
        bar = float(_AM_CONFIG["late_cash_gap"])
        if (gap >= bar
                and price_index <= float(_AM_CONFIG["late_protect_price_index"])
                and total_shed < _AM_CONFIG["shed_soft"] and cap <= 0):
            mode = "protect"
        elif (gap <= -bar
              and price_index >= float(_AM_CONFIG["late_convert_price_index"])
              and cap >= 0):
            mode = "convert"
    state["am_mode"] = mode
    state["am_epoch"] = epoch
    return mode


def _adaptive_market(obs, action, step, state):
    """boatlee V29-R1 `_adaptive_market` (see the _AM_CONFIG note above).

    Off (the default) this returns the action untouched and costs one
    attribute test per turn.
    """
    if not _AM_ON:
        return action
    try:
        market_ = _get(obs, "market", {{}}) or {{}}
        prices = _get(market_, "prices", {{}}) or {{}}
        inventory = _get(market_, "inventory", {{}}) or {{}}
        shed = _get(_get(obs, "private", {{}}) or {{}}, "shed", {{}}) or {{}}
        shops = tuple(_get(_get(obs, "town", {{}}) or {{}},
                           "unlocked_shops", []) or [])
        total_shed = sum(max(0, int(v or 0)) for v in dict(shed).values())

        # public-flow pressure: the town's inventory move that our own sells
        # and the town's own consumption do NOT explain is the OPPONENT.
        press = state.setdefault("am_press", {{}})
        if int(state.get("am_last", -1)) == step - 1:
            prev_inv = state.get("am_inv", {{}})
            prev_sold = state.get("am_sold", {{}})
            prev_shops = state.get("am_shops", ())
            for item in _AM_ITEMS:
                delta = (int(_get(inventory, item, 0) or 0)
                         - int(prev_inv.get(item, 0) or 0))
                ext = (delta + _am_town_demand(prev_shops, item, step - 1)
                       - int(prev_sold.get(item, 0) or 0))
                old = float(press.get(item, 0.0) or 0.0)
                press[item] = max(0.0, old * 0.72
                                  + min(24.0, max(0.0, float(ext))))
        state["am_last"] = step
        if state.get("am_mirror") is None or "am_mirror" not in state:
            if step >= int(_AM_CONFIG["mirror_latch_step"]):
                state["am_mirror"] = bool(_am_near_mirror(obs))

        mode = _am_late_mode(obs, state, step, total_shed, prices)
        added = state.setdefault("am_added", {{}})

        if step >= _AM_CONFIG["start_step"]:
            for item in _AM_ITEMS:
                stock = max(0, int(_get(shed, item, 0) or 0))
                pickup = _am_pickup_reserve(action, item)
                sched = min(max(0, stock - pickup),
                            _am_existing_sell(action, item))
                unscheduled = max(0, stock - pickup - sched)
                reserve = _am_reserve(step, item, total_shed)
                shop_demand = _am_shop_units(shops, item)
                if step < 684:
                    reserve += min(int(_AM_CONFIG["demand_reserve_cap"]),
                                   shop_demand * int(_AM_CONFIG["demand_reserve"]))
                urgent = (total_shed >= _AM_CONFIG["shed_hard"] or step >= 704)
                if not urgent and mode == "protect" and step < 648:
                    reserve += int(_AM_CONFIG["late_reserve_shift"])
                elif not urgent and mode == "convert":
                    reserve = max(0, reserve
                                  - int(_AM_CONFIG["late_reserve_shift"]))
                excess = max(0, unscheduled - reserve)
                if excess <= 0:
                    continue
                if state.get("am_mirror"):
                    cap = int(_AM_CONFIG["mirror_urgent_max_extra_per_item"
                                         if urgent
                                         else "mirror_max_extra_per_item"])
                else:
                    cap = int(_AM_CONFIG["max_extra_per_item"])
                budget = max(0, cap - int(added.get(item, 0) or 0))
                if budget <= 0:
                    continue
                ratio = (float(_get(prices, item, 0) or 0)
                         / _AM_BASE_PRICE[item])
                gate = float(_AM_CONFIG["price_gate"])
                if step < 684:
                    gate += min(float(_AM_CONFIG["demand_gate_cap"]),
                                shop_demand * float(_AM_CONFIG["demand_gate"]))
                if step >= 648:
                    gate -= 0.12
                if step >= 684:
                    gate -= 0.18
                if float(press.get(item, 0.0) or 0.0) >= _AM_CONFIG["pressure_trigger"]:
                    gate -= _AM_CONFIG["pressure_gate_cut"]
                if _am_capacity(obs, item) >= _AM_CONFIG["capacity_trigger"]:
                    gate -= _AM_CONFIG["capacity_gate_cut"]
                if not urgent and mode == "protect" and step < 648:
                    gate += float(_AM_CONFIG["late_gate_shift"])
                elif not urgent and mode == "convert":
                    gate -= float(_AM_CONFIG["late_gate_shift"])
                if not urgent and ratio < max(0.20, gate):
                    continue
                tranche = int(_AM_CONFIG["tranche"])
                if float(press.get(item, 0.0) or 0.0) >= _AM_CONFIG["pressure_trigger"]:
                    tranche += int(_AM_CONFIG["pressure_tranche"])
                if total_shed >= _AM_CONFIG["shed_soft"]:
                    tranche += int(_AM_CONFIG["shed_tranche"])
                if step >= 684:
                    tranche += int(_AM_CONFIG["tail_tranche"])
                if not urgent and mode == "convert":
                    tranche += int(_AM_CONFIG["late_tranche_shift"])
                qty = min(excess, max(1, tranche), budget)
                if qty <= 0:
                    continue
                orders = [list(o) for o in (action.get("market") or [])]
                hit = None
                for o in orders:
                    if (len(o) >= 3 and o[0] == "SELL" and o[1] == item):
                        hit = o
                        break
                if hit is not None:
                    hit[2] = max(0, int(hit[2] or 0)) + qty
                elif len(orders) < _MAX_ORDERS:
                    # premium cash ahead of hires and purchases; the relative
                    # order among the tape's own orders is untouched
                    orders.insert(0, ["SELL", item, qty])
                else:
                    continue        # never displace an order the tape queued
                action["market"] = orders[:_MAX_ORDERS]
                added[item] = int(added.get(item, 0) or 0) + qty

        state["am_inv"] = {{i: int(_get(inventory, i, 0) or 0)
                           for i in _AM_ITEMS}}
        state["am_sold"] = {{
            i: min(max(0, int(_get(shed, i, 0) or 0)),
                   _am_existing_sell(action, i)) for i in _AM_ITEMS}}
        state["am_shops"] = shops
    except Exception:                                          # noqa: BLE001
        pass
    return action


def _premium_lead(obs, action, step, state):
    """One-turn conservation lead + deposit advance (ported from the tape
    shell, 2026-08-30). Repays its own ledger first, every turn."""
    ledger = state.setdefault("plead_due", {{}})
    _net_due(ledger, step, action)
    if not _PLEAD or step < _PLEAD_START or step >= _PLEAD_STOP:
        return action
    if step + 1 >= len(_rt(state)) or step % 24 == 0 or step % 4 == 0:
        return action
    nxt = _rt(state)[step + 1]
    if not isinstance(nxt, dict):
        return action
    existing = list(action.get("market") or [])
    room = _MAX_ORDERS - len(existing)
    if room <= 0:
        return action
    already = {{o[1] for o in existing
               if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL"}}
    shed = _projected_shed(obs, action)
    inventory = _get(_get(obs, "market", {{}}) or {{}}, "inventory", {{}}) or {{}}
    owed_next = ledger.get(step + 1, {{}})
    advanced = False

    def _advance_deposit(item):
        nonlocal advanced
        if advanced:
            return
        me = int(_get(obs, "player", 0) or 0)
        farms = _get(obs, "farms", []) or []
        farm = farms[me] if me < len(farms) else {{}}
        size = len(_get(farm, "tiles", []) or []) or 10
        positions = [list(_get(farm, "farmer", [0, 0]) or [0, 0])]
        positions += [list(x) for x in (_get(farm, "hands", []) or [])]
        carried = _get(_get(obs, "private", {{}}) or {{}}, "inventories", []) or []
        ops = [action.get("farmer")] + list(action.get("hands") or [])
        for i, op in enumerate(ops):
            if not (isinstance(op, list) and op and op[0] == "PASS"):
                continue
            if i >= len(positions) or not _shed_adjacent(positions[i], size):
                continue
            inv2 = carried[i] if i < len(carried) else {{}}
            if int(_get(inv2 or {{}}, item, 0) or 0) <= 0:
                continue
            if i == 0:
                action["farmer"] = ["DROP"]
            else:
                hands = list(action.get("hands") or [])
                hands[i - 1] = ["DROP"]
                action["hands"] = hands
            advanced = True
            for what, count in (inv2 or {{}}).items():
                shed[what] = int(shed.get(what, 0) or 0) + int(count or 0)
            return

    new_orders = []
    for order in nxt.get("market") or []:
        if room <= 0:
            break
        if not (isinstance(order, list) and len(order) >= 3
                and order[0] == "SELL" and order[1] in _PLEAD_ITEMS):
            continue
        item = order[1]
        if item in already:
            continue
        try:
            planned = max(0, int(order[2])) - int(owed_next.get(item, 0) or 0)
        except (TypeError, ValueError):
            continue
        if planned <= 0:
            continue
        held = int(_get(inventory, item, 0) or 0)
        if _quote(item, held) < _PLEAD_MIN_QUOTE:
            continue
        if int(shed.get(item, 0) or 0) <= 0:
            _advance_deposit(item)
        qty = min(planned, _PLEAD_CAP, max(0, int(shed.get(item, 0) or 0)))
        if qty <= 0:
            continue
        new_orders.append(["SELL", item, qty])
        slot = ledger.setdefault(step + 1, {{}})
        slot[item] = slot.get(item, 0) + qty
        already.add(item)
        shed[item] = max(0, int(shed.get(item, 0) or 0) - qty)
        room -= 1
    if new_orders:
        action["market"] = (new_orders + existing)[:_MAX_ORDERS]
    return action


def _market_timing(obs, action, step, state):
    """Defer a scheduled SELL only while its own price is still climbing.

    Momentum, not mean reversion: the 22-replay measurement shows prices
    TREND (FERTILIZER 95 -> 10, WHEAT 32 -> 45) and do not recover after a
    dump, so the only sound reason to wait is that the price is going up right
    now. Releases the moment the climb stops, the horizon expires, the shed
    fills, cash runs short, or the endgame nears.

    Every guard the old layer lacked is here, because a deferral that starves
    the route is far more expensive than the ~2.5% it is chasing:
      * cash floor read from the RIGHT key, so the day-6 expansion (money
        drops to ~$45) can never be starved by a held sell;
      * shed room, since _SHED_CAP is 100 and a full shed makes DROP a silent
        no-op, which would jam the whole harvest chain;
      * order room computed BEFORE inserting releases, so a release can never
        evict a route BUY off the 10-order queue (the old release path did
        exactly that);
      * relay-raced products skipped -- racing IS selling early on purpose.
    """
    if not _TIMING:
        return action
    market_obs = _get(obs, "market", {{}}) or {{}}
    held = state.setdefault("timing_held", {{}})

    # Keep every eligible product's trend fresh, whether or not we sell it
    # this turn -- otherwise the window only fills on sell turns.
    trends = {{}}
    for item in _TIMING_PRODUCTS:
        trends[item] = _px_trend(obs, step, state, item)

    me = int(_get(obs, "player", 0) or 0)
    farms = _get(obs, "farms", []) or []
    farm = farms[me] if me < len(farms) else {{}}
    money = float(_get(farm, "money", 0) or 0)
    private = _get(obs, "private", {{}}) or {{}}
    shed_now = dict(_get(private, "shed", {{}}) or {{}})
    occupied = sum(int(v or 0) for v in shed_now.values())
    deferred_units = sum(int(r.get("qty") or 0) for r in held.values())

    # 1. releases: anything whose reason to wait has expired.
    release = []
    if held:
        projected = _projected_shed(obs, action)
        for item, rec in list(held.items()):
            tr = trends.get(item)
            expired = step - int(rec.get("since") or step) >= _TIMING_MAX_DEFER
            stalled = tr is None or tr <= 0.0
            pressed = occupied >= _SHED_CAP - _TIMING_SHED_MARGIN
            broke = money < _TIMING_CASH_FLOOR
            if not (expired or stalled or pressed or broke
                    or step + 2 >= _TIMING_END):
                continue
            qty = min(int(rec.get("qty") or 0),
                      max(0, int(projected.get(item, 0) or 0)))
            if qty > 0:
                release.append(["SELL", item, qty])
            del held[item]

    market = list(action.get("market") or [])

    # 2. capture: pull this turn's SELLs whose price is still climbing.
    out = []
    racing = set((state.get("relay_fired") or {{}}).keys())
    for slot in (state.get("relay_due") or {{}}).values():
        racing |= set(slot)
    for o in market:
        if not (isinstance(o, list) and len(o) >= 3 and o[0] == "SELL"
                and o[1] in _TIMING_PRODUCTS and o[1] not in racing):
            out.append(o)
            continue
        item = o[1]
        try:
            qty = max(0, int(o[2]))
        except (TypeError, ValueError):
            qty = 0
        if qty <= 0 or step >= _TIMING_END:
            out.append(o)
            continue
        tr = trends.get(item)
        if tr is None or tr < _TIMING_MIN_RISE:
            out.append(o)                      # not climbing: sell now
            continue
        if money < _TIMING_CASH_FLOOR:
            out.append(o)                      # cash-poor: never defer
            continue
        if occupied + qty > _SHED_CAP - _TIMING_SHED_MARGIN:
            out.append(o)                      # no shed room to hold it
            continue
        if deferred_units + qty > _TIMING_MAX_UNITS:
            out.append(o)                      # already holding enough
            continue
        if item in held:
            held[item]["qty"] = int(held[item]["qty"]) + qty
            deferred_units += qty
            continue                           # merge into the existing hold
        if len(held) >= _TIMING_MAX_ITEMS:
            out.append(o)
            continue
        held[item] = {{"qty": qty, "since": step}}
        deferred_units += qty

    # 3. re-insert releases WITHOUT evicting anything already queued.
    if release:
        room = max(0, _MAX_ORDERS - len(out))
        action["market"] = release[:room] + out
        for o in release[room:]:               # no room: keep holding, retry
            item = o[1]
            rec = held.setdefault(item, {{"qty": 0, "since": step}})
            rec["qty"] = int(rec.get("qty") or 0) + int(o[2])
    else:
        action["market"] = out
    return action


_POLICY_DELTA_CAP = 4      # units/product/turn the policy may add or remove


def _policy_vec(obs):
    """The 49-field state EXACTLY as src/turn_features.py captured it for
    training -- including its quirks (animals counted on PASTURE kind only),
    because the net learned the training extractor's world, and a 'fixed'
    runtime extractor would be train/serve skew wearing a virtuous face."""
    farms = _get(obs, "farms", []) or []
    if len(farms) < 2:
        return None
    market = _get(obs, "market", {{}}) or {{}}
    prices = _get(market, "prices", {{}}) or {{}}
    inv = _get(market, "inventory", {{}}) or {{}}
    priv = _get(obs, "private", {{}}) or {{}}
    shed = _get(priv, "shed", {{}}) or {{}}
    seeds = _get(priv, "seeds", {{}}) or {{}}
    me = 1 if int(_get(obs, "player", 0) or 0) == 1 else 0

    def fstats(farm):
        crops = animals = dry = starving = 0
        for row in (_get(farm, "tiles", []) or []):
            for t in (row if isinstance(row, list) else [row]):
                if not isinstance(t, dict):
                    continue
                if t.get("kind") == "PLANT" and t.get("crop"):
                    crops += 1
                    if (t.get("consecutive_unwatered") or 0) >= 1:
                        dry += 1
                elif t.get("kind") == "PASTURE" and t.get("animal"):
                    animals += 1
                    if (t.get("consecutive_unfed") or 0) >= 1:
                        starving += 1
        return [float(_get(farm, "money", 0) or 0) / 1e5,
                len(_get(farm, "hands", []) or []) / 20.0,
                len(_get(farm, "unlocked_quadrants", []) or []) / 4.0,
                crops / 100.0, animals / 100.0, dry / 100.0, starving / 100.0]

    products = _POLICY["products"]
    v = [float(_get(obs, "day", 0) or 0) / 30.0,
         float(_get(obs, "hour", 0) or 0) / 24.0,
         float(_get(obs, "step", 0) or 0) / 720.0]
    v += [float(_get(prices, p, 0) or 0) / _MARKET_PARAMS[p][0]
          for p in products]
    v += [float(_get(inv, p, 0) or 0) / 10000.0 for p in products]
    v += fstats(farms[me])
    v += fstats(farms[1 - me])
    v += [float(_get(shed, p, 0) or 0) / 100.0 for p in products]
    v += [float(_get(seeds, c, 0) or 0) / 20.0 for c in products[:5]]
    return v


def _policy_forward(x):
    """Pure-python forward of the exported BC net; equivalence to torch was
    asserted at export (worst |diff| 2.2e-06)."""
    P = _POLICY
    v = [(x[i] - P["x_mu"][i]) / P["x_sd"][i] for i in range(len(x))]
    W, B = P["W"], P["b"]
    for li in range(len(W)):
        out = []
        for o in range(len(B[li])):
            s = B[li][o]
            row = W[li][o]
            for i in range(len(v)):
                s += row[i] * v[i]
            out.append(s)
        v = [math.tanh(z) for z in out] if li < len(W) - 1 else out
    return [v[i] * P["a_sd"][i] + P["a_mu"][i] for i in range(len(v))]


def _policy_head(obs, action, step, state):
    """Bounded blend between the schedule and what a WINNING farm sells here.

    Return-conditioned: the net is asked for the sell vector of a game that
    ends at score 1.0. The delta against the scheduled sells is capped at
    +/-_POLICY_DELTA_CAP per product per turn, so a wrong prediction degrades
    the schedule smoothly instead of replacing it -- the discreteness that
    made arm COMMITS catastrophic is exactly what this head avoids. Runs
    BEFORE _safe_market, so additions clamp to the real shed and can never
    oversell."""
    try:
        vec = _policy_vec(obs)
        if vec is None:
            return action
        want = _policy_forward(vec + [1.0])
        market = list(action.get("market") or [])
        sched = {{}}
        for o in market:
            if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL":
                sched[o[1]] = sched.get(o[1], 0) + int(o[2] or 0)
        for pi, item in enumerate(_POLICY["products"]):
            w = max(0, int(round(want[pi])))
            d = w - sched.get(item, 0)
            d = max(-_POLICY_DELTA_CAP, min(_POLICY_DELTA_CAP, d))
            if d == 0:
                continue
            if d > 0:
                merged = False
                for o in market:
                    if (isinstance(o, list) and len(o) >= 3
                            and o[0] == "SELL" and o[1] == item):
                        o[2] = int(o[2]) + d
                        merged = True
                        break
                if not merged and len(market) < _MAX_ORDERS:
                    market.append(["SELL", item, d])
            else:
                need = -d
                for o in market:
                    if need <= 0:
                        break
                    if (isinstance(o, list) and len(o) >= 3
                            and o[0] == "SELL" and o[1] == item):
                        take = min(need, int(o[2] or 0))
                        o[2] = int(o[2]) - take
                        need -= take
                market = [o for o in market
                          if not (isinstance(o, list) and len(o) >= 3
                                  and o[0] == "SELL" and int(o[2] or 0) <= 0)]
        action["market"] = market[:_MAX_ORDERS]
    except Exception:                                          # noqa: BLE001
        pass
    return action


def _dbglog(msg):
    import os as _os
    path = _os.environ.get("KAGG_DEBUG_LOG")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(msg + chr(10))
    except OSError:
        pass


def agent(obs, config=None):
    try:
        seat = 1 if int(_get(obs, "player", 0) or 0) == 1 else 0
        step = int(_get(obs, "step", 0) or 0)
        state = _STATE[seat]
        if step == 0 or step < int(state.get("last", -1)):
            state = _STATE[seat] = {{"last": step}}
        state["last"] = step
        if _BRANCHES and "branch_done" not in state:
            _pick = None
            if _BRANCH_ON == "drain":
                # Day-6 DEMAND read. Every unlocked shop consumes one unit of
                # each product it lists every 4 steps, so the town's market
                # inventory drains by exactly 1 a turn at baseline and by >= 2
                # for a product some unlocked shop wants. The day-6 dawn tick
                # is the first turn carrying the extra draw: read at step 144
                # the rule is 69% accurate, at 145 it is 100% (Crop Dusta,
                # LB #2, reverse-engineered 134/134 on 2026-09-03 --
                # docs/history/cropdusta-adaptive-study-2026-09-03.md). Reading DEMAND
                # instead of the shop label is what makes it robust, and it is
                # causal: nothing before 145 reveals it.
                _inv = _get(_get(obs, "market", {{}}) or {{}},
                            "inventory", {{}}) or {{}}
                if step == _BRANCH_STEP - 1:
                    state["_pre"] = [int(_inv.get("WOOL") or 0),
                                     int(_inv.get("MILK") or 0)]
                elif step >= _BRANCH_STEP and "_pre" in state:
                    _pre = state.pop("_pre")
                    _dw = _pre[0] - int(_inv.get("WOOL") or 0)
                    _dm = _pre[1] - int(_inv.get("MILK") or 0)
                    state["branch_done"] = True
                    _pick = _BRANCHES.get(
                        "SHEEP" if _dw >= 2 else
                        ("COW" if _dm >= 2 else "GOOSE"))
            else:
                _town = _get(obs, "town", {{}}) or {{}}
                _shops = _get(_town, "unlocked_shops", []) or []
                if _shops:
                    state["branch_done"] = True
                    _pick = _BRANCHES.get(str(_shops[0]))
            if _pick is not None:
                # per-episode, NOT the module global: a swapped global
                # leaks into the next episode on serve process reuse
                state["route"] = _pick
        _route = state.get("route") or _ROUTE
        idx = min(max(0, step), len(_route) - 1)
        action = copy.deepcopy(_route[idx])

        arm = _bandit(obs, step, state)
        if arm is not None:
            override = _ARMS.get(arm, {{}}).get(str(step))
            if override is not None:
                action["market"] = copy.deepcopy(override)
                if not state.get("arm_ledgers_cleared"):
                    # repay ledgers were borrowed against the ROUTE's
                    # schedule, which this override replaces — netting
                    # them against the arm's sells would double-sell
                    state["arm_ledgers_cleared"] = True
                    for _lk in ("relay_due", "pull_due", "plead_due"):
                        state.pop(_lk, None)

        if _OBFUSCATE:
            # Episode-seeded deterministic jitter: hold each SELL 0..N turns.
            # Held orders re-enter on their release turn ahead of that turn's
            # own sells; _safe_market clamps everything to reality.
            if "obf_seed" not in state:
                state["obf_seed"] = (int(_get(_get(obs, "market", {{}}) or {{}},
                                              "prices", {{}}).get("WHEAT", 25))
                                     * 7919 + seat) % 104729
            pend = state.setdefault("obf_pending", {{}})
            market = []
            for o in (action.get("market") or []):
                if (isinstance(o, list) and len(o) >= 3 and o[0] == "SELL"
                        and step < 640):
                    h = (state["obf_seed"] * 31 + step * 7
                         + sum(ord(c) for c in str(o[1]))) % (_OBF_MAX_DELAY + 1)
                    if h:
                        pend.setdefault(str(step + h), []).append(list(o))
                        continue
                market.append(o)
            market = (pend.pop(str(step), []) + market)[:_MAX_ORDERS]
            action["market"] = market

        if _POLICY:
            action = _policy_head(obs, action, step, state)
        action = _tomato_read(obs, action, step, state)
        action = _align_hands(action, obs)
        action = _weed_repair(obs, action, step, state)
        action = _align_hands(action, obs)
        action = _feed_pull(obs, action, step, state)
        action = _safe_market(obs, action)
        action = _relay(obs, action, step, state)
        action = _pull_sells(obs, action, step, state)
        action = _premium_lead(obs, action, step, state)
        action = _adaptive_market(obs, action, step, state)
        action = _market_timing(obs, action, step, state)
        action = _sell_first(action)
        action = _impact_slots(obs, action)
        action = _threat_first(action, state, step)
        if step >= _LAST_STEP:
            action = _terminal_market(obs, action)
        state["our_last_sells"] = {{
            o[1]: int(o[2]) for o in (action.get("market") or [])
            if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL"
        }}
        return _align_hands(action, obs)
    except Exception as exc:                                   # noqa: BLE001
        # Fall back to the raw tape action, not PASS: a PASS on a planting
        # day skips mandatory same-day watering and compounds; an unclamped
        # SELL only partial-fills. PASS is the last resort.
        try:
            _step = int(_get(obs, "step", 0) or 0)
        except Exception:                                      # noqa: BLE001
            _step = -1
        _dbglog("AGENT-EXC step=" + str(_step) + " " + repr(exc)[:200])
        try:
            _idx = min(max(0, _step), len(_ROUTE) - 1)
            return _align_hands(copy.deepcopy(_ROUTE[_idx]), obs)
        except Exception:                                      # noqa: BLE001
            pass
        me = 0
        try:
            me = int(_get(obs, "player", 0) or 0)
            farms = _get(obs, "farms", []) or []
            hands = _get(farms[me], "hands", []) or []
        except Exception:                                      # noqa: BLE001
            hands = []
        return {{"farmer": ["PASS"], "hands": [["PASS"] for _ in hands],
                "market": []}}
'''


if __name__ == "__main__":
    main()
