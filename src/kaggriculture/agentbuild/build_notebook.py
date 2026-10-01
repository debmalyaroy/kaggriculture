"""Build the submission as a notebook: the agent, plus why it exists.

    python -m kaggriculture.agentbuild.build_notebook --agent agents/v17_route.py
    python -m kaggriculture.agentbuild.build_notebook --agent agents/v17_route.py --push
    python -m kaggriculture.agentbuild.build_notebook --agent agents/v17_route.py --push --submit

A bare `main.py` upload tells nobody anything, including us in a week. The
notebooks at the top of this leaderboard are write-ups first and artefacts
second, and that is not decoration -- it is how the mechanisms that actually
move the score get found, credited and re-tested. v22's price-impact ranking
reached us because its author published the ablation that isolated it, and we
reproduced their effect size to within a few hundred dollars *because* they
published the number rather than only the code.

So this generates a notebook that carries, in order:

1. the headline result and how it was measured, including what it lost;
2. a machine-readable contribution card -- upstream, provenance, what is new,
   what was tried and rejected -- so a fork can cite precisely;
3. EDA that motivates the mechanism rather than illustrating it after the fact:
   the official price curve, the freshness decay we measured, and the
   tournament that showed recorded bank is a bad selector;
4. the exact agent bytes, reconstructed and verified by SHA-256, and compiled;
5. provenance and credit for every borrowed mechanism.

`kaggle competitions submit -k <user>/<slug> -v <n>` is the documented notebook
submission path (AGENTS.md). This writes the kernel, pushes it, waits for the
run, and can then submit from it -- never without an explicit flag.
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import base64
import datetime as dt
import glob
import hashlib
import json
import os
import subprocess
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.data.episodes as E  # noqa: E402
import kaggriculture.data.registry as registry  # noqa: E402

OUT_DIR = os.path.join(ROOT, "notebooks", "submission")
SLUG = "kaggriculture-route-refresh-price-impact"
TITLE = "Kaggriculture | Route Refresh + Price Impact"


def chunk(text, width=100):
    return [text[i:i + width] for i in range(0, len(text), width)]


def md(*lines):
    return {"cell_type": "markdown", "metadata": {},
            "source": [l + "\n" for l in "\n".join(lines).split("\n")]}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": [l + "\n" for l in "\n".join(lines).split("\n")]}


def evidence(agent_name):
    """Pull the measured numbers out of the registry rather than retyping them."""
    card = registry.card(agent_name) or {}
    rows = []
    for e in card.get("evals") or []:
        rows.append({"opponent": os.path.basename(str(e.get("opponent", "?"))),
                     "win_rate": e.get("win_rate"), "margin": e.get("margin"),
                     "bank": e.get("bank"), "seeds": e.get("seeds"),
                     "note": e.get("note")})
    prov = card.get("route_provenance") or {}
    return card, rows, prov


def v22_cells():
    """EDA cells for the v22 build: measured losses by family, the trained
    counter-arms, and the in-game identification the bandit runs on.

    Everything is read from `models/v22/eda.json`, which the build pipeline
    writes from our own mined ladder games and the arm-training logs -- the
    numbers in these cells are measurements, not narrative. Silently returns
    nothing when the artifact is absent so older agents still build."""
    eda_path = os.path.join(ROOT, "models", "v22", "eda.json")
    if not os.path.exists(eda_path):
        return []
    eda = json.load(open(eda_path, encoding="utf-8"))
    cells = []

    fam = eda.get("loss_families") or {}
    lines = [
        "## 3c. EDA: who actually beats us, measured from our own games",
        "",
        "Every episode our submissions played was downloaded and each loss",
        "labelled with the nearest of the top-100 route *programs* (sales",
        "reconstructed from the shared market inventory, town consumption",
        "subtracted exactly). This is the evidence the adaptive design rests",
        "on: the ladder's losses are concentrated in a handful of identifiable",
        "programs, led by fresher editions of our own lineage.",
        "",
        "| submission | games | record | top loss families (count x team, total margin) |",
        "|---|---:|---|---|",
    ]
    for sub, r in fam.items():
        tops = "; ".join(f"{n}x {t} ({m:+,.0f})" for t, n, m in r.get("top", [])[:3])
        lines.append(f"| {sub} | {r.get('games')} | {r.get('record')} | {tops} |")
    lines += [
        "",
        "Three design decisions fall straight out of this table:",
        "",
        "1. **Per-turn move selection from the top-100 pool cannot help.** We",
        "   built it and measured it: with feed and inputs protected it is",
        "   *bit-identical* to the base route over 80 paired games (a frontier",
        "   route already sells everything the turn it lands); with them",
        "   unprotected it liquidates its own feed ($33k vs $171k). The value",
        "   is not in the turns -- it is in *which whole schedule* you play.",
        "2. **The counter must be per-family, chosen in-game.** No single",
        "   schedule beats every family (the rank-6 route beats Seb's family",
        "   100% and loses to ours), so the agent carries one trained",
        "   counter-schedule per major loss family.",
        "3. **Identification is reliable enough to bet on.** The same",
        "   reconstruction that labels this table runs live inside the agent.",
    ]
    cells.append(md(*lines))

    arms = eda.get("arms") or {}
    if arms:
        lines = [
            "### The trained counter-arms",
            "",
            "Each arm retimes the base route's own sells -- nothing created,",
            "nothing resized, every arm sharing the base's first 192 turns so",
            "switching is never a splice -- trained by evolutionary search",
            "(the policy-search form of RL) against tapes of the family it",
            "must beat, scored only by playing full episodes, with crowd",
            "guards penalising any regression:",
            "",
            "| arm | trained against | fitness gen0 -> best | market turns changed |",
            "|---|---|---|---:|",
        ]
        for name, a in arms.items():
            lines.append(f"| {name} | {a.get('vs')} | {a.get('gen0'):,.0f} -> "
                         f"**{a.get('best'):,.0f}** | {a.get('turns_changed')} |")
        lines += [
            "",
            "The in-game layer is a bandit over these arms: a posterior over",
            "program families updated every turn from the reconstruction",
            "residuals, committing to an arm only after two consecutive",
            "decisive reads (never before day 9), and abandoning it if the",
            "opponent later diverges from the matched program. Exploration",
            "cannot bankrupt the farm because every arm is a full schedule",
            "validated by played episodes before it ships.",
            "",
            "Not every trained arm shipped. The arm against our own lineage",
            "trained to +97k and did not generalise: half the ladder's",
            "programs look like that family mid-game, and the arm cost more",
            "against the lookalikes than it earned against the real thing.",
            "It was cut by paired validation, along with a threat-first",
            "reordering layer (-$1.8k against out-of-library programs) and a",
            "cheap-wheat purchase pull (no effect). What ships is exactly",
            "what measured >= the incumbent on every opponent: the",
            "identifier, the bandit, and the two arms that generalised.",
        ]
        cells.append(md(*lines))

    trace = eda.get("ident_trace") or []
    if trace:
        cells.append(md(
            "### Identification, demonstrated",
            "",
            "Match-distance ratio (best library program vs observed volume) by",
            "turn, from an instrumented episode against a live-loss opponent",
            "tape. The ratio collapses as evidence accumulates; the horizontal",
            "line is the decisiveness gate the bandit requires before acting.",
        ))
        cells.append(code(
            f"trace = {json.dumps(trace)}",
            "import pandas as pd",
            "s = pd.Series({t: r for t, r in trace})",
            "ax = s.plot(figsize=(8, 3), color='#2563eb')",
            "ax.axhline(0.35, color='#dc2626', ls='--', lw=1)",
            "ax.set(title='Opponent-program match ratio by turn (lower = surer)',",
            "       xlabel='turn', ylabel='distance / volume')",
            "plt.tight_layout()",
        ))
    return cells


def detailed_eda_cells():
    """PRIVATE appendix: our own broad fingerprint EDA, the field map, the
    obfuscation finding, and the latest local simulation. Rendered as static
    tables baked at build time so the notebook is self-contained (no private
    data attach needed). Never included in a public build."""
    cells = [md(
        "---", "",
        "# Appendix (PRIVATE): architecture, field map & obfuscation",
        "",
        "*This section is for our own record and is kept private. It names our",
        "opening clusters, our basin, and our identity exposure -- everything a",
        "competitor would need to reconstruct our approach. It must never be",
        "made public.*",
        "",
        "## A1. How the agent works, end to end",
        "",
        "A frozen stdlib-only runtime plays a **719-turn open-loop route** through",
        "a repair stack (weed-DIG, shed-clamp, sell-first ordering, price-impact",
        "SELL ranking, terminal liquidation). Over it sits a **Thompson bandit**",
        "that commits, only against a confidently *identified* counter-class, to a",
        "pre-trained counter-arm; an own-class guard and mismatch-abandon keep it",
        "bit-identical to the base route whenever it is unsure. The base route,",
        "counter-arms, the opponent **identifier** (7-class multinomial logistic",
        "over runtime-reconstructable prefix features), and the fire threshold are",
        "all trained offline and rendered into one `main.py` as compressed",
        "constants. Training changes the config; the runtime is frozen.",
    )]

    # A2. Broad opening-fingerprint field map (our own dataset)
    try:
        fp = json.load(open(os.path.join(ROOT, "data", "fingerprints",
                                         "summary.json"), encoding="utf-8"))
        rows = ["| # | routes | teams | win | median bank | public label | signature |",
                "|---|---|---|---|---|---|---|"]
        for c in fp["top_clusters"][:8]:
            rows.append(f"| {c['cluster']} | {c['routes']} | {c['teams']} | "
                        f"{c['win_rate']} | {c['median_bank']} | "
                        f"{c['public_label']} | {c['signature'][:48]} |")
        where = fp.get("where_we_sit", {}).get("v19_2_route", {})
        cells.append(md(
            "## A2. Our own broad opening-fingerprint dataset",
            "",
            f"`src/fingerprint_dataset.py` over **{fp['routes']} routes** (vs a",
            "public notebook's 15). First-48-turn BUILD signature, greedily",
            "clustered. The field map:",
            "",
            *rows,
            "",
            f"**Where we sit:** our base route is in cluster "
            f"{where.get('cluster','?')} (`{where.get('public_label','?')}`) -- "
            f"{where.get('saturation_note','?')}. That is the winningest basin,",
            "and we are identifiable *as a family*, not uniquely. Basin choice --",
            "not order-timing tricks -- is the only real identity lever, and the",
            "crown already selects for it.",
        ))
    except Exception:                                              # noqa: BLE001
        pass

    # A3. Obfuscation: measured dead end
    try:
        ob = json.load(open(os.path.join(ROOT, "models", "v22",
                                         "obfuscation_report.json"),
                            encoding="utf-8"))
        cost = ob.get("cost", {}) or {"sell_jitter+60": -133377,
                                      "seed_shuffle": -13254}
        rows = ["| transform | channel | id-conf | conf drop | engine cost |",
                "|---|---|---|---|---|"]
        chan = {"sell_jitter+2": "timing", "sell_jitter+30": "timing",
                "sell_jitter+60": "timing", "animal_swap": "farm-event",
                "seed_shuffle": "farm-event", "sellmix_shift": "sell-mix"}
        for t in ob["transforms"]:
            c = cost.get(t["transform"], "")
            cstr = f"${c:,}" if isinstance(c, (int, float)) else "-"
            rows.append(f"| {t['transform']} | {chan.get(t['transform'],'')} | "
                        f"{t['conf']} | {t['drop']} | {cstr} |")
        cells.append(md(
            "## A3. Signature obfuscation -- measured, and closed",
            "",
            f"Baseline: our base route is identified at "
            f"**{ob['base_conf']} confidence** (chance {ob['chance']}). We priced",
            "every candidate obfuscation on the engine:",
            "",
            *rows,
            "",
            "**Verdict:** no zero-cost runtime obfuscation exists. Sell-timing",
            "jitter provably cannot move the final-mix identity (it preserves the",
            "sell basket byte-for-byte); the only timing shift that moves our",
            "identifier costs **-$133k/episode**; and the farm-event edits a real",
            "detector keys on cannot change without farming differently. Identity",
            "is intrinsic to how we farm. `_OBFUSCATE` stays off permanently.",
        ))
    except Exception:                                              # noqa: BLE001
        pass

    # A4. Latest local simulation
    cells.append(md(
        "## A4. Latest local simulation (fixed panel, both seats)",
        "",
        "v22.2 (bandit) and v19.2 (base route), 4 seeds x 2 seats:",
        "",
        "| opponent | win% | margin |",
        "|---|---|---|",
        "| v18_route (prior) | 100% | +$39,148 |",
        "| v16_route (v23-fork-ish) | 100% | +$12,588 |",
        "| pass | 100% | +$153,443 |",
        "| each other | mirror | $0 |",
        "",
        "Both post an identical **$108,798** mean bank: the bandit is",
        "floor-neutral against this panel (arms fire only against identified",
        "counter-classes, none present here), confirming the floor property.",
    ))
    return cells


def family_doc_cells(kind):
    """PRIVATE: complete documentation for one agent family ('route' or
    'bandit') -- the step-by-step journey with every measured number, the
    architecture as shipped, the data/EDA state at build time, and the daily
    release runbook. Together with detailed_eda_cells() this makes each
    notebook the family's full documentation of record."""
    cells = [md(
        "---", "",
        f"# Part B (PRIVATE): the {kind} family, documented end to end",
    )]

    # ---- B1. The journey, step by step, measured at every fork ------------
    common = [
        "## B1. How we got here, step by step",
        "",
        "Every fork below was decided by a paired engine measurement, never by",
        "intuition. Differences under ~$3k/game are treated as noise; the",
        "ladder scores WINS, so tie-breaking beats margin-chasing.",
        "",
        "| # | Step | What we measured / learned |",
        "|---|------|---------------------------|",
        "| 1 | **Heuristic era (v0–v2)** | closed-loop job-valuation planner; tuned `PARAMS` via search. Ceiling: mid-ladder. |",
        "| 2 | **Route discovery (v14–v17)** | replaying a mined top route beats planning: the meta is an open-loop tape war. One medoid route per team; fit/validation/outer windows to avoid selecting on the test. |",
        "| 3 | **Freshness doctrine (v18–v19)** | a route is a perishable asset. Kaito's own ablation: stale medoid 19/46 vs current 40–41/46. Our v19 (THUNDER base + price-impact SELL ranking) hit 97.2% on the held-out top-100 panel. |",
        "| 4 | **Negative ablations (paid for, do not re-learn)** | glut guard, swap-advance, threat-first (−$1.8k), endgame pull, feed-pull: all measured ≤0. Freshness beats in-game cleverness. |",
        "| 5 | **Bandit era (v22)** | v19 base + 7-class opponent identifier + trained counter-arms + Thompson commit with own-class guard. Paired vs route: 124/208, zero regressions. Live peak 2809. |",
        "| 6 | **Fresh-crown automation (v19.x/v22.x)** | daily re-crown from newest data, refereed by tapes of our own live losses. v22.2/v19.2 reached 2788/2251 from provisional ~600 in 24h. |",
        "| 7 | **Basin map (2026-08-10)** | our own 2,562-route opening-fingerprint dataset: our base sits in `sheep_first_hybrid` — the winningest basin (75.5% win, 26 teams) vs the saturated `v23_fork` meta (117 teams, 35.3%). |",
        "| 8 | **Obfuscation closed (2026-08-10)** | no zero-cost identity hiding exists: timing jitter provably cannot move the sell-mix channel; the one transform that moves our identifier costs −$133,377/game. Defense = basin choice. |",
        "| 9 | **Relay era (2026-08-11)** | the public meta front-runs mirror dumps (boatlee 3-lead, 3046.4 public, 22 forks; shiv17 5-lead counter, 87.5% vs boatlee). Our v1 tie-break never fired (exact-equality incl. money). Relay v2: near-structural lock, FERTILIZER dumps, 2-turn lead, repayment ledger → **+202/game vs a clone, byte-identical outside**. Broad variant measured **−1,401** — narrow or nothing. |",
        "| 10 | **Model stack (2026-08-11)** | M1 learned detector = NULL (99% of pairs collide; identity IS the mirror model). M2 = per-class consensus dump schedules from the corpus. M2-RL/M3-RL = in-game learning (below). |",
    ]
    if kind == "bandit":
        common += [
            "| 11 | **v23.1_bandit (shipped)** | in-game RL: learns the opponent's actual dump schedule online + escalates the race lead when pre-empted. **+500/+339 vs relay-off clone; +263/+149 vs the fixed-lead relay (both seats); +37/+11 non-mirror.** |",
        ]
    else:
        common += [
            "| 11 | **v23_route (shipped)** | same proven base + relay v2 clone-lock. The route family is the redundancy arm: no identifier, no arms — a different failure mode from the bandit, same winning basin. |",
        ]
    cells.append(md(*common))

    # ---- B2. Architecture as shipped --------------------------------------
    if kind == "bandit":
        cells.append(md(
            "## B2. Architecture (bandit family), as shipped",
            "",
            "One self-contained `main.py`, stdlib only, frozen runtime +",
            "trained config rendered as compressed constants:",
            "",
            "```",
            "_ROUTE        719-turn crowned base route (daily re-crown)",
            "_IDMODEL      11-class multinomial logistic identifier",
            "              (runtime-reconstructable features ONLY: sell-curve",
            "              reconstruction + farm-event diffs; 0.900 held-out-day)",
            "_ARMS         counter-arm market overrides, (1+lambda) ES-trained",
            "_CLASS2ARM    identifier class -> arm (own class excluded)",
            "_FAMILY_DUMPS M2: per-class consensus dump schedules",
            "_MATCH_PROB   commit threshold (gates.json when >=25 judged commits)",
            "```",
            "",
            "Turn pipeline: route lookup → bandit (identify → streak-commit →",
            "mismatch-abandon; bit-identical floor when unsure) → weed-repair →",
            "feed-pull(off) → shed-clamp → **relay** → sell-first ordering →",
            "price-impact slot ranking → terminal liquidation.",
            "",
            "**The relay stack (M1/M2/M3):**",
            "1. *M1 predictive*: identifier `matched` class each step + money-free",
            "   structural near-mirror lock (tolerance 4, 12-turn streak) as fallback.",
            "2. *M2 what-to-counter*: online-learned opponent dump schedule",
            "   (observed ≥8-unit jumps in the reconstructed sell curve; phase mod 24",
            "   → predicted next dump; authoritative after 2 observations) →",
            "   static per-class schedule → clone assumption, in that order.",
            "   No collision → no pull → zero distortion.",
            "3. *M3 how-to-counter*: race outcomes are observable; being pre-empted",
            "   escalates the lead +2 (cap 8) — an online bandit over lead size.",
            "   Cross-game learning stays offline (fresh container per episode).",
        ))
    else:
        cells.append(md(
            "## B2. Architecture (route family), as shipped",
            "",
            "One self-contained `main.py`, stdlib only: the crowned 719-turn",
            "route replayed through the safety stack —",
            "",
            "1. weed-repair (DIG catch-up; an unwatered plant is a weed in 2 days)",
            "2. shed-clamp (`_safe_market`: SELL draws from the shed only)",
            "3. **near-mirror relay v2** (clone lock: money-free structural",
            "   near-match, FERTILIZER dumps ≥8, 2-turn lead, min-quote 3,",
            "   per-step repayment ledger — +202/game vs a clone, inert otherwise)",
            "4. sell-first ordering (quote before the mirror moves the market)",
            "5. price-impact slot ranking (10-order cap spent where it protects",
            "   the most price)",
            "6. terminal liquidation (everything unsold at step 718 scores 0)",
            "",
            "Deliberately NO identifier and NO arms: the route family is the",
            "portfolio's redundancy arm — a different failure mode from the",
            "bandit on the same winning basin.",
        ))

    # ---- B3. Data + EDA at build time (live numbers) ----------------------
    rows = ["## B3. Data & EDA at build time", ""]
    try:
        import kaggriculture.data.routes as R
        idx = R.load_index()
        n = len(idx["routes"])
        dates = [r.get("date", "") for r in idx["routes"].values()]
        rows += [f"- Route corpus: **{n} routes**, {min(dates)} → {max(dates)}."]
    except Exception:                                              # noqa: BLE001
        pass
    try:
        meta = json.load(open(os.path.join(ROOT, "models", "v22", "identifier",
                                           "meta.json"), encoding="utf-8"))
        rows += [f"- Identifier: {meta.get('classes')} classes, train acc "
                 f"{meta.get('train_acc'):.3f}, held-out-day acc "
                 f"{meta.get('heldout_acc'):.3f}, {meta.get('dims')} dims."]
    except Exception:                                              # noqa: BLE001
        pass
    try:
        fp = json.load(open(os.path.join(ROOT, "data", "fingerprints",
                                         "summary.json"), encoding="utf-8"))
        rows += [f"- Opening-fingerprint map: {fp['routes']} routes → "
                 f"{fp['clusters']} clusters; we sit in `sheep_first_hybrid` "
                 "(75.5% win basin)."]
    except Exception:                                              # noqa: BLE001
        pass
    try:
        fd = json.load(open(os.path.join(ROOT, "models", "relay",
                                         "family_dumps.json"), encoding="utf-8"))
        cl = fd.get("classes", {})
        rows += [f"- M2 dump schedules: {len(cl)} classes "
                 f"({', '.join('class ' + k + ': ' + str(len(v)) + ' events' for k, v in sorted(cl.items()))})."]
    except Exception:                                              # noqa: BLE001
        pass
    rows += [
        "- Same-day pipeline: leaderboard top-200 episode harvest + own-game",
        "  mining close the ~2-day archive lag (Kaggle's episode-listing lags",
        "  hours for fresh submissions — a low count is not 'no games').",
    ]
    cells.append(md(*rows))

    # ---- B4. The morning job ---------------------------------------------
    cells.append(md(
        "## B4. The daily release (05:00 morning job)",
        "",
        "`KaggricultureRefreshCycle` (Windows Task Scheduler) →",
        "`scripts/daily_release.bat` → `scripts/daily_release.py`:",
        "",
        "1. **FETCH** — `ourgames.py --all` (our own live games = the only real",
        "   measurement) + `fresh_data.py` (same-day scrape + leaderboard",
        "   top-200 episode harvest + identifier retrain), delta-only.",
        "2. **CYCLE** — `refresh_cycle.py --on-demand`: retrain identifier /",
        "   gates / surrogate on the refreshed corpus, build loss tapes from",
        "   our newest live losses, Bradley-Terry tournament refereed by those",
        "   tapes, crown the freshest winning base, build `v{x}.0_route` +",
        "   `v{x}.0_bandit` (both carry the relay stack), retrain arms.",
        "3. **GATE** — `submit.py --dry-run`: size, self-play Validation",
        "   Episode, latency. Any failure aborts the upload.",
        "4. **NOTEBOOKS** — rebuild + push THIS notebook and its sibling",
        "   (route ↔ bandit), same slugs, so every shipped agent adds one",
        "   version here.",
        "5. **UPLOAD** — submit both agents, each citing its notebook.",
        "",
        "Safeguards: HELD guard (no fresh crown → nothing uploads, the live",
        "pair stays), `last_release.json` for revert, quoted-env launcher",
        "(a Task Scheduler trailing-space once produced a fatal",
        "`invalid PYTHONUTF8` — fixed in the .bat).",
        "",
        "Versioning: `v{x}.{y}` — x steps daily, y per intra-day fix;",
        "self-determined by `refresh_cycle.next_version()`.",
    ))
    return cells


def release_cells(agent_name):
    """Dated evidence for THIS release, read from the decision artifacts.

    Everything the operator gates a ship on -- the gauntlet cross-play, the
    strict-future chronological veto, the engine-parity audit, and the
    second-slot decision -- rendered from the same JSON the gates read, so
    the notebook's claims can never drift from what was actually measured.
    Silently empty when an artifact is absent (older agents still build)."""
    cells = []
    lines = [f"## Release evidence -- {agent_name}", ""]

    # Engine parity: is the local engine provably the ladder's?
    try:
        par = json.load(open(os.path.join(ROOT, "models",
                                          "ladder_parity.json"),
                             encoding="utf-8"))
        lines += [f"**Engine parity** ({par.get('when', '?')}): "
                  f"{par.get('matches', 0)} fresh raw ladder replays "
                  f"re-simulated on the vendored engine "
                  f"{par.get('engine', '?')} to exact-bank matches, "
                  f"{par.get('mismatches', 0)} mismatches.", ""]
    except (OSError, ValueError):
        pass

    # Strict-future chronological veto for this candidate.
    try:
        gates = []
        for p in glob.glob(os.path.join(ROOT, "models", "strict_future",
                                        "gate_*.json")):
            g = json.load(open(p, encoding="utf-8"))
            if g.get("candidate") == agent_name:
                gates.append(g)
        if gates:
            g = max(gates, key=lambda g: str(g.get("when", "")))
            lines += [f"**Strict-future gate** (freeze at episode "
                      f"{g.get('cutoff_episode', '?')}, judged only on "
                      f"episodes recorded AFTER the freeze): "
                      f"**{g.get('verdict', '?')}** {g.get('w', '?')}-"
                      f"{g.get('l', '?')}-{g.get('d', 0)} over "
                      f"{g.get('tapes', '?')} post-cutoff tapes, score "
                      f"{g.get('score', 0):.3f}, sign p = "
                      f"{g.get('sign_p', 1):.5f}.", ""]
    except (OSError, ValueError):
        pass

    # Gauntlet cross-play: reactive public frontier agents, paired seeds,
    # both seats -- the ranking instrument (tapes over-predict; these react).
    try:
        best = None
        for p in glob.glob(os.path.join(ROOT, "models", "gauntlet",
                                        "gauntlet_*.json")):
            d = json.load(open(p, encoding="utf-8"))
            if agent_name in (d.get("results") or {}):
                if best is None or str(d.get("when")) > str(best.get("when")):
                    best = d
        if best:
            r = best["results"][agent_name]
            lines += [f"**Gauntlet cross-play** ({best.get('when')}; "
                      f"{len(best.get('seeds') or [])} paired seeds x both "
                      f"seats vs executable public frontier agents): overall "
                      f"**{r.get('overall', 0):.3f}** "
                      f"({r.get('w')}-{r.get('l')}-{r.get('d')}).", "",
                      "| opponent | record | score | mean margin |",
                      "|---|---|---:|---:|"]
            for opp, c in (r.get("per_opponent") or {}).items():
                lines.append(f"| {opp} | {c.get('w')}-{c.get('l')}-"
                             f"{c.get('d')} | {c.get('score', 0):.3f} | "
                             f"{c.get('mean_margin', 0):+,.0f} |")
            lines.append("")
    except (OSError, ValueError, KeyError):
        pass

    # The second-slot decision and any operator force, verbatim.
    try:
        ss = json.load(open(os.path.join(ROOT, "models", "second_slot.json"),
                            encoding="utf-8"))
        lines += [f"**Second-slot decision** (v{ss.get('version', '?')}, "
                  f"{ss.get('when', '?')}): {ss.get('why', '')}", ""]
    except (OSError, ValueError):
        pass

    if len(lines) > 2:
        cells.append(md(*lines))
    return cells


def build(agent_path, out_dir=OUT_DIR, slug=SLUG, title=TITLE,
          private=False):
    name = os.path.basename(agent_path)
    with open(agent_path, "rb") as fh:
        raw = fh.read()
    sha = hashlib.sha256(raw).hexdigest()
    payload = base64.b85encode(zlib.compress(raw, 9)).decode("ascii")
    card, rows, prov = evidence(name)
    today = dt.date.today().isoformat()

    cells = []

    cells.append(md(
        f"# {title}",
        "",
        "**A current mined route, replayed with a safety stack, and its already-planned",
        "SELLs ranked by the price damage they do to themselves.**",
        "",
        "Two separable contributions, measured separately, because conflating them is",
        "how a result stops being reusable:",
        "",
        "| change | effect |",
        "|---|---|",
        "| route refresh (stale &rarr; current) | the difference between never beating a top tape and beating it every game |",
        "| price-impact SELL ranking | **34 paired games, 34 wins, +$1,811 mean margin** |",
        "",
        "The second number is small on purpose. It only permutes orders we were",
        "already sending, in the slots they already occupy. It has never lost a",
        "paired game across three independent seed sets.",
    ))

    cells.extend(release_cells(name))

    cells.append(md(
        "## Machine-readable contribution card",
        "",
        "```yaml",
        "upstream_mechanism_sources:",
        "  - Kaito Fukami, v22 Price Impact: SELL ranking by self-induced quote drop",
        "  - Kaito Fukami, v21.1 Conditional Memory: order-only safety invariant",
        "  - fle3n, v21.1 No Preemption: published negative result on preemption",
        "  - llccqq624, Replay Data Miner: byte-level replay header parsing",
        "route_provenance:",
        f"  team: {prov.get('team', '?')}",
        f"  rank_at_capture: {prov.get('rank_at_mine') or prov.get('rank', '?')}",
        f"  episode: {prov.get('episode', '?')}",
        f"  seat: {prov.get('seat', '?')}",
        f"  window: {prov.get('window', '?')}",
        "  source: Kaggle public episode archive (kaggle/kaggriculture-episodes-*)",
        "new_contribution:",
        "  - route selection by playing candidates head to head, not by recorded bank",
        "  - independent verification of the price curve against the 1.32.4 interpreter",
        "  - an engine fingerprint check, because a Kaggle image has shipped a",
        "    different interpreter than the ladder",
        "  - a rating floor derived live from the score at rank N, not hardcoded",
        "explicitly_rejected:",
        "  - hazard-table preemption (published negative result, reproduced)",
        "  - opponent-move prediction as an action trigger",
        "  - XGBoost committee arbiter (50%, -$3,724)",
        "  - tabular MC control on asset bias (17%, -$8,383)",
        "  - endgame sale acceleration from day 24 (-$19,054)",
        "  - route splicing across field/market channels (banks collapse)",
        "  - per-turn consensus over 198 routes (-$167,032)",
        "  - counter-routing the rank-1 outlier (12-0 head to head, 72.1% vs 93.6% on the panel)",
        "measurement_caveat:",
        "  - a held-out tape panel over-predicts ladder win rate by ~18 points",
        "    (95.2% panel vs 77.6% real), because tapes do not react",
        "runtime_identity_fields: []",
        "runtime_opponent_private_fields: []",
        "```",
    ))

    cells.append(md(
        "## 1. How we got here: every version, and what each one measured",
        "",
        "Two architectures, eighteen versions, and a change of mind in the middle.",
        "The negative results are kept because most of this project's cost was",
        "re-testing ideas that were already disproved.",
        "",
        "### Phase one - a closed-loop planner (v0-v14)",
        "",
        "| ver | change | measured |",
        "|---|---|---|",
        "| v0 | carrot monoculture; proves the plumbing | ~$10k |",
        "| v1 | every job priced in dollars, Hungarian assignment | ~$51k, 100% vs v0 |",
        "| v2 | coordinate descent over 28 knobs | ~$83k, 100% vs v1 |",
        "| v3 | travel weight + Voronoi zoning | 81% vs v2 |",
        "| v4 | max-weight assignment instead of greedy pairs | public **683.7** - the field median |",
        "| v9 | **rebuilt economics**: fertilizer is sellable, livestock on turn 0, feed bought not grown, assets priced against projected supply on both farms | public **820.9** |",
        "| v14 | **two interpreter facts**: same-turn selling (units resolve at `:904`, market at `:910`) and sell-first ordering (the market pairs order *i* against order *i*) | public **847.8**, +$19,934 vs v9 |",
        "",
        "v4 sat exactly at the field median while local win rates kept improving.",
        "That disconnect was the whole problem: we were optimising against our own",
        "lineage. Profiling the top of the ladder showed the gap was never",
        "movement efficiency - it was the asset mix, and then the market channel.",
        "",
        "### The change of mind",
        "",
        "Decoding public top-10 agents showed they ship **no planner at all**: a",
        "recorded 719-turn route plus a small repair layer. The environment is",
        "deterministic apart from weed spawns and shop-unlock order, so one good",
        "trajectory transfers. That is why the top 20 sit within a few percent of",
        "each other.",
        "",
        "### Phase two - an open-loop route (v15-v18)",
        "",
        "| ver | change | measured |",
        "|---|---|---|",
        "| v15 | first mined route + safety stack (projected shed, weed repair, sell-first, terminal liquidation) | 100% vs v14 (+$54,325); vs a 2,900-rated tape, v14 was 0%/-$54,149, this was 17%/-$2,478 |",
        "| v16 | route chosen from the **freshest** archived day | **100%** vs that same tape (+$15,714) - the first win against a top tape |",
        "| v17 | **price-impact SELL ranking** | 34 paired games, 34 wins, +$1,811 |",
        "| v18 | route selected by **playing** candidates rather than by rank | 49/52 held-out opponents beaten outright, **93.6%** of 312 games; **77.6% on the real ladder** - see 3b |",
        "",
        "Three selectors were tried and only one works. **Recorded bank** fails: in",
        "our tournament the candidate with the highest mean bank ($142,753) had the",
        "second-*worst* win rate (11%). **Team rank** fails too: the same team's two",
        "routes scored 26.4% and 97.5% on an identical panel - 71 points apart,",
        "decided by nothing but which route was picked. Only playing them works.",
        "",
        "### Measured and rejected (do not retry without new evidence)",
        "",
        "| idea | result |",
        "|---|---|",
        "| XGBoost committee arbiter | 50%, -$3,724 |",
        "| tabular MC control on asset bias | 17%, -$8,383 |",
        "| opponent counter-play from market deltas | 56%, +$374 (noise) |",
        "| hazard-table preemption | published negative result, reproduced |",
        "| splicing one route's field channel onto another's market channel | **banks collapse from ~$146k to $12,900-$36,843** |",
        "| mirror tie-breaking against a fork | armed 456 turns, fired 0 - a good route already sells everything the turn it becomes available |",
        "| terminal glide-path floor (force-drain the last days) | +$170 over 44 games, under the noise bar |",
        "| **endgame accelerator** (pull shed sales forward from day 24) | **-$19,054**; mean bank ~$131k &rarr; ~$112k |",
        "| building from the rank-1 outlier's route | won the head-to-head 12-0, then **72.1% vs 93.6%** on the panel (see 3a) |",
        "| per-turn consensus over 198 routes | **-$167,032** (see 3a) |",
        "",
        "The last row is the most instructive, because it was built to fix a defect",
        "we had actually measured (see section 3b) and it still lost. Pulling sales",
        "forward floods the market *ourselves*: the route's schedule was already",
        "spreading the endgame load to avoid self-glut, and the accelerator undid",
        "it. That is the fourth independent confirmation of one rule -- **change the",
        "order, never the inventory.**",
        "",
        "## 2. Why route freshness, and how fast it decays",))

    cells.append(md(
        "### Freshness, measured on our own mine",))

    cells.append(md(
        "_(continued)_",
        "",
        "A route is a perishable asset. We measured this on our own mine rather than",
        "quoting it: eight candidate routes were built and played head to head, and",
        "**three routes from one team came 6th, 7th and 8th (22-44%) despite that team",
        "ranking 5-6 when the routes were captured.** That team is now rank 49.",
        "",
        "Mining the *freshest* published day returned routes from teams at live ranks",
        "2, 3, 4, 6, 7 and 8. The previous day's mine had topped out at rank 40.",
    ))

    cells.append(code(
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "",
        "# Candidate tournament: 8 mined routes + 2 incumbents, 2 seeds x 2 seats,",
        "# 36 games each. Ranks are the live leaderboard position at capture.",
        "tournament = pd.DataFrame([",
        '    ["90551213_s1", 7, "Dmitry Larko", 0.89, 120855],',
        '    ["90555869_s1", 3, "Raj Aryan", 0.89, 120623],',
        '    ["90557412_s0", 5, "Just a moroccan", 0.78, 120438],',
        '    ["90543557_s1", 6, "Akhil Chinta", 0.72, 120337],',
        '    ["90556633_s1", 8, "Chloe", 0.61, 120365],',
        '    ["90288945_s1", 6, "Konstantin03", 0.44, 111049],',
        '    ["90288225_s0", 5, "Konstantin03", 0.33, 111088],',
        '    ["90288253_s1", 6, "Konstantin03", 0.22, 107517],',
        '    ["older-route agent", 40, "incumbent", 0.11, 142753],',
        '    ["closed-loop planner", None, "incumbent", 0.00, 92217],',
        '], columns=["route", "rank at capture", "team", "win rate", "mean bank"])',
        "display(tournament)",
        "",
        "fig, ax = plt.subplots(figsize=(8.6, 4.2))",
        'ax.scatter(tournament["mean bank"], tournament["win rate"], s=70,'
        ' color="#2563eb")',
        'for _, r in tournament.iterrows():',
        '    ax.annotate(r["team"][:16], (r["mean bank"], r["win rate"]),',
        '                textcoords="offset points", xytext=(6, 4), fontsize=8)',
        'ax.set(xlabel="mean bank in the tournament",'
        ' ylabel="head-to-head win rate",',
        '       title="Recorded bank does not predict head-to-head strength")',
        'ax.grid(alpha=0.25)',
        "plt.tight_layout()",
    ))

    cells.append(md(
        "**Read the top-left point.** The incumbent with the *highest* mean bank",
        "($142,753) has the second-*worst* win rate (11%). A route that banks well",
        "against weak opposition loses to one that banks less against strong",
        "opposition, because recorded bank measures the game, not the trajectory.",
        "",
        "That is why selection here is by playing the candidates, and why the pool it",
        "draws from is disjoint from the window used to check the winner.",
    ))

    cells.append(md(
        "## 2. Why price impact, and what the curve actually looks like",
        "",
        "Every unit sold raises that product's market inventory and lowers its quote.",
        "The curves are **convex above equilibrium** and differ sharply by product:",
        "melon and wool are `sq`, milk and strawberry `linear`, wheat only `log`.",
        "",
        "So the order in which our own sells land changes what they fetch. A large",
        "melon order executed after a large wool order is quoted into a market our",
        "own wool has already moved. Ranking by `quantity x (quote now - quote after)`",
        "puts the most self-damaging order first, while it still has the good price.",
    ))

    cells.append(code(
        "import math",
        "",
        "# Transcribed from kaggriculture.py and verified against the vendored",
        "# 1.32.4 interpreter at 2,835 points: maximum difference zero.",
        "PARAMS = {",
        '    "WHEAT": (25, 400, "log", 0.2),',
        '    "STRAWBERRY": (120, 100, "linear", 1.6),',
        '    "MILK": (160, 122, "linear", 1.6),',
        '    "MELON": (250, 300, "sq", 3.6),',
        '    "WOOL": (200, 105, "sq", 3.2),',
        '    "FERTILIZER": (100, 200, "linear", 0.4),',
        "}",
        "",
        "def shape(name, x):",
        "    x = max(0.0, float(x))",
        '    return {"linear": x, "sq": x * x, "sqrt": math.sqrt(x),',
        '            "log": math.log1p(x)}[name]',
        "",
        "def quote(item, surplus):",
        "    base, scale, fn, target = PARAMS[item]",
        "    amp = target * base / shape(fn, scale)",
        "    return max(1, int(round(base - amp * shape(fn, surplus))))",
        "",
        "curve = pd.DataFrame(",
        "    [[item, s, quote(item, s)] for item in PARAMS for s in range(0, 201)],",
        '    columns=["product", "surplus inventory", "quote"])',
        'ax = curve.pivot(index="surplus inventory", columns="product",'
        ' values="quote").plot(figsize=(9, 4.4), linewidth=2)',
        'ax.set(title="Self-induced price damage above equilibrium",'
        ' ylabel="quote ($)")',
        "ax.grid(alpha=0.25)",
        "plt.tight_layout()",
        "",
        "# What one order costs itself, at a realistic size.",
        "impact = pd.DataFrame(",
        '    [[i, 30, quote(i, 0), quote(i, 30), 30 * (quote(i, 0) - quote(i, 30))]',
        "     for i in PARAMS],",
        '    columns=["product", "qty", "quote at equilibrium", "quote after",'
        ' "impact score"]).sort_values("impact score", ascending=False)',
        "display(impact)",
    ))

    cells.append(md(
        "The table is the ranking key. At 30 units, melon and wool damage their own",
        "quote by an order of magnitude more than wheat does -- so they go first, and",
        "wheat absorbs being last because its `log` curve barely moves.",
    ))

    cells.append(md(
        "## 3. What was measured, including what it lost",
        "",
        "The price-impact layer is a **pure re-ordering**: no order created, none",
        "resized, and every non-SELL slot keeps its index. That constraint is load",
        "bearing -- the published ablation where an agent *created* new early sells",
        "collapsed, and a separate author reached a verified 2830.4 public score by",
        "switching preemption off entirely. Predicting a sale does not establish the",
        "value of taking it early.",
    ))

    if rows:
        cells.append(code(
            "results = pd.DataFrame(" + json.dumps(rows, indent=1) + ")",
            'results["win %"] = (results["win_rate"] * 100).round(0)',
            'display(results[["opponent", "win %", "margin", "bank", "seeds", "note"]])',
        ))

    cells.append(md(
        "## 3a. The ladder is eleven route families, not two thousand agents",
        "",
        "Clustering all **525 mined routes** by action signature (cosine >= 0.995,",
        "single-linkage) collapses 89 teams into **11 families**:",
        "",
        "| routes | teams | rank range |",
        "|---:|---|---:|",
        "| 264 | **47 teams** | 3-99 |",
        "| 97 | 22 teams | 11-194 |",
        "| 70 | 20 teams | 21-190 |",
        "| 28 | 7 teams | 33-88 |",
        "| 15 | **one team** | **1-10** |",
        "| 12 | 2 teams | 1-24 |",
        "| 3 | one team | 26 |",
        "",
        "**Half the visible ladder runs a single lineage.** That is what compresses",
        "the top 20 into a 3.2% band.",
        "",
        "The three opponents this agent cannot beat map onto that structure exactly:",
        "two of them are the *solo* families at ranks 1 and 26, and the third shares",
        "**our own** family -- which is why that loss is only -$961, a near-mirror.",
        "",
        "There is a mechanical penalty for joining the crowd. The market inventory is",
        "shared, so when 47 teams run the same route they sell the same product on",
        "the same turn into the same price curve and gut it for each other. Route",
        "decay is not the route ageing; it is **adoption rising**. Our own two live",
        "submissions show it: the copied public notebook has drawn **19 games** where",
        "this agent has drawn 1. It keeps meeting its own clones.",
        "",
        "### Two things this suggested, both of which lost",
        "",
        "**Build from the rank-1 outlier.** We held 15 routes from the solo top-10",
        "family and had never used one; the archive showed it beating our lineage",
        "4/4 head to head (no seat bias: 47.1% vs 48.4%). Built it, and it beat our",
        "shipped agent **12-0, +$30,503**. Then on the held-out panel, same 52",
        "opponents and same seeds:",
        "",
        "| | counter-route | shipped |",
        "|---|---:|---:|",
        "| panel win rate | 72.1% | **93.6%** |",
        "| better / worse / equal, per opponent | **3 / 45 / 3** | |",
        "",
        "It genuinely fixes both unbeatable opponents (0% -> 50% and 0% -> 67%) and",
        "regresses against 45 others. **It is a counter to our lineage, not a better",
        "route.** Matchmaking pairs you by rating, so you meet the crowd, not the",
        "outlier -- and trading 45 opponents for 2 is a bad trade even when the 2",
        "include the one that humiliates you.",
        "",
        "The method lesson is the durable part: **a 12-0 head-to-head against the",
        "incumbent is not evidence of superiority.** Non-transitivity is strong here.",
        "Only a fixed opponent roster measures anything.",
        "",
        "**A per-turn consensus route.** Take the modal action at each of the 720",
        "turns across all 198 fit-window routes -- 'learn the best move for every",
        "turn from everyone'. Measured: **$12,085 against $179,117, 0% of 8 games**.",
        "The worst agent this project has produced, landing on the same floor as the",
        "channel splice ($12,900) by the same mechanism.",
        "",
        "| | |",
        "|---|---:|",
        "| modal whole-turn action share | mean **35.5%**, median 31.3% |",
        "| turns with a true majority (>50%) | **105 / 720** |",
        "| unit actions that are position-dependent | **87.8%** |",
        "| unit actions that are pure movement | 48.4% |",
        "",
        "At two turns in three there is no consensus to find, so the winner is a",
        "plurality of about a third and you are splicing at the finest possible",
        "granularity. And when two routes both say `WEST` at turn 400 they are not",
        "agreeing -- different farmers, different tiles, different destinations. The",
        "label matches; the plan does not.",
        "",
        "**A route is a plan carrying state, not a sequence of independent",
        "decisions.** You can choose between plans. You cannot average them.",
        "",
        "## 3b. What the ladder said, and where our own panel is wrong",
        "",
        "The held-out tape panel scored this agent at **95.2%** (400/420 games, 67",
        "of 70 opponents beaten outright). The real ladder, over ~70 episodes,",
        "returned **77.6%** (52W-14L-1T).",
        "",
        "| measurement | win rate |",
        "|---|---|",
        "| held-out tape panel, 70 top-100 opponents, both seats | 95.2% |",
        "| **public ladder, ~70 episodes** | **77.6%** |",
        "",
        "That ~18-point gap is not noise, and it is the most useful thing in this",
        "write-up. **Tapes do not react.** A recorded opponent floods the same",
        "premium market at the same turn whatever we do, so every layer that",
        "exploits a *predictable* opponent scores better against a recording than",
        "against a live agent. Treat a tape-panel win rate as an **upper bound**,",
        "never as an estimate.",
        "",
        "### Where the losses actually happen",
        "",
        "Re-reading all our own ladder games with a cash-by-day trace:",
        "",
        "| day | mean cash gap in wins | mean cash gap in losses |",
        "|---:|---:|---:|",
        "| 18 | -2,317 | -3,413 |",
        "| 21 | +9,781 | +1,451 |",
        "| 24 | +13,899 | **+39** |",
        "| 27 | +13,842 | **-2,909** |",
        "",
        "Through day 18 wins and losses are **indistinguishable**. We win by",
        "surging in days 21-24; in losses that surge simply never happens, and the",
        "deficit opens at median day 27.",
        "",
        "The tempting reading is a fixable endgame scheduling bug -- which is",
        "exactly what the accelerator in section 1 was built to fix, and it lost",
        "$19,054. The honest reading is the other one: **in losses we do not lose",
        "the endgame, we fail to gain it.** Across 14 losses to 14 different",
        "opponents with no repeat, a gap that opens late and stays flat is what",
        "being second-best looks like when both sides liquidate at once.",
    ))

    cells.extend(v22_cells())

    cells.append(md(
        "## 3d. The system behind the agent: one frozen runtime, a daily config",
        "",
        "From v23 onward every submission is produced by the same architecture:",
        "a **frozen runtime** (safety stack, exact price model, opponent-sales",
        "reconstruction, bandit with a decision-theoretic commit gate, and",
        "hand-rolled inference for exported model weights) rendered together",
        "with a **per-cycle config payload** (crowned base route, counter-arm",
        "overrides, behavioral cluster exemplars, identifier and value-model",
        "weights, gate parameters, config hash). The runtime changes only with",
        "code review and full regression; the config regenerates daily.",
        "",
        "| stage | what happens |",
        "|---|---|",
        "| mine | newest archive day: routes, replays, engine versions |",
        "| cluster | name-free behavioral families (broad features) |",
        "| diagnose | our own live games; every loss labelled by family |",
        "| train | identifier + value model + arms (PSRO best-response, surrogate-assisted ES) |",
        "| crown | fresh candidates, refereed by tapes of our own losses; >=10-pt gate vs incumbent |",
        "| build | runtime + config -> one `main.py`, hashed |",
        "| gate | paired validation, zero regressions, latency, self-play |",
        "| ship | this notebook gains a version; sha asserted; human submits |",
        "",
        "Fine-tuning is warm-started per component (arms resume from the",
        "previous genome; the identifier recalibrates on a held-out day); the",
        "base route is never tuned - it is re-crowned, because measured",
        "freshness decay (47% -> 86% -> 100% across three days of base age)",
        "makes tuning a stale base pointless. A/B runs at three tiers: local",
        "paired builds differing in one config entry; the two live slots as",
        "twins judged on game records (never early ratings); and full config",
        "lineage in a registry, so any historical agent is rebuildable from",
        "its hash. Training rows carry the engine version, after a",
        "mid-competition balance change taught us that old-game data poisons",
        "new-game models.",
    ))

    cells.append(md(
        "## 4. The agent",
        "",
        f"`{name}`, {len(raw):,} bytes, SHA-256 `{sha}`.",
        "",
        "The cell below reconstructs it byte for byte, checks the hash, and compiles",
        "it. A submission whose bytes are not verifiable is not reproducible.",
    ))

    cells.append(code(
        "import base64, hashlib, pathlib, tarfile, zlib",
        "",
        f'EXPECTED_SHA256 = "{sha}"',
        f"EXPECTED_BYTES = {len(raw)}",
        "PAYLOAD = (",
        *[f'    "{c}"' for c in chunk(payload)],
        ")",
        "",
        "raw = zlib.decompress(base64.b85decode(PAYLOAD))",
        "assert len(raw) == EXPECTED_BYTES, (len(raw), EXPECTED_BYTES)",
        "assert hashlib.sha256(raw).hexdigest() == EXPECTED_SHA256",
        'pathlib.Path("main.py").write_bytes(raw)',
        "",
        "# Kaggle accepts a tar.gz with main.py at the root (AGENTS.md).",
        'with tarfile.open("submission.tar.gz", "w:gz") as tar:',
        '    tar.add("main.py", arcname="main.py")',
        "",
        'compile(raw.decode("utf-8"), "main.py", "exec")',
        'print(f"main.py {len(raw):,} bytes, sha256 {EXPECTED_SHA256[:16]}, compiles")',
    ))

    cells.append(code(
        "# A real episode, under the stock configuration.",
        "!pip -q install 'kaggle-environments>=1.32.3'",
        "from kaggle_environments import make",
        "",
        'env = make("kaggriculture", configuration={"episodeSteps": 720}, debug=True)',
        'env.run(["main.py", "main.py"])',
        "final = env.steps[-1]",
        'print([(i, s["status"], s["reward"]) for i, s in enumerate(final)])',
    ))

    cells.append(md(
        "## 5. The model's knowledge graph",
        "",
        "Every agent in this project carries one, regenerated whenever the agent",
        "changes, and the pipeline refuses to gate a candidate without it. It",
        "records three things the prose cannot: the configuration in full, the",
        "decision path at each fork, and a real episode sampled turn by turn.",
        "",
        "For a route agent the configuration *is* the route, so the graph",
        "summarises what it builds, what it trades, and when it first sells each",
        "product. The cell below regenerates that summary from the embedded",
        "bytes, so it describes the agent in this notebook rather than one that",
        "was current when the notebook was written.",
    ))

    cells.append(code(
        "import base64, collections, json, re, zlib, gzip",
        "",
        "# PAYLOAD is the whole main.py; the primary route/tape is the first",
        "# compressed blob inside it. Agents encode differently (bandit: zlib+b85;",
        "# trackp: gzip+b64), so try each; this is a descriptive cell and must",
        "# never fail the run, so any miss degrades to a note.",
        "def _find_route(src):",
        "    for pat, dec, dcmp in (",
        "        (r'b85decode\\(\"([^\"]+)\"\\)', base64.b85decode, zlib.decompress),",
        "        (r'b85decode\\(\\'([^\\']+)\\'\\)', base64.b85decode, zlib.decompress),",
        "        (r'b64decode\\(\"([^\"]+)\"\\)', base64.b64decode, gzip.decompress),",
        "        (r'b64decode\\(\\'([^\\']+)\\'\\)', base64.b64decode, gzip.decompress),",
        "    ):",
        "        for m in re.finditer(pat, src):",
        "            try:",
        "                obj = json.loads(dcmp(dec(m.group(1))).decode())",
        "                if isinstance(obj, list) and obj and isinstance(obj[0], dict):",
        "                    return obj",
        "            except Exception:",
        "                continue",
        "    return None",
        "try:",
        "    _src = zlib.decompress(base64.b85decode(PAYLOAD)).decode()",
        "    route = _find_route(_src)",
        "    if route is None:",
        "        raise ValueError('no decodable route/tape blob found')",
        "    unit = collections.Counter(); market = collections.Counter()",
        "    first_sale, sell_days = {}, set()",
        "    for t, turn in enumerate(route):",
        "        for op in [turn.get('farmer')] + list(turn.get('hands') or []):",
        "            if isinstance(op, list) and op: unit[op[0]] += 1",
        "        for o in turn.get('market') or []:",
        "            if isinstance(o, list) and len(o) >= 3:",
        "                market[f'{o[0]} {o[1]}'] += int(o[2] or 0)",
        "                if o[0] == 'SELL':",
        "                    sell_days.add(t // 24); first_sale.setdefault(o[1], t)",
        "    print(f'route turns        {len(route)}')",
        "    print(f'days with a sale   {len(sell_days)} of 30')",
        "    print(f'max hands          {max(len(t.get(\"hands\") or []) for t in route)}')",
        "    print()",
        "    display(pd.DataFrame(market.most_common(12), columns=['market op', 'units']))",
        "    display(pd.DataFrame(",
        "        sorted(((k, v, v // 24) for k, v in first_sale.items()), key=lambda r: r[1]),",
        "        columns=['product', 'first SELL turn', 'day']))",
        "    ax = pd.Series(dict(unit.most_common(10))).plot.barh(",
        "        figsize=(8, 3.6), color='#2563eb')",
        "    ax.set(title='Unit operations across the season', xlabel='count')",
        "    ax.invert_yaxis(); plt.tight_layout()",
        "except Exception as _e:",
        "    print('route breakdown skipped:', _e)",
    ))

    cells.append(md(
        "## 6. Credit",
        "",
        "The route embedded here is a real public replay from the Kaggle episode",
        "archive; team and episode are named in the contribution card above. The",
        "archive is the sanctioned data source and this notebook does not claim the",
        "trajectory as invented work.",
        "",
        "| Public work | What it contributed here |",
        "|---|---|",
        "| Kaito Fukami, *v22 Price Impact* | the SELL ranking by self-induced quote drop; independently reimplemented and verified against the interpreter |",
        "| Kaito Fukami, *v21.1 Conditional Memory* | the order-only safety invariant: change order, never inventory |",
        "| fle3n, *v21.1 No Preemption* | the published negative result on repeated preemption, which we did not re-test |",
        "| Rayk Kretzschmar, *Findings from Zero to Top Meta* | both-seat promotion discipline and honest replay/LB boundaries |",
        "| llccqq624, *Replay Data Miner* | byte-level replay header parsing, ~1,600x faster than a full parse |",
        "| boatlee, *Order-Safe Premium Control* | the warning not to move BUY/HIRE slots |",
        "",
        f"Built {today} by `src/kaggriculture/agentbuild/build_notebook.py`.",
        "",
        "**Caveat, stated plainly.** Local counterfactual results are not a public",
        "leaderboard score. Replay opponents execute fixed recorded trajectories and",
        "do not react to what we do, so these numbers screen a policy; they do not",
        "predict a rating.",
    ))

    # Detailed architecture + EDA appendix -- PRIVATE builds only. This section
    # names our clusters, basin, and obfuscation findings, so it must never
    # appear in a public kernel (would let a reader reconstruct our model).
    if private:
        cells.extend(detailed_eda_cells())
        cells.extend(family_doc_cells(
            "bandit" if "bandit" in os.path.basename(agent_path) else "route"))

    os.makedirs(out_dir, exist_ok=True)
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3",
                                      "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    nb_path = os.path.join(out_dir, slug + ".ipynb")
    with open(nb_path, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1)

    meta = {
        "id": f"{_username()}/{slug}",
        "title": title,
        "code_file": slug + ".ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": bool(private),
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,      # needed for pip install kaggle-environments
        "competition_sources": ["kaggriculture"],
        "dataset_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    with open(os.path.join(out_dir, "kernel-metadata.json"), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)

    print(f"wrote {os.path.relpath(nb_path, ROOT)} "
          f"({os.path.getsize(nb_path):,} bytes, {len(cells)} cells)")
    print(f"  agent    {name}  {len(raw):,} bytes  sha256 {sha[:16]}")
    print(f"  kernel   {meta['id']}")
    return nb_path, meta


def _username():
    for path in (os.path.expanduser("~/.kaggle/kaggle.json"),):
        if os.path.exists(path):
            try:
                return json.load(open(path, encoding="utf-8"))["username"]
            except Exception:                                      # noqa: BLE001
                pass
    return os.environ.get("KAGGLE_USERNAME", "USERNAME")


def push(out_dir=OUT_DIR):
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    proc = subprocess.run(E.kaggle_cmd() + ["kernels", "push", "-p", out_dir],
                          capture_output=True, timeout=900, env=env)
    out = ((proc.stdout or b"") + (proc.stderr or b"")).decode("utf-8", "replace")
    print(out.strip())
    return proc.returncode == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agent", default=os.path.join(ROOT, "agents", "v17_route.py"))
    ap.add_argument("--slug", default=SLUG)
    ap.add_argument("--title", default=TITLE)
    ap.add_argument("--private", action="store_true",
                    help="publish the kernel privately (default is public, "
                         "which is the point of a write-up)")
    ap.add_argument("--push", action="store_true", help="upload the kernel")
    ap.add_argument("--submit", action="store_true",
                    help="after pushing, submit the competition entry from it")
    ap.add_argument("--message", default=None)
    args = ap.parse_args()

    agent = args.agent if os.path.isabs(args.agent) else os.path.join(ROOT, args.agent)
    if not os.path.exists(agent):
        sys.exit(f"no such agent: {agent}")

    nb_path, meta = build(agent, slug=args.slug, title=args.title,
                          private=args.private)
    if not args.push:
        print("\nnot pushed. Add --push to upload the kernel, then --submit to "
              "enter it.\n  preview: " + os.path.relpath(nb_path, ROOT))
        return 0

    if not push():
        print("\nkernel push failed -- not submitting")
        return 1
    print(f"\npushed {meta['id']}")
    print("  Kaggle runs it now; the version must finish before it can be "
          "submitted.")
    if not args.submit:
        print("\nnot submitted. Re-run with --submit once the version has run, "
              "or use:\n"
              f"  kaggle competitions submit kaggriculture -k {meta['id']} "
              f"-v 1 -m \"...\"")
        return 0

    # Deliberately no auto-confirm here: this is the one irreversible step, and
    # only the latest two submissions stay active.
    msg = args.message or f"{os.path.basename(agent)} via notebook {meta['id']}"
    print("\nTo submit this notebook version, run:")
    print(f"  kaggle competitions submit kaggriculture -k {meta['id']} "
          f"-v <version> -m \"{msg}\"")
    print("\n(Not run automatically: a submission is irreversible and only the "
          "latest two stay active.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
