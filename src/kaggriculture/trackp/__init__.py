"""Track P -- the closed-loop planner lineage.

ISOLATION RULE (operator order 2026-08-14): this package shares NO Python
code with the bandit/route tracks. Everything here is freshly written.
What it MAY touch:
  * the Rust engine binary (shared infrastructure, PRE-RANKER ONLY)
  * the vendored official interpreter (it IS the ladder's engine)
  * data files (replays, traces, family_dumps_v2.json, episodes.csv) as DATA

Layout:
  common.py       engine constants + replay IO + paths (the only shared base)
  trace_v2.py     P0.1  versioned ~115-dim per-turn trace extraction
  fetch.py        P0.1  Track P's own bounded replay fetcher (delta, polite)
  insight.py      P0.2  the four correlation passes (weekly, dated)
  families.py     P0.5  opening-hash-anchored family labels
  projections.py  L2    price-at-harvest projector
  macro.py        P3.2  macro action space + decision dataset
  build_agent.py  P1    emits the single-file planner agent
  arena.py        --    own official-engine paired match runner
  panel.py        P1.6  regime/shop-draw stratified kill-switch panel
  validity.py     P4.1  open-loop vs closed-loop ranking correlation
  search.py       P4.2  CMA-ES over planner PARAMS
  serve_env.py    P3.1  client for `kagg serve`
  league.py       P3.3  PFSP league (anchors / past selves / exploiters)
  ppo.py          P3.3  PPO+GAE trainer over macro decisions
  iql.py          P3.3b IQL warm start on the elite macro corpus
  exploiters.py   P3.4  exploiter training = the exploitability meter
  export_policy.py P3.5 torch -> pure-python export w/ equivalence assert
  guard.py        P3.6  PLANNER GUARD (sign-tested, generation-scoped)
  graduation.py   P5    the 5-condition graduation gate (report-only)
  generator.py    P2    return+shop-conditioned route generator
  gen_funnel.py   P2    sample -> repair -> rust pre-rank -> official confirm
  pipeline.py     --    the Track P orchestrator (never submits)
"""

__version__ = "0.1.0"
