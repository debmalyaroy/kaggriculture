"""Counterfactual policy diff: same board, same state — what would WE have done?

A replay records, for every one of the 720 turns, the complete observation *and*
the action the player actually took. So for any strong opponent we can replay
their trajectory, hand each state to our agent, and ask: given this exact board,
this exact market, this exact shed — what would our bot do instead?

That isolates *policy* from *outcome*. It is not "they ended richer"; it is
"on turn 312, holding 14 cows and $4,100, they hired and we watered".

Method note: states always come from *their* trajectory, so errors never
compound the way they would in a rollout. The flip side is that our agent is
being asked about boards it would never have built, and disagreement is not
automatically error — it is a *hypothesis* about where the policies part ways.

    python -m kaggriculture.measure.action_diff data/episodes/top/2026-08-03/89619023.json
    python -m kaggriculture.measure.action_diff replay.json --agent agents/v2_tuned.py --out docs/action-diff.md
"""
from kaggriculture.paths import ROOT
import argparse
import collections
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
import kaggriculture.engine._vendor as _vendor  # noqa: F401,E402

MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}


def load_agent(path):
    spec = importlib.util.spec_from_file_location("cand", os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "agent"):
        sys.exit(f"{path} defines no agent()")
    return mod.agent, mod.agent.__code__.co_argcount > 1


def obs_for(steps, i, p):
    """Rebuild the observation player p saw on turn i.

    kaggle-environments stores *shared* observation keys only on index 0 and the
    private half on each player's own entry.
    """
    shared = (steps[i][0] or {}).get("observation") or {}
    own = (steps[i][p] or {}).get("observation") or {}
    obs = {
        "player": p,
        "step": shared.get("step", i),
        "day": shared.get("day", i // 24),
        "hour": shared.get("hour", i % 24),
        "farms": shared.get("farms"),
        "market": shared.get("market"),
        "town": shared.get("town"),
        "private": own.get("private") or shared.get("private"),
    }
    return obs if obs["farms"] and obs["private"] else None


def norm_unit(op):
    return op[0] if isinstance(op, list) and op else "PASS"


def norm_market(orders):
    c = collections.Counter()
    for o in orders or []:
        if isinstance(o, list) and o:
            c[o[0]] += 1
    return c


def analyse(replay_path, agent_path, focus="winner", limit=None):
    with open(replay_path, encoding="utf-8") as f:
        rep = json.load(f)
    steps = rep.get("steps") or []
    if not steps:
        sys.exit("replay has no steps")
    rewards = rep.get("rewards") or [0, 0]
    cfg = rep.get("configuration") or {}
    names = (rep.get("info") or {}).get("TeamNames") or ["p0", "p1"]

    if focus == "winner":
        p = 0 if rewards[0] >= rewards[1] else 1
    else:
        p = int(focus)

    agent, two_arg = load_agent(agent_path)

    farmer_cm = collections.Counter()     # (theirs, ours)
    market_cm = collections.Counter()
    agree = collections.Counter()
    total = collections.Counter()
    by_phase = collections.defaultdict(lambda: [0, 0])
    hands_theirs, hands_ours = [], []
    errors = 0

    n = len(steps) if limit is None else min(limit, len(steps))
    for i in range(n):
        obs = obs_for(steps, i, p)
        if obs is None:
            continue
        # steps[i]["action"] produced state i; the action taken *from* state i
        # is recorded one index later.
        nxt = steps[i + 1] if i + 1 < len(steps) else None
        theirs = ((nxt[p] or {}).get("action") if nxt and p < len(nxt) else None)
        if not isinstance(theirs, dict):
            continue
        try:
            ours = agent(obs, cfg) if two_arg else agent(obs)
        except Exception:                                          # noqa: BLE001
            errors += 1
            continue
        if not isinstance(ours, dict):
            continue

        t_f, o_f = norm_unit(theirs.get("farmer")), norm_unit(ours.get("farmer"))
        farmer_cm[(t_f, o_f)] += 1
        total["farmer"] += 1
        if t_f == o_f:
            agree["farmer"] += 1
        phase = "early (d0-9)" if obs["day"] < 10 else (
            "mid (d10-19)" if obs["day"] < 20 else "late (d20-29)")
        by_phase[phase][1] += 1
        by_phase[phase][0] += 1 if t_f == o_f else 0

        tm, om = norm_market(theirs.get("market")), norm_market(ours.get("market"))
        for op in set(tm) | set(om):
            d = om[op] - tm[op]
            if d:
                market_cm[op] += d
        hands_theirs.append(len(theirs.get("hands") or []))
        hands_ours.append(len(ours.get("hands") or []))

    return {
        "episode": (rep.get("info") or {}).get("EpisodeId"),
        "seed": (rep.get("info") or {}).get("seed"),
        "focus_player": p, "focus_name": names[p] if p < len(names) else f"p{p}",
        "focus_bank": rewards[p], "other_bank": rewards[1 - p],
        "farmer_cm": farmer_cm, "market_cm": market_cm,
        "agree": agree, "total": total, "by_phase": dict(by_phase),
        "hands_theirs": hands_theirs, "hands_ours": hands_ours,
        "errors": errors, "steps": n,
    }


def render(res, agent_path, out=None):
    L = []
    A = L.append
    tot = res["total"]["farmer"] or 1
    ag = 100.0 * res["agree"]["farmer"] / tot
    A(f"# Policy diff — same board, different move\n")
    A(f"Episode `{res['episode']}` · seed `{res['seed']}` · "
      f"{res['steps']} turns replayed\n")
    A(f"Reference player: **{res['focus_name']}** "
      f"(${res['focus_bank']:,.0f} vs ${res['other_bank']:,.0f})  \n"
      f"Our agent: `{os.path.basename(agent_path)}`\n")
    A(f"\n**Farmer-op agreement: {ag:.1f}%** "
      f"({res['agree']['farmer']:,} of {tot:,} turns)\n")
    if res["errors"]:
        A(f"\n_{res['errors']} turns where our agent raised — investigate._\n")

    A("\n## Agreement by phase\n")
    A("| phase | agreement | turns |")
    A("|---|---:|---:|")
    for ph in ("early (d0-9)", "mid (d10-19)", "late (d20-29)"):
        if ph in res["by_phase"]:
            a, t = res["by_phase"][ph]
            A(f"| {ph} | {100.0*a/max(1,t):.1f}% | {t:,} |")

    A("\n## Where we diverge — they did X, we would have done Y\n")
    A("| they did | we would | turns | share |")
    A("|---|---|---:|---:|")
    diff = [(k, v) for k, v in res["farmer_cm"].items() if k[0] != k[1]]
    diff.sort(key=lambda kv: -kv[1])
    for (t_op, o_op), cnt in diff[:20]:
        A(f"| `{t_op}` | `{o_op}` | {cnt:,} | {100.0*cnt/tot:.1f}% |")

    A("\n## Net action budget — ours minus theirs\n")
    A("Positive = our agent spends *more* turns on this op than the reference "
      "player did, over the same 720 states.\n")
    A("| op | net turns |")
    A("|---|---:|")
    net = collections.Counter()
    for (t_op, o_op), cnt in res["farmer_cm"].items():
        net[o_op] += cnt
        net[t_op] -= cnt
    for op, v in sorted(net.items(), key=lambda kv: -abs(kv[1]))[:14]:
        if v:
            A(f"| `{op}` | {v:+,} |")
    mv_ours = sum(v for k, v in net.items() if k in MOVES)
    A(f"\nNet movement turns: **{mv_ours:+,}** "
      f"({'we walk more' if mv_ours > 0 else 'we walk less'}).\n")

    if res["market_cm"]:
        A("\n## Market orders — ours minus theirs (whole episode)\n")
        A("| order | net |")
        A("|---|---:|")
        for op, v in sorted(res["market_cm"].items(), key=lambda kv: -abs(kv[1])):
            A(f"| `{op}` | {v:+,} |")

    ht, ho = res["hands_theirs"], res["hands_ours"]
    if ht:
        A(f"\n## Crew size\n")
        A(f"Reference player carried a mean of **{sum(ht)/len(ht):.1f}** hands; "
          f"our action dict supplied **{sum(ho)/len(ho):.1f}** ops per turn.\n")

    A("\n## Reading this honestly\n")
    A("- States come from *their* trajectory, so errors do not compound — but our "
      "agent is being asked about boards it would never have built. Disagreement "
      "is a hypothesis, not a verdict.\n")
    A("- The useful signal is a *systematic* skew: if we spend hundreds more turns "
      "walking, or never issue an order they issue constantly, that is a real "
      "policy gap worth testing in self-play.\n")
    A("- Every change suggested here still has to clear three independent seed "
      "sets in `src/kaggriculture/measure/evaluate.py` before it is believed.\n")

    text = "\n".join(L)
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("replay")
    ap.add_argument("--agent", default=None, help="default: newest agents/agent_v*.py")
    ap.add_argument("--focus", default="winner", help="'winner', '0' or '1'")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    agent = args.agent
    if not agent:
        import glob
        cands = sorted(glob.glob(os.path.join(ROOT, "agents", "agent_v*.py")))
        agent = cands[-1] if cands else os.path.join(ROOT, "agents", "v2_tuned.py")
    res = analyse(args.replay, agent, args.focus, args.limit)
    print(render(res, agent, args.out))
    if args.out:
        print(f"\n[written to {args.out}]", file=sys.stderr)


if __name__ == "__main__":
    main()
