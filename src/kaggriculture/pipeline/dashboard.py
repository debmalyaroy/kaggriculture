"""Elo dashboard: every bot, its rating, and which one to submit.

Reads `models/elo/ladder.json` -- the running scoreboard that src/kaggriculture/measure/elo.py
maintains -- and writes a self-contained HTML page.

    python -m kaggriculture.pipeline.dashboard                       # -> docs/history/elo-dashboard.html
    python -m kaggriculture.pipeline.dashboard --open                # and open it
    python -m kaggriculture.pipeline.dashboard --json                # machine-readable, for scripts

Why Elo and not bank: the Kaggle leaderboard is a skill rating built from
head-to-head results. The public meta write-up names the failure mode directly
-- optimise mean bank against the starter bot and you get a high local bank and
a mediocre Elo. This page deliberately shows no coin totals above the fold.

The confidence column is the part people skip and shouldn't. An Elo from 2
games is noise; the +/- shown is a standard-error style bound (400/sqrt(games)),
so a bot on 2 games carries roughly +/-283 and a bot on 32 carries +/-71. Two
bots whose intervals overlap are not distinguishable, however different their
point ratings look.
"""
from kaggriculture.paths import ROOT
import argparse
import datetime as dt
import glob
import html
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LADDER = os.path.join(ROOT, "models", "elo", "ladder.json")
OUT = os.path.join(ROOT, "docs", "elo-dashboard.html")

# A rating needs this many games before it is worth acting on. Below it the
# error bar is wider than the gap between our best and worst agents.
MIN_GAMES = 16
# Head-to-head margin over the incumbent that counts as a real improvement.
MIN_WIN_RATE = 0.55


def load_ladder():
    if not os.path.exists(LADDER):
        return {"rating": {}, "games": {}, "wins": {}, "history": []}
    with open(LADDER, encoding="utf-8") as f:
        return json.load(f)


def confidence(games):
    """Rough +/- on the rating. Elo's own K-factor logic: uncertainty falls as
    1/sqrt(n). 400 is the Elo scale constant, so this reads in rating points."""
    return 400.0 / (games ** 0.5) if games else float("inf")


def collect(lad):
    rating = lad.get("rating", {})
    games = lad.get("games", {})
    wins = lad.get("wins", {})

    rows = []
    for name, elo in rating.items():
        g = int(games.get(name, 0))
        w = float(wins.get(name, 0.0))
        path = os.path.join("agents", name)
        rows.append({
            "name": name,
            "path": path,
            "exists": os.path.exists(os.path.join(ROOT, path)),
            "elo": float(elo),
            "games": g,
            "wins": w,
            "win_rate": (w / g) if g else 0.0,
            "conf": confidence(g),
            "rated": g >= MIN_GAMES,
            "mtime": (os.path.getmtime(os.path.join(ROOT, path))
                      if os.path.exists(os.path.join(ROOT, path)) else 0),
        })
    rows.sort(key=lambda r: (-r["elo"], -r["games"]))

    # Agents that exist but have never been rated -- easy to miss otherwise.
    known = {r["name"] for r in rows}
    for p in sorted(glob.glob(os.path.join(ROOT, "agents", "*.py"))):
        n = os.path.basename(p)
        if n in known or n.startswith("__"):
            continue
        rows.append({"name": n, "path": f"agents/{n}", "exists": True,
                     "elo": 0.0, "games": 0, "wins": 0.0, "win_rate": 0.0,
                     "conf": float("inf"), "rated": False,
                     "mtime": os.path.getmtime(p), "unrated": True})
    return rows


def recommend(rows):
    """Which bot to submit, and why -- or why none of them qualifies yet."""
    eligible = [r for r in rows if r["exists"] and r["rated"]]
    if not eligible:
        best_any = max((r for r in rows if r["exists"]),
                       key=lambda r: r["elo"], default=None)
        return {
            "pick": best_any["name"] if best_any else None,
            "confident": False,
            "why": (f"No bot has the {MIN_GAMES} rated games needed to trust its "
                    f"Elo. Highest point rating is "
                    f"{best_any['name'] if best_any else 'n/a'}, but on "
                    f"{best_any['games'] if best_any else 0} games that number is "
                    f"noise. Run more Elo rounds before submitting on this basis."),
        }
    top = eligible[0]
    rivals = [r for r in eligible[1:]
              if r["elo"] + r["conf"] >= top["elo"] - top["conf"]]
    if rivals:
        return {
            "pick": top["name"],
            "confident": False,
            "why": (f"{top['name']} leads on points, but its interval overlaps "
                    f"{', '.join(r['name'] for r in rivals[:3])}. That is a tie, "
                    f"not a lead. Settle it with a direct head-to-head over at "
                    f"least {MIN_GAMES} paired matches before spending a "
                    f"submission slot."),
        }
    return {
        "pick": top["name"],
        "confident": True,
        "why": (f"{top['name']} leads by more than the combined uncertainty of "
                f"the field, on {top['games']} rated games "
                f"({top['win_rate']:.0%} wins). Confirm with a direct "
                f"head-to-head, then submit."),
    }


def sparkline(history, width=520, height=90):
    """Inline SVG of every agent's rating over the ladder's recorded history."""
    if not history:
        return '<p class="muted">No history yet -- run src/kaggriculture/measure/elo.py a few times.</p>'
    series = {}
    for i, snap in enumerate(history):
        table = snap.get("table") or {}
        for name, val in table.items():
            v = val.get("rating") if isinstance(val, dict) else val
            try:
                series.setdefault(name, []).append((i, float(v)))
            except (TypeError, ValueError):
                continue
    if not series:
        return '<p class="muted">History present but no ratings recorded in it.</p>'

    xs = [p[0] for pts in series.values() for p in pts]
    ys = [p[1] for pts in series.values() for p in pts]
    x0, x1 = min(xs), max(xs) or 1
    y0, y1 = min(ys), max(ys)
    if y1 - y0 < 1:
        y0, y1 = y0 - 20, y1 + 20
    pad = 8

    def sx(x):
        return pad + (x - x0) / max(1, (x1 - x0)) * (width - 2 * pad)

    def sy(y):
        return height - pad - (y - y0) / (y1 - y0) * (height - 2 * pad)

    colours = ["#4f8ef7", "#39b98a", "#e0994a", "#c96ec9", "#d7605f",
               "#63b6c9", "#9a9a9a"]
    paths = []
    for i, (name, pts) in enumerate(sorted(series.items())):
        pts.sort()
        d = " ".join(("M" if j == 0 else "L") + f"{sx(x):.1f},{sy(y):.1f}"
                     for j, (x, y) in enumerate(pts))
        paths.append(f'<path d="{d}" fill="none" stroke="{colours[i % len(colours)]}" '
                     f'stroke-width="2" stroke-linejoin="round"/>')
    legend = " ".join(
        f'<span class="key"><i style="background:{colours[i % len(colours)]}"></i>'
        f'{html.escape(n)}</span>'
        for i, n in enumerate(sorted(series)))
    return (f'<svg viewBox="0 0 {width} {height}" class="spark">{"".join(paths)}</svg>'
            f'<div class="legend">{legend}</div>'
            f'<p class="muted">{y0:.0f} - {y1:.0f} Elo over '
            f'{len(history)} recorded round(s).</p>')


def render(rows, rec, lad):
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    body = []
    for i, r in enumerate(rows, 1):
        if r.get("unrated"):
            body.append(
                f'<tr class="unrated"><td>-</td><td class="name">{html.escape(r["name"])}</td>'
                f'<td class="num">-</td><td class="num">0</td><td class="num">-</td>'
                f'<td><span class="badge grey">never rated</span></td></tr>')
            continue
        cls = 'pick' if r["name"] == rec["pick"] else ''
        badge = ('<span class="badge green">rated</span>' if r["rated"]
                 else f'<span class="badge amber">only {r["games"]} games</span>')
        if not r["exists"]:
            badge += ' <span class="badge grey">file missing</span>'
        conf = '&plusmn;&infin;' if r["conf"] == float("inf") else f'&plusmn;{r["conf"]:.0f}'
        body.append(
            f'<tr class="{cls}"><td>{i}</td>'
            f'<td class="name">{html.escape(r["name"])}</td>'
            f'<td class="num big">{r["elo"]:.0f}</td>'
            f'<td class="num">{r["games"]}</td>'
            f'<td class="num">{r["win_rate"]:.0%}</td>'
            f'<td>{conf} &nbsp; {badge}</td></tr>')

    verdict_cls = 'good' if rec["confident"] else 'warn'
    verdict = ('Ready to submit' if rec["confident"]
               else 'Not settled yet')

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kaggriculture - bot ratings</title>
<style>
 :root {{ color-scheme: light dark; }}
 body {{ font: 15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
        margin: 0; padding: 32px; background:#0f1115; color:#e6e8eb; }}
 h1 {{ font-size: 22px; margin:0 0 4px; }}
 h2 {{ font-size: 15px; text-transform: uppercase; letter-spacing:.08em;
       color:#8b93a1; margin: 32px 0 10px; font-weight:600; }}
 .muted {{ color:#8b93a1; font-size:13px; }}
 table {{ border-collapse: collapse; width:100%; max-width: 860px; margin-top:8px; }}
 th,td {{ text-align:left; padding:9px 12px; border-bottom:1px solid #232733; }}
 th {{ font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:#8b93a1; }}
 td.num {{ text-align:right; font-variant-numeric: tabular-nums; }}
 td.big {{ font-size:17px; font-weight:600; }}
 td.name {{ font-family: ui-monospace,SFMono-Regular,Menlo,monospace; font-size:13px; }}
 tr.pick {{ background:#132a1f; }}
 tr.pick td.name::after {{ content:" <- recommended"; color:#39b98a; font-weight:600; }}
 tr.unrated {{ opacity:.55; }}
 .badge {{ display:inline-block; padding:1px 7px; border-radius:10px; font-size:11px;
           font-weight:600; }}
 .green {{ background:#123a2a; color:#4ad598; }}
 .amber {{ background:#3a3212; color:#e3c04a; }}
 .grey  {{ background:#26292f; color:#98a0ad; }}
 .card {{ border:1px solid #232733; border-left-width:4px; border-radius:8px;
          padding:14px 18px; max-width:860px; margin-top:8px; }}
 .card.good {{ border-left-color:#39b98a; }}
 .card.warn {{ border-left-color:#e3c04a; }}
 .card h3 {{ margin:0 0 6px; font-size:15px; }}
 code {{ background:#1a1d24; padding:2px 6px; border-radius:4px; font-size:13px; }}
 pre {{ background:#1a1d24; padding:12px 14px; border-radius:8px; overflow:auto;
        max-width:860px; font-size:13px; }}
 .spark {{ width:100%; max-width:560px; height:auto; margin-top:6px; }}
 .legend {{ font-size:12px; color:#8b93a1; margin-top:2px; }}
 .key {{ margin-right:12px; white-space:nowrap; }}
 .key i {{ display:inline-block; width:9px; height:9px; border-radius:2px;
           margin-right:4px; }}
 ol {{ max-width:860px; }} li {{ margin:6px 0; }}
</style></head><body>

<h1>Kaggriculture &mdash; bot ratings</h1>
<p class="muted">Local Elo ladder &middot; generated {now} &middot;
  {len(rows)} bot(s) &middot; source <code>models/elo/ladder.json</code></p>

<h2>Recommendation</h2>
<div class="card {verdict_cls}">
  <h3>{verdict}: {html.escape(rec['pick'] or 'none')}</h3>
  <p>{html.escape(rec['why'])}</p>
</div>

<h2>Ladder</h2>
<table>
<thead><tr><th>#</th><th>Bot</th><th>Elo</th><th>Games</th><th>Win rate</th>
<th>Confidence</th></tr></thead>
<tbody>
{chr(10).join(body)}
</tbody></table>
<p class="muted">Elo starts at 600, matching Kaggle's initial rating for a new
submission. Every pairing is played in <em>both seats</em> per seed &mdash; the
shared market makes the game asymmetric, so a one-seat test can reverse the
apparent winner.</p>

<h2>Rating history</h2>
{sparkline(lad.get('history', []))}

<h2>How to choose what to submit</h2>
<ol>
  <li><strong>It must pass the contract tests first.</strong>
      <code>.\\scripts\\test.ps1 -Full</code>. An agent that emits one illegal
      action or overruns the 1&nbsp;second <code>actTimeout</code> loses on
      Kaggle without telling you why. No rating is worth anything until this
      passes.</li>
  <li><strong>Rank by Elo, not by bank.</strong> The leaderboard is a skill
      rating: a 10-coin win and a 10,000-coin win score identically. Coin margin
      is only a tie-break between agents with the same record.</li>
  <li><strong>Ignore any rating with fewer than {MIN_GAMES} games.</strong> The
      confidence column is the gate. Two bots whose intervals overlap are tied,
      no matter how different the point ratings look. Fix it with
      <code>python tools\\elo.py --rounds 4</code>, not by guessing.</li>
  <li><strong>Confirm head-to-head against the bot you currently have on
      Kaggle.</strong> <code>python tools\\evaluate.py &lt;new&gt; --vs &lt;current&gt;
      -n 16</code>. Ladder Elo is transitive and can hide a bad matchup;
      the direct result cannot. Want &ge;{MIN_WIN_RATE:.0%} over &ge;{MIN_GAMES}
      paired matches.</li>
  <li><strong>Distrust improvements measured on fewer than 16 matches.</strong>
      This project has a worked example:
      <code>cost_per_animal_day</code> 4.5&rarr;3.5 looked like 80% over 10
      matches and was 56% over 16. Ten matches is a coin toss wearing a
      lab coat.</li>
  <li><strong>Then spend the slot.</strong> Five submissions a day, only the
      latest two stay active &mdash; so a submission you are unsure about costs
      you an active slot as well as a daily one.</li>
</ol>

<h2>Refresh</h2>
<pre>python tools\\elo.py --rounds 4          # play more rated games
python tools\\elo.py --only agents\\&lt;new&gt;.py   # rate one newcomer
python tools\\dashboard.py               # regenerate this page
.\\scripts\\submit.ps1 -Best -Live        # submit the top-rated bot</pre>

</body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--json", action="store_true",
                    help="print the ladder as JSON instead (used by submit.ps1)")
    ap.add_argument("--open", action="store_true",
                    help="open the page in a browser once it is written")
    args = ap.parse_args()

    lad = load_ladder()
    rows = collect(lad)
    rec = recommend(rows)

    if args.json:
        print(json.dumps({"rows": rows, "recommendation": rec,
                          "min_games": MIN_GAMES}, indent=1, default=str))
        return 0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(render(rows, rec, lad))
    print(f"wrote {os.path.relpath(args.out, ROOT)}  ({len(rows)} bots)")
    print(f"recommendation: {rec['pick']} "
          f"({'confident' if rec['confident'] else 'NOT settled'})")
    for r in rows[:8]:
        flag = "" if r.get("rated") else "   (low confidence)"
        print(f"  {r['elo']:>7.0f}  {r['name']:<34} {r['games']:>3} games{flag}")
    if args.open:
        import webbrowser
        webbrowser.open("file://" + os.path.abspath(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
