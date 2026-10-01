"""Exploratory data analysis over a Kaggriculture episode dataset.

Input : data/episodes.csv  (src/kaggriculture/data/build_dataset.py, or src/kaggriculture/data/gen_local_dataset.py)
Output: a self-contained HTML report with inline SVG charts -- no JS, no CDN.

Sections
  1  dataset summary and provenance
  2  outcome distribution: what banks win, and by how much
  3  paired winner-minus-loser deltas (the robust view)
  4  win rate by feature quintile, for every actionable feature
  5  asset allocation vs outcome
  6  labour efficiency (the movement finding, checked against data)
  7  strategy archetypes (k-means on portfolio mix) and their win rates
  8  what to change, with the caveats that matter

    python -m kaggriculture.pipeline.eda_report --data data/episodes.csv --out docs/history/eda-report.html
"""
from kaggriculture.paths import ROOT
import argparse
import csv
import html
import math
import os
import sys

SKIP = {"episode_id", "date", "player", "won", "final_bank", "opp_bank", "margin"}

PAL = ["#4c9be8", "#5fbf8f", "#d9a441", "#a985d9", "#e0715f", "#5bb8c4", "#c98bb0"]
WIN, LOSE = "#5fbf8f", "#e0715f"


# ----------------------------------------------------------------- helpers
def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            if k == "date":
                continue
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                r[k] = 0.0
    return rows


def med(xs):
    xs = sorted(xs)
    n = len(xs)
    return 0.0 if not n else (xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2]))


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def quintiles(rows, feat):
    vals = sorted({r[feat] for r in rows})
    if len(vals) < 5:
        buckets = [[r for r in rows if r[feat] == v] for v in vals]
        labels = [f"{v:g}" for v in vals]
        return labels, buckets
    srt = sorted(rows, key=lambda r: r[feat])
    n = len(srt)
    buckets, labels = [], []
    for i in range(5):
        chunk = srt[i * n // 5:(i + 1) * n // 5]
        if not chunk:
            continue
        buckets.append(chunk)
        labels.append(f"{chunk[0][feat]:g}–{chunk[-1][feat]:g}")
    return labels, buckets


# -------------------------------------------------------------------- svg
def bar_chart(pairs, width=560, bar_h=22, fmt="{:.0f}", colour=None, ref=None,
              ref_label=""):
    """pairs = [(label, value)]. Diverging around 0 if any value is negative."""
    if not pairs:
        return "<p class='dim'>no data</p>"
    vals = [v for _l, v in pairs]
    lo, hi = min(0.0, min(vals)), max(0.0, max(vals))
    span = (hi - lo) or 1.0
    left = 190
    plot = width - left - 60
    h = len(pairs) * bar_h + 26
    zero = left + (0 - lo) / span * plot
    out = [f"<svg viewBox='0 0 {width} {h}' width='100%' role='img'>"]
    if ref is not None:
        x = left + (ref - lo) / span * plot
        out.append(f"<line x1='{x:.1f}' y1='14' x2='{x:.1f}' y2='{h-12}' "
                   f"stroke='var(--faint)' stroke-dasharray='3 3'/>")
        out.append(f"<text x='{x+4:.1f}' y='12' class='axis'>{html.escape(ref_label)}</text>")
    for i, (label, v) in enumerate(pairs):
        y = 20 + i * bar_h
        c = colour(v) if callable(colour) else (colour or PAL[0])
        x0 = min(zero, left + (v - lo) / span * plot)
        w = abs((v - 0) / span * plot)
        out.append(f"<rect x='{x0:.1f}' y='{y}' width='{max(w,1):.1f}' height='{bar_h-7}' "
                   f"rx='3' fill='{c}' opacity='.85'/>")
        out.append(f"<text x='{left-8}' y='{y+bar_h-12}' text-anchor='end' "
                   f"class='lbl'>{html.escape(label)}</text>")
        tx = x0 + w + 5 if v >= 0 else x0 - 5
        anc = "start" if v >= 0 else "end"
        out.append(f"<text x='{tx:.1f}' y='{y+bar_h-12}' text-anchor='{anc}' "
                   f"class='val'>{fmt.format(v)}</text>")
    out.append(f"<line x1='{zero:.1f}' y1='14' x2='{zero:.1f}' y2='{h-12}' "
               f"stroke='var(--line)'/>")
    out.append("</svg>")
    return "".join(out)


def hist(win_vals, lose_vals, width=560, height=180, bins=16):
    allv = win_vals + lose_vals
    if not allv:
        return "<p class='dim'>no data</p>"
    lo, hi = min(allv), max(allv)
    if hi == lo:
        hi = lo + 1
    step = (hi - lo) / bins
    def counts(vs):
        c = [0] * bins
        for v in vs:
            i = min(bins - 1, int((v - lo) / step))
            c[i] += 1
        return c
    cw, cl = counts(win_vals), counts(lose_vals)
    top = max(max(cw or [0]), max(cl or [0])) or 1
    pad, bw = 34, (width - 44) / bins
    out = [f"<svg viewBox='0 0 {width} {height}' width='100%' role='img'>"]
    for i in range(bins):
        for c, col, off in ((cw[i], WIN, 0), (cl[i], LOSE, bw / 2)):
            hgt = (c / top) * (height - pad - 18)
            out.append(f"<rect x='{34+i*bw+off:.1f}' y='{height-pad-hgt:.1f}' "
                       f"width='{bw/2-1.5:.1f}' height='{max(hgt,0):.1f}' "
                       f"fill='{col}' opacity='.85' rx='2'/>")
    out.append(f"<line x1='30' y1='{height-pad}' x2='{width-8}' y2='{height-pad}' "
               f"stroke='var(--line)'/>")
    out.append(f"<text x='30' y='{height-12}' class='axis'>${lo:,.0f}</text>")
    out.append(f"<text x='{width-8}' y='{height-12}' text-anchor='end' "
               f"class='axis'>${hi:,.0f}</text>")
    out.append(f"<text x='30' y='14' class='axis'>"
               f"<tspan fill='{WIN}'>■</tspan> winners  "
               f"<tspan fill='{LOSE}'>■</tspan> losers</text>")
    out.append("</svg>")
    return "".join(out)


def scatter(pts, xlab, ylab, width=560, height=230):
    if not pts:
        return "<p class='dim'>no data</p>"
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs); y0, y1 = min(ys), max(ys)
    if x1 == x0: x1 = x0 + 1
    if y1 == y0: y1 = y0 + 1
    L, B = 52, 34
    out = [f"<svg viewBox='0 0 {width} {height}' width='100%' role='img'>"]
    for x, y, won in pts:
        px = L + (x - x0) / (x1 - x0) * (width - L - 16)
        py = height - B - (y - y0) / (y1 - y0) * (height - B - 18)
        out.append(f"<circle cx='{px:.1f}' cy='{py:.1f}' r='3.4' "
                   f"fill='{WIN if won else LOSE}' opacity='.75'/>")
    out.append(f"<line x1='{L}' y1='{height-B}' x2='{width-8}' y2='{height-B}' stroke='var(--line)'/>")
    out.append(f"<line x1='{L}' y1='12' x2='{L}' y2='{height-B}' stroke='var(--line)'/>")
    out.append(f"<text x='{L}' y='{height-12}' class='axis'>{x0:.2f}</text>")
    out.append(f"<text x='{width-8}' y='{height-12}' text-anchor='end' class='axis'>{x1:.2f} {html.escape(xlab)}</text>")
    out.append(f"<text x='4' y='16' class='axis'>{ylab} {y1:,.0f}</text>")
    out.append("</svg>")
    return "".join(out)


# ------------------------------------------------------------------ stats
def paired(rows, feats):
    by = {}
    for r in rows:
        by.setdefault(r["episode_id"], []).append(r)
    res = {f: [] for f in feats}
    n = 0
    for pair in by.values():
        if len(pair) != 2 or pair[0]["won"] == pair[1]["won"]:
            continue
        w, l = (pair[0], pair[1]) if pair[0]["won"] else (pair[1], pair[0])
        n += 1
        for f in feats:
            res[f].append(w[f] - l[f])
    out = {}
    for f, ds in res.items():
        if not ds:
            continue
        hi = sum(1 for d in ds if d > 0); lo = sum(1 for d in ds if d < 0)
        dec = hi + lo
        out[f] = {"delta": med(ds), "pct": 100.0 * hi / dec if dec else 50.0, "n": dec}
    return out, n


def kmeans(rows, feats, k=4, iters=40):
    import numpy as np
    X = np.array([[r[f] for f in feats] for r in rows], float)
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
    Z = (X - mu) / sd
    idx = [int(i * len(Z) / k) for i in range(k)]
    C = Z[idx].copy()
    lab = np.zeros(len(Z), int)
    for _ in range(iters):
        d = ((Z[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(k):
            if (lab == j).any():
                C[j] = Z[lab == j].mean(0)
    return lab


# ----------------------------------------------------------------- report
def build(rows, out_path, source):
    feats = [k for k in rows[0] if k not in SKIP and k != "date"]
    n_ep = len({r["episode_id"] for r in rows})
    wins = [r for r in rows if r["won"]]
    losses = [r for r in rows if not r["won"]]
    pstats, n_dec = paired(rows, feats)

    S = []
    A = S.append
    A(f"<h2>1 · Dataset</h2>")
    A(f"<div class='grid'>")
    for lbl, val in (("player-rows", f"{len(rows):,}"), ("episodes", f"{n_ep:,}"),
                     ("decided episodes", f"{n_dec:,}"), ("features", f"{len(feats)}"),
                     ("median winner bank", f"${med([r['final_bank'] for r in wins]):,.0f}"),
                     ("median loser bank", f"${med([r['final_bank'] for r in losses]):,.0f}")):
        A(f"<div class='kpi'><span>{lbl}</span><b>{val}</b></div>")
    A("</div>")
    A(f"<p class='dim'>Source: {html.escape(source)}</p>")
    if n_dec < 25:
        A("<div class='warn'><b>Small sample.</b> Below ~25 decided episodes every "
          "number here is inside the noise band. Treat this as a smoke test of the "
          "pipeline, not evidence. Fetch more with "
          "<code>src/kaggriculture/data/build_dataset.py</code>.</div>")

    A("<h2>2 · Outcome distribution</h2>")
    A("<p class='dim'>Final bank, winners vs losers. If the distributions overlap "
      "heavily, matches are close and small edges decide them.</p>")
    A(hist([r["final_bank"] for r in wins], [r["final_bank"] for r in losses]))
    margins = [abs(r["margin"]) for r in rows if r["won"]]
    if margins:
        A(f"<p class='dim'>Median winning margin <b>${med(margins):,.0f}</b>. "
          f"Remember the ladder scores <b>win/loss only</b> — margin buys nothing.</p>")

    A("<h2>3 · Paired winner-minus-loser deltas</h2>")
    A("<p class='dim'>Both players in an episode faced the same seed, town and "
      "market, so differencing inside the episode removes all of it. This is the "
      "view to trust. Bars show how often the winner had more of the feature; "
      "50% is noise.</p>")
    ranked = sorted(pstats.items(), key=lambda kv: -abs(kv[1]["pct"] - 50))[:18]
    A(bar_chart([(f, s["pct"] - 50) for f, s in ranked],
                fmt="{:+.0f} pts",
                colour=lambda v: WIN if v > 0 else LOSE,
                ref=0, ref_label="no signal"))

    A("<h2>4 · Win rate by quintile</h2>")
    A("<p class='dim'>For each knob-relevant feature: split players into five "
      "equal groups and show the win rate of each. A monotone rise means more is "
      "better; a hump means there is an optimum.</p>")
    interesting = [f for f in ("peak_herd", "peak_hands", "peak_crops", "move_frac",
                              "quadrants", "peak_COW", "peak_SHEEP", "peak_GOOSE",
                              "peak_MELON", "peak_STRAWBERRY", "peak_WHEAT")
                   if f in feats]
    A("<div class='cols'>")
    for f in interesting:
        labels, buckets = quintiles(rows, f)
        pairs = [(labels[i], 100.0 * mean([r["won"] for r in b]))
                 for i, b in enumerate(buckets) if b]
        A(f"<div class='card'><h4>{f}</h4>"
          + bar_chart(pairs, width=430, bar_h=20, fmt="{:.0f}%",
                      colour=lambda v: WIN if v >= 50 else LOSE,
                      ref=50, ref_label="50%") + "</div>")
    A("</div>")

    A("<h2>5 · Labour efficiency</h2>")
    A("<p class='dim'>Share of unit-turns spent walking, against final bank. "
      "The v3 change was built on this relationship.</p>")
    A(scatter([(r["move_frac"], r["final_bank"], r["won"]) for r in rows],
              "move_frac", "bank"))
    mw, ml = mean([r["move_frac"] for r in wins]), mean([r["move_frac"] for r in losses])
    A(f"<p class='dim'>Mean movement share — winners <b>{mw*100:.1f}%</b>, "
      f"losers <b>{ml*100:.1f}%</b>.</p>")

    A("<h2>6 · Strategy archetypes</h2>")
    port = [f for f in feats if f.startswith("tiledays_")]
    try:
        lab = kmeans(rows, port, k=min(4, max(2, len(rows) // 4)))
        groups = {}
        for r, g in zip(rows, lab):
            groups.setdefault(int(g), []).append(r)
        A("<p class='dim'>k-means over portfolio mix (tile-days per asset). "
          "Each cluster is a way of playing; the win rate says whether it works.</p>")
        A("<table><tr><th>archetype</th><th class='n'>players</th>"
          "<th class='n'>win rate</th><th class='n'>median bank</th><th>dominant assets</th></tr>")
        for g, rs in sorted(groups.items(), key=lambda kv: -mean([r["won"] for r in kv[1]])):
            tops = sorted(((mean([r[p] for r in rs]), p.replace("tiledays_", ""))
                           for p in port), reverse=True)[:3]
            A(f"<tr><td>cluster {g}</td><td class='n'>{len(rs)}</td>"
              f"<td class='n'><b>{100*mean([r['won'] for r in rs]):.0f}%</b></td>"
              f"<td class='n'>${med([r['final_bank'] for r in rs]):,.0f}</td>"
              f"<td>{', '.join(f'{n} ({v:.0f})' for v, n in tops if v > 0)}</td></tr>")
        A("</table>")
    except Exception as exc:                                       # noqa: BLE001
        A(f"<p class='dim'>clustering unavailable: {html.escape(str(exc))}</p>")

    A("<h2>7 · What this suggests</h2>")
    sug = [(f, s) for f, s in pstats.items() if abs(s["pct"] - 50) >= 15 and s["n"] >= 8]
    sug.sort(key=lambda kv: -abs(kv[1]["pct"] - 50))
    if sug:
        A("<table><tr><th>feature</th><th class='n'>winner higher</th>"
          "<th class='n'>median Δ</th><th>read</th></tr>")
        for f, s in sug[:14]:
            A(f"<tr><td><code>{f}</code></td><td class='n'>{s['pct']:.0f}%</td>"
              f"<td class='n'>{s['delta']:+.1f}</td>"
              f"<td>{'winners do more' if s['delta']>0 else 'winners do less'}</td></tr>")
        A("</table>")
    else:
        A("<p class='dim'>Nothing clears the noise floor at this sample size.</p>")
    A("<div class='warn'><b>Before acting on any of this.</b> This project has a "
      "list of signals that looked decisive and measured negative: strawberry "
      "scarcity (0% win rate), rescue pens for stranded animals (62.5% vs 81%), "
      "and cheaper animal labour (80% over 10 matches, 56% over 16). Correlation "
      "here is not causation — feed these to <code>src/kaggriculture/train/tune.py</code> as priors "
      "and let self-play decide, over <b>at least three independent seed sets</b>.</div>")

    css = """
:root{--bg:#0d1117;--panel:#161b22;--line:#30363d;--ink:#e6edf3;--dim:#9aa7b4;--faint:#6e7b8a}
@media(prefers-color-scheme:light){:root{--bg:#f7f9fc;--panel:#fff;--line:#d3dae3;--ink:#111820;--dim:#4c5866;--faint:#7b8794}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:32px 22px 80px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:44px 0 8px;padding-bottom:8px;border-bottom:1px solid var(--line)}
h4{margin:0 0 6px;font-size:13px;color:var(--dim)}
p.dim,.dim{color:var(--dim)}
.grid{display:flex;flex-wrap:wrap;gap:12px;margin:14px 0}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:10px 14px;min-width:150px}
.kpi span{display:block;font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--faint)}
.kpi b{font-size:19px;font-variant-numeric:tabular-nums}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:12px}
table{width:100%;border-collapse:collapse;margin:12px 0;font-size:13px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line)}
th{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.06em}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
code{background:rgba(127,127,127,.16);padding:1px 5px;border-radius:4px;font-size:12px}
.warn{background:var(--panel);border:1px solid var(--line);border-left:3px solid #d9a441;
border-radius:8px;padding:13px 16px;margin:18px 0;color:var(--dim)}
.warn b{color:var(--ink)}
text.lbl{fill:var(--dim);font-size:11px}text.val{fill:var(--ink);font-size:11px;font-variant-numeric:tabular-nums}
text.axis{fill:var(--faint);font-size:10px}
footer{margin-top:48px;color:var(--faint);font-size:12px;border-top:1px solid var(--line);padding-top:14px}
"""
    doc = (f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>Kaggriculture EDA</title><style>{css}</style></head><body><div class='wrap'>"
           f"<h1>Kaggriculture — exploratory data analysis</h1>"
           f"<p class='dim'>{n_ep:,} episodes · {len(rows):,} player-rows · {len(feats)} features</p>"
           + "".join(S) +
           "<footer>Generated by src/kaggriculture/pipeline/eda_report.py · rebuild after every "
           "src/build_dataset.py run.</footer></div></body></html>")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "episodes.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "eda-report.html"))
    args = ap.parse_args()
    if not os.path.exists(args.data):
        sys.exit(f"no dataset at {args.data}\n"
                 f"  python -m kaggriculture.data.build_dataset --days 3 --per-day 50   (needs network)\n"
                 f"  python -m kaggriculture.data.gen_local_dataset --matches 12        (offline)")
    rows = load(args.data)
    p = build(rows, args.out, args.data)
    print(f"wrote {p}  ({os.path.getsize(p):,} bytes)")


if __name__ == "__main__":
    main()
