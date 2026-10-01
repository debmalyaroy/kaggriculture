"""Build a PUBLIC, generic knowledge notebook on the Kaggriculture episode data.

Deliberately non-revealing. It teaches only things a newcomer would want and
that are already public (how to read the daily episode replays, market-price
mechanics, score distributions, the announced balance change). It contains
NOTHING about how we build our agent: no behavioural feature engineering, no
family clustering, no opponent identification, no freshness/selection
strategy, no reference to our submissions. Anyone reading it learns the data
format and basic mechanics, not our approach.

Data: auto-discovers whatever `*episodes*` dataset is attached (no hardcoded
dates), so it stays runnable as the daily datasets roll.

    python src/build_eda_notebook.py --push
"""
from kaggriculture.paths import ROOT
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

OUT_DIR = os.path.join(ROOT, "notebooks", "eda")
SLUG = "kaggriculture-episode-data-a-gentle-eda"
TITLE = "Kaggriculture Episode Data: A Gentle EDA"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {},
            "source": [l + "\n" for l in "\n".join(lines).split("\n")]}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": [l + "\n" for l in "\n".join(lines).split("\n")]}


def newest_datasets(n=2):
    """The n most recent daily episode datasets that actually exist."""
    import datetime as dt
    found = []
    day = dt.date.today()
    for _ in range(14):
        ref = f"kaggle/kaggriculture-episodes-{day.isoformat()}"
        try:
            out = subprocess.run(["kaggle", "datasets", "files", ref],
                                 capture_output=True, timeout=60,
                                 env=dict(os.environ, PYTHONUTF8="1")
                                 ).stdout.decode("utf-8", "replace")
            if ".json" in out:
                found.append(ref)
        except Exception:                                          # noqa: BLE001
            pass
        if len(found) >= n:
            break
        day -= dt.timedelta(days=1)
    return found or ["kaggle/kaggriculture-episodes-index"]


def build():
    cells = []
    cells.append(md(
        "# Kaggriculture Episode Data: A Gentle EDA",
        "",
        "Kaggle publishes the previous day's ladder games as a public dataset",
        "(`kaggle/kaggriculture-episodes-<date>`). Each file is a full replay",
        "of one two-player, 720-turn match. This notebook is a friendly",
        "starting point for anyone new to the data: how to open a replay, what",
        "the market looks like over a season, how scores are distributed, and",
        "what the recent balance change did to prices.",
        "",
        "It is intentionally about the *data and mechanics*, not any strategy.",
    ))
    cells.append(code(
        "import glob, json, os, random",
        "import numpy as np, pandas as pd",
        "import matplotlib.pyplot as plt",
        "",
        "# Auto-discover whatever episode dataset is attached (no hardcoded",
        "# dates, so this keeps working as the daily datasets roll).",
        "def episode_files():",
        "    hits = []",
        "    for root in glob.glob('/kaggle/input/*'):",
        "        js = glob.glob(os.path.join(root, '**', '*.json'), recursive=True)",
        "        hits += [p for p in js if os.path.basename(p)[:-5].isdigit()]",
        "    return sorted(set(hits))",
        "",
        "FILES = episode_files()",
        "print(f'{len(FILES)} episode replays found across attached datasets')",
        "if not FILES:",
        "    print('No episode files attached. Add a kaggriculture-episodes-<date>',",
        "          'dataset via Add Input -> Datasets, then re-run.')",
        "random.seed(0)",
        "SAMPLE = random.sample(FILES, min(60, len(FILES)))",
    ))
    cells.append(md(
        "## 1. Opening one replay",
        "",
        "A replay is a JSON with a `steps` array. `steps[i]` holds both",
        "players' state after turn `i`; the final step carries each player's",
        "`reward` (their final bank). One subtlety worth knowing: in a stored",
        "replay `steps[i][seat]['action']` is the action that *produced* state",
        "`i` (an off-by-one that trips people up when reading actions back).",
    ))
    cells.append(code(
        "def load(p):",
        "    with open(p) as f: return json.load(f)",
        "",
        "def final_banks(r):",
        "    last = (r.get('steps') or [[]])[-1]",
        "    return [s.get('reward') or 0 for s in last]",
        "",
        "if FILES:",
        "    ex = load(FILES[0])",
        "    print('steps:', len(ex.get('steps') or []))",
        "    print('final banks:', final_banks(ex))",
        "    obs0 = ex['steps'][1][0]['observation']",
        "    print('observation keys:', list(obs0.keys()))",
        "    print('market products:', list((obs0.get('market') or {}).get('prices', {}).keys()))",
    ))
    cells.append(md(
        "## 2. How are final scores distributed?",
        "",
        "A simple but useful orientation: the spread of end-of-game banks",
        "across a day's episodes.",
    ))
    cells.append(code(
        "banks = []",
        "for p in SAMPLE:",
        "    try:",
        "        b = final_banks(load(p))",
        "        if len(b) == 2: banks.append(b)",
        "    except Exception: pass",
        "banks = np.array(banks) if banks else np.zeros((0,2))",
        "if len(banks):",
        "    plt.figure(figsize=(8,3))",
        "    plt.hist(banks.flatten(), bins=30, color='#4C78A8')",
        "    plt.xlabel('final bank'); plt.ylabel('players')",
        "    plt.title(f'Final-bank distribution ({len(banks)} games)')",
        "    plt.tight_layout(); plt.show()",
        "    print(f'median {np.median(banks):,.0f}, top decile {np.percentile(banks,90):,.0f}')",
    ))
    cells.append(md(
        "## 3. The market over a season",
        "",
        "Prices move as both players sell into a shared market. Watching a few",
        "products across the 30 days shows the shape every agent prices",
        "against: premium goods drift as supply accumulates.",
    ))
    cells.append(code(
        "PRODUCTS = ['STRAWBERRY','MELON','MILK','WOOL','WHEAT']",
        "def price_trace(p):",
        "    r = load(p); out = {k: [] for k in PRODUCTS}",
        "    for s in r['steps'][::24]:",
        "        pr = (s[0].get('observation') or {}).get('market', {}).get('prices', {})",
        "        for k in PRODUCTS: out[k].append(pr.get(k, np.nan))",
        "    return out",
        "",
        "agg = {k: [] for k in PRODUCTS}",
        "for p in SAMPLE[:30]:",
        "    try:",
        "        tr = price_trace(p)",
        "        for k in PRODUCTS:",
        "            if len(tr[k]) >= 30: agg[k].append(tr[k][:30])",
        "    except Exception: pass",
        "plt.figure(figsize=(9,4))",
        "for k in PRODUCTS:",
        "    if agg[k]: plt.plot(np.nanmean(agg[k], axis=0), label=k)",
        "plt.legend(); plt.xlabel('day'); plt.ylabel('mean price')",
        "plt.title('Average market price by day'); plt.tight_layout(); plt.show()",
    ))
    cells.append(md(
        "## 4. When does selling happen?",
        "",
        "Aggregate sell volume by day across the sample -- a high-level look at",
        "the season's rhythm. This is a descriptive view of the field as a",
        "whole, nothing agent-specific.",
    ))
    cells.append(code(
        "vol = np.zeros(30)",
        "for p in SAMPLE:",
        "    try:",
        "        r = load(p)",
        "        for i in range(1, len(r['steps'])):",
        "            day = i // 24",
        "            if day >= 30: break",
        "            for seat in (0,1):",
        "                for o in (r['steps'][i][seat].get('action') or {}).get('market', []):",
        "                    if isinstance(o, list) and len(o) >= 3 and o[0]=='SELL':",
        "                        vol[day] += max(0, int(o[2] or 0))",
        "    except Exception: pass",
        "plt.figure(figsize=(9,3))",
        "plt.bar(range(30), vol, color='#54A24B')",
        "plt.xlabel('day'); plt.ylabel('total units sold (sample)')",
        "plt.title('Sell volume by day (all players, sampled)')",
        "plt.tight_layout(); plt.show()",
    ))
    cells.append(md(
        "## 5. The balance change, if two eras are attached",
        "",
        "On 2026-08-07 the hosts reduced Town Center demand and made shops",
        "sample with replacement (announced in the competition discussion, PR",
        "#1394; engine >= 1.32.6). If you attach one pre- and one post-change",
        "day, mean prices show the late-season floor arriving harder. Public",
        "information -- included so newcomers know to re-check mechanics after",
        "any balance patch.",
    ))
    cells.append(code(
        "from collections import defaultdict",
        "by_dir = defaultdict(list)",
        "for p in FILES: by_dir[os.path.basename(os.path.dirname(p))].append(p)",
        "days = sorted(by_dir)",
        "print('attached day-datasets:', days)",
        "if len(days) >= 2:",
        "    plt.figure(figsize=(9,4))",
        "    for d in days[:2]:",
        "        tr = []",
        "        for p in by_dir[d][:20]:",
        "            try:",
        "                t = price_trace(p)['STRAWBERRY']",
        "                if len(t) >= 29: tr.append(t[:29])",
        "            except Exception: pass",
        "        if tr: plt.plot(np.nanmean(tr, axis=0), label=d)",
        "    plt.legend(); plt.xlabel('day'); plt.ylabel('mean STRAWBERRY price')",
        "    plt.title('Strawberry price by day, per attached dataset')",
        "    plt.tight_layout(); plt.show()",
        "else:",
        "    print('Attach a second day-dataset to compare eras.')",
    ))
    cells.append(md(
        "## Takeaways for newcomers",
        "",
        "1. Each replay is a full 720-turn match; the last step holds final banks.",
        "2. The market is shared, so prices reflect *both* players' selling.",
        "3. Scores span a wide range within a single day -- context matters.",
        "4. Re-verify mechanics against the current engine after any balance",
        "   patch (the 2026-08-07 change altered late-season economics).",
        "",
        "*Shared to help people get started with the episode datasets. Happy",
        "farming!*",
    ))

    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3",
                                      "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.10"}},
          "nbformat": 4, "nbformat_minor": 4}
    os.makedirs(OUT_DIR, exist_ok=True)
    nb_path = os.path.join(OUT_DIR, SLUG + ".ipynb")
    json.dump(nb, open(nb_path, "w", encoding="utf-8"), indent=1)

    user = json.load(open(os.path.expanduser("~/.kaggle/kaggle.json"),
                          encoding="utf-8"))["username"]
    datasets = newest_datasets(2)
    print("attaching datasets:", datasets)
    meta = {"id": f"{user}/{SLUG}", "title": TITLE,
            "code_file": SLUG + ".ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": "false",
            "enable_gpu": "false", "enable_internet": "false",
            "dataset_sources": datasets, "competition_sources": ["kaggriculture"],
            "kernel_sources": []}
    json.dump(meta, open(os.path.join(OUT_DIR, "kernel-metadata.json"), "w",
                         encoding="utf-8"), indent=1)
    print(f"wrote {os.path.relpath(nb_path, ROOT)} ({len(cells)} cells)")
    return nb_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()
    build()
    if args.push:
        env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        p = subprocess.run(["kaggle", "kernels", "push", "-p", OUT_DIR],
                           capture_output=True, timeout=900, env=env)
        print(((p.stdout or b"") + (p.stderr or b"")).decode("utf-8", "replace"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
