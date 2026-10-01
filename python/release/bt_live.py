"""Where does v63.16 finish? A Bradley-Terry projection from live data.

    python python/release/bt_live.py [--opp-subs 30]

1. Leaderboard (kaggle competitions leaderboard -d): every team's current score + rank.
2. EpisodeService.ListEpisodes for our live subs (v63.16 56720196, v63.14 56718979) and for the --opp-subs most frequent
   opponent submissions: every finished game with both submissions' rewards and scores. Cached in .local/bt/episodes.json.
3. Scale: fit s in P(A beats B) = 1 / (1 + exp(-(rA - rB) / s)) by maximum likelihood over every game NOT involving us,
   rA/rB = each side's CURRENT leaderboard score (teams), draws = half a win.
4. Our BT strength: the r that maximises the likelihood of our own results against the same opponents' current scores
   (Newton on one parameter, s fixed), with a Fisher-information 90% interval; projected rank = position of r among the
   leaderboard scores (and of the interval ends).
Writes .local/bt/report.json.
"""
import argparse, collections, glob, json, math, os, subprocess, sys, tempfile, zipfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'top50'))
import fetch  # noqa: E402  (auth, list_episodes)

OUT = '.local/bt'
OURS = {'56720196': 'v63.16', '56718979': 'v63.14'}


def board():
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(['kaggle', 'competitions', 'leaderboard', 'kaggriculture', '-d', '-p', tmp], capture_output=True)
        z = glob.glob(os.path.join(tmp, '*.zip'))[0]
        zipfile.ZipFile(z).extractall(tmp)
        import pandas as pd
        lb = pd.read_csv(glob.glob(os.path.join(tmp, '*.csv'))[0])
    lb.to_csv(f'{OUT}/leaderboard.csv', index=False)
    return lb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--opp-subs', type=int, default=30)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    lb = board()
    score = dict(zip(lb['TeamId'], lb['Score']))
    name = dict(zip(lb['TeamId'], lb['TeamName']))
    hdr = fetch.auth()
    cache = f'{OUT}/episodes.json'
    eps = json.load(open(cache)) if os.path.exists(cache) else {}

    def pull(sid):
        got = fetch.list_episodes(sid, hdr) or []
        for e in got:
            ag = e.get('agents') or []
            if len(ag) != 2 or any(x.get('reward') is None for x in ag):
                continue
            eps[str(e['id'])] = [{k: x.get(k) for k in ('submissionId', 'teamId', 'reward', 'initialScore', 'updatedScore')} for x in ag]
        return got

    for s in OURS:
        pull(s)
    opp = collections.Counter()
    for g in eps.values():
        if any(str(x['submissionId']) in OURS for x in g):
            for x in g:
                if str(x['submissionId']) not in OURS:
                    opp[x['submissionId']] += 1
    for sid, _ in opp.most_common(a.opp_subs):
        pull(sid)
    json.dump(eps, open(cache, 'w'))

    our_team = None
    games = []  # (teamA, teamB, resultA)
    ours = []   # (opp team, result, sub)
    for eid, (x, y) in eps.items():
        ra, rb = float(x['reward']), float(y['reward'])
        res = 1.0 if ra > rb else 0.0 if ra < rb else 0.5
        if str(x['submissionId']) in OURS or str(y['submissionId']) in OURS:
            me, ot, r = (x, y, res) if str(x['submissionId']) in OURS else (y, x, 1 - res)
            our_team = me['teamId']
            ours.append((ot['teamId'], r, OURS[str(me['submissionId'])], eid))
        else:
            games.append((x['teamId'], y['teamId'], res))
    games = [g for g in games if g[0] in score and g[1] in score and our_team not in (g[0], g[1])]

    def nll_s(s):
        t = 0.0
        for A, B, r in games:
            p = 1 / (1 + math.exp(-(score[A] - score[B]) / s))
            p = min(max(p, 1e-9), 1 - 1e-9)
            t -= r * math.log(p) + (1 - r) * math.log(1 - p)
        return t
    grid = [10 * i for i in range(5, 301)]
    s = min(grid, key=nll_s)

    def fit(rows):
        r = sum(score.get(o, 0) for o, _, _, _ in rows) / max(1, len(rows))
        for _ in range(100):
            g = h = 0.0
            for o, res, _, _ in rows:
                if o not in score:
                    continue
                p = 1 / (1 + math.exp(-(r - score[o]) / s))
                g += (res - p) / s
                h -= p * (1 - p) / s ** 2
            if h == 0:
                break
            step = g / h
            r -= step
            if abs(step) < 1e-6:
                break
        se = 1 / math.sqrt(-h) if h < 0 else float('inf')
        return r, se

    scores = sorted(lb['Score'], reverse=True)
    rank = lambda r: 1 + sum(x > r for x in scores)
    rep = {'teams_on_board': len(lb), 'scale_s': s, 'calibration_games': len(games)}
    our_lb = lb[lb['TeamId'] == our_team]
    if len(our_lb):
        rep['our_board_score'] = float(our_lb['Score'].iloc[0])
        rep['our_board_rank'] = int(our_lb['Rank'].iloc[0]) if 'Rank' in our_lb else rank(float(our_lb['Score'].iloc[0]))
    for tag in ('v63.16', 'v63.14', 'both'):
        rows = [x for x in ours if tag == 'both' or x[2] == tag]
        rows = [x for x in rows if x[0] in score]
        if not rows:
            continue
        r, se = fit(rows)
        w = sum(x[1] == 1 for x in rows); l = sum(x[1] == 0 for x in rows)
        rep[tag] = {'games': len(rows), 'W': w, 'L': l, 'D': len(rows) - w - l,
                    'bt_rating': round(r, 1), 'se': round(se, 1),
                    'rank': rank(r), 'rank_90': [rank(r + 1.645 * se), rank(r - 1.645 * se)],
                    'opp_median_score': sorted(score[x[0]] for x in rows)[len(rows) // 2]}
    losses = [(name.get(o, o), score.get(o), eid) for o, res, tag, eid in ours if tag == 'v63.16' and res == 0]
    rep['v63.16_losses'] = sorted(losses, key=lambda t: -(t[1] or 0))
    json.dump(rep, open(f'{OUT}/report.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != 'v63.16_losses'}, indent=1))
    print('v63.16 losses (opponent, its score, episode):')
    for t in rep['v63.16_losses']:
        print('  ', t)


if __name__ == '__main__':
    main()
