"""Classify every live loss of a submission by mechanism (needs a traced replay: KRL_TRACE_DIR run of the matching config).
    python python/release/live_loss_classify.py RESULTS.tsv TRACE_DIR
Classes (first match wins):
  cash_starved  our money < $50 at some day start in days 3-8 while the rival had > 2x ours, and the rival sold an item at a
                day start that we did not (our layers withheld a route sale) -- the mirror-race starvation
  collapse      >= 20 empty unlocked tiles or >= 3 animals stuck in the shed at days 18-24
  late          ahead at day 24 (sales / endgame race)
  midgame       behind from the middle (days 6-24) with a healthy farm
"""
import collections, sys

res, trd = sys.argv[1], sys.argv[2]
rows = [l.split('\t') for l in open(res).read().splitlines()[1:]]
ANIM = ('COW', 'SHEEP', 'GOOSE', 'CHICKEN')
out = collections.defaultdict(list)
for x in rows:
    if float(x[6]) > 0:
        continue
    D, M = {}, collections.defaultdict(dict)
    for l in open(f'{trd}/{x[0]}.trace.tsv'):
        p = l.rstrip('\n').split('\t')
        if p[0] == 'D':
            D[(int(p[1]), p[2])] = p
        elif p[0] == 'M':
            M[int(p[1])][p[2]] = p[3]
    starved = any(float(D[(s, 'us')][3]) < 50 and float(D[(s, 'them')][3]) > 2 * float(D[(s, 'us')][3])
                  for s in range(72, 193, 24) if (s, 'us') in D and (s, 'them') in D)
    withheld = sum(1 for s in range(48, 200) if s % 24 == 0 and 'SELL' in M[s].get('them', '') and 'SELL' not in M[s].get('us', ''))
    stuck = idle = 0
    for s in (432, 576):
        if (s, 'us') in D:
            p = D[(s, 'us')]
            shed = dict(y.split(':') for y in p[13].split(',') if y)
            stuck = max(stuck, sum(int(shed.get(a, 0)) for a in ANIM))
            idle = max(idle, int(p[6]))
    g24 = float(x[14]) - float(x[15])
    c = ('cash_starved' if starved and withheld else 'collapse' if idle >= 20 or stuck >= 3 else 'late' if g24 > 0 else 'midgame')
    out[c].append((x[0], x[3], x[7], float(x[6]), g24))
n = sum(len(v) for v in out.values())
print(f'{n} losses')
for c in ('cash_starved', 'collapse', 'late', 'midgame'):
    v = out.get(c, [])
    print(f'{c:13} {len(v):3}  median gap {sorted(t[3] for t in v)[len(v) // 2] if v else 0:+8.0f}')
    for t in sorted(v, key=lambda t: t[3]):
        print(f'     {t[0]:13} {t[1]:30} r{t[2]:5} gap {t[3]:+8.0f}  d24 {t[4]:+8.0f}')
