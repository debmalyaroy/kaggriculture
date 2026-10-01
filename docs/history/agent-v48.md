# v48 / v48.1 pair (2026-09-08)

## v48.0 (shipped 56089706 bandit / 56089711 trackp)
- bandit: dispatcher + full pluggable layers (23 flags). sha 3b85eb0f.
- trackp: value-net searcher (sim_search OFF — refuted). main 04052657.

## v48.1 (the "much better pair" — OUR OWN aggressive economy)
Root: the 93.8% public-state router beat both v48 seats 0-8..0-12. Decoded
its edge = aggressive EARLY economy (turn-1 heavy hires + 2 cows/2 sheep +
rapid melons + wheat-arb). Evolved OUR OWN aggressive opening (full [0:144]
window, closed-loop vs the router, intermediate-save; models/trackp/
aggressive_opening.json: vs router -11,358 from base -23k, kaito +19,181).

Built v48.1_bandit (c871865f) + v48.1_trackp (bb618ff4) = aggressive economy
+ our chassis (+ value-net searcher for trackp). GATES:
| gate | v48.1 | v48.0 |
|---|---|---|
| vs 93.8 router | 4-8 (-7,044) | 0-12 (-15,323) |
| vs mid-field kaito | 12-0 (+17,075) | 12-0 |
| reactive panel | PASS 26-0, held-out 12-0 | PASS 26-0 |
| upset gauntlet | 3/17 | 3/17 |
| elite band | 71/120 (+3,292) | 71/120 |
| legality | passed | passed |
STRICT DOMINANCE: better vs the frontier router (4-8 from 0-12), equal
everywhere else. First time OUR OWN economy beats the router at all.

## The winning architecture (proven)
Portfolio {base, aggressive} + learned dispatcher: no single economy beats
all (aggressive beats router/loses mid-field-hard; base opposite). The
aggressive economy narrows the router gap while holding mid-field -> a net
upgrade. Next: add tree_dispatch to SELECT aggressive vs router-class and
base vs mid-field per opponent (dispatcher built, CV 0.798).
