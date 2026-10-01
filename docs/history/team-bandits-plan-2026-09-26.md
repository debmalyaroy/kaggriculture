# Top-40 study and the team-bandit league: breakdown and plan (2026-09-26)

Status: **EXECUTED 2026-09-26/27. Result: negative.** League (22 agents, 22,176 Rust games, all 64 realized worlds):
v63 0.988, v62.1 0.957, best team bandit 0.800 (Kaggledew + v63 shell) with 0.00 vs v63; no team bandit beats v63 in any
realized world. v63's shell helps 7 of 10 team chassis (DECEM +560/-146), hurts mtmr_s1. Tape-routed top teams keep their
economy only in their own games; the league cannot rank candidates at 2500+. See docs/history/issues-and-improvements.md
(2026-09-26/27).

## 1. Data and tools

| What | Where |
|---|---|
| Source | GM's dataset `D:\gm_dataset` (georgymamarin/kaggriculture-episodes, rebuilt 2026-09-26 01:39Z; replays to 2026-09-22) |
| Top-40 | teams.csv `ladder_score` on 2026-09-26: DSM 3100 down to AI是我的豆包 2802 |
| Tapes | `python -m kaggriculture.bandit.top_field --top 40` -> `data/field/top40/tapes/` (3,957 game sides, 39 teams, 947 submissions). Replaying both recorded sides reproduces the real banks exactly (60/60 checked) |
| Daily state + choices | Rust `tapedump` -> `data/field/top40/days.jsonl` (237,420 player-days: board, shed, seeds, prices, rival board, copy share, unit ops, buys, sells, sale prices, final banks) |
| Per-game table | `python -m kaggriculture.bandit.top_study` -> `data/field/top40/games.parquet`; team profiles `report_data.json` |
| Determinism / transplant | `.local/field/determinism_sameworld.parquet`, `.local/field/transplant/result.csv` |

Realized world = the first two shops the game actually unlocked. In every one of the 3,957 games the first shop is
visible at the start of day 3 (step 72) and the second at the start of day 6 (step 144).

## 2. How the top 40 play

### 2.1 One meta, a few different economies

- Most top teams run the same public economy: day 10 = 37 plants, 13 animals; ~280 hires; 2 land buys. Win 0.60.
- A crop-heavy economy (plants on day 10 >= 45: carrots 29 vs 9 per game, tomatoes 7 vs 0, geese 2 vs 0, sheep 4 vs 6)
  wins 0.75, in all 8 first-shop worlds (0.69-0.87 vs 0.57-0.62), with median sales $98k vs $73k. Inside the same team
  it wins more in 14 of 18 teams (THUNDER 0.92 vs 0.56, tetsuya 0.75 vs 0.47, RS Turley 0.72 vs 0.48, DSM 0.84 vs 0.78).
  It is never deep-copied (0% vs 10%). Caveat: a big day-10 farm is partly a result, not only a choice.
- Farm SIZE hardly depends on the world (share of day-10 plant variance explained by the first shop ~0 for all teams
  except KawattaTaido 0.42 and Sida Zuo 0.24). The world changes timing and mix, not size.

### 2.2 Copies

- When a rival copies a top player move-for-move for 300+ steps, the top player wins only 0.36-0.40 (342 games).
  Players who break the mirror on day 2-4 win 0.69-0.77.
- Copies split most often on day 6 (89 of 488 decided mirrors), then days 8, 13, 14. On the split day the winner SELLS
  more (fertilizer 91 vs 38 games, milk 55/32, wheat 47/24, wool 46/27) and drops more into the shed (48/16).

### 2.3 By opponent band (rating at game time)

| | < 2500 | 2500-2800 | > 2800 |
|---|---|---|---|
| games | 3,574 | 336 | 47 |
| win | 0.62 | 0.57 | 0.60 |
| plants / animals day 10 | 37 / 13 | 37 / 12 | 38 / 12 |
| deep-copied (300+ identical steps) | 9% | 3% | 0% |
| rival sells X -> they sold X the same day | 81% | 82% | 85% |

Same team, weaker vs stronger opponents (14 teams): the chassis does not change (plants, animals, mix, hires, land,
end-game share all within noise). The only change: +7.5 points same-day selling with the rival (13/14 teams, t 3.7).
No team has a "strong-opponent mode"; the reactive layer fires more because strong rivals sell in the same windows.
THUNDER THUNDER (0.70 / 0.43 / 0.29) and Planned Economy (0.77 / 0.47 / 0.25) lose their edge against stronger
opponents; DECEM (0.79 / 0.63 / 0.83), Arda Ceylan, 摆烂小分队 hold.

### 2.4 World-indexed logic: where each team's tapes branch

Median first step at which two tapes of the same submission differ, split by (different first shop / same first
shop, different second / same world), and where the branch falls (before step 72 = before any shop is known,
72-143 = after the first shop, 144+ = after both):

| team | different first shop | same first, different second | same world | reading |
|---|---|---|---|---|
| DSM (#1) | 358 (97% after 144) | 358 | 358 | **world-agnostic until ~day 15**, then diverges for non-world reasons |
| DECEM (#2) | 29 (53% before 72) | 25 | 361 | branches on day 1 before any shop is known: **reactive** (rival or seed) |
| Unknown Mother-Goose | 177 | 121 (57% at 72-143) | - | first-shop + second-shop routing |
| M & M & P & Q | 24 (74% before 72) | 24 | 24 | **reactive from day 1** (crop-heavy) |
| Kaggledew Valley | 719 | 719 | 719 | **one tape for every world**: no routing, no reaction |
| 吃白饭的大肥鱼 | 96 (97% at 72-143) | 108 | 120 | **routes on the FIRST shop** as soon as it opens (crop-heavy) |
| mtmr_s1 | 263 | 436 | - | late world routing |
| THIRD FARM CLUB | 144 | 145 | 299 | our-style router at step 144 |
| tetsuya & yuanzhe & guoqin | 144 | 168 | - | router at 144 |
| Azat Akhtyamov | 72 | 122 | 150 | first-shop router + reactive |
| Arda Ceylan | 341 | 534 | - | late |
| Densike | 690 | 693 | 693 | one tape to the last day |
| Smackaveli | 144 | 144 | 247 | router at 144 |
| Navier-stokes | 150 | 150 | 223 | router at ~144-150 (our lineage) |
| ymg_aq | 145 | 120 | - | first/second-shop router |
| Artem The Farmer | 150 | 150 | 217 | router at 144 (crop-heavy) |

Four families: (a) one tape everywhere (Kaggledew, Densike to day 29); (b) world-agnostic opening to day 12-15, then
state-driven (DSM, Arda, mtmr); (c) shop routers at step 144 like ours (Navier, THIRD FARM, tetsuya, Smackaveli, Artem)
or on the first shop at ~96-120 (吃白饭的大肥鱼, Mother-Goose, ymg); (d) reactive from day 1 (M & M & P & Q, DECEM, Azat).

### 2.5 Can their play be simulated from their tapes?

- Determinism: same submission + same realized world -> identical moves for the whole game in 59% of pairs
  (Kaggledew 99%); identical to day 5.7 in 94%.
- Transplant (a team's tape from another of its games in the SAME world, replayed against this game's real opponent):
  keeps 97% of the real bank (100% from the same submission), but win falls 0.64 -> 0.44. A tape from a different
  world: 93% / 0.37. So their ECONOMY is reproducible per world; ~20 win points are in-game reaction.
- Not tape-simulable (tape keeps only 69-87% of bank): THUNDER, DECEM, M & M & P & Q, Artem, HowardLeeTW.

## 3. The plan: ten team bandits in a league

Idea (operator, 2026-09-26): our bandit is a tape router plus a reactive shell. Build one bandit per top team: its
routes are that team's own tapes, its router dispatches them per world, and our reactive shell rides on top. All
routes of one agent come from one player (ideally one submission), so their prefixes agree and desync risk is low.
Then run them against each other, v63 and v62.1.

### 3.0 Data freshness (operator, 2026-09-26)

The leaderboard moves every day, so the agents are built from THIS WEEK's top 50 and each team's CURRENTLY ACTIVE
submissions, not from their whole history. GM's dataset is not enough for that: it indexes only a few games per
submission for most top teams (DSM 6 games from its last-7-day submissions, DECEM 6, M & M & P & Q 2; exceptions
Navier-stokes 386, ActiveMusyoku 124, Mother-Goose 93) and has no replays from 2026-09-23 on. Source instead, via the
Kaggle CLI, run by the operator (Claude does not download):
1. `.local/field/week/1_list.sh`: `leaderboard --show` (top 50 + team ids) -> `team-submissions <team_id>` (active
   submissions) -> `episodes <submission_id>` (their games). Listing calls only.
2. Claude builds `worklist.txt`: completed ladder games of those submissions, newest first, capped per submission
   (default 60), minus games GM already holds.
3. `.local/field/week/2_download.sh`: `replay <episode_id>` per worklist line, gzipped on arrival (~2 MB each;
   ~6,000 replays ~ 12 GB on disk at most). Resumable.
Then the same tools (top_field -> tapes, tapedump, top_study) run on the week's replays, and agents are keyed to
active submissions. Team membership of the top 50 is re-read each time the league is rebuilt.

### 3.1 The ten agents

Provisional (from GM history; to be re-selected from this week's top 50 and their active submissions once
section 3.0's data is in). Chosen for coverage: one submission with tapes in >= 40 realized worlds, so the agent is one algorithm, not a blend.

| agent | team (rank) | submission tapes / worlds | family |
|---|---|---|---|
| TB-DSM | DSM (1) | 230 / 61 | world-agnostic to day 15 |
| TB-DECEM | DECEM (2) | 72 / 42 | reactive from day 1 |
| TB-MMPQ | M & M & P & Q (6) | 105 / 53 | reactive, crop-heavy |
| TB-KAGG | Kaggledew Valley (7) | 590 / 64 | one tape |
| TB-DENS | Densike (16) | 108 / 52 | one tape to day 29, few animals |
| TB-NAVI | Navier-stokes (19) | 360 / 64 | router at 144 |
| TB-THUN | THUNDER THUNDER (34) | 137 / 61 | reactive |
| TB-PLAN | Planned Economy (39) | 80 / 42 | tape |
| TB-ARDA | Arda Ceylan (15) | team: 106 / 49 (48 subs) | late router; multi-submission = higher desync risk |
| TB-BAIL | 摆烂小分队 (31) | team: 158 / 59 (86 subs) | multi-submission |

Crop-heavy 吃白饭的大肥鱼 (12 tapes in its best submission) and Artem (7) have too few tapes per submission for a
64-world agent; they are optional extras with a "nearest world" fallback.


### 3.1a Selection from THIS WEEK's top 50 (supersedes 3.1)

Leaderboard read 2026-09-26 ~14:40Z (`.local/field/week/top50_lb.json`): heavy churn vs GM's 01:39Z snapshot (19 new
teams; DSM, GM's #1, is no longer in the top 50; #1 Boey 3076.8, #2 Fourth Quadrant, #3 Vadim Vasilenko, #4 M & M & P & Q,
#5 DECEM; #50 = 2735.6). Their CURRENT submissions are days old and mostly absent from GM, but their PREVIOUS
submissions are in GM (operator: build from those). Coverage (`.local/field/week/top50_gm_coverage.csv`): 29 teams have
>= 30 GM replays; only 8 have a single submission with >= 30; the top 3 have 3 / 7 / 2 replays; Pii, sekai013, IsaiahP,
YumeNeko have none. So each agent POOLS the team's submissions. The trie router keeps pooling desync-safe: it switches
only to a tape whose recorded moves so far equal what the agent actually played, so versions add branches, never
splices.

Tapes: `python -m kaggriculture.bandit.top_field --top 50 --leaderboard .local/field/week/top50_lb.json --out data/field/week50`.

Agents = the ten highest-ranked teams THIS WEEK with >= 45 pooled replays (world coverage checked after extraction):
M & M & P & Q (#4, 148, crop-heavy), DECEM (#5, 184), Yizhou (#9, 53), THIRD FARM CLUB (#12, 92), mtmr_s1 (#13, 52),
tetsuya & yuanzhe & guoqin (#15, 117), My second life (#17, 65), kigasudayooo (#18, 86), We wanna be tomatos (#20, 69),
Kaggledew Valley (#21, 624). Optional crop-heavy extras: 吃白饭的大肥鱼 (#10, 31), Artem (#23, 28). Not buildable: Boey,
Fourth Quadrant, Vadim Vasilenko, Majkel1337 (<10 replays each).

### 3.2 Building a team base (Python, `kaggriculture.bandit.team_base`)

For each agent: `configs/bandit/bases/team/<agent>/routes.json` = the team's recorded action streams for its seat
(same 719-step format as our routes) and `router.json` in a new `"mode": "trie"`:
- every route carries its realized world and its game's result;
- the prefix tree of the routes (the steps where they branch) is precomputed.

### 3.3 The trie router (Rust, `agent/src/router.rs`, config-driven, off by default)

At each step the agent follows its current tape. At a branch point it may switch only to a tape whose recorded
moves so far equal what it has actually played AND whose unit squares equal the live ones (exact sync, so the
switch cannot desync). Among those it picks by: (1) the realized world so far (shops seen), (2) nearest recorded
situation (money, lead, board, rival board: the fieldplay distance), (3) the tape's game result. Branches before a
shop is visible (DECEM, M & M & P & Q) fall back to (2)-(3). A tape whose world never matches keeps its nearest-world
continuation. The existing v61.1 router stays untouched; `mode` absent = today's behaviour bit for bit.

### 3.4 Shell profiles

| profile | what runs on top of the team tape |
|---|---|
| S0 | nothing: the team's tapes as routed (fidelity baseline) |
| S1 | market shell: sell_lead, racepx, v92 rival forecaster (library = all league routes), end-game planner/tsell, hfeed |
| S2 | full v63 shell (adds weed repair, hand align, clone controls) |

Layers that assume the v61.1 base (PIPE rewriting route 0 at steps 57/91, r36 route indices) are switched off for
team bases by knob.

### 3.5 Gates before the league (each agent)

1. **Fidelity (S0):** in the team's own recorded games (same seed, recorded opponent), in-sample the router must
   pick the game's own tape and reproduce the real bank exactly; leave-one-out must reach the transplant level
   (>= 95% of real bank).
2. **Desync:** in new seeds, share of steps where the live unit squares differ from the followed tape's recorded
   squares, and count of engine-rejected unit actions. Target < 2% of steps; report per agent.
3. **Latency:** worst turn < 100 ms (Rust).

### 3.6 The league (Rust `selfplay`, both seats, two different bases)

- Players: 10 team bandits x {S0, S1} + v63 + v62.1 = 22 agents.
- Worlds: seeds chosen by REALIZED world (record the pair each game actually played; fill every one of the 64
  pairs, >= 2 seeds each) - the tournament's PASS-labels do not match real play (0/64), so no labels are trusted.
- Size: 231 pairings x 128 seeds x 2 seats ~ 59k games; Rust ~0.5 s/game on 16 threads ~ 30-35 min.
- Report: win matrix, Elo-style ranking, per realized world, per family; paired S1-vs-S0 for each team (does our
  shell help other chassis?), and each team bandit vs v63 / v62.1.

### 3.7 What we learn and what would ship

- Does our shell add wins on other chassis (S1 vs S0, paired)? If yes, the shell is chassis-independent.
- Which chassis + our shell is strongest, per world? A team bandit that beats v63 across worlds is a candidate NEW
  CHASSIS (the crop-heavy M & M & P & Q is the one to watch).
- A per-world chassis choice (dispatch the best team's routes per realized world) is the natural v64: the D6
  dispatch layer already swaps routes at step 144, and all routes would come from one team per world.
- Any shipping candidate still goes through the realized-world tournament, the official-engine tarball check and
  the operator's go. Nothing ships from the league alone.

### 3.8 Risks

- Their tapes were recorded against other opponents; a tape-routed team is weaker than the real team by ~20 win
  points (transplant). The league ranks CHASSIS, not the real players.
- Reactive teams (DECEM, M & M & P & Q, THUNDER) branch before any world signal; the trie falls back to nearest
  situation, which may desync more (gate 2 measures it).
- The v92 forecaster's library changes (league routes), so S1 is not exactly v63's shell.
- License/fair-use: these are public ladder replays from a public dataset; the agents stay local and private.

### 3.9 Order of work and ETA

1. team_base builder + 10 bases (1 h)
2. trie router in Rust + unit test "mode absent = unchanged" (md5 of existing screens) (2 h)
3. gates 1-3 per agent (30 min)
4. seed search by realized world (Rust, 15 min)
5. league S0 + S1 (35 min at 16 threads, low priority) and report (30 min)

About 5 hours to the first league result.
