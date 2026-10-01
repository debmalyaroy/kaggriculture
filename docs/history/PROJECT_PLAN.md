# PROJECT_PLAN.md

Updated 2026-08-14. This file is a pointer, not the plan — the living
documents are:

| document | role |
|---|---|
| `CLAUDE.md` | operating rules, constraints, tool map |
| `.local/docs/pipeline.html` | **design document**: the pipeline and agents as built, with the measurement behind every choice |
| `.local/docs/track-p.html` | **the forward plan**: closed-loop planner lineage (Track P, v5) — agent architecture, model zoo, self-play spec, go/no-go calendar to 23 Sep |
| `.local/docs/plan.html` | the 13-Aug execution plan — **CLOSED 2026-08-14**, every item executed/measured/parked |
| `docs/history/issues-and-improvements.md` | every measurement and reversal, chronologically, with numbers |
| `BUILD_JOURNAL.md` | day-by-day chronology |
| `MEMORY.md` | current state of play |

## Position (2026-08-14 night)

Live pair: **v25.0_route** (peaked 2406, project best) + **v25.1_bandit**
(first GRU deployment). The daily 04:30 release **publishes unattended**
(gate / already-live / quota / sha rails). Ship strategy: keep shipping
{bandit, route}; train the closed-loop planner internally; on graduation the
pair becomes **{bandit, closed-loop}**. Entry deadline: 23 Sep 2026.
