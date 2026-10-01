# Kaggriculture

Agents and tooling for the Kaggle **[Kaggriculture](https://www.kaggle.com/competitions/kaggriculture)** simulation
competition: a two-player farming game (30 days = 720 turns) where the larger final bank balance wins and the
leaderboard is a skill rating built from head-to-head results.

The competition is closed. The final agents were Rust "route + layers" agents (v63.x): a recorded top-player
**route** per world, a **chassis** of repair guards that keeps the route legal, and a chain of **reactive layers**
(sale timing, races, endgame, economy) configured by managers in `agent.json`.

| Release | Submission | What it added | Live result |
|---|---|---|---|
| v63.14_rl_f898 | 56718979 | f898 chassis + land_repair + milk routes + endgame/economy managers | 41 W / 12 L / 1 D (54 games) |
| v63.16_rl_f898 | 56720196 | 4 world routes + `wb3` wheat-buy reorder only on signal with a cash guard | 53 W / 11 L (65 games) |
| v63.17_rl_f898 | built, not submittable (deadline passed) | sale cash floor + 10 re-screened world routes | 343 / 356 replayed live games (v63.16: 328) |

Details and lessons: [docs/results.md](docs/results.md).

## Repository

```
crates/              Rust workspace: agent (chassis, managers, layers), engine, runner (tape/field tools), policy, ...
rustengine/          legacy `kagg` engine port + serve/batch/pre-ranker CLI used by the Python package
src/kaggriculture/   Python package: engine bridge, data fetch, measurement, training, pipeline, agent builds
python/              RL-era Python tools: tapes, route screens, ladder pulls, field harness, release helpers
configs/             agent / base / profile / shell / opponent configs (small JSON; generated tables are not kept)
scripts/             build, submission and scheduling scripts (PowerShell / bash / batch)
kaggle/              Kaggle notebook + kernel templates (private submission kernel, payload dataset metadata)
tests/               test suites (python tests/run_all.py)
docs/                documentation; docs/history/ keeps the dated design notes and journals
research/            research notes and the standalone OR tape-generator crate
```

Full map: [docs/repo-layout.md](docs/repo-layout.md). How the agent works:
[docs/agent-architecture.md](docs/agent-architecture.md); the system around it: [docs/architecture.md](docs/architecture.md). Agent working rules: [AGENTS.md](AGENTS.md).

## Quick start

```bash
pip install -e . && pip install -r requirements.txt      # Python package + kaggle-environments
cargo build --release                                    # Rust workspace -> target/release/{agent-stdio, chassis-vs-tapes, ...}
(cd rustengine && cargo build --release)                 # kagg engine CLI
python tests/run_all.py --fast
```

The repository ships **no data** (replays, tapes, route tables, weights, builds). Regenerate what you need first:
[docs/data.md](docs/data.md). Build, measure and release workflows: [docs/workflows.md](docs/workflows.md),
[docs/release.md](docs/release.md).

## License

MIT — see [LICENSE](LICENSE).
