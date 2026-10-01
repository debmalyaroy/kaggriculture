# AGENTS.md

Working rules for coding agents in this repo. Details live in [docs/](docs/README.md).

## Workspace
- `D:\codebase\kaggriculture` is the only workspace. Build, edit and keep everything here.
- Scratch, logs and experiment output go in **`.local/`** (gitignored). Never use a home or temp directory.
- Assistant memory lives in **`.local/memory/`** (index: `.local/memory/MEMORY.md`). Read it at session start; never
  write memory under `~/.claude/projects/...`.
- The repo ships **no data**: replays, tapes, routes, weights, builds and generated agents are regenerated
  (see [docs/data.md](docs/data.md)). Never commit them; never download a whole daily dataset.
- Sibling repos `D:\codebase\kaggriculture-sim` (public) and `D:\codebase\kaggle-sim-framework`: when you fix an
  engine / loader / wire / harness bug here, check whether they need the same patch.

## Layout (one repo)
`crates/` Rust agent workspace (agent, engine, runner, ...) · `rustengine/` legacy `kagg` engine CLI ·
`src/kaggriculture/` Python package · `python/` RL-era Python tools (tapes, routes, ladder, release) ·
`configs/` · `scripts/` · `kaggle/` notebooks/kernels · `tests/` · `docs/`. Full map: [docs/repo-layout.md](docs/repo-layout.md).

## Commands
```bash
cargo build --release                         # Rust workspace (crates/*)
(cd rustengine && cargo build --release)      # kagg engine CLI used by the Python package
pip install -e . && pip install -r requirements.txt
python tests/run_all.py --fast                # needs built binaries + regenerated data for the full set
```

## Agent constraints (submission)
- One submission = `main.py` bridge + static Linux `agent-stdio` + its config files, as `submission.tar.gz`.
- Max 10 market orders per turn; `hands` align positionally with `farms[me]["hands"]`; illegal ops are silent no-ops.
- 1 s `actTimeout` per turn. The bridge allows 5 s for the first turn (spawn + route load; ~550 ms on Kaggle) and
  0.25 s for every later turn before it falls back to `fallback.py`.

## Measurement discipline
- The currency is **wins**, never mean margin. Compare candidates **paired** (same tapes / seeds / seats) and decide
  with the sign test (McNemar). Single games and dollar means decide nothing.
- Tape replays are only valid when the **realized world** (first two shops) is unchanged: split every tape A/B into
  same-world vs changed-world games before claiming a gain.
- Validate route/config picks on **held-out** tapes plus real ladder games; confirm in the field (both sides react).
- A layer stays on only if it does not hurt; a reactive layer must act on a signal and never block route actions
  it did not create.

## Operator rules
- Never submit to Kaggle without explicit operator approval; name the live submission that would retire.
- Report times in IST. Keep CPU use within the operator's thread budget (16 threads).
- Keep `docs/` in step with behaviour changes, in the same pass.
