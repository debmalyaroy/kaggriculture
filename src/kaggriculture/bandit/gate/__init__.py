"""Bandit gate/tuning harness (tracked source; scratch outputs go to .local).

Reusable RC-pipeline utilities:
  * harness    -- shared config injection, world-diverse seeds, build+eval
  * rail_sweep -- general knob-grid sweep, world-gated, margin-aware
  * world_gate -- NN-on vs NN-off candidate gate, per-world breakdown

The isolated Rust binary and per-stage build artifacts stay under .local
(regenerable, gitignored); the LOGIC lives here in source.
"""
