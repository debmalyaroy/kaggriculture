"""Win-plan harness (2026-09-24): automated measurement for the plan to win.

Measurement only -- harvest public agents, gate candidates on the Rust serve
engine, round-robins, loss traces, top-team replay mining. It never submits.

    python -m kaggriculture.winplan.runner status
    python -m kaggriculture.winplan.runner run            # every ready task
    python -m kaggriculture.winplan.runner run S1.1       # one task
"""
