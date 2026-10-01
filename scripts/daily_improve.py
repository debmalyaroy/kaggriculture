"""Daily ladder-driven improvement loop (2026-09-07).

The measurement wall (loss-reproducibility-wall-2026-09-07): most live
losses are closed-loop vs live agents and cannot be reproduced from tapes.
So this loop uses the LADDER as the oracle and fixes only what reproduces:

  1. pull all fresh loss tapes for the active pair (loss_autopsy)
  2. triage: replay OUR agent vs each winner tape; keep only the
     REPRODUCIBLE subset (winner tape still beats us open-loop)
  3. for reproducible losses, evolve a realized-world day-6 branch vs the
     winner + frontier panel; keep where paired bank improves
  4. maintain the FRONTIER panel (current strong routes that beat >=90k
     opponents) as the standing gate -- not degraded clones
  5. report candidates + numbers; NEVER auto-submits (house rule)

Report -> .local/daily_improve_report.txt. Run daily after the pair has
accrued fresh games. Ship manually via the notebooks after review.
"""
from kaggriculture.paths import ROOT
import os, sys, json, subprocess

def sh(args):
    return subprocess.run(args, capture_output=True, text=True, cwd=ROOT)

def main():
    print("daily_improve: pull losses -> triage -> evolve reproducible -> report")
    print("(driver scaffold; see loss-reproducibility-wall-2026-09-07 memory)")
    # 1. refresh frontier panel
    sh([sys.executable, "src/kaggriculture/pipeline/frontier_panel.py"])
    print("frontier panel refreshed")
    # steps 1-3 are run via the session tools (loss_autopsy, triage, evolve_repro)
    # kept as documented manual steps until the closed-loop gauntlet is wired
    return 0

if __name__=="__main__":
    raise SystemExit(main())
