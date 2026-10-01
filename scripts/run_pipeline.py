"""Single entry point for the whole pipeline: fetch -> train -> crown ->
build -> gate. This is what the daily scheduler runs and what the dashboard
'Run pipeline' button invokes. Never submits (house rule); writes a report
with the submit commands and registers the produced model.

    python scripts/run_pipeline.py                 # full daily cycle
    python scripts/run_pipeline.py --on-demand     # mid-day: fetch delta first
    python scripts/run_pipeline.py --fresh-only     # just refresh data + models
"""
from kaggriculture.paths import ROOT
import argparse
import os
import subprocess
import sys

SRC = os.path.join(ROOT, "src")
ENV = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")


def run(cmd, timeout=None):
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, env=ENV, timeout=timeout).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--on-demand", action="store_true",
                    help="fetch delta data before the cycle (mid-day builds)")
    ap.add_argument("--fresh-only", action="store_true",
                    help="only refresh data + retrain models, no crown/build")
    ap.add_argument("--version", default=None, metavar="X.Y",
                    help="explicit version for the built pair (passed through "
                         "to refresh_cycle)")
    args = ap.parse_args()

    if args.fresh_only:
        return run([sys.executable, os.path.join(ROOT, "src", "kaggriculture", "data", "fresh_data.py")])

    # Continuous Rust-engine parity (plan D2): a few episodes through both
    # engines, bit-identical state per step, EVERY day. On mismatch the
    # pre-ranker's recall evidence is deleted so train_gates.preranker_allowed()
    # gates the funnel shut -- the release itself continues, because every
    # release decision runs on the official engine and is untouched by drift.
    kagg = os.path.join(ROOT, "rustengine", "target", "release",
                        "kagg.exe" if os.name == "nt" else "kagg")
    if os.path.exists(kagg):
        code = run([sys.executable,
                    os.path.join(ROOT, "tests", "test_rust_engine.py"),
                    "--episodes", "1", "--replays", "2"], timeout=1800)
        if code != 0:
            recall = os.path.join(ROOT, "models", "lab",
                                  "preranker_recall.json")
            if os.path.exists(recall):
                os.remove(recall)
            print("ALARM: Rust engine parity FAILED -- pre-ranker evidence "
                  "revoked; funnel stays on the official engine only",
                  flush=True)

    cmd = [sys.executable, os.path.join(ROOT, "src", "kaggriculture", "pipeline", "refresh_cycle.py")]
    if args.on_demand:
        cmd.append("--on-demand")
    if args.version:
        cmd += ["--version", args.version]
    return run(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
