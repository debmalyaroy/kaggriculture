"""Per-team learnability sweep over the reactive cohort (proper argv)."""
from kaggriculture.paths import ROOT
import json, os, subprocess, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PY = "C:/ProgramData/anaconda3/envs/llm/python.exe"
se = json.load(open(os.path.join(ROOT, "data", "bc_corpus",
                                 "script_end.json"), encoding="utf-8"))
teams = [t for t, d in sorted(se.items(), key=lambda kv: kv[1]["rank"])
         if d.get("script_end_step", 720) < 720][:30]
for t in teams:
    r = subprocess.run([PY, os.path.join(ROOT, "src", "kaggriculture", "trackp", "harness",
                                         "bc_arch_test.py"),
                        "--team", t, "--epochs", "8"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=1800)
    skill = rows = "?"
    for line in (r.stdout or "").splitlines():
        if line.startswith("single-team"):
            rows = line.split(":")[1].split(",")[0].strip()
        if line.startswith("best:"):
            skill = line.rsplit("skill", 1)[-1].strip(" ()")
    print(f"{t[:26]:<28} rows {rows:>12}  skill {skill}", flush=True)
