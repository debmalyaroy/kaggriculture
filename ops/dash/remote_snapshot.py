"""Box side of the RL dashboard (ops/dash/build.py runs this over ssh): one JSON snapshot on stdout.

    python ops/dash/remote_snapshot.py

PPO metrics of the newest runs, the branch-oracle batch log, gate/band/tournament results, candidates,
queue status and box load. Read-only: it never writes anything.
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def jl(path, keep=None):
    out = []
    try:
        for ln in open(path, encoding="utf-8"):
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass
    except OSError:
        pass
    return out[-keep:] if keep else out


def main():
    snap = {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    runs = sorted(glob.glob(os.path.join(RL, "weights", "ppo", "ppo-*")), key=os.path.getmtime)
    snap["runs"] = []
    for r in runs[-3:]:
        snap["runs"].append({"name": os.path.basename(r), "metrics": jl(os.path.join(r, "metrics.jsonl")),
                             "snapshots": sorted(os.path.basename(s) for s in glob.glob(os.path.join(r, "snapshots", "*.bin")))})
    # oracle batches: "[oracle] TAG: N games, D decisions, U where the choice matters, Ss"
    orc = []
    pat = re.compile(r"\[oracle\] (\S+): (\d+) games, (\d+) decisions, (\d+) where the choice matters, (\d+)s")
    try:
        for ln in open(os.path.join(RL, "data", "ops", "logs", "Q19.log"), encoding="utf-8", errors="replace"):
            m = pat.search(ln)
            if m:
                orc.append({"tag": m[1], "games": int(m[2]), "decisions": int(m[3]), "useful": int(m[4]), "sec": int(m[5])})
    except OSError:
        pass
    snap["oracle"] = orc[-200:]
    gates = []
    for f in sorted(glob.glob(os.path.join(RL, "data", "gates", "*.json")), key=os.path.getmtime)[-40:]:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        gates.append({"file": os.path.basename(f), **{k: v for k, v in d.items() if not isinstance(v, (list, dict))}})
    snap["gates"] = gates
    cands = []
    for d in sorted(glob.glob(os.path.join(RL, "data", "candidates", "*"))):
        rep = {}
        try:
            rep = json.load(open(os.path.join(d, "report.json"), encoding="utf-8"))
        except (OSError, ValueError):
            pass
        cands.append({"name": os.path.basename(d), "pending": os.path.exists(os.path.join(d, "RC_PENDING.json")),
                      **{k: v for k, v in rep.items() if not isinstance(v, (list, dict))}})
    snap["candidates"] = cands
    q = subprocess.run([sys.executable, os.path.join(RL, "ops", "queue.py"), "status"], capture_output=True, text=True, cwd=RL)
    snap["queue"] = q.stdout[-6000:]
    mem = {}
    for ln in open("/proc/meminfo"):
        k, v = ln.split(":")
        mem[k] = int(v.split()[0]) / 1024 / 1024
    cpu = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    time.sleep(2)
    cpu2 = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    busy = [b - a for a, b in zip(cpu, cpu2)]
    idle = busy[3] + busy[4]
    du = shutil.disk_usage(RL)
    snap["box"] = {"load1": os.getloadavg()[0], "ncpu": os.cpu_count(), "cpu_pct": round(100 * (1 - idle / max(1, sum(busy))), 1),
                   "mem_total_gb": round(mem["MemTotal"], 1), "mem_used_gb": round(mem["MemTotal"] - mem["MemAvailable"], 1),
                   "disk_free_gb": round(du.free / 2 ** 30, 1)}
    json.dump(snap, sys.stdout)


if __name__ == "__main__":
    main()
