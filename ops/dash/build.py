"""Laptop side of the RL dashboard: pull a snapshot from the box and render one self-contained page.

    python ops/dash/build.py [--host ec2-user@<box-ip>] [--key ~/.ssh/aicm-key-openssh.pem]

Writes .local/dash/rl-learning.html (published as a private Artifact by the monitoring session).
Keeps a box-load history in .local/dash/box_history.jsonl (one row per poll). Annotations come from
ops/dash/events.json (dated fixes/restarts); the written reading of the numbers from .local/dash/notes.json
({"t": ..., "headline": ..., "points": [...]}) which the monitoring session writes each poll.
"""
import argparse
import json
import os
import subprocess
import sys

RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(RL, ".local", "dash")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="ec2-user@<box-ip>")
    ap.add_argument("--key", default=os.path.expanduser("~/.ssh/aicm-key-openssh.pem"))
    ap.add_argument("--snap", help="use this snapshot file instead of ssh")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.snap:
        snap = json.load(open(a.snap, encoding="utf-8"))
    else:
        r = subprocess.run(["ssh", "-i", a.key, "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=20", a.host,
                            "cd ~/krl && .venv/bin/python ops/dash/remote_snapshot.py"], capture_output=True, text=True, timeout=180)
        if r.returncode != 0:
            sys.exit(f"snapshot failed: {r.stderr[-500:]}")
        snap = json.loads(r.stdout)
    hist_p = os.path.join(OUT, "box_history.jsonl")
    with open(hist_p, "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": snap["t"], **snap["box"]}) + "\n")
    snap["box_history"] = [json.loads(x) for x in open(hist_p, encoding="utf-8").read().splitlines()[-500:]]
    try:
        snap["events"] = json.load(open(os.path.join(RL, "ops", "dash", "events.json"), encoding="utf-8"))
    except (OSError, ValueError):
        snap["events"] = []
    try:
        snap["notes"] = json.load(open(os.path.join(OUT, "notes.json"), encoding="utf-8"))
    except (OSError, ValueError):
        snap["notes"] = {}
    prof = json.load(open(os.path.join(RL, "configs", "profiles", "rl3.json"), encoding="utf-8"))["profiles"]
    snap["profile_names"] = [p.get("name", str(i)) for i, p in enumerate(prof)]
    tpl = open(os.path.join(RL, "ops", "dash", "template.html"), encoding="utf-8").read()
    html = tpl.replace("/*__DATA__*/null", json.dumps(snap, separators=(",", ":")).replace("</", "<\\/"))
    out = os.path.join(OUT, "rl-learning.html")
    open(out, "w", encoding="utf-8").write(html)
    print(f"[dash] {out} ({len(html) // 1024} KB) snapshot {snap['t']}")


if __name__ == "__main__":
    main()
