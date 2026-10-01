"""Release-candidate assembly (queue Q18): the verified RC base tarball + a policy's weights.

    python python/learn/rc.py build --weights weights/ppo/<run>/best.bin --name v63.0_bandit
    python python/learn/rc.py daily                     # the 18:00Z decision (see below)

The RC base (data/builds/rc_base/submission.tar.gz, built on the laptop by
scripts/build_submission.ps1 -Policy ... and checked on the official engine: 719/719 turns through
the Rust bridge, 0 fallback) holds the x86_64 static agent with --policy support, the bridge,
base tables, profiles_v2 and the Python v61.1 fallback. A candidate = that tarball with
policy.bin replaced, so no x86 build is needed on the ARM box. File modes are kept (agent-stdio
0755). Output: data/builds/<name>/submission.tar.gz + build.json (shas, weights, verdicts).

`daily` picks the best tournament candidate whose verdicts ALL pass (tournament vs v61.1; panel gate vs
v61.1 with no worse opponent; beats v62 and v62.1 head-to-head; per-band objective no worse than
v62.x; and beats the previous RC if one exists), builds it, and writes data/builds/RC_PENDING.json
for the operator. It never submits.
"""
import argparse
import datetime as dt
import glob
import shutil
import hashlib
import io
import json
import os
import sys
import tarfile

RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(RL, "data", "builds", "rc_base", "submission.tar.gz")
BUILDS = os.path.join(RL, "data", "builds")
GATES = os.path.join(RL, "data", "gates")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def build(weights, name):
    wb = open(weights, "rb").read()
    out_dir = os.path.join(BUILDS, name)
    os.makedirs(out_dir, exist_ok=True)
    buf = io.BytesIO()
    with tarfile.open(BASE, "r:gz") as src, tarfile.open(fileobj=buf, mode="w:gz") as dst:
        for m in src.getmembers():
            if m.name == "policy.bin":
                continue
            data = src.extractfile(m).read() if m.isfile() else None
            if m.name == "main.py":
                data = data.replace(b'BUILD = "', f'BUILD = "{name} w={sha(wb)[:12]} '.encode(), 1)
                m.size = len(data)
            dst.addfile(m, io.BytesIO(data) if data is not None else None)
        info = tarfile.TarInfo("policy.bin")
        info.size, info.mode, info.mtime = len(wb), 0o644, int(dt.datetime.now().timestamp())
        dst.addfile(info, io.BytesIO(wb))
    tgz = buf.getvalue()
    open(os.path.join(out_dir, "submission.tar.gz"), "wb").write(tgz)
    meta = {"name": name, "built": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "weights": os.path.relpath(weights, RL), "weights_sha": sha(wb), "tarball_sha": sha(tgz),
            "base_sha": sha(open(BASE, "rb").read()), "profile": None, "policy": "policy.bin"}
    json.dump(meta, open(os.path.join(out_dir, "build.json"), "w"), indent=1)
    print(f"[rc] built {name}: tarball {meta['tarball_sha'][:12]} weights {meta['weights_sha'][:12]} -> {out_dir}")
    return meta


CANDS = os.path.join(RL, "data", "candidates")


def hand_off(weights, name, tournament, panel):
    """No RC base on this machine (the ARM box): copy the candidate's weights + its verdicts to
    data/candidates/<box>-<name>/ and append it to data/candidates/RC_PENDING.json."""
    box = os.environ.get("KRL_BOX", "A")
    d = os.path.join(CANDS, f"{box}-{name}")
    os.makedirs(d, exist_ok=True)
    blob = open(weights, "rb").read()
    open(os.path.join(d, "policy.bin"), "wb").write(blob)
    # for the laptop's x86 Rust==torch check (export_check.py --weights <dir>): the deployed bytes as
    # weights.bin plus the torch checkpoint they came from (a PPO best.pt, or a BC run's ckpt.pt)
    open(os.path.join(d, "weights.bin"), "wb").write(blob)
    src = os.path.dirname(weights)
    for ck in ([os.path.join(src, "best.pt")] if os.path.basename(weights) == "best.bin" else []) + [os.path.join(src, "ckpt.pt")]:
        if os.path.exists(ck):
            shutil.copy(ck, os.path.join(d, "ckpt.pt"))
            break
    meta = {"name": f"{box}-{name}", "box": box, "dir": os.path.relpath(d, RL).replace(os.sep, "/"), "weights": weights,
            "weights_sha": sha(blob), "candidate": tournament["candidate"], "tournament": tournament,
            "panel": panel, "t": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "pulled": False}
    json.dump(meta, open(os.path.join(d, "report.json"), "w"), indent=1)
    idx = os.path.join(CANDS, "RC_PENDING.json")
    pend = json.load(open(idx)) if os.path.exists(idx) else []
    pend.append({k: meta[k] for k in ("name", "box", "dir", "weights_sha", "candidate", "t", "pulled")})
    json.dump(pend, open(idx + ".tmp", "w"), indent=1)
    os.replace(idx + ".tmp", idx)
    return meta


def latest(pattern):
    fs = sorted(glob.glob(os.path.join(GATES, pattern)))
    return json.load(open(fs[-1])) if fs else None


def daily(name):
    """Build an RC only when every verdict for the tournament's best candidate passes."""
    t = json.load(open(os.path.join(GATES, "tournament_latest.json"))) if os.path.exists(os.path.join(GATES, "tournament_latest.json")) else None
    if not t:
        print("[rc] no tournament yet; nothing to build")
        return
    for c in t["ranking"]:
        if c["verdict"] != "PASS":
            continue
        panel = latest(f"panel__{c['candidate']}__*.json")
        if not panel:
            print(f"[rc] {c['candidate']}: tournament PASS, waiting for its panel gate")
            continue
        checks = panel.get("checks", {})
        failed = [k for k, v in checks.items() if not v]
        if failed:
            print(f"[rc] {c['candidate']}: panel checks failed: {failed}")
            continue
        band = latest(f"band__{c['candidate']}__*.json")
        if not band:
            print(f"[rc] {c['candidate']}: waiting for its band gate")
            continue
        if not (band["pass"] or band["no_worse_than_ref"]):
            print(f"[rc] {c['candidate']}: band gate: {band['losses_below_2500']} losses below 2500 (ref {band['ref_losses_below_2500']}), "
                  f"2500+ {band['win_rate_2500plus']:.3f} (ref {band['ref_win_rate_2500plus']:.3f}) -- worse than the reference (v63)")
            continue
        checks = dict(checks, band_objective=band["pass"], band_no_worse_than_ref=band["no_worse_than_ref"])
        panel = dict(panel, band=band)
        w = os.path.join(RL, c["weights"]) if not os.path.isabs(c["weights"]) else c["weights"]
        if not os.path.exists(BASE):
            # the AWS box: hand the candidate to the laptop (aws/pull_rc.sh), which builds the x86 tarball,
            # runs the full band-scored tournament and the official-engine check, and asks the operator
            meta = hand_off(w, name, c, panel)
            print(f"[rc] CANDIDATE READY for the laptop tournament: {name} ({c['candidate']}) -> {meta['dir']}")
            return
        meta = build(w, name)
        meta.update(candidate=c["candidate"], tournament=c, panel_checks=checks)
        json.dump(meta, open(os.path.join(BUILDS, "RC_PENDING.json"), "w"), indent=1)
        print(f"[rc] RC READY for operator approval: {name} ({c['candidate']})")
        return
    print("[rc] no candidate passes every check today; the live pair stays")


def cycle(name, top=2):
    """After a tournament: panel-gate its top PASS candidates (skipping ones already gated today), then `daily`."""
    import subprocess
    t = os.path.join(GATES, "tournament_latest.json")
    if not os.path.exists(t):
        print("[rc] no tournament yet")
        return
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    passing = [c for c in json.load(open(t))["ranking"] if c["verdict"] == "PASS"][:top]
    for c in passing:
        if glob.glob(os.path.join(GATES, f"panel__{c['candidate']}__{today.replace('-', '')}*.json")):
            continue
        w = c["weights"] if os.path.isabs(c["weights"]) else os.path.join(RL, c["weights"])
        workers = os.environ.get("KRL_THREADS") or str(max(2, (os.cpu_count() or 4) - 2))
        subprocess.run([sys.executable, os.path.join(RL, "python", "panel_gate.py"), "--cand", w, "--name", c["candidate"],
                        "--date", today, "--workers", workers], check=True)
        # the objective gate: real ladder players by rating band vs v63 (python/band_gate.py, profile 35)
        subprocess.run([sys.executable, os.path.join(RL, "python", "band_gate.py"), "--cand", w, "--name", c["candidate"],
                        "--ref-profile", "35", "--threads", workers], check=True)
    daily(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "daily", "cycle"])
    ap.add_argument("--weights")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    name = a.name or f"rc_{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%MZ}"
    if a.cmd == "build":
        if not a.weights:
            sys.exit("--weights required")
        build(a.weights, name)
    elif a.cmd == "daily":
        daily(name)
    else:
        cycle(name)


if __name__ == "__main__":
    main()
