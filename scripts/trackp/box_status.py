#!/usr/bin/env python
"""(6) BOX STATUS  --  local; read-only SSH pull of a detailed training status.

Same tight block the 30-min monitor prints. For BC: stage / progress% / sps /
ETA / GPU%/RAM / alive. For RL: also gstep vs target, mean reward + win-rate and
env-steps/s from metrics.jsonl, plus ETA to target.

    python scripts/trackp/box_status.py
    python scripts/trackp/box_status.py --rl-target 1000000000
"""
from __future__ import annotations
import argparse, subprocess

HOST = "root@<box-ip>"
PORT = "55632"
KEY = "~/.ssh/vast_knee"
CK = "/root/kaggriculture/ckpts"

REMOTE = r'''
CK=/root/kaggriculture/ckpts
echo "STAGE|$(tail -n 40 $CK/cloud_train.log 2>/dev/null | grep -E '\[[0-9]_' | tail -1)"
echo "BC|$(grep -a '\[bc\]' $CK/pipeline.log 2>/dev/null | tail -1)"
echo "RL|$(grep -a '\[rl\]' $CK/pipeline.log 2>/dev/null | tail -1)"
echo "MET_BC|$(grep -a '"kind": "bc"' $CK/metrics.jsonl 2>/dev/null | tail -1)"
echo "MET_RL|$(grep -a '"kind": "rl"' $CK/metrics.jsonl 2>/dev/null | tail -1)"
echo "PROCS|run=$(pgrep -c -f run_pipeline.sh) ct=$(pgrep -c -f cloud_train) bc=$(pgrep -c -f bc_train) rl=$(pgrep -c -f rl_selfplay)"
echo "GPU|$(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader | head -1)"
echo "RAM|$(free -g | awk '/Mem:/{print $4"/"$2"G free"}')"
echo "GS|$(PYTHONPATH=/root/kaggriculture/src /venv/main/bin/python -c "import torch,glob,os;fs=[f for f in ['$CK/policy_rl.pt','$CK/policy_bc.pt'] if os.path.exists(f)];print(int(torch.load(fs[0],map_location=\"cpu\",weights_only=False).get(\"global_step\",0)) if fs else 0, fs[0].split('/')[-1] if fs else '-')" 2>/dev/null)"
echo "CRASH|$(grep -aiE 'traceback|exited non-zero|killed|oom' $CK/pipeline.log 2>/dev/null | tail -1 | cut -c1-70)"
echo "PBEST|$(cat $CK/.panel_best.json 2>/dev/null | tr -d '\n')"
echo "PLAST|$(tail -n 1 $CK/panel_eval.jsonl 2>/dev/null)"
echo "TIME|$(date -u +%H:%M:%SZ)"
'''


def _ssh():
    cmd = ["ssh", "-p", PORT, "-i", KEY, "-o", "ConnectTimeout=20", HOST, "bash -s"]
    # Send the script as BYTES, not via text=True. On Windows, subprocess text
    # mode translates every \n in `input` to \r\n when writing the child's
    # stdin, so bash sees "CK=/root/kaggriculture/ckpts\r" -- a carriage return
    # embedded in every $CK/<file> path. The log greps then read nonexistent
    # files and stage/RL/gstep come back empty, while nvidia-smi/free/pgrep
    # (which don't touch $CK) still work. Bytes in => no newline translation.
    payload = REMOTE.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    r = subprocess.run(cmd, input=payload, capture_output=True)
    d = {}
    for ln in r.stdout.decode("utf-8", "replace").splitlines():
        if "|" in ln and not ln.startswith(("Welcome", "Have", "AI agents", "authentication")):
            k, _, v = ln.partition("|"); d[k] = v.strip()
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rl-target", type=int, default=500_000_000)
    ap.add_argument("--bc-target", type=int, default=134_000_000)
    a = ap.parse_args()
    d = _ssh()
    if not d:
        print("no response from box"); return

    stage = d.get("STAGE", "").split("] ")[-1] or "?"
    procs = d.get("PROCS", "")
    rl_running = "rl=1" in procs or "rl=2" in procs
    print(f"=== trackp box status  ({d.get('TIME','?')} UTC) ===")
    print(f"stage: {stage}")

    if rl_running or d.get("RL"):
        rl = d.get("RL", "")
        met = _json(d.get("MET_RL", ""))
        gs = _int(d.get("GS", "0").split()[0] if d.get("GS") else 0)
        pct = 100.0 * gs / max(a.rl_target, 1)
        eps = met.get("eps")
        eta = _eta(a.rl_target - gs, eps)
        print(f"RL: gstep={gs:,} / {a.rl_target:,} = {pct:.1f}%  "
              f"meanR={met.get('meanR')} win={_pct(met.get('winrate'))} "
              f"loss={met.get('loss')}  {eps} eps/s  ETA {eta}")
        print(f"    last log: {rl[:90]}")
    else:
        bc = d.get("BC", "")
        met = _json(d.get("MET_BC", ""))
        seen = _grab(bc, "seen=") or 0
        sps = _grab(bc, "(", "/s")
        pct = 100.0 * seen / max(a.bc_target, 1)
        eta = _eta(a.bc_target - seen, sps)
        print(f"BC: seen={seen:,} / {a.bc_target:,} = {pct:.1f}%  "
              f"loss={met.get('loss')}  {sps or '?'}/s  ETA {eta}")
        print(f"    last log: {bc[:90]}")

    print(f"GPU: {d.get('GPU','?')} | RAM {d.get('RAM','?')} | procs {procs}")

    pb = _json(d.get("PBEST", ""))
    pl = _json(d.get("PLAST", ""))
    if pb or pl:
        last = (f"gstep={_int(pl.get('gstep', 0)):,} score={pl.get('score', '?')} "
                f"(bank {pl.get('mean_bank', '?')} vs {pl.get('mean_opp', '?')})"
                if pl else "no eval yet")
        best = (f"score={pb.get('score', '?')} @ {_int(pb.get('gstep', 0)):,}"
                if pb else "?")
        print(f"PANEL: last {last} | best {best}")

    if d.get("CRASH"):
        print(f"CRASH?: {d['CRASH']}")


def _json(s):
    import json
    try:
        return json.loads(s)
    except Exception:
        return {}


def _int(x):
    try:
        return int(str(x).replace(",", ""))
    except Exception:
        return 0


def _grab(s, pre, post=" "):
    try:
        seg = s.split(pre, 1)[1]
        seg = seg.split(post, 1)[0]
        return _int(seg)
    except Exception:
        return None


def _pct(x):
    return f"{100*x:.0f}%" if isinstance(x, (int, float)) else "?"


def _eta(remaining, rate):
    try:
        h = remaining / float(rate) / 3600.0
        return f"~{h:.1f}h" if h >= 0 else "done"
    except Exception:
        return "?"


if __name__ == "__main__":
    main()
