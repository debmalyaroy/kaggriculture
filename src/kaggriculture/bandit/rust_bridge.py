"""Kaggle-style agent(obs, configuration) backed by the v62 Rust bandit (rustengine/v62 agent-stdio).

One persistent process per agent instance: each call writes the observation as one JSON line and reads
one action line back. `extra` carries the knob flags (--profile N, --group/--endgame/--jitter/...).
"""
import json
import os
import subprocess

from kaggriculture.paths import ROOT

EXE = os.path.join(ROOT, "rustengine", "v62", "target", "release", "agent-stdio" + (".exe" if os.name == "nt" else ""))
BASE = os.path.join(ROOT, "configs", "bandit", "bases", "v61.1")
PROFILES = os.path.join(ROOT, "configs", "bandit", "profiles", "v2.json")


def _plain(o):
    return json.loads(json.dumps(o, default=lambda s: s.__dict__))


class RustBandit:
    def __init__(self, extra=(), exe=EXE, base=BASE, profiles=PROFILES):
        cmd = [exe, "--base", base, "--profiles", profiles] + list(extra)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     text=True, bufsize=1)

    def __call__(self, observation, configuration=None):
        self.proc.stdin.write(json.dumps(_plain(observation), separators=(",", ":")) + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())
