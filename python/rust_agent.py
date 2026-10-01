"""Bridge: a Kaggle-style `agent(obs, configuration)` backed by the Rust `agent-stdio` process.

One persistent process per agent instance; each call writes the observation as one JSON line
and reads one action line back. Used by closed-loop parity (python/closed_loop.py) and as the
template for the submission's main.py.
"""
import json, os, subprocess

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_EXE = os.path.join(RL, "target-dev", "release", "agent-stdio.exe")
DEFAULT_BASE = os.path.join(RL, "configs", "bases", "v61.1")


def _plain(o):
    return json.loads(json.dumps(o, default=lambda s: s.__dict__))


class RustAgent:
    def __init__(self, exe=DEFAULT_EXE, base=DEFAULT_BASE, cut="full", profiles=None, profile=None, clone_profile=None, clone_strict=False, policy=None):
        cmd = [exe, "--base", base, "--cut", cut]
        if profiles is not None:
            cmd += ["--profiles", profiles] + (["--profile", str(profile)] if profile is not None else [])
            cmd += ["--clone-profile", str(clone_profile)] if clone_profile is not None else []
            cmd += ["--clone-strict"] if clone_strict else []
            cmd += ["--policy", policy] if policy else []
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def __call__(self, observation, configuration=None):
        self.proc.stdin.write(json.dumps(_plain(observation), separators=(",", ":")) + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
