"""P3.1 -- Python client for `kagg serve` (step-wise env over stdio).

The observation arrives in the OFFICIAL interpreter's observation schema
(plus both seats' private and a "done" flag), so agent code written against
the ladder parses it unchanged. Verified: driving serve with a real replay's
actions reproduces `kagg batch` banks exactly, and batch reproduces recorded
ladder banks exactly (6/6 sampled 1.32.6 episodes).

PRE-RANKER / TRAINING ONLY: nothing stepped here decides a release.
"""
from __future__ import annotations

import json
import os
import subprocess

from . import common


class ServeEnv:
    """One persistent kagg process; many episodes over its lifetime."""

    def __init__(self, kagg_path: str = ""):
        self._exe = kagg_path or common.KAGG
        self._proc = None

    def _ensure(self):
        if self._proc is None or self._proc.poll() is not None:
            self._proc = subprocess.Popen(
                [self._exe, "serve"], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, text=True, bufsize=1,
                encoding="utf-8")

    def _rpc(self, line: str) -> dict:
        self._ensure()
        self._proc.stdin.write(line + "\n")
        self._proc.stdin.flush()
        resp = self._proc.stdout.readline()
        if not resp:
            raise RuntimeError("kagg serve closed the pipe")
        obs = json.loads(resp)
        if "error" in obs:
            raise RuntimeError(f"kagg serve: {obs['error']}")
        return obs

    def reset(self, seed: int, opp_tape: str = "", opp_seat: int = 1) -> dict:
        """Start an episode. With opp_tape, that seat replays the tape and
        step() drives only the free seat; without, use step_both()."""
        if opp_tape:
            return self._rpc(
                f"RESET {seed} OPP {opp_seat} {os.path.abspath(opp_tape)}")
        return self._rpc(f"RESET {seed}")

    def step(self, action: dict) -> dict:
        """Step with OUR action (dict in official schema); opponent scripted."""
        return self._rpc("STEP " + common.action_to_tape_lines(action))

    def step_both(self, action0: dict, action1: dict) -> dict:
        return self._rpc(
            "STEP2 " + common.action_to_tape_lines(action0)
            + "\x1e" + common.action_to_tape_lines(action1))

    def close(self):
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.stdin.write("QUIT\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
        self._proc = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def seat_view(obs: dict, seat: int) -> dict:
    """The official per-seat observation an agent(obs) would receive."""
    return {
        "step": obs["step"], "day": obs["day"], "hour": obs["hour"],
        "player": seat, "farms": obs["farms"], "market": obs["market"],
        "town": obs["town"], "private": obs["private"][seat],
    }
