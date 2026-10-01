"""Platform-neutral paths for the RL pipeline (laptop = Windows x86_64, AWS = Linux ARM64/x86_64).

  RL        repo root
  BIN_DIR   built binaries: $KRL_BIN, else <RL>/target-dev/release
  EXE       ".exe" on Windows, "" elsewhere
  PY        the Python for training jobs: $KRL_PY, else the llm conda env on the laptop, else this one
  bin(name) full path of a built binary
The queue exports KRL_BIN / KRL_EXE / KRL_PY to every job and expands {BIN} {EXE} {PY} in commands.
"""
import os
import sys

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = ".exe" if os.name == "nt" else ""
BIN_DIR = os.environ.get("KRL_BIN") or os.path.join(RL, "target-dev", "release")
CARGO_TARGET = os.environ.get("CARGO_TARGET_DIR") or os.path.join(RL, "target-dev")
_LAPTOP_PY = "C:/ProgramData/anaconda3/envs/llm/python.exe"
PY = os.environ.get("KRL_PY") or (_LAPTOP_PY if os.name == "nt" and os.path.exists(_LAPTOP_PY) else sys.executable)


def bin(name):
    return os.path.join(BIN_DIR, name + EXE)


def env():
    """The variables every queue job gets."""
    return {"KRL_BIN": BIN_DIR, "KRL_EXE": EXE, "KRL_PY": PY, "CARGO_TARGET_DIR": CARGO_TARGET}
