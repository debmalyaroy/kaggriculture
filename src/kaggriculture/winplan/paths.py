"""Where the win-plan harness keeps things. Data under data/winplan (gitignored),
run state under .local/winplan, the plan's knobs in configs/winplan.json."""
import json
import os

from kaggriculture.paths import ROOT

DATA = os.path.join(ROOT, "data", "winplan")
AGENTS = os.path.join(DATA, "agents")          # unpacked public packages, one dir each
FIELD = os.path.join(DATA, "field")            # one loader shim per agent (the gate roster)
GATES = os.path.join(DATA, "gates")            # per-game gate results (jsonl)
REPLAYS = os.path.join(DATA, "replays")        # downloaded ladder replays
CRAWL = os.path.join(DATA, "crawl")            # EpisodeService crawl output
NBRUN = os.path.join(ROOT, ".local", "winplan", "nbrun")   # notebook packaging sandboxes
STATE_DIR = os.path.join(ROOT, ".local", "winplan")
STATE = os.path.join(STATE_DIR, "state.json")
EVENTS = os.path.join(STATE_DIR, "events.log")
CONFIG = os.path.join(ROOT, "configs", "winplan.json")
LADDER_CONFIG = os.path.join(DATA, "ladder_config.json")
VENDOR = os.path.join(ROOT, "vendor")

for _d in (DATA, AGENTS, FIELD, GATES, REPLAYS, CRAWL, NBRUN, STATE_DIR):
    os.makedirs(_d, exist_ok=True)


def config():
    with open(CONFIG, encoding="utf-8") as fh:
        return json.load(fh)


def rel(path):
    """Project-relative path for display and state files."""
    try:
        return os.path.relpath(path, ROOT).replace("\\", "/")
    except ValueError:
        return path
