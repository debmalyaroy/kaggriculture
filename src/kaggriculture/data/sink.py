"""The episode sink: every locally played game becomes a training row.

Append-only JSONL under data/sink/, one file per day, one row per episode.
Rows carry the engine fingerprint so models can filter by game version --
the 2026-08-07 balance change made version-mixing a real poisoning risk.

    import kaggriculture.data.sink as sink
    sink.record(left="agents/v23_route.py", right="tape.py", seed=60000,
                left_bank=101000, right_bank=99000, source="evaluate")
"""
from kaggriculture.paths import ROOT
import datetime as dt
import json
import os
import threading

SINK = os.path.join(ROOT, "data", "sink")
_LOCK = threading.Lock()
_FINGERPRINT = None


def _fingerprint():
    global _FINGERPRINT
    if _FINGERPRINT is None:
        try:
            base = json.load(open(os.path.join(ROOT, "data",
                                               "engine_baseline.json"),
                                  encoding="utf-8"))
            _FINGERPRINT = base.get("fingerprint", "unknown")
        except Exception:                                          # noqa: BLE001
            _FINGERPRINT = "unknown"
    return _FINGERPRINT


def _route_identity(path):
    """Base-route id recoverable from an agent artifact, or None.

    routes.py --build stamps "Route: <id>" into the agent docstring, and
    candidate/tape filenames carry the id too. Persisted at WRITE time
    because filename archaeology later only resolved 104/1865 rows
    (surrogate-v2 postmortem, 2026-08-13) -- the surrogate needs real
    features for every row the day the arms program returns."""
    import re as _re
    base = os.path.basename(str(path or ""))
    m = _re.search(r"(\d{6,}_s[01])", base)
    if m:
        return m.group(1)
    m = _re.search(r"episode-(\d{6,})-replay_s([01])", base)
    if m:
        return f"{m.group(1)}_s{m.group(2)}"
    try:
        head = open(path, encoding="utf-8").read(2000)
        m = _re.search(r"^Route:\s*(\S+)", head, _re.M)
        if m:
            return m.group(1)
    except OSError:
        pass
    return None


def record(**row):
    """Append one episode row. Never raises -- the sink must not break play."""
    try:
        row.setdefault("ts", dt.datetime.now().isoformat(timespec="seconds"))
        row.setdefault("engine", _fingerprint())
        for side in ("agent", "opponent"):
            if row.get(side) and f"{side}_route" not in row:
                rid = _route_identity(row[side])
                if rid:
                    row[f"{side}_route"] = rid
        path = os.path.join(SINK, f"{dt.date.today().isoformat()}.jsonl")
        os.makedirs(SINK, exist_ok=True)
        line = json.dumps(row, separators=(",", ":"), default=str)
        with _LOCK, open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:                                              # noqa: BLE001
        pass


def rows(days=None):
    """Yield rows, newest files first. days=None -> all."""
    files = sorted((f for f in os.listdir(SINK) if f.endswith(".jsonl")),
                   reverse=True) if os.path.isdir(SINK) else []
    if days is not None:
        files = files[:days]
    for f in files:
        with open(os.path.join(SINK, f), encoding="utf-8") as fh:
            for line in fh:
                try:
                    yield json.loads(line)
                except Exception:                                  # noqa: BLE001
                    continue
