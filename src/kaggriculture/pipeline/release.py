"""Release management: versioning + config-LOCK + private-notebook submit-prep.

The harnesses (bandit, trackp) are CONFIG-DRIVEN and the CODE rarely changes. The
daily pipeline's job is to find the OPTIMAL CONFIG via train->test->improve; once
the gate passes, the config is LOCKED and that locked config IS the release. The
operator then pushes it to the harness's PRIVATE notebook and submits from there.

Versioning (operator rule): a new MAJOR version each day; intraday fixes bump the
MINOR. Next version is v59. So the first release today is v59 (==v59.0); a second
release the same day is v59.1; tomorrow's is v60.

Submission notebooks (operator rule): each harness ships from its OWN private
notebook:  kaggriculture-private-submission-bandit / -trackp.

Nothing here calls Kaggle. It writes the locked config + a submit checklist.
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import json
import os
import time

REL_DIR = os.path.join(ROOT, "models", "rl", "releases")
VER_STATE = os.path.join(REL_DIR, "version_state.json")
START_MAJOR = 59                              # operator: next version is v59
NOTEBOOK = {"bandit": "kaggriculture-private-submission-bandit",
            "trackp": "kaggriculture-private-submission-trackp"}


def _load_ver():
    if os.path.exists(VER_STATE):
        try:
            return json.load(open(VER_STATE, encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return {"major": START_MAJOR - 1, "minor": 0, "date": ""}


def next_version() -> str:
    """MAJOR bump on a new day, else MINOR bump. Returns e.g. 'v59' or 'v59.1'."""
    st = _load_ver()
    today = time.strftime("%Y-%m-%d")
    if st["date"] != today:
        st = {"major": max(st["major"] + 1, START_MAJOR), "minor": 0, "date": today}
    else:
        st["minor"] += 1
    os.makedirs(REL_DIR, exist_ok=True)
    json.dump(st, open(VER_STATE, "w", encoding="utf-8"), indent=1)
    return f"v{st['major']}" + (f".{st['minor']}" if st["minor"] else "")


def peek_version() -> str:
    st = _load_ver()
    today = time.strftime("%Y-%m-%d")
    if st["date"] != today:
        return f"v{max(st['major'] + 1, START_MAJOR)}"
    return f"v{st['major']}" + (f".{st['minor'] + 1}" if st['minor'] or st['date'] == today else "")


def lock_config(harness: str, version: str, config: dict, artifact: str,
                gate: dict) -> str:
    """Freeze the gated config as the release for a harness. Returns the path."""
    assert harness in NOTEBOOK, harness
    os.makedirs(REL_DIR, exist_ok=True)
    rec = {"harness": harness, "version": version,
           "locked_at": time.strftime("%Y-%m-%d %H:%M"),
           "notebook": NOTEBOOK[harness], "artifact": artifact,
           "gate": gate, "config": config}
    p = os.path.join(REL_DIR, f"{harness}_{version}.json")
    json.dump(rec, open(p, "w", encoding="utf-8"), indent=1)
    # a stable 'latest' pointer per harness
    json.dump(rec, open(os.path.join(REL_DIR, f"{harness}_latest.json"), "w",
                        encoding="utf-8"), indent=1)
    return p


def submit_checklist(version: str, seats: dict) -> str:
    """seats = {'bandit': {...}, 'trackp': {...}} with artifact/gate/ship/locked."""
    lines = [f"# Release {version} — submit checklist ({time.strftime('%Y-%m-%d %H:%M')})",
             "",
             "The pipeline optimised + LOCKED the configs. It does NOT submit.",
             "Each harness ships from its OWN private notebook, versioned "
             f"{version} (major=daily, minor=intraday).", ""]
    for h in ("bandit", "trackp"):
        s = seats.get(h) or {}
        ship = s.get("ship")
        lines += [f"## {h.upper()} — notebook `{NOTEBOOK[h]}`",
                  f"- Release candidate: **{'YES (gate passed)' if ship else 'NO (gate not passed / not built)'}**",
                  f"- Locked config: `{s.get('locked') or '—'}`",
                  f"- Artifact: `{s.get('artifact') or '—'}`",
                  f"- Gate vs refs: {s.get('gate')}",
                  "- To ship (only if release candidate = YES):",
                  f"  1. Open the private notebook `{NOTEBOOK[h]}`.",
                  f"  2. Update it to the LOCKED config `{s.get('locked') or '<locked json>'}` "
                  "(code is stable; only the config changes).",
                  f"  3. Set the version to **{version}** and Save & Run (commit).",
                  "  4. Submit that notebook version to the competition.",
                  "  5. Latest-2 rule: confirm the pair keeps two strong agents.",
                  ""]
    p = os.path.join(ROOT, ".local", "candidates", "SUBMIT_INSTRUCTIONS.md")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return p


def main():
    print("next version would be:", peek_version())
    print("notebooks:", NOTEBOOK)


if __name__ == "__main__":
    main()
