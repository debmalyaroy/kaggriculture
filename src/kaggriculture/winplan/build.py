"""Compose a candidate agent: base agent file + layer snippets + baked knobs.

    python -m kaggriculture.winplan.build rsa3 --base agents/bandit_chassis_m4afl.py \
        --layer rsa --set _RSA_LOOK=3 [--queue]

Writes agents/cand_<label>.py (single self-contained file, same imports as the
base), smoke-plays one full self-play game on the Rust engine, and with
--queue adds it to the harness's candidate list for gating (S1.10).
Layer snippets live in src/kaggriculture/winplan/layers/<name>.py and wrap the
module-level `agent` of whatever precedes them.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

from kaggriculture.paths import ROOT

LAYERS = os.path.join(os.path.dirname(__file__), "layers")

# Kaggle's loader (kaggle_environments.agent.get_last_callable) runs the LAST
# callable in the module dict by insertion order, not the name `agent`. A
# re-bound `agent` keeps the dict slot of its FIRST binding, so a layer's
# helpers defined after it would be run instead (v61 notebook, 2026-09-24).
ENTRY_FOOTER = """
# --- entry point: re-insert `agent` so it is the module's LAST callable ---
_kaggle_entry = agent
del agent
agent = _kaggle_entry
"""


def compose(label, base, layers, knobs):
    src = open(os.path.join(ROOT, base) if not os.path.isabs(base) else base, encoding="utf-8").read()
    parts = [src.rstrip() + "\n"]
    for name in layers:
        body = open(os.path.join(LAYERS, name + ".py"), encoding="utf-8").read()
        for k, v in knobs.items():
            body, n = re.subn(rf"^{re.escape(k)}\s*=.*$", f"{k} = {v}", body, flags=re.M)
        parts.append(body)
    unset = [k for k in knobs if not any(re.search(rf"^{re.escape(k)}\s*=", p, re.M) for p in parts[1:])]
    if unset:
        raise SystemExit(f"knobs not found in layers: {unset}")
    parts.append(ENTRY_FOOTER)
    out = os.path.join(ROOT, "agents", f"cand_{label}.py")
    open(out, "w", encoding="utf-8").write("\n".join(parts))
    check_entry(out)
    return out


def check_entry(path):
    """Fail unless Kaggle's loader would pick the module's `agent`."""
    import contextlib
    import io
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(open(path, encoding="utf-8").read(), path, "exec"), ns)
    last = [v for v in ns.values() if callable(v)][-1]
    if last is not ns.get("agent"):
        raise SystemExit(f"{path}: Kaggle would run {getattr(last, '__name__', last)!r}, not agent")


def smoke(path):
    code = ("import contextlib,io;from kaggriculture.bandit.gate import loss_forensics as LF, harness as H\n"
            "w,s=H.world_seeds(1)[0]\n"
            "with contextlib.redirect_stdout(io.StringIO()):\n"
            f"    _,_,a,b=LF.capture_game(LF.load_pyagent({path!r}),{path!r},s,0)\n"
            "print('SMOKE', a, b)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("SMOKE")]
    if r.returncode or not line:
        raise SystemExit("smoke failed:\n" + r.stderr[-1500:])
    return line[0]


def from_config(path):
    """Build an agent from a JSON config: {"label", "base", "layers": [...],
    "knobs": {NAME: value}, "overrides": {NAME: value}, "out": optional path}.
    `knobs` are baked into the layer snippets (must exist there); `overrides`
    re-bind module globals of the base (e.g. a base layer's switch) in a block
    before the entry footer. Returns the written path."""
    import json
    cfg = json.load(open(path if os.path.isabs(path) else os.path.join(ROOT, path), encoding="utf-8"))
    knobs = {k: repr(v) for k, v in (cfg.get("knobs") or {}).items()}
    out = compose(cfg["label"], cfg["base"], cfg.get("layers") or [], knobs)
    over = cfg.get("overrides") or {}
    if over:
        src = open(out, encoding="utf-8").read()
        foot = ENTRY_FOOTER.strip().splitlines()[0]
        block = "# --- config overrides ---\n" + "".join(f"{k} = {v!r}\n" for k, v in over.items())
        src = src.replace(foot, block + foot, 1)
        open(out, "w", encoding="utf-8").write(src)
        check_entry(out)
    if cfg.get("out"):
        import shutil
        dst = os.path.join(ROOT, cfg["out"])
        shutil.copyfile(out, dst)
        check_entry(dst)
        return dst
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("label", nargs="?")
    ap.add_argument("--config", help="configs/agents/<name>.json (base + layers + knobs + overrides)")
    ap.add_argument("--base", default="agents/bandit_chassis_m4afl.py")
    ap.add_argument("--layer", action="append", default=[])
    ap.add_argument("--set", action="append", default=[], help="KNOB=value baked into the layers")
    ap.add_argument("--queue", action="store_true")
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    if a.config:
        out = from_config(a.config)
        print("wrote", out)
        print(smoke(out))
        return
    knobs = dict(s.split("=", 1) for s in a.set)
    out = compose(a.label, a.base, a.layer, knobs)
    print("wrote", out)
    print(smoke(out))
    if a.queue:
        from kaggriculture.winplan import runner
        rel = os.path.relpath(out, ROOT).replace("\\", "/")
        runner.main(["candidate", a.label, rel, "--note", a.note or f"{a.base} + {'+'.join(a.layer)} {knobs}"])
        print("queued", a.label)


if __name__ == "__main__":
    main()
