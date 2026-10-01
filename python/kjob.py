"""Shared Kaggle private-notebook job machinery (used by delta.py and league_kaggle.py).

  make_kernel(name, sources, code)       write kaggle/kernels/<name>/ (private, internet off)
  push(name)                             push with retries while Kaggle's 5 CPU sessions are busy
  status(name)                           KernelWorkerStatus (RUNNING / COMPLETE / ERROR / ...)
  fetch(name, dest, pattern)             paged `kernels output` download of files matching pattern
  download(name, dest, manifest, pushed_utc)
                                         run.json + manifest first; refuse a failed or stale output
                                         (finished before the push); then exactly the missing files
Every notebook is created PRIVATE; make_kernel refuses anything else.
"""
import datetime as dt
import json
import os
import re
import subprocess
import time

RL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNELS = os.path.join(RL, "kaggle", "kernels")
OWNER = "debmalya84"
POLL_SEC = 120


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg):
    print(f"{dt.datetime.now(dt.timezone.utc).strftime('%H:%M:%SZ')} {msg}", flush=True)


def kaggle(args, timeout=900):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    p = subprocess.run(["kaggle"] + args, capture_output=True, timeout=timeout, env=env)
    return p.returncode, (p.stdout or b"").decode("utf-8", "replace") + (p.stderr or b"").decode("utf-8", "replace")


def make_kernel(name, sources, code):
    d = os.path.join(KERNELS, name)
    os.makedirs(d, exist_ok=True)
    meta = {"id": f"{OWNER}/{name}", "title": name, "code_file": f"{name}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": False, "dataset_sources": sources, "competition_sources": [],
            "kernel_sources": [], "model_sources": []}
    assert meta["is_private"] is True
    open(os.path.join(d, "kernel-metadata.json"), "w", newline="\n").write(json.dumps(meta, indent=1))
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                     "source": code.splitlines(keepends=True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 4}
    open(os.path.join(d, f"{name}.ipynb"), "w", encoding="utf-8", newline="\n").write(json.dumps(nb, indent=1))
    return d


def status(name):
    rc, out = kaggle(["kernels", "status", f"{OWNER}/{name}"])
    m = re.search(r"KernelWorkerStatus\.(\w+)", out)
    return m.group(1) if m else ("MISSING" if "404" in out or "not found" in out.lower() else "UNKNOWN")


def push(name, tries=60):
    """Push kaggle/kernels/<name>; returns the push time (UTC) or raises."""
    d = os.path.join(KERNELS, name)
    meta = json.load(open(os.path.join(d, "kernel-metadata.json")))
    assert meta.get("is_private") is True, f"{name} is not private"
    for _ in range(tries):
        rc, out = kaggle(["kernels", "push", "-p", d])
        if "successfully pushed" in out:
            log(f"PUSHED {name}")
            return now()
        if "Maximum batch CPU session" in out:
            log(f"{name}: Kaggle's 5 CPU sessions are busy, retrying in {POLL_SEC}s")
            time.sleep(POLL_SEC)
            continue
        raise RuntimeError(f"push {name} failed: {out.strip()[:300]}")
    raise RuntimeError(f"push {name}: sessions stayed busy")


def fetch(name, dest, pattern):
    os.makedirs(dest, exist_ok=True)
    tok = None
    for _ in range(200):
        args = ["kernels", "output", f"{OWNER}/{name}", "-p", dest, "--page-size", "200", "--file-pattern", pattern]
        if tok:
            args += ["--page-token", tok]
        # Kaggle answers 429 when too many listing pages are requested; wait and repeat the same page
        for wait in (30, 60, 120, 240, 480, 0):
            rc, out = kaggle(args, timeout=3600)
            if "429" not in out or not wait:
                break
            log(f"{name}: Kaggle rate limit (429), retrying this page in {wait}s")
            time.sleep(wait)
        m = re.search(r"Next Page Token\s*=\s*(\S+)", out)
        if not m:
            break
        tok = m.group(1)


def download(name, dest, manifest, pushed_utc):
    """Fetch run.json + `manifest` (relative path), validate, then every listed file we lack."""
    fetch(name, dest, r"^(run\.json|" + re.escape(manifest) + r")$")
    rj, man = os.path.join(dest, "run.json"), os.path.join(dest, manifest)
    if not (os.path.exists(rj) and os.path.exists(man)):
        raise RuntimeError(f"{name}: run.json/{manifest} not in the output")
    run = json.load(open(rj))
    if run.get("rc") != 0:
        raise RuntimeError(f"{name}: job rc={run.get('rc')}")
    if run.get("finished_utc", "") < pushed_utc:
        raise RuntimeError(f"{name}: output finished {run.get('finished_utc')} predates push {pushed_utc} (stale)")
    want = [ln.strip() for ln in open(man) if ln.strip()]
    for attempt in range(8):
        missing = [f for f in want if not os.path.exists(os.path.join(dest, f))]
        if not missing:
            log(f"DOWNLOADED {name} {len(want)} files")
            return run
        log(f"{name}: {len(missing)} of {len(want)} files missing, fetching (round {attempt + 1})")
        for i in range(0, len(missing), 40):
            fetch(name, dest, "^(" + "|".join(re.escape(f) for f in missing[i:i + 40]) + ")$")
    missing = [f for f in want if not os.path.exists(os.path.join(dest, f))]
    raise RuntimeError(f"{name}: still missing {len(missing)} files after retries: {missing[:5]}")
