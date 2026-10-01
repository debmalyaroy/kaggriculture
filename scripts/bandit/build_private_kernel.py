"""Write (and optionally push) a new version of the PRIVATE submission notebook that emits a verified
submission.tar.gz.

    python scripts/bandit/build_private_kernel.py --tar data/bandit_builds/v622_c13/submission.tar.gz \
        --label v62 --subdir v62 --desc "..." [--push]

The tarball is read from a PRIVATE dataset (--dataset, file <subdir>/submission.tar.gz.bin -- Kaggle
unpacks .tar.gz files in datasets, so the verified bytes are stored under a .bin name). Embedding a 1.4 MB tarball
made the notebook too large to push (HTTP 400 on SaveKernel). The notebook checks it by size and
sha256 against the local build, writes /kaggle/working/submission.tar.gz, then unpacks it and plays
a full self-play episode on Kaggle's official engine: it must end DONE, and the Rust bridge must
not fall back when its beacons are visible. Only submission.tar.gz is left in /kaggle/working.

The kernel metadata MUST be the bandit slug with is_private: true. The script refuses otherwise and
re-asserts it right before pushing.

Submitting is a separate, operator-approved step:
    kaggle competitions submit kaggriculture -k debmalya84/kaggriculture-adaptive-bandit-private \
        -v <version> -f submission.tar.gz -m "..."
"""
import argparse
import hashlib
import json
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DIR = os.path.join(ROOT, ".local", "submissions", "adaptive_bandit_private_kernel")
SLUG = "debmalya84/kaggriculture-adaptive-bandit-private"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


SELF_CHECK = """import io, contextlib
AG = '/kaggle/working/_agent'
shutil.rmtree(AG, ignore_errors=True); os.makedirs(AG)
with tarfile.open('/kaggle/working/submission.tar.gz') as t:
    t.extractall(AG)
print(sorted(os.listdir(AG)))
from kaggle_environments import make
env = make('kaggriculture', debug=True)
err = io.StringIO()
with contextlib.redirect_stderr(err):
    env.run([os.path.join(AG, 'main.py'), os.path.join(AG, 'main.py')])
st = [s.get('status') for s in env.steps[-1]]
rw = [s.get('reward') for s in env.steps[-1]]
beacons = [l for l in err.getvalue().splitlines() if l.startswith('RUSTV61 end')]
print('statuses', st); print('rewards ', rw)
for b in beacons[-2:]:
    print(b)
assert all(s == 'DONE' for s in st), st
stats = [json.loads(b.split(' ', 2)[2]) for b in beacons[-2:]]
if stats:
    assert all(s['fallback'] == 0 for s in stats), stats
else:
    print('WARNING: no RUSTV61 beacons captured; the DONE check is still enforced')
shutil.rmtree(AG, ignore_errors=True)
print('SELF-CHECK PASS; outputs:', os.listdir('/kaggle/working'))
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True, help="the verified local build's submission.tar.gz")
    ap.add_argument("--label", required=True)
    ap.add_argument("--desc", required=True)
    ap.add_argument("--subdir", required=True, help="dataset folder holding this version's tarball, e.g. v62")
    ap.add_argument("--dataset", default="debmalya84/kaggriculture-bandit-payload")
    ap.add_argument("--kernel-dir", default=DEFAULT_DIR)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()

    meta_path = os.path.join(a.kernel_dir, "kernel-metadata.json")
    meta = json.load(open(meta_path, encoding="utf-8"))
    if meta.get("id") != SLUG or meta.get("is_private") is not True:
        raise SystemExit(f"REFUSING: metadata must be {SLUG} with is_private=true; got {meta.get('id')} private={meta.get('is_private')}")
    if a.dataset not in meta.get("dataset_sources", []):
        meta["dataset_sources"] = list(meta.get("dataset_sources", [])) + [a.dataset]
    meta["is_private"] = True
    open(meta_path, "w", encoding="utf-8", newline="\n").write(json.dumps(meta, indent=2))

    data = open(a.tar, "rb").read()
    sha = hashlib.sha256(data).hexdigest()
    build = json.load(open(os.path.join(os.path.dirname(a.tar), "build.json"), encoding="utf-8-sig"))
    cells = [
        md(f"# Kaggriculture Adaptive Bandit (private) - {a.label}\n\n{a.desc}\n\n"
           f"`submission.tar.gz` = `main.py` stdio bridge + static Linux `agent-stdio` (v62 Rust bandit, lever "
           f"profile {build.get('profile')} of {build.get('profiles')}) + route tables + profile table + Python v61.1 fallback.\n\n"
           f"{len(data):,} bytes, sha256 `{sha}` (binary `{build.get('binary_sha')}`); read from the private dataset "
           f"`{a.dataset}`, folder `{a.subdir}`."),
        code(f"import glob, hashlib, json, os, shutil, tarfile, zipfile\n"
             f"TAR_SHA256 = '{sha}'\nTAR_SIZE = {len(data)}\nSUBDIR = {a.subdir!r}\n"),
        md("## payload"),
        # stored as submission.tar.gz.bin: Kaggle unpacks .tar.gz files in datasets on upload
        code("hits = [p for p in glob.glob('/kaggle/input/**/submission.tar.gz.bin', recursive=True) if f'/{SUBDIR}/' in p]\n"
             "if not hits:  # the folder may be mounted as a zip\n"
             "    for z in glob.glob(f'/kaggle/input/**/{SUBDIR}.zip', recursive=True):\n"
             "        zipfile.ZipFile(z).extractall('/tmp/payload')\n"
             "    hits = glob.glob('/tmp/payload/**/submission.tar.gz.bin', recursive=True)\n"
             "print('payload', hits)\n"
             "assert len(hits) == 1, hits\n"
             "data = open(hits[0], 'rb').read()\n"
             "assert len(data) == TAR_SIZE, (len(data), TAR_SIZE)\n"
             "assert hashlib.sha256(data).hexdigest() == TAR_SHA256\n"
             "open('/kaggle/working/submission.tar.gz', 'wb').write(data)\n"
             "print('wrote submission.tar.gz', len(data), 'bytes; sha', TAR_SHA256)\n"),
        md("## self-check: unpack and play a full self-play episode on the official engine"),
        code(SELF_CHECK),
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
    path = os.path.join(a.kernel_dir, meta["code_file"])
    open(path, "w", encoding="utf-8", newline="\n").write(json.dumps(nb, indent=1))
    print(f"wrote {path} ({os.path.getsize(path):,} bytes) for {a.label}; tar sha {sha}")
    if a.push:
        meta = json.load(open(meta_path, encoding="utf-8"))
        assert meta.get("id") == SLUG and meta.get("is_private") is True, "must be the private bandit kernel"
        r = subprocess.run(["kaggle", "kernels", "push", "-p", a.kernel_dir], capture_output=True, text=True)
        print((r.stdout + r.stderr).strip())


if __name__ == "__main__":
    main()
