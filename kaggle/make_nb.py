"""Generate a PRIVATE Kaggle extraction notebook (kernel dir) that runs corpus-extract --mode slim.

    python kaggle/make_nb.py <name> --kind gm|daily [--days 2026-08-16,2026-08-17,...]
                             [--since D] [--until D] [--limit N] [--threads 4]

Writes kaggle/kernels/<name>/{kernel-metadata.json, <name>.ipynb}. The kernel:
  - copies the static binary from the private dataset debmalya84/kaggriculture-rl-bin,
  - finds its inputs by globbing (robust to Kaggle mount layouts),
  - streams the extractor log, writes /kaggle/working/slim/** + run.json,
  - fails the run (non-zero) if the extractor fails.
Output = kernel output (no dataset is created). is_private is always true.
"""
import argparse, json, os

ROOT = os.path.dirname(os.path.abspath(__file__))
BIN_DS = "debmalya84/kaggriculture-rl-bin"
GM_DS = "georgymamarin/kaggriculture-episodes"

CODE = r'''
import os, glob, shutil, subprocess, time, json
T0 = time.time()
KIND, EXTRA, THREADS = __KIND__, __EXTRA__, "__THREADS__"
os.makedirs('/kaggle/working/bin', exist_ok=True)
src = sorted(glob.glob('/kaggle/input/**/corpus-extract', recursive=True))[0]
BIN = '/kaggle/working/bin/corpus-extract'
shutil.copy(src, BIN); os.chmod(BIN, 0o755)
sha = open(os.path.join(os.path.dirname(src), 'SHA256.txt')).read().strip()
print('binary', src, sha, flush=True)
if KIND == 'gm':
    gm = sorted(os.path.dirname(p) for p in glob.glob('/kaggle/input/**/episodes.csv', recursive=True))
    assert gm, 'GM dataset not mounted'
    args = ['--gm', gm[0]]
else:
    days = sorted({os.path.dirname(p) for p in glob.glob('/kaggle/input/**/manifest.csv', recursive=True)
                   if 'kaggriculture-episodes-20' in p})
    print('daily dirs:', len(days), days[:3], '...', flush=True)
    assert days, 'no daily datasets mounted'
    args = sum((['--daily', d] for d in days), [])
OUT = '/kaggle/working/slim'
cmd = [BIN, '--mode', 'slim', '--out', OUT, '--threads', THREADS] + args + EXTRA
print(' '.join(cmd[:8]), '...', ' '.join(EXTRA), flush=True)
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    print(line, end='', flush=True)
rc = p.wait()
files = [f for f in glob.glob(OUT + '/**/*', recursive=True) if os.path.isfile(f)]
tot = sum(os.path.getsize(f) for f in files)
json.dump({'rc': rc, 'secs': round(time.time() - T0, 1), 'bytes': tot, 'files': len(files),
           'kind': KIND, 'extra': EXTRA, 'binary_sha': sha}, open('/kaggle/working/run.json', 'w'), indent=1)
shutil.rmtree('/kaggle/working/bin', ignore_errors=True)
print('RC', rc, 'files', len(files), 'bytes', tot, 'secs', round(time.time() - T0, 1), flush=True)
assert rc == 0, 'corpus-extract failed'
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--kind", choices=["gm", "daily"], required=True)
    ap.add_argument("--days", default="")
    ap.add_argument("--since")
    ap.add_argument("--until")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    extra = []
    if a.since:
        extra += ["--since", a.since]
    if a.until:
        extra += ["--until", a.until]
    if a.limit:
        extra += ["--limit", str(a.limit)]
    code = (CODE.replace("__KIND__", repr(a.kind)).replace("__EXTRA__", repr(extra))
            .replace("__THREADS__", str(a.threads)))
    sources = [BIN_DS]
    if a.kind == "gm":
        sources.append(GM_DS)
    else:
        sources += [f"kaggle/kaggriculture-episodes-{d}" for d in a.days.split(",") if d]
    d = os.path.join(ROOT, "kernels", a.name)
    os.makedirs(d, exist_ok=True)
    nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": code}],
          "metadata": {"kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 4}
    json.dump(nb, open(os.path.join(d, f"{a.name}.ipynb"), "w"))
    meta = {"id": f"debmalya84/{a.name}", "title": a.name, "code_file": f"{a.name}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": False, "dataset_sources": sources, "competition_sources": [],
            "kernel_sources": [], "model_sources": []}
    json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1)
    print(d, "sources:", len(sources), "extra:", extra)


if __name__ == "__main__":
    main()
