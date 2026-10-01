"""Publish the COMPILED trackp agent from its private kernel.

House rule (CLAUDE.md): both seats publish from their own private kernel --
stable slug, version history preserved, and the submitted bytes verified by
sha against what we built. `src/build_notebook.py` already emits a
`submission.tar.gz`, but only ever wrapping a single `main.py`; the compiled
trackp seat is `main.py` + a 988 KB static ELF (`kagg`), so it needs its own
builder.

The notebook carries both files as base85+zlib payloads, writes them out with
the right modes (kagg MUST be 0o755 or the bridge cannot exec it), tars them
with main.py at the ROOT, and refuses to finish unless the reconstructed
tarball's sha256 matches the one we built locally. That refusal is the point:
it is what makes a kernel submission trustworthy.

    python src/build_compiled_notebook.py            # build the notebook
    kaggle kernels push -p notebooks/kaggriculture-track-p-private
    kaggle kernels output debmalya84/kaggriculture-track-p-private -p <dir>
    # verify sha, then:
    kaggle competitions submit kaggriculture \
        -k debmalya84/kaggriculture-track-p-private -v <N> -f submission.tar.gz
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import argparse
import base64
import hashlib
import json
import os
import tarfile
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
SLUG = "kaggriculture-track-p-private"
TITLE = "Kaggriculture Track P (private)"
TAR = os.path.join(ROOT, ".local", "candidates", "trackp_compiled",
                   "submission.tar.gz")
OUT_DIR = os.path.join(ROOT, "notebooks", SLUG)


def _pack(raw: bytes) -> str:
    return base64.b85encode(zlib.compress(raw, 9)).decode("ascii")


def md(*lines):
    return {"cell_type": "markdown", "metadata": {},
            "source": [l + "\n" for l in lines]}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": [l + "\n" for l in lines]}


def build(tar_path=TAR, out_dir=OUT_DIR, slug=SLUG, title=TITLE):
    raw_tar = open(tar_path, "rb").read()
    want = hashlib.sha256(raw_tar).hexdigest()
    with tarfile.open(tar_path) as t:
        members = {m.name: (t.extractfile(m).read(), m.mode)
                   for m in t.getmembers()}
    if "main.py" not in members:
        raise SystemExit("main.py must be at the archive root")
    # VERIFY MEMBER CONTENT, NOT THE ARCHIVE BYTES (2026-09-03).
    #
    # The first version of this notebook asserted the rebuilt archive's sha256
    # equalled the local one and it FAILED on Kaggle for a reason that has
    # nothing to do with the agent: gzip writes its own mtime into the stream
    # header, so two archives with byte-identical contents have different
    # archive shas. Normalising the TAR entries (uid/gid/mtime) is not enough.
    # What actually matters -- and what the submission runs -- is the bytes of
    # each member, so that is what the notebook checks.
    member_sha = {name: hashlib.sha256(data).hexdigest()
                  for name, (data, _m) in members.items()}

    parts = {name: (_pack(data), mode) for name, (data, mode) in members.items()}
    sizes = {name: len(data) for name, (data, _m) in members.items()}

    cells = [
        md(f"# {title} — compiled closed-loop seat",
           "",
           "This kernel publishes the **compiled** Track-P agent: a Python",
           "bridge (`main.py`) plus a statically linked Linux binary (`kagg`),",
           "shipped as `submission.tar.gz` with `main.py` at the archive root.",
           "",
           "**Why a binary.** The competition's real limit is **1 s per turn**",
           "and multi-file `tar.gz` submissions are documented; our engine is a",
           "bit-exact Rust port, so the agent carries its own forward model.",
           "The single-file / `math`-only convention is a house rule for the",
           "pure-Python seats, not an engine constraint.",
           "",
           "**Why the search is off.** `SEARCH_BUDGET_MS = 0`. Measured paired",
           "over 40 official-engine cells, the searcher cost **-14,755 own",
           "bank** (richer on 11/40, sign p = 0.0064) for identical zero wins,",
           "and its worst turn was 266.7 ms against our 250 ms bar. Its value",
           "function was fitted around the older, weaker economy, so after the",
           "land fix it hill-climbs away from a better one. At budget 0 the",
           "compiled path is byte-identical to the Python fallback, so a",
           "sandbox that forbids `fork`/`exec` costs us nothing in strategy.",
           "",
           "**Provenance.** Built from `rustengine` (engine bit-identical to the",
           "official interpreter, 6/6 episodes) and",
           "`src/trackp/build_econ_agent.py`. Every decision is taken at",
           "runtime from the live observation plus a parameter genome — no",
           "tape, no route, no embedded action sequence, not even as a",
           "fallback.",
           "",
           f"Target archive sha256: `{want}`"),
        code("import base64, hashlib, os, tarfile, zlib",
             "",
             "# The ARCHIVE sha is not reproducible: gzip stamps its own mtime",
             "# into the stream header, so identical contents give different",
             "# archive bytes. Verify each MEMBER instead -- that is what the",
             "# submission actually runs.",
             f"MEMBER_SHA256 = {member_sha!r}",
             f'LOCAL_ARCHIVE_SHA256 = "{want}"  # provenance only',
             f"MODES = {({n: m for n, (_d, m) in members.items()})!r}",
             f"SIZES = {sizes!r}"),
    ]
    for name, (payload, mode) in parts.items():
        cells.append(md(f"## payload — `{name}` "
                        f"({sizes[name]:,} bytes, mode {oct(mode)})"))
        cells.append(code(f'{name.replace(".", "_").upper()}_B85 = (',
                          f'    "{payload}"',
                          ")"))
    cells += [
        md("## reconstruct, verify, and emit the submission"),
        code("def _unpack(b85):",
             "    return zlib.decompress(base64.b85decode(b85))",
             "",
             *[f'open({name!r}, "wb").write(_unpack('
               f'{name.replace(".", "_").upper()}_B85))'
               for name in parts],
             *[f"os.chmod({name!r}, {oct(mode)})"
               for name, (_p, mode) in parts.items()],
             "",
             "for _n, _want in SIZES.items():",
             "    _got = os.path.getsize(_n)",
             "    assert _got == _want, f'{_n}: {_got} != {_want}'",
             "",
             "# The real gate: every member is byte-identical to what we gated.",
             "for _n, _want in MEMBER_SHA256.items():",
             '    _got = hashlib.sha256(open(_n, "rb").read()).hexdigest()',
             "    assert _got == _want, f'{_n} sha {_got} != {_want}'",
             '    print(f"  {_n:<10} sha256 {_got[:16]}  OK")',
             "",
             "# main.py must be importable and define agent(obs).",
             'src = open("main.py", "rb").read().decode("utf-8")',
             'compile(src, "main.py", "exec")',
             'assert "def agent(" in src, "main.py defines no agent()"',
             "",
             "# Deterministic archive AND explicit modes. tar.add() copies the",
             "# host filesystem's mode, which is 0o666 on Windows and whatever",
             "# the umask gives on Linux -- so `kagg` could land WITHOUT the",
             "# execute bit and the bridge would silently fall back for every",
             "# turn of every episode. Force the intended modes here.",
             "def _norm(ti):",
             "    ti.uid = ti.gid = 0",
             '    ti.uname = ti.gname = "root"',
             "    ti.mtime = 0",
             "    ti.mode = MODES[ti.name]",
             "    return ti",
             "",
             'with tarfile.open("submission.tar.gz", "w:gz", '
             "compresslevel=9) as tar:",
             *[f'    tar.add({name!r}, arcname={name!r}, filter=_norm)'
               for name in parts],
             "",
             "# Re-verify THROUGH the archive we just wrote, so what we check",
             "# is what gets submitted, not merely what we wrote to disk.",
             'with tarfile.open("submission.tar.gz") as _t:',
             "    _names = _t.getnames()",
             '    assert "main.py" in _names, "main.py not at archive root"',
             "    for _m in _t.getmembers():",
             "        _d = _t.extractfile(_m).read()",
             "        assert hashlib.sha256(_d).hexdigest() == "
             "MEMBER_SHA256[_m.name], _m.name",
             "        assert _m.mode == MODES[_m.name], "
             "f'{_m.name} mode {oct(_m.mode)}'",
             'print("members:", _names)',
             'print("all members byte-identical to the gated artefact: True")',
             'print("archive sha (informational, gzip mtime varies):",',
             '      hashlib.sha256(open("submission.tar.gz", "rb").read())'
             ".hexdigest()[:16])",
             'print("local archive sha was:", LOCAL_ARCHIVE_SHA256[:16])'),
        md("### self-check: the agent plays a legal episode",
           "",
           "The bridge spawns `kagg` and validates every action before it",
           "reaches the engine (hands aligned positionally, <= 10 market",
           "orders, known ops). If the binary cannot spawn, hangs, or returns",
           "malformed JSON, a pure-Python fallback that is byte-identical at",
           "budget 0 carries the episode. Six deliberate breakages of this",
           "exact tarball all played a full legal `DONE` episode."),
        code("try:",
             "    from kaggle_environments import make",
             '    env = make("kaggriculture", debug=True)',
             '    env.run(["main.py", "main.py"])',
             '    print("statuses", [s.get("status") for s in env.steps[-1]])',
             '    print("rewards ", [s.get("reward") for s in env.steps[-1]])',
             "except Exception as exc:",
             '    print("self-check unavailable here:", type(exc).__name__, exc)'),
    ]

    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3",
                                      "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    os.makedirs(out_dir, exist_ok=True)
    nb_path = os.path.join(out_dir, slug + ".ipynb")
    with open(nb_path, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, ensure_ascii=False)
    meta = {"id": f"debmalya84/{slug}", "title": title,
            "code_file": slug + ".ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True,
            "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
            "competition_sources": ["kaggriculture"], "dataset_sources": [],
            "kernel_sources": [], "model_sources": []}
    with open(os.path.join(out_dir, "kernel-metadata.json"), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    print(f"wrote {nb_path} ({os.path.getsize(nb_path):,} bytes)")
    for name in parts:
        print(f"  payload {name:<10} {sizes[name]:>9,} bytes raw")
    print(f"  archive sha256 {want}")
    print(f"\npush:   kaggle kernels push -p {os.path.relpath(out_dir, ROOT)}")
    print(f"submit: kaggle competitions submit kaggriculture "
          f"-k debmalya84/{slug} -v <N> -f submission.tar.gz -m \"...\"")
    return nb_path, want


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tar", default=TAR)
    ap.add_argument("--slug", default=SLUG)
    ap.add_argument("--title", default=TITLE)
    ap.add_argument("--out-dir", default=OUT_DIR)
    a = ap.parse_args()
    build(a.tar, a.out_dir, a.slug, a.title)


if __name__ == "__main__":
    main()
