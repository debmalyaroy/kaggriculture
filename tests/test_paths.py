"""Every path the code constructs must exist.

The gap this closes: the 2026-08-10 restructure moved `tools/` to `src/`, and
39 constructed paths kept pointing at the old directory. `tests/test_agents.py`
was among them, so the contract suite had been UNRUNNABLE for days -- and the
house rule says to run it after every behavioural change. Nothing detected that,
because a stale path only fails when the line executes.

This walks the tree for path construction that names a top-level project
directory and asserts the directory is real. It is deliberately shallow: it
catches whole-directory drift (the failure that actually happened) rather than
trying to resolve every dynamic filename.

    python tests/test_paths.py
"""
from kaggriculture.paths import ROOT
import ast
import os
import re
import sys

SCAN_DIRS = ("src", "scripts", "tests", "dashboard", "python", "kaggle", "ops")
SKIP_PARTS = ("__pycache__", ".local", "vendor", "node_modules")

# Directories the project is structured around (AGENTS.md / docs/repo-layout.md).
KNOWN = {"src", "agents", "models", "configs", "data", "scripts", "dashboard", "docs", "tests", "notebooks",
         "rustengine", "opponents", "crates", "python", "kaggle", "ops", "aws", "research", "weights"}
# Names that look like directories but are legitimately absent: generated-data folders are created on demand by the
# code that writes them (the repo carries no data since the 2026-10-01 purge), and vendor/ is an optional local copy
# of kaggle-environments (normally pip-installed).
ALLOW_MISSING = {"vendor", "data", "models", "notebooks", "opponents", "weights"}


def py_files():
    for d in SCAN_DIRS:
        base = os.path.join(ROOT, d)
        for dirpath, dirnames, filenames in os.walk(base):
            if any(p in dirpath for p in SKIP_PARTS):
                continue
            dirnames[:] = [x for x in dirnames if x not in SKIP_PARTS]
            for f in filenames:
                # Skip this file: it necessarily contains the very literals it
                # searches for ("tools", the KNOWN set), so scanning itself
                # reports a false positive on its own pattern strings.
                if f.endswith(".py") and f != os.path.basename(__file__):
                    yield os.path.join(dirpath, f)


def referenced_dirs(path):
    """Top-level project dirs named in os.path.join(...) / literal paths."""
    src = open(path, encoding="utf-8", errors="replace").read()
    found = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return found
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        is_join = (isinstance(fn, ast.Attribute) and fn.attr == "join")
        is_insert = (isinstance(fn, ast.Attribute) and fn.attr == "insert")
        if not (is_join or is_insert):
            continue
        for a in node.args:
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                v = a.value.strip().strip("/\\")
                if v in KNOWN:
                    found.add(v)
    # plus obvious literal relative paths like "src/kaggriculture/measure/evaluate.py"
    for m in re.finditer(r'["\']([a-z_]+)[/\\][A-Za-z0-9_./\\-]+["\']', src):
        if m.group(1) in KNOWN:
            found.add(m.group(1))
    return found


def test_no_stale_directory_references():
    bad = []
    checked = 0
    for f in py_files():
        for d in referenced_dirs(f):
            checked += 1
            if d in ALLOW_MISSING:
                continue
            if not os.path.isdir(os.path.join(ROOT, d)):
                bad.append((os.path.relpath(f, ROOT), d))
    if bad:
        print(f"{len(bad)} reference(s) to directories that do not exist:")
        for f, d in sorted(set(bad))[:30]:
            print(f"  {f}: {d}/")
        raise AssertionError(f"{len(bad)} stale directory reference(s)")
    print(f"{checked} directory references across "
          f"{len(list(py_files()))} files -- all resolve")


def test_the_restructure_is_complete():
    """`tools/` is gone; nothing may construct a path into it."""
    assert not os.path.isdir(os.path.join(ROOT, "tools")), \
        "tools/ exists again -- the restructure invariant changed"
    offenders = []
    for f in py_files():
        src = open(f, encoding="utf-8", errors="replace").read()
        for m in re.finditer(r'(os\.path\.join\([^)]*|sys\.path\.insert\([^)]*)'
                             r'["\']tools["\']', src):
            offenders.append(os.path.relpath(f, ROOT))
    assert not offenders, f"still building paths into tools/: {set(offenders)}"
    print("no code constructs a path into the removed tools/ directory")


def test_key_project_dirs_present():
    missing = [d for d in ("src", "agents", "scripts", "docs", "tests", "configs", "crates", "python", "rustengine")
               if not os.path.isdir(os.path.join(ROOT, d))]
    assert not missing, missing
    print("core project directories present")


if __name__ == "__main__":
    for fn in (test_key_project_dirs_present,
               test_the_restructure_is_complete,
               test_no_stale_directory_references):
        fn()
    print("\nall path checks passed")
