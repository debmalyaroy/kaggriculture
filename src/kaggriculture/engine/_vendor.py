"""Make `kaggle_environments` importable without a network install.

Prefers whatever is installed system-wide; falls back to the slimmed copy in
`vendor/` (the competition package with the visualiser assets and other
games' data removed -- 4.8 MB instead of 135 MB). Import this before importing
kaggle_environments.

To (re)create the vendored copy on a machine that does have network:

    pip download --no-deps kaggle-environments==1.32.3 -d /tmp/w
    cd /tmp && mkdir v && cd v && unzip -q /tmp/w/kaggle_environments-*.whl
    rm -rf kaggle_environments/envs/open_spiel_env
    find kaggle_environments -type d -name visualizer -exec rm -rf {} +
    tar czf ke_slim.tgz kaggle_environments   # -> vendor/
"""
from kaggriculture.paths import ROOT
import os
import sys

VENDOR = os.path.join(ROOT, "vendor")


def _missing_module(exc):
    """The module name out of an ImportError, if it names one."""
    name = getattr(exc, "name", None)
    if name:
        return name
    text = str(exc)
    if "No module named" in text:
        return text.split("No module named", 1)[1].strip().strip("'\"")
    return None


def ensure():
    """Make kaggle_environments importable, and say *why* if it is not.

    The failure this used to produce was misleading. `import
    kaggle_environments` can fail for two completely different reasons -- the
    package is absent, or the package is present and one of *its* dependencies
    (jsonschema, requests) is not -- and reporting both as "not available" sent
    people off to reinstall something they already had.
    """
    first = None
    vendored = os.path.isdir(os.path.join(VENDOR, "kaggle_environments"))
    if not vendored:
        try:
            import kaggle_environments  # noqa: F401
            return "system"
        except ImportError as exc:
            first = exc

    # The vendored copy WINS when present (2026-09-24). It is the one
    # scripts/engine_swap_1327.py keeps on the ladder's ruleset; a system /
    # user-site install can be stale -- this box had a 1.32.2 kaggriculture.py
    # in user site-packages (CARROT on the old sqrt curve, not the 1.32.7
    # hinge) that silently halved banks and made test_rust_engine.py "fail"
    # a correct Rust engine. Preferring the system install hid that.
    if vendored:
        if sys.path[:1] != [VENDOR]:
            if VENDOR in sys.path:
                sys.path.remove(VENDOR)
            sys.path.insert(0, VENDOR)
            # The stale-import trap (Rayk Kretzschmar's write-up, confirmed
            # here): a pip-installed kaggle_environments earlier on sys.path
            # can shadow the vendored engine while version checks read the
            # new metadata. Assert the import actually resolves to us.
            import importlib
            ke = importlib.import_module("kaggle_environments")
            got = os.path.normcase(os.path.dirname(ke.__file__))
            want = os.path.normcase(os.path.join(VENDOR, "kaggle_environments"))
            assert got == want, (
                f"kaggle_environments resolved to {got}, not the vendored "
                f"copy {want} -- a stale install is shadowing the engine")
        try:
            import kaggle_environments  # noqa: F401
            return "vendored"
        except ImportError as exc:
            first = exc

    missing = _missing_module(first)
    if missing and missing.split(".")[0] != "kaggle_environments":
        # The package is here; one of its dependencies is not. Say so, because
        # "reinstall kaggle-environments" would not fix it.
        raise ImportError(
            f"kaggle_environments is present but cannot import: it needs "
            f"'{missing}', which is not installed for this interpreter "
            f"({sys.executable}).\n"
            f"  fix:  python -m pip install {missing}\n"
            f"  or all of them:  python -m pip install jsonschema requests\n"
            f"  original error: {first}") from first

    raise ImportError(
        "kaggle_environments is not available.\n"
        f"  interpreter: {sys.executable}\n"
        f"  vendored copy present: {vendored} ({VENDOR})\n"
        "  fix:  python -m pip install -r requirements.txt\n"
        f"  original error: {first}")


ensure()
