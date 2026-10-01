"""Read-only access to the mined route store, for the Track P lane.

Why this file exists
--------------------
`tests/test_trackp.py::test_isolation` pins the lane rule from CLAUDE.md:
**Track P shares zero code with the bandit/route lane.** `src/routes.py` is a
bandit/route module and, worse, importing it drags in `_vendor`, `episodes`
and `opponents` -- i.e. the whole scraping stack -- for what the factory,
the GA and the shop-branch study actually want, which is two file reads.

So this is the *store format*, re-implemented against the on-disk layout, not
a wrapper around `routes.py`. It is deliberately READ-ONLY: nothing in Track P
mines or writes routes, and a writer here would be a second author of
`index.json` (see `routes.save_index`'s lock, which exists because two writers
once lost 196 same-day routes).

The two behaviours that must not drift from `src/routes.py`:

* `route_path()` sanitises the id into a filename the same way -- a "?" is a
  legal JSON key and an illegal Windows path component, and that difference
  once killed a 12 GB mine;
* `load_index()` answers an empty index rather than raising when the store is
  absent or half-written, so a Track P tool on a fresh checkout reports "no
  routes" instead of a traceback.

`tests/test_trackp.py::test_routes_io` pins both against `src/routes.py` when
that module is importable, and skips when it is not.
"""
from __future__ import annotations
from kaggriculture.paths import ROOT

import gzip
import json
import os

WORK = os.path.join(ROOT, "data", "routes")
INDEX = os.path.join(WORK, "index.json")


def route_path(route_id) -> str:
    """The on-disk path for a route id. Byte-for-byte `routes.route_path`."""
    safe = "".join(c if (c.isalnum() or c in "._-") else "_"
                   for c in str(route_id))
    return os.path.join(WORK, f"{safe or 'unknown'}.json.gz")


def load_index() -> dict:
    """The route index, or an empty one if it is missing or unparseable."""
    if os.path.exists(INDEX):
        try:
            with open(INDEX, encoding="utf-8") as fh:
                return json.load(fh)
        except ValueError:
            pass
    return {"routes": {}, "mined": [], "updated": None}


def load_route(route_id) -> list:
    """The per-turn action tape for one route id."""
    with gzip.open(route_path(route_id), "rt", encoding="utf-8") as fh:
        return json.load(fh)


def has_route(route_id) -> bool:
    return os.path.exists(route_path(route_id))
