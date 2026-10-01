"""TPP dataset reader: the fixed-size records of crates/runner/src/bin/bcdump.rs (layout = crates/agent/src/tpp.rs)."""
import glob
import os

import numpy as np

B, C, G, MAXU, UF, MAXM = 10, 29, 97, 17, 18, 10
NUNIT, NUQ, NMKT, NMQ = 44, 8, 34, 11
REC = np.dtype([("game", "<u4"), ("step", "<u2"), ("nu", "u1"), ("nm", "u1"), ("res", "<f4"), ("margin", "<f4"),
                ("board", "u1", (C, B, B)), ("glob", "<f4", (G,)), ("units", "u1", (MAXU, UF)),
                ("ulab", "u1", (MAXU, 2)), ("mlab", "u1", (MAXM, 2))])
assert REC.itemsize == 3664, REC.itemsize


def open_parts(d):
    """Memory-mapped record arrays (one per part file) -- nothing is loaded into RAM up front."""
    return [np.memmap(f, dtype=REC, mode="r") for f in sorted(glob.glob(os.path.join(d, "part_*.bin"))) if os.path.getsize(f) >= REC.itemsize]
