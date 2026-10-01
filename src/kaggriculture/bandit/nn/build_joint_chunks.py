"""Build joint_Y chunks = [sale_Y (9) | opp_y (1)] for the merged multiclass head.

The joint head shares sale_X as its input (identical to opp_X), so we only need to
stack the labels. Writes joint_Y_NNN.npy beside the sale_X_NNN.npy the trainer will
glob (train_stream --kind joint uses xpat=sale_X, ypat=joint_Y). No X duplication.

Run: python -m kaggriculture.bandit.nn.build_joint_chunks
"""
import glob, os
import numpy as np
from kaggriculture.paths import ROOT

CH = os.path.join(ROOT, ".local", "nn", "chunks")


def main():
    sx = sorted(glob.glob(os.path.join(CH, "sale_X_*.npy")))
    sy = sorted(glob.glob(os.path.join(CH, "sale_Y_*.npy")))
    oy = sorted(glob.glob(os.path.join(CH, "opp_y_*.npy")))
    assert len(sx) == len(sy) == len(oy) and sx, f"chunk count mismatch in {CH}"
    n = 0
    for i, (fsy, foy) in enumerate(zip(sy, oy)):
        S = np.load(fsy).astype(np.float32)             # (rows, 9)
        O = np.load(foy).reshape(-1, 1).astype(np.float32)  # (rows, 1)
        assert len(S) == len(O), f"row mismatch chunk {i}: {len(S)} vs {len(O)}"
        J = np.concatenate([S, O], axis=1)              # (rows, 10)
        np.save(os.path.join(CH, f"joint_Y_{i:03d}.npy"), J)
        n += len(J)
    print(f"wrote {len(sx)} joint_Y chunks ({n} rows, 10 cols = 9 sale + 1 opp) -> {CH}", flush=True)


if __name__ == "__main__":
    main()
