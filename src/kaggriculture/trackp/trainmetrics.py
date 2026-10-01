"""Tiny append-only metrics log written NEXT TO every checkpoint, so we can see
how BC / RL are actually learning without re-parsing stdout.

One JSON record per line in ``<ckpt_dir>/metrics.jsonl``:

  BC : {"kind":"bc", "gstep", "epoch", "shard", "loss", "sps"}
  RL : {"kind":"rl", "gstep", "meanR", "winrate", "loss", "kl", "vloss", "ent", "eps"}

Plot/inspect with:  python -m kaggriculture.trackp.trainmetrics <ckpt_dir>
(prints the last N records + a simple trend). Never raises into the training loop.
"""
from __future__ import annotations
import json, os, time


def append(ckpt_path_or_dir: str, kind: str, **fields) -> None:
    """Append one metrics record. Accepts a checkpoint FILE path or a directory;
    the log always lands as ``metrics.jsonl`` in the checkpoint directory."""
    try:
        d = ckpt_path_or_dir
        if os.path.splitext(d)[1]:            # a file path -> its dir
            d = os.path.dirname(os.path.abspath(d)) or "."
        os.makedirs(d, exist_ok=True)
        rec = {"t": round(time.time(), 1),
               "iso": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind}
        rec.update(fields)
        with open(os.path.join(d, "metrics.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception:
        pass                                   # metrics must NEVER break training


def read(ckpt_dir: str, kind: str | None = None, last: int | None = None):
    p = os.path.join(ckpt_dir, "metrics.jsonl")
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if kind and r.get("kind") != kind:
            continue
        out.append(r)
    return out[-last:] if last else out


def _main():
    import argparse
    ap = argparse.ArgumentParser(description="show training metrics")
    ap.add_argument("ckpt_dir")
    ap.add_argument("--kind", default=None, help="bc | rl")
    ap.add_argument("--last", type=int, default=15)
    a = ap.parse_args()
    recs = read(a.ckpt_dir, a.kind, a.last)
    if not recs:
        print("(no metrics yet)"); return
    for r in recs:
        k = r.get("kind")
        if k == "bc":
            print(f"{r['iso']}  BC  gstep={r.get('gstep'):>8}  loss={r.get('loss')}  "
                  f"shard={r.get('shard')}  {r.get('sps')}/s")
        elif k == "rl":
            print(f"{r['iso']}  RL  gstep={r.get('gstep'):>10}  meanR={r.get('meanR')}  "
                  f"win={r.get('winrate')}  loss={r.get('loss')}  kl={r.get('kl')}  "
                  f"{r.get('eps')}eps/s")
        else:
            print(r)
    # crude trend: first vs last of the numeric learning signal
    if len(recs) >= 2:
        key = "loss" if recs[0].get("kind") == "bc" else "meanR"
        a0, a1 = recs[0].get(key), recs[-1].get(key)
        if isinstance(a0, (int, float)) and isinstance(a1, (int, float)):
            print(f"\ntrend {key}: {a0} -> {a1}  ({'+' if a1 >= a0 else ''}{round(a1 - a0, 4)})")


if __name__ == "__main__":
    _main()
