"""Advantage-weighted BC for the macro policy + value and aux heads (tasks P6.3, queue Q11/Q12).

    python python/learn/bc.py                        # full run (resumes an unfinished one)
    python python/learn/bc.py --from latest --delta-only   # fine-tune the newest teacher on new data

Data (python/learn/cache.py):
  league  our random-profile games: policy target = the profile played each day, weighted by
          exp((score - V)/beta) (advantage-weighted regression); value target = score.
  corpus  real ladder games: value (score), aux rival-next-day sales, aux opponent band.
Held out: episode_id % 10 == 0 (corpus), seed % 10 == 0 (league) -- never trained on.

Every run writes weights/bc/<BCID>/ {ckpt.pt (model+optimizer+step+rng, atomic), weights.bin +
meta.json (crates/policy format), metrics.json, DONE}; weights/bc/LATEST names the newest finished
run and weights/registry.json lists every run. An unfinished run is resumed from its checkpoint.
Fine-tune with no new data exits 0 without a new teacher.
"""
import argparse
import datetime as dt
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import MacroNet, NF, profile_names  # noqa: E402

RL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN = os.path.join(RL, "data", "train")
WDIR = os.path.join(RL, "weights", "bc")
REG = os.path.join(RL, "weights", "registry.json")
PROFILES = os.path.join(RL, "configs", "profiles", "rl3.json")
FEAT_VERSION = 2  # = crates/dayobs FEAT_VERSION


def now():
    return dt.datetime.now(dt.timezone.utc)


def load_data():
    d = {}
    if os.path.exists(os.path.join(TRAIN, "corpus_obs.npy")):
        d["c_obs"] = np.load(os.path.join(TRAIN, "corpus_obs.npy"), mmap_mode="r")
        d["c_rn"] = np.load(os.path.join(TRAIN, "corpus_rnext.npy"), mmap_mode="r")
        d["c_meta"] = dict(np.load(os.path.join(TRAIN, "corpus_meta.npz")))
    if os.path.exists(os.path.join(TRAIN, "league_obs.npy")):
        d["l_obs"] = np.load(os.path.join(TRAIN, "league_obs.npy"), mmap_mode="r")
        d["l_meta"] = dict(np.load(os.path.join(TRAIN, "league_meta.npz")))
    return d


def registry():
    try:
        return json.load(open(REG, encoding="utf-8"))
    except (OSError, ValueError):
        return {"runs": []}


def save_registry(r):
    os.makedirs(os.path.dirname(REG), exist_ok=True)
    tmp = REG + ".tmp"
    json.dump(r, open(tmp, "w", encoding="utf-8"), indent=1)
    os.replace(tmp, REG)


def latest():
    try:
        return open(os.path.join(WDIR, "LATEST")).read().strip()
    except OSError:
        return None


def batch(obs, idx, dev):
    idx = np.sort(idx)
    return torch.from_numpy(np.asarray(obs[idx], dtype=np.float32)).to(dev), idx


def evaluate(net, d, dev, beta):
    net.eval()
    out = {}
    with torch.no_grad():
        if "l_obs" in d:
            m = d["l_meta"]
            ho = np.where(m["seed"] % 10 == 0)[0][:4000]
            if len(ho):
                x, ho = batch(d["l_obs"], ho, dev)
                o = net(x)
                a = torch.from_numpy(m["sched"][ho].astype(np.int64)).to(dev)
                y = torch.from_numpy(m["score"][ho]).to(dev)[:, None].expand(-1, 30)
                v = torch.sigmoid(o["v"])
                w = torch.clamp(torch.exp((y - v) / beta), max=20.0)
                ce = F.cross_entropy(o["pi"].reshape(-1, net.n_act), a.reshape(-1), reduction="none").reshape(a.shape)
                out["league_wce"] = float((w * ce).sum() / w.sum())
                out["league_value_brier"] = float(((v - y) ** 2).mean())
                out["league_heldout_games"] = int(len(ho))
        if "c_obs" in d:
            m = d["c_meta"]
            ho = np.where((m["episode_id"] % 10 == 0) & np.isfinite(m["score"]) & (m["days"] >= 30))[0][:6000]
            if len(ho):
                x, ho = batch(d["c_obs"], ho, dev)
                o = net(x)
                y = torch.from_numpy(m["score"][ho]).to(dev)
                v = torch.sigmoid(o["v"])
                for day in (0, 3, 10, 20, 29):
                    out[f"corpus_value_brier_d{day}"] = float(((v[:, day] - y) ** 2).mean())
                    out[f"corpus_value_acc_d{day}"] = float((((v[:, day] > .5).float() == y) | (y == .5)).float().mean())
                band = torch.from_numpy(m["opp_band"][ho].astype(np.int64)).to(dev)
                k = band >= 0
                if k.any():
                    out["band_acc_d3"] = float((o["band"][:, 3].argmax(-1)[k] == band[k]).float().mean())
                    out["band_acc_d10"] = float((o["band"][:, 10].argmax(-1)[k] == band[k]).float().mean())
                rn = torch.from_numpy(np.asarray(d["c_rn"][ho], dtype=np.float32)).to(dev)
                out["rnext_mse"] = float(((o["rnext"][:, :29] - rn[:, :29]) ** 2).mean())
                out["corpus_heldout_rows"] = int(len(ho))
    net.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--from", dest="parent", default=None, help="'latest' or a BCID to fine-tune")
    ap.add_argument("--delta-only", action="store_true", help="fine-tune on data newer than the parent's")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=500)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = load_data()
    if "l_obs" not in d and "c_obs" not in d:
        sys.exit("no training caches (run python/learn/cache.py)")
    n_act = len(profile_names(PROFILES))

    # -------- run directory: resume an unfinished run of the same kind, else start one
    parent = None
    if a.parent:
        parent = latest() if a.parent == "latest" else a.parent
        if not parent:
            sys.exit("no parent teacher yet")
    kind = "ft" if parent else "full"
    run = None
    for r in sorted(glob.glob(os.path.join(WDIR, "*", "ckpt.pt"))):
        rd = os.path.dirname(r)
        if not os.path.exists(os.path.join(rd, "DONE")) and json.load(open(os.path.join(rd, "run.json")))["kind"] == kind:
            run = rd
    c_meta, l_meta = d.get("c_meta"), d.get("l_meta")
    c_ok = np.zeros(0, np.int64) if c_meta is None else np.where((c_meta["episode_id"] % 10 != 0) & np.isfinite(c_meta["score"]) & (c_meta["days"] >= 2))[0]
    l_ok = np.zeros(0, np.int64) if l_meta is None else np.where(l_meta["seed"] % 10 != 0)[0]
    pmeta = None
    if parent:
        pmeta = json.load(open(os.path.join(WDIR, parent, "meta.json")))
        if a.delta_only:
            if c_meta is not None:
                c_ok = c_ok[c_meta["end_date"][c_ok] > int(pmeta.get("corpus_to", 0))]
            if l_meta is not None:
                l_ok = l_ok[l_meta["batch"][l_ok] > int(pmeta.get("league_to", -1))]
            if len(c_ok) == 0 and len(l_ok) == 0:
                print(f"[bc] nothing newer than parent {parent}; no new teacher")
                return
    corpus_to = int(c_meta["end_date"].max()) if c_meta is not None else 0
    league_to = int(l_meta["batch"].max()) if l_meta is not None else -1
    if run is None:
        bcid = f"bc-gru64-f{FEAT_VERSION}-p{n_act}-s1-to{corpus_to}-{now():%Y%m%dT%H%MZ}" + ("-ft" if parent else "")
        run = os.path.join(WDIR, bcid)
        os.makedirs(run, exist_ok=True)
        json.dump({"kind": kind, "parent": parent, "started": f"{now():%Y-%m-%dT%H:%M:%SZ}", "args": vars(a)},
                  open(os.path.join(run, "run.json"), "w"), indent=1)
    bcid = os.path.basename(run)

    # -------- model: normalization from the data (or the parent's), init from parent
    if parent:
        mu = np.fromfile(os.path.join(WDIR, parent, "weights.bin"), dtype="<f4", count=NF)
        sdv = np.fromfile(os.path.join(WDIR, parent, "weights.bin"), dtype="<f4", count=2 * NF)[NF:]
    else:
        src = d["c_obs"] if "c_obs" in d else d["l_obs"]
        rows = np.sort(np.random.default_rng(0).choice(len(src), size=min(20000, len(src)), replace=False))
        s = np.asarray(src[rows], dtype=np.float32).reshape(-1, NF)
        s = s[np.abs(s).sum(1) > 0]
        mu, sdv = s.mean(0), np.maximum(s.std(0), 1e-3)
    net = MacroNet(n_act, mu, sdv).to(dev)
    if parent:
        net.load_state_dict(torch.load(os.path.join(WDIR, parent, "ckpt.pt"), map_location=dev)["model"])
    opt = torch.optim.Adam(net.parameters(), lr=a.lr if not parent else a.lr / 4)
    step = 0
    ck = os.path.join(run, "ckpt.pt")
    if os.path.exists(ck):
        s = torch.load(ck, map_location=dev, weights_only=False)
        net.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        step = s["step"]
        np.random.set_state(s["np_rng"])
        print(f"[bc] resumed {bcid} at step {step}", flush=True)
    steps = a.steps if not parent else min(a.steps, 1500)
    rng = np.random.default_rng(step + 17)
    print(f"[bc] {bcid}: device {dev}; league train games {len(l_ok)}; corpus train rows {len(c_ok)}; steps {step}->{steps}", flush=True)

    def save():
        tmp = ck + ".tmp"
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "step": step, "np_rng": np.random.get_state()}, tmp)
        os.replace(tmp, ck)

    while step < steps:
        loss = torch.zeros((), device=dev)
        parts = {}
        if len(l_ok):
            idx = rng.choice(l_ok, size=min(a.bs, len(l_ok)), replace=False)
            x, idx = batch(d["l_obs"], idx, dev)
            o = net(x)
            act = torch.from_numpy(l_meta["sched"][idx].astype(np.int64)).to(dev)
            y = torch.from_numpy(l_meta["score"][idx]).to(dev)[:, None].expand(-1, 30)
            v = torch.sigmoid(o["v"])
            w = torch.clamp(torch.exp((y - v.detach()) / a.beta), max=20.0)
            ce = F.cross_entropy(o["pi"].reshape(-1, n_act), act.reshape(-1), reduction="none").reshape(act.shape)
            lp = (w * ce).sum() / w.sum()
            lv = F.binary_cross_entropy_with_logits(o["v"], y)
            fam = torch.from_numpy(l_meta["opp_fam"][idx].astype(np.int64)).to(dev) if "opp_fam" in l_meta else None
            lf = torch.zeros((), device=dev)
            if fam is not None and (fam >= 0).any():
                kf = fam >= 0
                lf = F.cross_entropy(o["fam"][kf].reshape(-1, o["fam"].shape[-1]), fam[kf][:, None].expand(-1, 30).reshape(-1))
                # the plan's target: the family is identified by day 3 (index 3 = the day-3 decision)
                parts["fam_acc_d3"] = float((o["fam"][kf][:, 3].argmax(-1) == fam[kf]).float().mean())
            loss = loss + lp + 0.5 * lv + 0.2 * lf
            parts.update(pi=float(lp), v_l=float(lv), fam=float(lf))
        if len(c_ok):
            idx = rng.choice(c_ok, size=min(a.bs, len(c_ok)), replace=False)
            x, idx = batch(d["c_obs"], idx, dev)
            o = net(x)
            days = torch.from_numpy(c_meta["days"][idx].astype(np.int64)).to(dev)
            tt = torch.arange(30, device=dev)[None]
            valid = (tt < days[:, None]).float()
            y = torch.from_numpy(c_meta["score"][idx]).to(dev)[:, None].expand(-1, 30)
            lv = (F.binary_cross_entropy_with_logits(o["v"], y, reduction="none") * valid).sum() / valid.sum()
            rn = torch.from_numpy(np.asarray(d["c_rn"][idx], dtype=np.float32)).to(dev)
            vr = (tt < days[:, None] - 1).float()
            lr_ = (((o["rnext"] - rn) ** 2).mean(-1) * vr).sum() / vr.sum().clamp(min=1)
            band = torch.from_numpy(c_meta["opp_band"][idx].astype(np.int64)).to(dev)
            kb = band >= 0
            lb = torch.zeros((), device=dev)
            if kb.any():
                bl = F.cross_entropy(o["band"][kb].reshape(-1, 5), band[kb][:, None].expand(-1, 30).reshape(-1), reduction="none")
                lb = (bl.reshape(-1, 30) * valid[kb]).sum() / valid[kb].sum()
            loss = loss + 0.5 * lv + 0.1 * lr_ + 0.2 * lb
            parts.update(v_c=float(lv), rnext=float(lr_), band=float(lb))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 100 == 0:
            print(f"[bc] step {step}/{steps} " + " ".join(f"{k} {v:.4f}" for k, v in parts.items()), flush=True)
        if step % a.eval_every == 0 or step == steps:
            save()
            ev = evaluate(net, d, dev, a.beta)
            print(f"[bc] eval step {step}: " + json.dumps(ev), flush=True)
            json.dump({"step": step, **ev}, open(os.path.join(run, "metrics.json"), "w"), indent=1)

    meta = {"bcid": bcid, "parent": parent, "feat_version": FEAT_VERSION, "profiles": "configs/profiles/rl3.json",
            "corpus_to": corpus_to, "league_to": league_to, "steps": steps,
            "finished": f"{now():%Y-%m-%dT%H:%M:%SZ}"}
    net.export(run, meta)
    open(os.path.join(run, "DONE"), "w").write(meta["finished"])
    open(os.path.join(WDIR, "LATEST.tmp"), "w").write(bcid)
    os.replace(os.path.join(WDIR, "LATEST.tmp"), os.path.join(WDIR, "LATEST"))
    reg = registry()
    reg.setdefault("runs", []).append({**meta, "metrics": json.load(open(os.path.join(run, "metrics.json"))), "status": "teacher"})
    save_registry(reg)
    print(f"[bc] DONE {bcid} -> weights/bc/LATEST", flush=True)


if __name__ == "__main__":
    main()
