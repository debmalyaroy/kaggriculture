"""MacroNet: the macro policy. Linear(93->64) + ReLU -> GRUCell(64) over days -> heads.

  pi     [A]  logits over lever profiles (configs/profiles/*.json)
  v      [1]  win-probability logit
  rnext  [9]  rival's next-day sales (slog units), aux
  band   [5]  opponent rating band, aux
  fam    [5]  opponent family (league ground truth), aux, training only -- not in the export

Export format (weights.bin, little-endian f32, read by crates/policy): the tensors in EXPORT_ORDER,
flattened row-major, preceded by nothing; shapes/versions live in meta.json. The Rust forward
pass must reproduce the torch outputs (python/learn/export_check.py).
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn

NF = 93  # = crates/dayobs N (FEAT_VERSION 2)
H = 64
N_BAND = 5
N_FAM = 5  # python/learn/cache.py FAMILIES; aux, training only (not exported: play never reads it)
def profile_names(path=None):
    """Profile names in id order from configs/profiles/rl3.json ({"version", "note", "profiles": [...]})."""
    # KRL_PROFILES selects the action table (PPO3 plays configs/profiles/rl4.json: rl3 + project / endgame options)
    path = path or os.environ.get("KRL_PROFILES") or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs", "profiles", "rl3.json")
    d = json.load(open(path, encoding="utf-8"))
    ps = d["profiles"] if isinstance(d, dict) else d
    return [p.get("name", str(i)) if isinstance(p, dict) else str(i) for i, p in enumerate(ps)]


EXPORT_ORDER = ["mu", "sd", "inp.weight", "inp.bias", "gru.weight_ih", "gru.weight_hh", "gru.bias_ih", "gru.bias_hh",
                "pi.weight", "pi.bias", "v.weight", "v.bias", "rnext.weight", "rnext.bias", "band.weight", "band.bias"]


class MacroNet(nn.Module):
    def __init__(self, n_act, mu=None, sd=None):
        super().__init__()
        self.n_act = n_act
        self.register_buffer("mu", torch.zeros(NF) if mu is None else torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer("sd", torch.ones(NF) if sd is None else torch.as_tensor(sd, dtype=torch.float32))
        self.inp = nn.Linear(NF, H)
        self.gru = nn.GRUCell(H, H)
        self.pi = nn.Linear(H, n_act)
        self.v = nn.Linear(H, 1)
        self.rnext = nn.Linear(H, 9)
        self.band = nn.Linear(H, N_BAND)
        self.fam = nn.Linear(H, N_FAM)

    def forward(self, x, detach_v=False):
        """x [B, T, 93] -> dict of [B, T, *] (the hidden state after day t's observation).
        detach_v: the value head reads a detached trunk, so critic updates cannot move the policy
        (PPO; a shared-trunk critic drifted a 'frozen' policy to KL 3.3 in public runs)."""
        B, T, _ = x.shape
        z = torch.relu(self.inp((x - self.mu) / self.sd))
        h = x.new_zeros(B, H)
        hs = []
        for t in range(T):
            h = self.gru(z[:, t], h)
            hs.append(h)
        hs = torch.stack(hs, 1)
        hv = hs.detach() if detach_v else hs
        return {"pi": self.pi(hs), "v": self.v(hv).squeeze(-1), "rnext": self.rnext(hs), "band": self.band(hs), "fam": self.fam(hs)}

    def export(self, out_dir, meta):
        os.makedirs(out_dir, exist_ok=True)
        sd = self.state_dict()
        with open(os.path.join(out_dir, "weights.bin"), "wb") as fh:
            for k in EXPORT_ORDER:
                fh.write(sd[k].detach().cpu().float().contiguous().numpy().astype("<f4").tobytes())
        m = dict(meta)
        m.update(arch="gru64-v1", nf=NF, hidden=H, n_act=self.n_act, n_band=N_BAND,
                 order=EXPORT_ORDER, shapes={k: list(sd[k].shape) for k in EXPORT_ORDER})
        json.dump(m, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
