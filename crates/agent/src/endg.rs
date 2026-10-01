//! Learned endgame controller (rlV2 workstream D, docs/endgame-model-2026-09-27.md).
//!
//! At ONE decision step `from` (default 673 = day 28 hour 1, right after the day's profile switch; a proposal may
//! start later with its own `from`) the controller
//! picks an ENDGAME PROPOSAL: a patch over the endgame knobs (terminal closure planner `term_*`, final-day sale
//! advance `tsell_*`). The patch is applied on top of whatever profile is active on every later step (PPO keeps
//! choosing the day's profile; only the endgame knobs are replaced), so the choice holds to the end of the game.
//!
//! Two small MLPs (JSON, trained by python/endg/train.py on exact counterfactual labels from
//! crates/runner/src/bin/endg-label.rs):
//!   proposal   x -> K logits (prior over proposals: which are worth considering here)
//!   objective  x -> 2K (win logit per proposal, margin delta vs proposal 0 in units of $2,000)
//! Modes: observe (inputs only), force K (the labeller), rerank (the top `top_m` proposals by prior + proposal 0;
//! argmax utility, kept only when it beats proposal 0 by `gate`), replace (argmax utility over all K).
//! Utility u_k = sigmoid(win_k) + lam * tanh(dm_k), or with `"util": "updown"` (objective = up / down logits per
//! proposal vs proposal 0): u_k = sigmoid(up_k) - sigmoid(down_k), u_0 = 0.
//!
//! Inputs: the reactive shell's GLOBAL inputs (35) and 10 per-item inputs x 9 items (+ presence flag) of the
//! PREVIOUS turn (the shell runs in observe mode when the agent has no shell v2), plus 6 knobs of the active profile.
use crate::layers::knobs::Knobs;
use crate::rshell::{ItemIn, ITEMF, NG, NI};
use kagg_engine::json::{self, Json};
use std::sync::Arc;

pub const ITEM_FEATS: [&str; 10] = ["stock", "px", "inv", "rival_sold_24", "plan_24", "rev_0", "rev_4", "best_wait", "need", "surplus"];
pub const NKF: usize = 6;
pub const NF: usize = NG + NI * (ITEM_FEATS.len() + 1) + NKF;
/// Standardisation guard (python/endg/train.py `standardise` must match): std at or below STD_CONST = constant in training.
pub const STD_CONST: f32 = 2e-3;
pub const ZMAX: f32 = 8.0;

/// The model input at the decision step.
pub fn features(g: &[f32; NG], last: &[Option<ItemIn>], k: &Knobs) -> Vec<f32> {
    let mut x = Vec::with_capacity(NF);
    x.extend_from_slice(g);
    for i in 0..NI {
        match last.get(i).and_then(|o| o.as_ref()) {
            Some(it) => {
                x.push(1.0);
                for f in ITEM_FEATS {
                    let j = ITEMF.iter().position(|y| *y == f).unwrap();
                    x.push(it.x[j]);
                }
            }
            None => x.extend(std::iter::repeat_n(0.0, ITEM_FEATS.len() + 1)),
        }
    }
    x.push(k.term_on as i32 as f32);
    x.push(k.term_start as f32 - 712.0);
    x.push((k.term_sims as f32).max(1.0).ln());
    x.push(k.tsell_on as i32 as f32);
    x.push(k.tsell_model as f32);
    x.push(k.v92_on as i32 as f32);
    debug_assert_eq!(x.len(), NF);
    x
}

/// A dense ReLU MLP with input standardisation.
#[derive(Clone, Debug)]
pub struct Mlp {
    mean: Vec<f32>,
    std: Vec<f32>,
    layers: Vec<(Vec<Vec<f32>>, Vec<f32>)>,
}

fn fv(j: &Json) -> Vec<f32> {
    j.arr().iter().map(|v| v.f64() as f32).collect()
}

impl Mlp {
    pub fn from_json(j: &Json) -> Result<Mlp, String> {
        let layers: Vec<(Vec<Vec<f32>>, Vec<f32>)> = j.get("layers").arr().iter().map(|l| (l.get("w").arr().iter().map(fv).collect(), fv(l.get("b")))).collect();
        if layers.is_empty() {
            return Err("mlp without layers".into());
        }
        let m = Mlp { mean: fv(j.get("mean")), std: fv(j.get("std")), layers };
        if m.mean.len() != NF || m.std.len() != NF || m.layers[0].0.first().map(|r| r.len()) != Some(NF) {
            return Err(format!("mlp input width != {NF}"));
        }
        Ok(m)
    }

    pub fn run(&self, x: &[f32]) -> Vec<f32> {
        // A feature that never varied in training (std at the trainer's 1e-3 floor) carries no learned signal: it enters
        // as 0 (its training value). Everything else is clamped to +-ZMAX. Without this, an input the training set held
        // constant (v1: opponent group, lineage flags -- trained on copy games only) arrives thousands of sigmas out,
        // saturates every sigmoid and the utility collapses to 0 for all proposals (v63.8..v63.10: model never fired).
        let mut h: Vec<f32> = x.iter().zip(&self.mean).zip(&self.std).map(|((v, m), s)| if *s <= STD_CONST { 0.0 } else { ((v - m) / s).clamp(-ZMAX, ZMAX) }).collect();
        let n = self.layers.len();
        for (li, (w, b)) in self.layers.iter().enumerate() {
            let mut o: Vec<f32> = w.iter().zip(b).map(|(row, bb)| row.iter().zip(&h).map(|(a, c)| a * c).sum::<f32>() + bb).collect();
            if li + 1 < n {
                o.iter_mut().for_each(|v| *v = v.max(0.0));
            }
            h = o;
        }
        h
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode {
    Observe,
    Force(usize),
    Rerank,
    Replace,
}

#[derive(Clone, Debug)]
pub struct EndgCfg {
    pub from: i64,
    pub names: Vec<String>,
    pub patches: Vec<Json>,
    /// First step each proposal's patch applies (default `from`; a proposal may start later, never earlier).
    pub starts: Vec<i64>,
    pub mode: Mode,
    pub gate: f32,
    pub top_m: usize,
    pub lam: f32,
    /// Utility form: false = sigmoid(win_k) + lam * tanh(dm_k) (objective outputs win logits + margin deltas);
    /// true ("util": "updown") = sigmoid(up_k) - sigmoid(down_k), the predicted chance that proposal k turns the result
    /// UP vs proposal 0 minus the chance it turns it DOWN (u_0 = 0).
    pub updown: bool,
    pub objective: Option<Mlp>,
    pub proposal: Option<Mlp>,
}

impl EndgCfg {
    /// `{"from":673,"mode":"rerank"|"replace"|"observe"|"force:K","gate":0.02,"top_m":6,"lam":0.1,
    ///   "proposals":[{"name":..,"knobs":{..}},..],"objective":MLP,"proposal":MLP}`
    pub fn load(path: &str) -> Result<EndgCfg, String> {
        let j = json::parse(&std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?)?;
        let props = j.get("proposals").arr();
        if props.is_empty() {
            return Err(format!("{path}: no proposals"));
        }
        let mode = match j.get("mode").str() {
            "" | "rerank" => Mode::Rerank,
            "replace" => Mode::Replace,
            "observe" => Mode::Observe,
            m if m.starts_with("force:") => Mode::Force(m[6..].parse().map_err(|_| format!("bad mode {m}"))?),
            m => return Err(format!("bad endg mode {m}")),
        };
        let num = |k: &str, d: f64| if j.get(k).is_num() { j.get(k).f64() } else { d };
        let mlp = |k: &str| -> Result<Option<Mlp>, String> { if j.get(k).is_obj() { Mlp::from_json(j.get(k)).map(Some) } else { Ok(None) } };
        let cfg = EndgCfg {
            from: num("from", 673.0) as i64,
            names: props.iter().map(|p| p.get("name").str().to_string()).collect(),
            patches: props.iter().map(|p| p.get("knobs").clone()).collect(),
            starts: vec![],
            mode,
            gate: num("gate", 0.02) as f32,
            top_m: num("top_m", 6.0).max(1.0) as usize,
            lam: num("lam", 0.1) as f32,
            updown: j.get("util").str() == "updown",
            objective: mlp("objective")?,
            proposal: mlp("proposal")?,
        };
        let mut cfg = cfg;
        cfg.starts = props.iter().map(|p| if p.get("from").is_num() { (p.get("from").i64()).max(cfg.from) } else { cfg.from }).collect();
        // validate every patch against the knob parser once
        for p in &cfg.patches {
            Knobs::default().with(p)?;
        }
        Ok(cfg)
    }
}

#[derive(Clone, Debug)]
pub struct EndgCtl {
    pub cfg: Arc<EndgCfg>,
    /// per seat: (player, chosen proposal, inputs, utilities)
    pub seats: Vec<(i64, usize, Vec<f32>, Vec<f32>)>,
}

fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

impl EndgCtl {
    pub fn new(cfg: Arc<EndgCfg>) -> EndgCtl {
        EndgCtl { cfg, seats: vec![] }
    }

    /// The proposal chosen for `player` (None before the decision step).
    pub fn chosen(&self, player: i64) -> Option<usize> {
        self.seats.iter().find(|s| s.0 == player).map(|s| s.1)
    }

    /// Decide once for `player` from the inputs `x`; returns the proposal index.
    pub fn decide(&mut self, player: i64, x: Vec<f32>) -> usize {
        let c = &self.cfg;
        let k = c.patches.len();
        let (pick, util) = match c.mode {
            Mode::Observe => (0, vec![]),
            Mode::Force(f) => (f.min(k - 1), vec![]),
            Mode::Rerank | Mode::Replace => match c.objective.as_ref() {
                None => (0, vec![]),
                Some(obj) => {
                    let o = obj.run(&x);
                    let at = |i: usize| o.get(i).copied().unwrap_or(0.0);
                    let u: Vec<f32> = (0..k)
                        .map(|i| {
                            if c.updown {
                                if i == 0 { 0.0 } else { sigmoid(at(i)) - sigmoid(at(k + i)) }
                            } else {
                                sigmoid(at(i)) + c.lam * at(k + i).tanh()
                            }
                        })
                        .collect();
                    let mut cands: Vec<usize> = (0..k).collect();
                    if c.mode == Mode::Rerank {
                        if let Some(pm) = c.proposal.as_ref() {
                            let pr = pm.run(&x);
                            cands.sort_by(|a, b| pr.get(*b).copied().unwrap_or(f32::MIN).partial_cmp(&pr.get(*a).copied().unwrap_or(f32::MIN)).unwrap());
                            cands.truncate(c.top_m);
                        }
                        if !cands.contains(&0) {
                            cands.push(0);
                        }
                    }
                    let best = *cands.iter().max_by(|a, b| u[**a].partial_cmp(&u[**b]).unwrap().then(b.cmp(a))).unwrap();
                    let pick = if c.mode == Mode::Rerank && u[best] - u[0] < c.gate { 0 } else { best };
                    (pick, u)
                }
            },
        };
        if let Ok(path) = std::env::var("KRL_ENDG_LOG") {
            // decision log (default off): inputs, utilities, pick -> python/endg/parity.py checks the Rust model against torch
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                let xs: Vec<String> = x.iter().map(|v| format!("{v}")).collect();
                let us: Vec<String> = util.iter().map(|v| format!("{v:.6}")).collect();
                let _ = writeln!(f, "{{\"player\":{player},\"pick\":{pick},\"u\":[{}],\"x\":[{}]}}", us.join(","), xs.join(","));
            }
        }
        self.seats.push((player, pick, x, util));
        pick
    }

    /// The knobs for this step: `k` with the chosen patch applied (unchanged before the decision, before the
    /// proposal's own start step, or when its patch is empty).
    pub fn patch(&self, player: i64, step: i64, k: &Knobs) -> Option<Knobs> {
        let p = self.chosen(player)?;
        let j = &self.cfg.patches[p];
        if j.obj().is_empty() || step < self.cfg.starts[p] {
            return None;
        }
        k.with(j).ok()
    }
}
