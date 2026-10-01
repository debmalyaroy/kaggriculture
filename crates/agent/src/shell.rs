//! Learned sales shell (kaggriculture-rl): per product and turn, hold / sell part / sell all.
//!
//! One feature function for training and play: `branch` (crates/runner) writes these exact vectors
//! for every decision it labels, python/top50/train_shell.py fits the model on them, and the agent
//! applies the model to the same vectors at play time. The model only overrides the chain's SELL
//! quantity for an item when it is at least `tau` confident and disagrees with the chain.
//!
//! Classes: 0 = hold (sell 0), 1 = sell part (half the shed stock, rounded up), 2 = sell all.
use crate::act::{Action, Cmd};
use crate::view::{View, PRODUCTS};
use kagg_engine::json::{self, Json};

/// Products the shell decides (FERTILIZER is left to the chain).
pub const SHELL_ITEMS: [&str; 8] = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL"];
pub const NF: usize = 12 + SHELL_ITEMS.len();
pub const FEATURES: [&str; 12] = ["day", "hour", "hours_left", "stock", "mkt_inv", "px", "px_rel", "dpx", "dinv", "money_gap", "shop_item", "stock_share"];

fn shop_items(shop: &str) -> &'static [&'static str] {
    match shop {
        "BAKERY" => &["EGG", "WHEAT"],
        "PIZZA_SHOP" => &["MILK", "TOMATO", "WHEAT"],
        "BRUNCH_SPOT" => &["EGG", "WHEAT", "STRAWBERRY"],
        "YARN_STORE" => &["WOOL"],
        "ICE_CREAM_SHOP" => &["STRAWBERRY", "MILK", "WHEAT"],
        "PET_CAFE" => &["CARROT"],
        "SMOOTHIE_SHOP" => &["STRAWBERRY", "MILK"],
        "FARMERS_MARKET" => &["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
        _ => &[],
    }
}

/// What the shell remembers between turns (from public observations only).
#[derive(Clone, Debug, Default)]
pub struct ShellMem {
    prev_inv: [f64; 9],
    prev_px: [f64; 9],
    px_sum: [f64; 9],
    n: f64,
}

impl ShellMem {
    /// Feature vector for SHELL_ITEMS[k] at this turn (uses memory up to the previous turn).
    pub fn features(&self, v: &View, k: usize) -> [f32; NF] {
        let item = SHELL_ITEMS[k];
        let p = PRODUCTS.iter().position(|x| *x == item).unwrap_or(0);
        let step = v.step.max(0) as f64;
        let inv = crate::obs::qget(&v.obs.mkt_inventory, item) as f64;
        let px = v.price(item) as f64;
        let mean = if self.n > 0.0 { self.px_sum[p] / self.n } else { px };
        let stock = v.shed(item) as f64;
        let total = v.shed_total().max(1) as f64;
        let shop = v.shops().iter().any(|s| shop_items(s).contains(&item));
        let first = self.n == 0.0;
        let mut x = [0f32; NF];
        let vals = [
            (step / 24.0).floor(),
            step % 24.0,
            720.0 - step,
            stock,
            inv,
            px,
            px / mean.max(1.0),
            if first { 0.0 } else { px - self.prev_px[p] },
            if first { 0.0 } else { inv - self.prev_inv[p] },
            v.money() - v.rival().money,
            shop as i32 as f64,
            stock / total,
        ];
        for (i, val) in vals.iter().enumerate() {
            x[i] = *val as f32;
        }
        x[12 + k] = 1.0;
        x
    }

    /// Record this turn's public market (call once per turn, after `features`).
    pub fn observe(&mut self, v: &View) {
        for (p, item) in PRODUCTS.iter().enumerate() {
            let px = v.price(item) as f64;
            self.prev_inv[p] = crate::obs::qget(&v.obs.mkt_inventory, item) as f64;
            self.prev_px[p] = px;
            self.px_sum[p] += px;
        }
        self.n += 1.0;
    }
}

/// Total SELL quantity for `item` in an action's market list.
pub fn sell_qty(a: &Action, item: &str) -> i64 {
    a.market.iter().filter(|c| c.is_sell3() && c.s(1) == item).map(|c| c.n(2).max(0)).sum()
}

/// The class an action plays for `item` given the shed stock.
pub fn class_of(a: &Action, item: &str, stock: i64) -> usize {
    let q = sell_qty(a, item);
    if q <= 0 {
        0
    } else if q >= stock {
        2
    } else {
        1
    }
}

pub fn qty_of(class: usize, stock: i64) -> i64 {
    match class {
        0 => 0,
        1 => (stock + 1) / 2,
        _ => stock,
    }
}

/// Rewrite `a` so its SELL quantity for `item` is `q` (first SELL order of the item carries it, later
/// ones go to 0 so their positional slots stay; a new order is appended only when there is room).
pub fn set_sell(a: &mut Action, item: &str, q: i64) {
    let mut done = false;
    for c in a.market.iter_mut() {
        if c.is_sell3() && c.s(1) == item {
            c.set_n(2, if done { 0 } else { q });
            done = true;
        }
    }
    if !done && q > 0 && a.market.len() < 10 {
        a.market.push(Cmd::order("SELL", item, q));
    }
}

#[derive(Clone, Debug)]
struct Dense {
    w: Vec<Vec<f32>>, // out x in
    b: Vec<f32>,
}

/// A small MLP over the normalized features (ReLU hidden layers, 3 logits).
#[derive(Clone, Debug)]
pub struct ShellModel {
    mean: Vec<f32>,
    std: Vec<f32>,
    layers: Vec<Dense>,
    pub tau: f32,
    /// 0 = override either way, 1 = only sell MORE than the chain, 2 = only sell LESS.
    pub mode: u8,
    /// Opponent groups the shell acts against (0 DIFFERENT, 1 PARTIAL, 2 COPY; layers/group.rs, fixed at
    /// step 25); empty = every group. Before the group is known the shell does not act.
    pub groups: Vec<usize>,
}

impl ShellModel {
    pub fn load(path: &str) -> Result<ShellModel, String> {
        let j = json::parse(&std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?)?;
        let vf = |x: &Json| x.arr().iter().map(|v| v.f64() as f32).collect::<Vec<f32>>();
        let m = ShellModel {
            mean: vf(j.get("mean")),
            std: vf(j.get("std")),
            layers: j.get("layers").arr().iter().map(|l| Dense { w: l.get("w").arr().iter().map(vf).collect(), b: vf(l.get("b")) }).collect(),
            tau: if j.get("tau").is_null() { 0.6 } else { j.get("tau").f64() as f32 },
            mode: if j.get("mode").is_null() { 0 } else { j.get("mode").i64().clamp(0, 2) as u8 },
            groups: j.get("groups").arr().iter().map(|g| g.i64().clamp(0, 2) as usize).collect(),
        };
        if m.mean.len() != NF || m.std.len() != NF || m.layers.is_empty() || m.layers.last().map(|l| l.b.len()) != Some(3) {
            return Err(format!("{path}: shell model shape (want {NF} features, 3 outputs)"));
        }
        Ok(m)
    }

    pub fn probs(&self, x: &[f32; NF]) -> [f32; 3] {
        let mut h: Vec<f32> = x.iter().zip(&self.mean).zip(&self.std).map(|((v, m), s)| (v - m) / s.max(1e-6)).collect();
        for (i, l) in self.layers.iter().enumerate() {
            let mut o: Vec<f32> = l.w.iter().zip(&l.b).map(|(row, b)| row.iter().zip(&h).map(|(w, v)| w * v).sum::<f32>() + b).collect();
            if i + 1 < self.layers.len() {
                o.iter_mut().for_each(|v| *v = v.max(0.0));
            }
            h = o;
        }
        let mx = h.iter().cloned().fold(f32::MIN, f32::max);
        let e: Vec<f32> = h.iter().map(|v| (v - mx).exp()).collect();
        let s: f32 = e.iter().sum();
        [e[0] / s, e[1] / s, e[2] / s]
    }
}

/// The shell as the agent's last market step.
#[derive(Clone, Debug)]
pub struct ShellCtl {
    pub model: std::sync::Arc<ShellModel>,
    pub mem: ShellMem,
    /// Overrides made this game (diagnostics).
    pub overrides: usize,
}

impl ShellCtl {
    pub fn new(model: std::sync::Arc<ShellModel>) -> ShellCtl {
        ShellCtl { model, mem: ShellMem::default(), overrides: 0 }
    }

    pub fn apply(&mut self, a: &mut Action, v: &View, group: Option<usize>) {
        let group_ok = self.model.groups.is_empty() || group.is_some_and(|g| self.model.groups.contains(&g));
        if v.step < crate::view::LAST_ACT_STEP && group_ok {
            for (k, item) in SHELL_ITEMS.iter().enumerate() {
                let stock = v.shed(item);
                if stock <= 0 {
                    continue;
                }
                let p = self.model.probs(&self.mem.features(v, k));
                let (best, pb) = p.iter().enumerate().fold((0, 0f32), |acc, (i, &x)| if x > acc.1 { (i, x) } else { acc });
                let cur = class_of(a, item, stock);
                let dir_ok = match self.model.mode {
                    1 => best > cur,
                    2 => best < cur,
                    _ => true,
                };
                if pb >= self.model.tau && best != cur && dir_ok {
                    set_sell(a, item, qty_of(best, stock));
                    self.overrides += 1;
                }
            }
        }
        self.mem.observe(v);
    }
}
