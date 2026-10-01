//! Reactive shell v2 (kaggriculture-rl): the market-race ideas of the v61.1 rule shell (RACE, R36/R37,
//! AFR, v92, OR2, glut gates, lead-sell windows, tsell) as ONE learned, config-driven sale controller
//! that sits after the chain, for the PPO-driven agent. Design: docs/rshell-v2.md.
//!
//! Per turn it builds GLOBAL inputs (time, money, opponent identity: group, cluster, clone gate,
//! position streaks, layout similarity, step-1 mirror, LINEAGE match against our own route library,
//! race state, today's profile, shed room) and per-ITEM inputs for 9 products (8 + FERTILIZER: stock
//! after this turn's unit actions, market, town demand, the rival's recovered sales, its visible
//! production, pre-emptions, rival-sale forecasts from three models: copy = our own plan, v92,
//! lineage = the matched route's tape; our plan, the chain's order and debts, and the PRICE
//! CALCULATOR: exact engine price curve, inventory simulated over the horizon, revenue of each sale
//! fraction now + the rest at the best later step).
//!
//! Decision per item: a sale fraction of the projected shed (`fracs`: 0, 1/4, 1/2, 3/4, all) and a
//! priority (slot order among our SELLs). Score per class
//!     logit[c] = model_w * net(x)[c] + bias[c] + stay * [c == chain's class]
//!              + beta_price * (rev[c] - max rev) / scale + urge * (2 frac[c] - 1)
//!     urge     = sum_j urge_w[j] * signal_j
//! The net (optional) is a shared per-item encoder + mean/max pooled cross-item context + heads for
//! the 5 class logits, a priority score and the 5 predicted margins (a confidence gate).
//!
//! HARD LIMITS (never learned): only SELL quantities and their order change (never units, BUYs or what a
//! later tape step picks up); quantities are clamped to the projected shed; FERTILIZER is never sold
//! below its committed need (fertilizer tour + V219 + the tape's future pickups/applications); the 10-
//! order cap holds; off from `to` (<= 717) and before `from`. With no config: byte-identical chain.
use crate::act::{Action, Cmd};
use crate::chassis::{base_price, Chassis};
use crate::obs::{qadd, qget, Qty};
use crate::shell::{sell_qty, set_sell};
use crate::view::View;
use kagg_engine::json::{self, Json};

pub const ITEMS: [&str; 9] = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"];
pub const NI: usize = ITEMS.len();
pub const FERT: usize = 8;
pub const NC: usize = 5;
pub const GLOBAL: [&str; 35] = [
    "day", "hour", "hours_left", "hour23", "to_term", "money_gap", "our_cash_24", "rival_cash_24", "g_diff", "g_partial",
    "g_copy", "g_known", "cluster_m", "clone_gate", "poseq_6", "poseq_24", "similarity", "mirror", "lin_sim", "lin_same",
    "lin_streak", "lin_known", "lin_fit", "race_h", "r37_h", "race_lost", "afr_24", "v92_on", "k_race_clone", "k_afr_on", "k_rsa_look",
    "room", "carried", "overflow", "profile",
];
pub const ITEMF: [&str; 52] = [
    "stock", "stock_raw", "in_hands", "share", "inv", "px", "px_base", "px_mean", "dpx", "dinv", "glut", "shop_item",
    "town_rate", "to_draw", "rival_sold_1", "rival_sold_24", "rival_ticks_24", "since_rival", "rival_tiles", "rival_ready",
    "preempt_24", "fc_v92", "fc_lin_1", "fc_lin_4", "fc_lin_8", "fc_lin_24", "plan_1", "plan_3", "plan_8", "plan_24",
    "to_plan", "cur_qty", "cur_slot", "debt", "rev_0", "rev_1", "rev_2", "rev_3", "rev_4", "best_wait", "pf_1", "pf_4",
    "pf_8", "pf_24", "impact", "need", "surplus", "cur_cls", "chain_sells", "px_now_frac", "arrived", "rival_first_48",
];
pub const NG: usize = GLOBAL.len();
pub const NIF: usize = ITEMF.len();
/// Full per-item model input: item features + global features + item one-hot.
pub const NX: usize = NIF + NG + NI;

/// Signals the urgency score weights (each roughly unit scale).
pub const SIGNALS: [&str; 15] = [
    "clone", "copy", "lineage", "preempt", "rival_sold_1", "fc_lineage", "fc_v92", "glut", "late", "money_behind", "room_tight", "partial",
    // RCA 2026-09-27 (public-25 losses): stock that just arrived (the field sells milk on arrival), the rival selling
    // this product before us lately, and the final two days' race
    "arrived", "rival_first", "final_days",
];

fn gi(n: &str) -> usize {
    GLOBAL.iter().position(|f| *f == n).unwrap()
}
fn ii(n: &str) -> usize {
    ITEMF.iter().position(|f| *f == n).unwrap()
}

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

/// Town demand drawn from the market between `t0` and `t0 + 1` (4-step shop draws, day-start draw).
fn town_draw(t0: i64, shops: &[&'static str], item: &str) -> i64 {
    let mut n = 0;
    if t0 % 4 == 0 {
        for s in shops {
            let it = shop_items(s);
            if it.contains(&item) {
                n += if it.len() == 1 { 2 } else { 1 };
            }
        }
    }
    if t0 % 24 == 0 && item != "FERTILIZER" {
        n += 1;
    }
    n
}

fn product_of(t: &crate::obs::Tile) -> Option<&'static str> {
    if !t.is_dict() {
        return None;
    }
    if t.kind == "PLANT" && !t.crop.is_empty() {
        return ITEMS.iter().copied().find(|i| *i == t.crop);
    }
    match t.animal {
        "GOOSE" => Some("EGG"),
        "COW" => Some("MILK"),
        "SHEEP" => Some("WOOL"),
        _ => None,
    }
}

// ---------------------------------------------------------------------------------------------
// the network

#[derive(Clone, Debug)]
struct Dense {
    w: Vec<Vec<f32>>,
    b: Vec<f32>,
}

fn mlp(layers: &[Dense], x: &[f32], relu_last: bool) -> Vec<f32> {
    let mut h = x.to_vec();
    for (i, l) in layers.iter().enumerate() {
        let mut o: Vec<f32> = l.w.iter().zip(&l.b).map(|(row, b)| row.iter().zip(&h).map(|(w, v)| w * v).sum::<f32>() + b).collect();
        if i + 1 < layers.len() || relu_last {
            o.iter_mut().for_each(|v| *v = v.max(0.0));
        }
        h = o;
    }
    h
}

/// Shared per-item encoder -> [e_i, mean_j e_j, max_j e_j] -> head -> 5 logits, priority, 5 margins.
#[derive(Clone, Debug)]
pub struct Net {
    mean: Vec<f32>,
    std: Vec<f32>,
    enc: Vec<Dense>,
    head: Vec<Dense>,
}

pub struct NetOut {
    pub logits: [f32; NC],
    pub prio: f32,
    pub margin: [f32; NC],
}

impl Net {
    fn from_json(j: &Json) -> Result<Net, String> {
        let vf = |x: &Json| x.arr().iter().map(|v| v.f64() as f32).collect::<Vec<f32>>();
        let dl = |x: &Json| x.arr().iter().map(|l| Dense { w: l.get("w").arr().iter().map(vf).collect(), b: vf(l.get("b")) }).collect::<Vec<Dense>>();
        let n = Net { mean: vf(j.get("mean")), std: vf(j.get("std")), enc: dl(j.get("enc")), head: dl(j.get("head")) };
        let e = n.enc.last().map(|l| l.b.len()).unwrap_or(0);
        if n.mean.len() != NX || n.std.len() != NX || e == 0 || n.head.first().map(|l| l.w.first().map(|r| r.len())) != Some(Some(3 * e)) || n.head.last().map(|l| l.b.len()) != Some(2 * NC + 1) {
            return Err(format!("rshell net shape: want {NX} inputs, encoder E, head 3E -> {}", 2 * NC + 1));
        }
        Ok(n)
    }

    pub fn run(&self, xs: &[Option<[f32; NX]>]) -> Vec<Option<NetOut>> {
        let enc: Vec<Option<Vec<f32>>> = xs
            .iter()
            .map(|x| x.as_ref().map(|x| mlp(&self.enc, &x.iter().zip(&self.mean).zip(&self.std).map(|((v, m), s)| (v - m) / s.max(1e-6)).collect::<Vec<f32>>(), true)))
            .collect();
        let live: Vec<&Vec<f32>> = enc.iter().flatten().collect();
        if live.is_empty() {
            return xs.iter().map(|_| None).collect();
        }
        let e = live[0].len();
        let mut mean = vec![0f32; e];
        let mut max = vec![f32::MIN; e];
        for v in &live {
            for i in 0..e {
                mean[i] += v[i] / live.len() as f32;
                max[i] = max[i].max(v[i]);
            }
        }
        enc.iter()
            .map(|ei| {
                ei.as_ref().map(|ei| {
                    let mut h = ei.clone();
                    h.extend_from_slice(&mean);
                    h.extend_from_slice(&max);
                    let o = mlp(&self.head, &h, false);
                    let mut l = [0f32; NC];
                    let mut m = [0f32; NC];
                    l.copy_from_slice(&o[..NC]);
                    m.copy_from_slice(&o[NC + 1..2 * NC + 1]);
                    NetOut { logits: l, prio: o[NC], margin: m }
                })
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------------------------
// configuration: every tunable

#[derive(Clone, Debug)]
pub struct RConfig {
    /// false = track and expose inputs only (labelling), never change the action.
    pub on: bool,
    pub net: Option<Net>,
    pub model_w: f32,
    pub fracs: [f32; NC],
    pub tau: f32,
    pub tau_item: [f32; NI],
    pub bias: [f32; NC],
    pub stay: f32,
    pub beta_price: f32,
    pub urge_w: [f32; SIGNALS.len()],
    /// Per-item urgency bias (e.g. milk sells on arrival, wool holds), added to the urgency score.
    pub item_urge: [f32; NI],
    /// Per opponent group (DIFFERENT, PARTIAL, COPY): replaces `item_urge` / `urge_w` against that group (None = the
    /// global ones). RCA 29 Sep: v63.10's tuned urges hold strawberry / wheat too long against chassis copies (a copy
    /// dumps one step earlier and takes the peak) but are right against top players, so the fix is per group.
    pub group_item_urge: [Option<[f32; NI]>; 3],
    pub group_urge_w: [Option<[f32; SIGNALS.len()]>; 3],
    /// Per opponent group: a complete replacement config used against that group (`"group_cfg": {"2": "file.json"}`,
    /// path relative to this config). 29 Sep: v63.7's shell settings beat v63.8+'s against chassis copies (COPY), and
    /// the ladder fell 2110 -> 1874 when v63.8 replaced them for every group.
    pub group_cfg: [Option<std::sync::Arc<RConfig>>; 3],
    /// Per opponent group (DIFFERENT, PARTIAL, COPY): acts, tau offset, direction (0 either, 1 more, 2 less).
    pub group_on: [bool; 3],
    pub group_tau: [f32; 3],
    pub group_mode: [u8; 3],
    pub pre_group: bool,
    pub from: i64,
    pub to: i64,
    pub skip_hour23: bool,
    /// Never sell MORE than the chain below this fraction of the base price.
    pub glut_floor: f32,
    /// Predicted-margin gate: override only when net margin[best] - margin[chain] >= this (needs a net).
    pub margin_gate: f32,
    /// Reorder our SELL slots by priority (net priority + prio_urge * urge).
    pub reorder: bool,
    pub prio_urge: f32,
    /// Urgency above which the item's SELLs move ahead of every other order (large = never).
    pub front: f32,
    pub item_on: [bool; NI],
    /// Price calculator: horizon and the rival-forecast mix (copy, v92, lineage, recent rate).
    pub horizon: i64,
    pub fc_w: [f32; 4],
    /// Lineage: route layouts per day (from the lineage-table bin) and the match threshold.
    pub lineage: Option<std::sync::Arc<Lineage>>,
    pub lin_min: f32,
    /// Fertilizer: extra units kept on top of the committed need.
    pub fert_reserve: i64,
    /// Shell v3 buy head (BUY_PRODUCT WHEAT): predicted margin gain per quantity; off without a net.
    pub buy_net: Option<BuyNet>,
    pub buy_gate: f32,
    pub buy_from: i64,
    pub buy_to: i64,
    pub buy_cash: f32,
    /// at most this many buys per game day (labels value ONE buy at a sampled step; firing every turn drained
    /// the cash: preview test 27 Sep lineage +35/-582)
    pub buy_per_day: i64,
}

/// Buy head: a small MLP over the WHEAT item input (NX) -> predicted final-margin gain of buying each of
/// `qs` units now vs not buying (exact counterfactual labels from the buylabel runner).
#[derive(Clone, Debug)]
pub struct BuyNet {
    pub qs: Vec<i64>,
    mean: Vec<f32>,
    std: Vec<f32>,
    layers: Vec<(Vec<Vec<f32>>, Vec<f32>)>,
}

impl BuyNet {
    pub fn from_json(j: &Json) -> Result<BuyNet, String> {
        let v = |x: &Json| x.arr().iter().map(|y| y.f64() as f32).collect::<Vec<f32>>();
        let layers = j.get("layers").arr().iter().map(|l| (l.get("w").arr().iter().map(|r| v(r)).collect(), v(l.get("b")))).collect::<Vec<_>>();
        let n = BuyNet { qs: j.get("qs").arr().iter().map(|x| x.i64()).collect(), mean: v(j.get("mean")), std: v(j.get("std")), layers };
        if n.mean.len() != NX || n.std.len() != NX {
            return Err(format!("buy net: expected {NX} inputs, got {}", n.mean.len()));
        }
        Ok(n)
    }
    pub fn run(&self, x: &[f32; NX]) -> Vec<f32> {
        let mut h: Vec<f32> = x.iter().zip(&self.mean).zip(&self.std).map(|((a, m), s)| (a - m) / s.max(1e-6)).collect();
        let nl = self.layers.len();
        for (i, (w, b)) in self.layers.iter().enumerate() {
            let mut o: Vec<f32> = w.iter().zip(b).map(|(row, bb)| row.iter().zip(&h).map(|(a, c)| a * c).sum::<f32>() + bb).collect();
            if i + 1 < nl {
                o.iter_mut().for_each(|z| *z = z.max(0.0));
            }
            h = o;
        }
        h
    }
}

impl Default for RConfig {
    fn default() -> Self {
        RConfig {
            on: true,
            net: None,
            model_w: 1.0,
            fracs: [0.0, 0.25, 0.5, 0.75, 1.0],
            tau: 0.6,
            tau_item: [0.0; NI],
            bias: [0.0; NC],
            stay: 2.0,
            beta_price: 0.0,
            urge_w: [0.0; SIGNALS.len()],
            item_urge: [0.0; NI],
            group_item_urge: [None; 3],
            group_urge_w: [None; 3],
            group_cfg: [None, None, None],
            group_on: [true; 3],
            group_tau: [0.0; 3],
            group_mode: [0; 3],
            pre_group: false,
            from: 25,
            to: 717,
            skip_hour23: false,
            glut_floor: 0.0,
            margin_gate: f32::MIN,
            reorder: false,
            prio_urge: 0.0,
            front: 99.0,
            item_on: [true; NI],
            horizon: 48,
            fc_w: [0.0, 1.0, 1.0, 0.0],
            lineage: None,
            lin_min: 0.9,
            fert_reserve: 0,
            buy_net: None,
            buy_gate: 150.0,
            buy_from: 144,
            buy_to: 700,
            buy_cash: 0.5,
            buy_per_day: 1,
        }
    }
}

fn farr<const N: usize>(j: &Json, dflt: [f32; N]) -> Result<[f32; N], String> {
    if j.is_null() {
        return Ok(dflt);
    }
    let v = j.arr();
    if v.len() != N {
        return Err(format!("expected {N} values, got {}", v.len()));
    }
    let mut o = dflt;
    for (i, x) in v.iter().enumerate() {
        o[i] = x.f64() as f32;
    }
    Ok(o)
}

impl RConfig {
    pub fn load(path: &str) -> Result<RConfig, String> {
        let j = json::parse(&std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?)?;
        let dir = std::path::Path::new(path).parent().map(|p| p.to_path_buf()).unwrap_or_default();
        RConfig::from_json(&j, &dir)
    }

    pub fn from_json(j: &Json, dir: &std::path::Path) -> Result<RConfig, String> {
        let mut c = RConfig::default();
        let rd = |f: &str| -> Result<Json, String> { json::parse(&std::fs::read_to_string(dir.join(f)).map_err(|e| format!("{f}: {e}"))?) };
        for (k, v) in j.obj() {
            match k.as_str() {
                "name" | "note" | "features" | "signals" | "fitness" | "trained" | "cma" => {}
                "on" => c.on = v.bool(),
                "net" => c.net = if v.is_null() { None } else { Some(Net::from_json(v)?) },
                "net_file" => c.net = if v.is_null() { None } else { Some(Net::from_json(&rd(v.str())?)?) },
                "model_w" => c.model_w = v.f64() as f32,
                "fracs" => c.fracs = farr(v, c.fracs)?,
                "tau" => c.tau = v.f64() as f32,
                "tau_item" => c.tau_item = farr(v, c.tau_item)?,
                "bias" => c.bias = farr(v, c.bias)?,
                "stay" => c.stay = v.f64() as f32,
                "beta_price" => c.beta_price = v.f64() as f32,
                "urge_w" => {
                    // configs from before the 3 RCA signals carry 12 weights: pad with zeros
                    let arr = v.arr();
                    let mut w = [0f32; SIGNALS.len()];
                    if arr.len() > SIGNALS.len() {
                        return Err(format!("urge_w: at most {} values", SIGNALS.len()));
                    }
                    for (i, x) in arr.iter().enumerate() {
                        w[i] = x.f64() as f32;
                    }
                    c.urge_w = w;
                }
                "item_urge" => c.item_urge = farr(v, c.item_urge)?,
                "group_urge" => {
                    // {"2": {"item_urge": [9], "urge_w": [15]}, ...} keyed by group index
                    for (gk, gv) in v.obj() {
                        let g: usize = gk.parse().map_err(|_| format!("group_urge: bad group {gk}"))?;
                        if g > 2 {
                            return Err(format!("group_urge: group {g} > 2"));
                        }
                        if !gv.get("item_urge").is_null() {
                            c.group_item_urge[g] = Some(farr(gv.get("item_urge"), [0.0; NI])?);
                        }
                        if !gv.get("urge_w").is_null() {
                            c.group_urge_w[g] = Some(farr(gv.get("urge_w"), [0.0; SIGNALS.len()])?);
                        }
                    }
                }
                "group_on" => {
                    let g = farr(v, [1.0; 3])?;
                    c.group_on = [g[0] > 0.5, g[1] > 0.5, g[2] > 0.5];
                }
                "group_tau" => c.group_tau = farr(v, c.group_tau)?,
                "group_mode" => {
                    let g = farr(v, [0.0; 3])?;
                    c.group_mode = g.map(|x| x.round().clamp(0.0, 2.0) as u8);
                }
                "pre_group" => c.pre_group = v.bool(),
                "from" => c.from = v.i64(),
                "to" => c.to = v.i64().min(crate::view::LAST_ACT_STEP - 1),
                "skip_hour23" => c.skip_hour23 = v.bool(),
                "glut_floor" => c.glut_floor = v.f64() as f32,
                "margin_gate" => c.margin_gate = v.f64() as f32,
                "reorder" => c.reorder = v.bool(),
                "prio_urge" => c.prio_urge = v.f64() as f32,
                "front" => c.front = v.f64() as f32,
                "item_on" => {
                    let g = farr(v, [1.0; NI])?;
                    c.item_on = g.map(|x| x > 0.5);
                }
                "horizon" => c.horizon = v.i64().clamp(1, 96),
                "fc_w" => c.fc_w = farr(v, c.fc_w)?,
                "lineage_file" => c.lineage = if v.is_null() { None } else { Some(std::sync::Arc::new(Lineage::from_json(&rd(v.str())?)?)) },
                "lin_min" => c.lin_min = v.f64() as f32,
                "fert_reserve" => c.fert_reserve = v.i64().max(0),
                "buy_net_file" => c.buy_net = if v.is_null() { None } else { Some(BuyNet::from_json(&rd(v.str())?)?) },
                "buy_gate" => c.buy_gate = v.f64() as f32,
                "buy_from" => c.buy_from = v.i64(),
                "buy_to" => c.buy_to = v.i64(),
                "buy_cash" => c.buy_cash = v.f64() as f32,
                "buy_per_day" => c.buy_per_day = v.i64().max(0),
                "group_cfg" => {
                    for (gk, gv) in v.obj() {
                        let g: usize = gk.parse().map_err(|_| format!("group_cfg: bad group {gk}"))?;
                        if g > 2 {
                            return Err(format!("group_cfg: group {g} > 2"));
                        }
                        c.group_cfg[g] = Some(std::sync::Arc::new(RConfig::load(&dir.join(gv.str()).to_string_lossy())?));
                    }
                }
                other => return Err(format!("unknown rshell key {other:?}")),
            }
        }
        c.fracs.iter_mut().for_each(|x| *x = x.clamp(0.0, 1.0));
        Ok(c)
    }
}

// ---------------------------------------------------------------------------------------------
// lineage: our own route library's farm layouts per day

/// `routes[k] = (route id, per day: sorted (tile index, product code))` from the lineage-table bin.
#[derive(Clone, Debug, Default)]
pub struct Lineage {
    pub routes: Vec<(i64, Vec<Vec<(u16, u8)>>)>,
}

fn code(t: &crate::obs::Tile) -> u8 {
    match product_of(t) {
        Some(p) => 1 + ITEMS.iter().position(|x| *x == p).unwrap_or(0) as u8 + if t.animal.is_empty() { 0 } else { 16 },
        None => 0,
    }
}

/// A farm's layout: occupied tiles with a product code.
pub fn layout(f: &crate::obs::FarmObs) -> Vec<(u16, u8)> {
    f.tiles.iter().enumerate().filter_map(|(i, t)| {
        let c = code(t);
        (c > 0).then_some((i as u16, c))
    }).collect()
}

fn lsim(a: &[(u16, u8)], b: &[(u16, u8)]) -> f64 {
    let (mut i, mut j, mut m, mut tot) = (0, 0, 0usize, 0usize);
    while i < a.len() || j < b.len() {
        if j >= b.len() || (i < a.len() && a[i].0 < b[j].0) {
            tot += 1;
            i += 1;
        } else if i >= a.len() || b[j].0 < a[i].0 {
            tot += 1;
            j += 1;
        } else {
            tot += 1;
            m += (a[i].1 == b[j].1) as usize;
            i += 1;
            j += 1;
        }
    }
    if tot >= 8 { m as f64 / tot as f64 } else { 0.0 }
}

impl Lineage {
    pub fn from_json(j: &Json) -> Result<Lineage, String> {
        let mut routes = vec![];
        for (k, days) in j.get("routes").obj() {
            let id: i64 = k.parse().map_err(|_| format!("lineage route id {k:?}"))?;
            let d: Vec<Vec<(u16, u8)>> = days.arr().iter().map(|day| day.arr().iter().map(|p| {
                let a = p.arr();
                (a[0].i64() as u16, a[1].i64() as u8)
            }).collect()).collect();
            routes.push((id, d));
        }
        Ok(Lineage { routes })
    }

    /// Layout similarity of the rival to every route on `day` (best of days day-1..=day).
    pub fn sims(&self, rival: &[(u16, u8)], day: i64) -> Vec<(i64, f64)> {
        self.routes
            .iter()
            .map(|(id, days)| {
                let s = [day - 1, day].iter().filter(|d| **d >= 0 && (**d as usize) < days.len()).map(|d| lsim(rival, &days[*d as usize])).fold(0.0, f64::max);
                (*id, s)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------------------------
// what the chain knows

#[derive(Clone, Debug, Default)]
pub struct ChainSig {
    pub race_h: i64,
    pub r37_h: i64,
    pub race_lost: bool,
    pub clone: bool,
    pub similarity: f64,
    pub group: Option<usize>,
    pub afr_24: usize,
    pub v92: Option<Qty>,
    pub route: Option<i64>,
    pub k_race_clone: i64,
    pub k_v92_on: bool,
    pub k_afr_on: bool,
    pub k_rsa_look: i64,
    pub term_start: i64,
    pub profile: usize,
    /// Fertilizer committed to projects (fertilizer tour undelivered + V219 dedicated).
    pub fert_committed: i64,
    /// R36 debts / suppressions still due (item -> units).
    pub debts: Qty,
}

fn tape_sells(ch: &Chassis, route: Option<i64>, item: &str, from: i64, to: i64) -> i64 {
    let Some(route) = route else { return 0 };
    let mut n = 0;
    for t in from.max(0)..to.min(719) {
        let r = if t >= 648 { 2 } else { route };
        if ch.route_idx(r).is_none() {
            continue;
        }
        if let Some(a) = ch.route(r).tape.get(t as usize) {
            n += a.market.iter().filter(|o| o.is_sell3() && o.s(1) == item).map(|o| o.n(2).max(0)).sum::<i64>();
        }
    }
    n
}

fn first_sell(ch: &Chassis, route: Option<i64>, item: &str, from: i64, cap: i64) -> i64 {
    for d in 0..cap {
        if tape_sells(ch, route, item, from + d, from + d + 1) > 0 {
            return d;
        }
    }
    cap
}

/// Fertilizer the tape still needs from the shed: future PICKUP FERTILIZER and FERTILIZE uses.
fn tape_fert_need(ch: &Chassis, route: Option<i64>, from: i64) -> i64 {
    let Some(route) = route else { return 0 };
    let mut n = 0;
    for t in from.max(0)..719 {
        let r = if t >= 648 { 2 } else { route };
        if ch.route_idx(r).is_none() {
            continue;
        }
        if let Some(a) = ch.route(r).tape.get(t as usize) {
            for u in a.units() {
                if u.is_empty() {
                    continue;
                }
                if u.op() == "PICKUP" && u.len() > 1 && u.s(1) == "FERTILIZER" {
                    n += if u.len() > 2 { u.n(2).max(1) } else { 1 };
                }
            }
        }
    }
    n
}

// ---------------------------------------------------------------------------------------------
// memory

#[derive(Clone, Debug, Default)]
struct Prev {
    step: i64,
    inv: Qty,
    shops: Vec<&'static str>,
    our_sold: Qty,
    shed: Qty,
    ahead: [bool; NI],
}

#[derive(Clone, Debug, Default)]
pub struct RMem {
    prev: Option<Prev>,
    px_hist: Vec<[f64; NI]>,
    prev_px: [f64; NI],
    prev_inv: [f64; NI],
    rival: Vec<(i64, [i64; NI])>,
    /// Our executed sales per step (units, clamped to the shed), aligned with `rival` (recovered one step late).
    ours: Vec<(i64, [i64; NI])>,
    preempt: Vec<(i64, usize)>,
    mirror: bool,
    poseq: Vec<bool>,
    day5_eq: (usize, usize),
    money: Vec<(f64, f64)>,
    lin: Option<(i64, f64)>,
    lin_fit: f64,
    lin_streak: i64,
    lin_day: i64,
}

impl RMem {
    fn observe_start(&mut self, v: &View) {
        let step = v.step;
        if step == 1 {
            self.mirror = (v.rival().money - v.farm().money).abs() < 0.5;
        }
        let (own, rival) = (v.farm(), v.rival());
        let eq = own.farmer == rival.farmer && own.hands == rival.hands;
        self.poseq.push(eq);
        if (120..144).contains(&step) {
            self.day5_eq.0 += eq as usize;
            self.day5_eq.1 += 1;
        }
        self.money.push((own.money, rival.money));
        if self.money.len() > 25 {
            self.money.remove(0);
        }
        // recover the rival's sales of the last step
        if let Some(p) = self.prev.as_ref().filter(|p| p.step + 1 == step) {
            let mut row = [0i64; NI];
            for (k, item) in ITEMS.iter().enumerate() {
                let inv_before = qget(&p.inv, item);
                let ours = qget(&p.our_sold, item);
                let draw = town_draw(p.step, &p.shops, item).min(inv_before + ours);
                row[k] = (qget(&v.obs.mkt_inventory, item) - inv_before + draw - ours).max(0);
                if row[k] > 0 && qget(&p.shed, item) > 0 && p.ahead[k] {
                    self.preempt.push((step, k));
                }
            }
            self.rival.push((step, row));
            if self.rival.len() > 240 {
                self.rival.remove(0);
            }
        }
    }

    /// Once a day: routes whose layout matches the rival's (>= lin_min, else the best layout) are
    /// ranked by how well their tape's sale steps explain the rival's recovered sales of the last 240
    /// steps (a hit = the route sells that item within +-1 step; a route sale with no rival sale near
    /// it = a false alarm, half weight). Layouts alone do not separate our routes (median 0.93).
    fn lineage(&mut self, v: &View, ch: &Chassis, lin: Option<&Lineage>, lin_min: f64) {
        let Some(l) = lin else { return };
        let day = v.step / 24;
        if v.step % 24 != 12 || day == self.lin_day {
            return;
        }
        self.lin_day = day;
        let sims = l.sims(&layout(v.rival()), day);
        let top = sims.iter().map(|x| x.1).fold(0.0, f64::max);
        let cands: Vec<(i64, f64)> = sims.into_iter().filter(|x| x.1 >= lin_min.min(top)).collect();
        let step = v.step;
        let sold: Vec<(i64, usize)> = self.rival.iter().flat_map(|(t, r)| r.iter().enumerate().filter(|(_, n)| **n > 0).map(move |(k, _)| (*t, k))).collect();
        let base = step - 242;
        let mut seen = vec![[false; NI]; 245];
        for (t, k) in &sold {
            if *t >= base && ((t - base) as usize) < seen.len() {
                seen[(t - base) as usize][*k] = true;
            }
        }
        let mut best: Option<(i64, f64, f64)> = None;
        for (id, ls) in cands {
            if ch.route_idx(id).is_none() {
                continue;
            }
            let sells = |t: i64, k: usize| -> bool {
                let r = if t >= 648 { 2 } else { id };
                ch.route_idx(r).is_some() && ch.route(r).tape.get(t.max(0) as usize).is_some_and(|a| a.market.iter().any(|o| o.is_sell3() && o.s(1) == ITEMS[k] && o.n(2) > 0))
            };
            let hits = sold.iter().filter(|(t, k)| (t - 2..=*t).any(|u| sells(u, *k))).count() as f64;
            let mut false_alarm = 0.0;
            for t in (step - 240).max(1)..step - 1 {
                for k in 0..NI {
                    if sells(t, k) && !(t..=t + 2).any(|u| u >= base && ((u - base) as usize) < seen.len() && seen[(u - base) as usize][k]) {
                        false_alarm += 1.0;
                    }
                }
            }
            let fit = if sold.is_empty() { 0.0 } else { (hits - 0.5 * false_alarm).max(0.0) / sold.len() as f64 };
            if best.is_none_or(|b| fit > b.2 || (fit == b.2 && ls > b.1)) {
                best = Some((id, ls, fit));
            }
        }
        self.lin_fit = best.map(|b| b.2).unwrap_or(0.0);
        let b = best.map(|b| (b.0, b.1));
        match (b, self.lin) {
            (Some(n), Some(o)) if n.0 == o.0 => self.lin_streak += 1,
            _ => self.lin_streak = 0,
        }
        self.lin = b;
    }
}

// ---------------------------------------------------------------------------------------------
// the price calculator

/// Revenue of selling `q` units starting at market inventory `x` (the engine sells unit by unit).
struct Curve {
    lo: i64,
    cum: Vec<f64>,
}

impl Curve {
    fn new(item: &str, lo: i64, hi: i64) -> Curve {
        let lo = lo.max(0);
        let idx = crate::market::item_index(item);
        let mut cum = vec![0.0];
        for x in lo..=hi.max(lo) {
            let p = crate::market::price_i(idx, item, x) as f64;
            cum.push(cum.last().unwrap() + p);
        }
        Curve { lo, cum }
    }
    fn rev(&self, x: i64, q: i64) -> f64 {
        if q <= 0 {
            return 0.0;
        }
        let a = (x.max(self.lo) - self.lo) as usize;
        let b = (a + q as usize).min(self.cum.len() - 1);
        self.cum[b] - self.cum[a.min(self.cum.len() - 1)]
    }
}

pub struct Priced {
    pub rev: [f64; NC],
    pub best_wait: i64,
    pub pf: [f64; 4],
}

/// For each fraction: sell q now, the rest at the best later step within the horizon (inventory moved
/// by the rival forecast and the town draw; anything past step 718 is worth nothing).
fn price_calc(item: &str, stock: i64, inv0: i64, step: i64, h: i64, shops: &[&'static str], rival_fc: &[f64], fracs: &[f32; NC]) -> Priced {
    let h = h.min(718 - step).max(0);
    let mut path = vec![inv0 as f64];
    let mut x = inv0 as f64;
    for t in 0..h {
        x += rival_fc.get(t as usize).copied().unwrap_or(0.0);
        x -= town_draw(step + t, shops, item) as f64;
        x = x.max(0.0);
        path.push(x);
    }
    let lo = path.iter().cloned().fold(f64::MAX, f64::min).floor() as i64;
    let hi = path.iter().cloned().fold(0.0, f64::max).ceil() as i64 + 2 * stock + 2;
    let c = Curve::new(item, lo, hi);
    let mut rev = [0f64; NC];
    let mut best_wait = 0;
    for (k, f) in fracs.iter().enumerate() {
        let q = if *f <= 0.0 || stock <= 0 { 0 } else { ((f * stock as f32).ceil() as i64).clamp(1, stock) };
        let now = c.rev(inv0, q);
        let rest = stock - q;
        let mut later = 0.0;
        for t in 1..path.len() {
            let r = c.rev(path[t].round() as i64 + q, rest);
            if r > later {
                later = r;
                if k == 0 {
                    best_wait = t as i64;
                }
            }
        }
        rev[k] = now + later;
    }
    let at = |t: usize| path.get(t).or(path.last()).map(|x| crate::market::price(item, *x) as f64).unwrap_or(0.0);
    Priced { rev, best_wait, pf: [at(1), at(4), at(8), at(24)] }
}

// ---------------------------------------------------------------------------------------------
// the controller

pub fn class_of(fracs: &[f32; NC], q: i64, stock: i64) -> usize {
    if stock <= 0 || q <= 0 {
        return 0;
    }
    let f = (q.min(stock) as f32) / stock as f32;
    let mut best = 0;
    for (i, x) in fracs.iter().enumerate() {
        if (x - f).abs() < (fracs[best] - f).abs() {
            best = i;
        }
    }
    best
}

pub fn qty_of(fracs: &[f32; NC], c: usize, stock: i64) -> i64 {
    let f = fracs[c.min(NC - 1)];
    if f <= 0.0 || stock <= 0 {
        0
    } else {
        ((f * stock as f32).ceil() as i64).clamp(1, stock)
    }
}

/// One turn's inputs for one item (for the labeller and the model).
#[derive(Clone, Debug)]
pub struct ItemIn {
    pub x: [f32; NX],
    pub stock: i64,
    pub cur: usize,
    pub rev: [f64; NC],
    pub cap: i64,
}

#[derive(Clone, Debug)]
pub struct RShellCtl {
    pub cfg: std::sync::Arc<RConfig>,
    pub mem: RMem,
    /// This turn's inputs per item (None = not decided); read by the branch labeller.
    pub last: Vec<Option<ItemIn>>,
    /// This turn's GLOBAL inputs (always set; read by the endgame model, crate::endg).
    pub last_g: [f32; NG],
    /// This turn's WHEAT input for the buy head, computed even with no wheat in the shed (buylabel reads it).
    pub buy_x: Option<[f32; NX]>,
    /// Layer-log timing (us, whole game): observe+lineage, projected shed, global, items, decide.
    pub prof: [f64; 5],
    /// buylabel sets this: compute buy_x even without a buy net (computing it is not free of side effects)
    pub want_buy_x: bool,
    pub buys: usize,
    /// (day, buys that day)
    pub buy_day: (i64, i64),
    pub overrides: usize,
    pub reorders: usize,
    pub fronts: usize,
}

fn signals(g: &[f32; NG], it: &[f32; NIF]) -> [f32; SIGNALS.len()] {
    [
        g[gi("clone_gate")],
        g[gi("g_copy")],
        g[gi("lin_known")],
        it[ii("preempt_24")].min(3.0),
        (it[ii("rival_sold_1")] > 0.0) as i32 as f32,
        (it[ii("fc_lin_4")] > 0.0) as i32 as f32,
        (it[ii("fc_v92")] > 0.0) as i32 as f32,
        it[ii("glut")],
        (1.0 - g[gi("hours_left")] / 720.0).max(0.0),
        (-g[gi("money_gap")] / 5000.0).clamp(-2.0, 2.0),
        (1.0 - g[gi("room")] / 99.0).clamp(0.0, 1.0),
        g[gi("g_partial")],
        (it[ii("arrived")] > 0.0) as i32 as f32,
        (it[ii("rival_first_48")] / 4.0).clamp(-2.0, 2.0),
        (g[gi("day")] >= 28.0) as i32 as f32,
    ]
}

impl RShellCtl {
    pub fn new(cfg: std::sync::Arc<RConfig>) -> RShellCtl {
        RShellCtl { cfg, mem: RMem::default(), last: vec![None; NI], last_g: [0.0; NG], buy_x: None, prof: [0.0; 5], want_buy_x: false, buys: 0, buy_day: (-1, 0), overrides: 0, reorders: 0, fronts: 0 }
    }

    fn global(&self, v: &View, sig: &ChainSig, proj: &Qty) -> [f32; NG] {
        let m = &self.mem;
        let step = v.step.max(0);
        let n6 = m.poseq.iter().rev().take(6).filter(|b| **b).count();
        let n24 = m.poseq.iter().rev().take(24).filter(|b| **b).count();
        let (c0, c1) = (m.money.first().copied().unwrap_or((0.0, 0.0)), m.money.last().copied().unwrap_or((0.0, 0.0)));
        let lin = m.lin.unwrap_or((-1, 0.0));
        let total: i64 = proj.iter().map(|(_, n)| *n).sum();
        let carried: i64 = v.obs.invs.iter().map(|q| q.iter().map(|(_, n)| (*n).max(0)).sum::<i64>()).sum();
        let g = sig.group;
        let vals: [f64; NG] = [
            (step / 24) as f64,
            (step % 24) as f64,
            (720 - step) as f64,
            (step % 24 == 23) as i32 as f64,
            (sig.term_start - step) as f64,
            v.money() - v.rival().money,
            c1.0 - c0.0,
            c1.1 - c0.1,
            (g == Some(0)) as i32 as f64,
            (g == Some(1)) as i32 as f64,
            (g == Some(2)) as i32 as f64,
            g.is_some() as i32 as f64,
            (step >= 144 && m.day5_eq.1 > 0 && m.day5_eq.0 as f64 >= 0.9 * m.day5_eq.1 as f64) as i32 as f64,
            sig.clone as i32 as f64,
            n6 as f64,
            n24 as f64,
            sig.similarity,
            m.mirror as i32 as f64,
            lin.1,
            (sig.route == Some(lin.0)) as i32 as f64,
            m.lin_streak as f64,
            (lin.1 >= self.act_cfg(sig).lin_min as f64) as i32 as f64,
            m.lin_fit,
            sig.race_h as f64,
            sig.r37_h as f64,
            sig.race_lost as i32 as f64,
            sig.afr_24 as f64,
            sig.k_v92_on as i32 as f64,
            sig.k_race_clone as f64,
            sig.k_afr_on as i32 as f64,
            sig.k_rsa_look as f64,
            (99 - total).max(0) as f64,
            carried as f64,
            if step % 24 == 23 { (total + carried - 99).max(0) as f64 } else { 0.0 },
            sig.profile as f64,
        ];
        vals.map(|x| x as f32)
    }

    #[allow(clippy::too_many_arguments)]
    fn item(&self, v: &View, ch: &Chassis, sig: &ChainSig, a: &Action, proj: &Qty, k: usize, fert_need: i64) -> ([f32; NIF], ItemIn) {
        let c = &self.act_cfg(sig);
        let m = &self.mem;
        let item = ITEMS[k];
        let step = v.step.max(0);
        let inv = qget(&v.obs.mkt_inventory, item);
        let px = v.price(item) as f64;
        let hist: Vec<f64> = m.px_hist.iter().rev().take(24).map(|r| r[k]).collect();
        let mean = if hist.is_empty() { px } else { hist.iter().sum::<f64>() / hist.len() as f64 };
        let stock = qget(proj, item);
        let total = proj.iter().map(|(_, n)| *n).sum::<i64>().max(1);
        let first = m.px_hist.is_empty();
        let r1 = m.rival.last().filter(|(t, _)| *t == step).map(|(_, r)| r[k]).unwrap_or(0);
        let recent: Vec<i64> = m.rival.iter().filter(|(t, _)| *t > step - 24).map(|(_, r)| r[k]).collect();
        let since = m.rival.iter().rev().find(|(_, r)| r[k] > 0).map(|(t, _)| step - t).unwrap_or(240).min(240);
        let town_rate: f64 = (0..24).map(|t| town_draw(step + t, v.shops(), item)).sum::<i64>() as f64;
        let to_draw = (4 - step % 4) % 4;
        let (mut rt, mut rr) = (0i64, 0i64);
        for t in &v.rival().tiles {
            if product_of(t) == Some(item) {
                rt += 1;
                rr += t.yield_units.max(0);
            }
        }
        let lin_route = m.lin.filter(|l| l.1 >= c.lin_min as f64).map(|l| l.0);
        let lin = |h: i64| tape_sells(ch, lin_route, item, step + 1, step + 1 + h) as f64;
        let v92 = sig.v92.as_ref().map(|q| qget(q, item)).unwrap_or(0) as f64;
        let plan = |h: i64| tape_sells(ch, sig.route, item, step + 1, step + 1 + h) as f64;
        let cur_q = sell_qty(a, item);
        let cur_slot = a.market.iter().position(|o| o.is_sell3() && o.s(1) == item).map(|i| i as f64).unwrap_or(10.0);
        // price calculator: the rival forecast per future step
        let hz = c.horizon.min(718 - step).max(0);
        let rate = recent.iter().sum::<i64>() as f64 / 24.0;
        let fc: Vec<f64> = (0..hz)
            .map(|t| {
                let tt = step + 1 + t;
                c.fc_w[0] as f64 * tape_sells(ch, sig.route, item, tt, tt + 1) as f64
                    + c.fc_w[1] as f64 * if t == 0 { v92 } else { 0.0 }
                    + c.fc_w[2] as f64 * tape_sells(ch, lin_route, item, tt, tt + 1) as f64
                    + c.fc_w[3] as f64 * rate
            })
            .collect();
        let pr = price_calc(item, stock.max(0), inv, step, hz, v.shops(), &fc, &c.fracs);
        let rmax = pr.rev.iter().cloned().fold(f64::MIN, f64::max).max(1.0);
        let cap = if k == FERT { (stock - fert_need - c.fert_reserve).max(0) } else { stock };
        let cur = class_of(&c.fracs, cur_q, stock);
        let debt = qget(&sig.debts, item);
        let vals: [f64; NIF] = [
            stock as f64,
            v.shed(item) as f64,
            v.in_hands(item) as f64,
            stock as f64 / total as f64,
            inv as f64,
            px,
            px / (base_price(item).max(1) as f64),
            px / mean.max(1.0),
            if first { 0.0 } else { px - m.prev_px[k] },
            if first { 0.0 } else { inv as f64 - m.prev_inv[k] },
            (px <= base_price(item) as f64) as i32 as f64,
            v.shops().iter().any(|s| shop_items(s).contains(&item)) as i32 as f64,
            town_rate,
            to_draw as f64,
            r1 as f64,
            recent.iter().sum::<i64>() as f64,
            recent.iter().filter(|x| **x > 0).count() as f64,
            since as f64,
            rt as f64,
            rr as f64,
            m.preempt.iter().filter(|(t, i)| *i == k && *t > step - 24).count() as f64,
            v92,
            lin(1),
            lin(4),
            lin(8),
            lin(24),
            plan(1),
            plan(3),
            plan(8),
            plan(24),
            first_sell(ch, sig.route, item, step + 1, 48) as f64,
            cur_q as f64,
            cur_slot,
            debt as f64,
            pr.rev[0] / rmax,
            pr.rev[1] / rmax,
            pr.rev[2] / rmax,
            pr.rev[3] / rmax,
            pr.rev[4] / rmax,
            pr.best_wait as f64,
            pr.pf[0] / px.max(1.0),
            pr.pf[1] / px.max(1.0),
            pr.pf[2] / px.max(1.0),
            pr.pf[3] / px.max(1.0),
            if px > 0.0 { (px - crate::market::price(item, (inv + stock) as f64) as f64) / px } else { 0.0 },
            if k == FERT { fert_need as f64 } else { 0.0 },
            if k == FERT { cap as f64 } else { 0.0 },
            cur as f64,
            a.market.iter().filter(|o| o.is_sell3()).count() as f64,
            if stock > 0 { cur_q.min(stock) as f64 / stock as f64 } else { 0.0 },
            m.prev.as_ref().map(|p| (stock - (qget(&p.shed, item) - qget(&p.our_sold, item))).max(0)).unwrap_or(0) as f64,
            {
                // rival sold it at step t-1 (recovered at t) while we did not sell it at t-1, minus the reverse
                let ours_at = |t: i64| m.ours.iter().find(|(s, _)| *s == t).map(|(_, r)| r[k]).unwrap_or(0);
                let mut bal = 0i64;
                for (t, r) in m.rival.iter().filter(|(t, _)| *t > step - 48) {
                    let us = ours_at(t - 1);
                    if r[k] > 0 && us == 0 {
                        bal += 1;
                    } else if r[k] == 0 && us > 0 {
                        bal -= 1;
                    }
                }
                bal as f64
            },
        ];
        let itf = vals.map(|x| x as f32);
        (itf, ItemIn { x: [0.0; NX], stock, cur, rev: pr.rev, cap })
    }

    /// The config in force against this rival: its group's `group_cfg` if set, else the base config.
    fn act_cfg(&self, sig: &ChainSig) -> std::sync::Arc<RConfig> {
        match sig.group {
            Some(g) => self.cfg.group_cfg[g.min(2)].clone().unwrap_or_else(|| self.cfg.clone()),
            None => self.cfg.clone(),
        }
    }

    pub fn apply(&mut self, a: &mut Action, v: &View, ch: &Chassis, sig: &ChainSig) {
        let c = self.act_cfg(sig);
        let p0 = std::time::Instant::now();
        self.mem.observe_start(v);
        self.mem.lineage(v, ch, c.lineage.as_deref(), c.lin_min as f64);
        let p1 = std::time::Instant::now();
        let proj = ch.projected_shed(a, v);
        let p2 = std::time::Instant::now();
        let step = v.step;
        self.last = vec![None; NI];
        let g = self.global(v, sig, &proj);
        let p3 = std::time::Instant::now();
        self.last_g = g;
        let fert_need = sig.fert_committed + tape_fert_need(ch, sig.route, step + 1);
        let mut ins: Vec<Option<ItemIn>> = vec![None; NI];
        let mut sigs: Vec<[f32; SIGNALS.len()]> = vec![[0.0; SIGNALS.len()]; NI];
        for k in 0..NI {
            if qget(&proj, ITEMS[k]) <= 0 {
                continue;
            }
            let (itf, mut it) = self.item(v, ch, sig, a, &proj, k, fert_need);
            sigs[k] = signals(&g, &itf);
            it.x[..NIF].copy_from_slice(&itf);
            it.x[NIF..NIF + NG].copy_from_slice(&g);
            it.x[NIF + NG + k] = 1.0;
            ins[k] = Some(it);
        }
        self.last = ins.clone();
        // buy head input: the WHEAT item vector, also when the shed holds none
        self.buy_x = match ins[0].as_ref() {
            _ if c.buy_net.is_none() && !self.want_buy_x => None,
            Some(it) => Some(it.x),
            None => {
                let (itf, mut it) = self.item(v, ch, sig, a, &proj, 0, fert_need);
                it.x[..NIF].copy_from_slice(&itf);
                it.x[NIF..NIF + NG].copy_from_slice(&g);
                it.x[NIF + NG] = 1.0;
                Some(it.x)
            }
        };
        let p4 = std::time::Instant::now();
        for (i, (a0, b0)) in [(p0, p1), (p1, p2), (p2, p3), (p3, p4)].iter().enumerate() {
            self.prof[i] += (*b0 - *a0).as_secs_f64() * 1e6;
        }
        let group_ok = match sig.group {
            Some(gr) => c.group_on[gr.min(2)],
            None => c.pre_group,
        };
        let live = c.on && group_ok && (c.from..=c.to).contains(&step) && step < crate::view::LAST_ACT_STEP && !(c.skip_hour23 && step % 24 == 23);
        if live {
            let outs = c.net.as_ref().map(|n| n.run(&ins.iter().map(|o| o.as_ref().map(|i| i.x)).collect::<Vec<_>>()));
            let gix = sig.group.unwrap_or(0).min(2);
            let mut prio: Vec<(&'static str, f32)> = vec![];
            let mut front: Vec<&'static str> = vec![];
            for k in 0..NI {
                let Some(it) = ins[k].as_ref() else { continue };
                if !c.item_on[k] {
                    continue;
                }
                let item = ITEMS[k];
                let (uw, iu) = match sig.group {
                    Some(_) => (c.group_urge_w[gix].as_ref().unwrap_or(&c.urge_w), c.group_item_urge[gix].as_ref().unwrap_or(&c.item_urge)),
                    None => (&c.urge_w, &c.item_urge),
                };
                let urge: f32 = sigs[k].iter().zip(uw).map(|(s, w)| s * w).sum::<f32>() + iu[k];
                let net = outs.as_ref().and_then(|o| o[k].as_ref());
                let rmax = it.rev.iter().cloned().fold(f64::MIN, f64::max);
                let scale = (it.stock as f64 * base_price(item).max(1) as f64).max(1.0);
                let mut l = [0f32; NC];
                for i in 0..NC {
                    l[i] = c.bias[i]
                        + if i == it.cur { c.stay } else { 0.0 }
                        + c.beta_price * ((it.rev[i] - rmax) / scale) as f32
                        + urge * (2.0 * c.fracs[i] - 1.0)
                        + net.map(|n| c.model_w * n.logits[i]).unwrap_or(0.0);
                }
                let mx = l.iter().cloned().fold(f32::MIN, f32::max);
                let e: Vec<f32> = l.iter().map(|x| (x - mx).exp()).collect();
                let z: f32 = e.iter().sum();
                let (best, pb) = e.iter().enumerate().fold((0, 0f32), |acc, (i, &y)| if y / z > acc.1 { (i, y / z) } else { acc });
                let more = c.fracs[best] > c.fracs[it.cur];
                let dir_ok = match c.group_mode[gix] {
                    1 => more,
                    2 => !more,
                    _ => true,
                };
                let glut_ok = !more || (v.price(item) as f32) >= c.glut_floor * base_price(item) as f32;
                let gate_ok = net.map(|n| n.margin[best] - n.margin[it.cur] >= c.margin_gate).unwrap_or(true);
                let tau = c.tau + c.tau_item[k] + if sig.group.is_some() { c.group_tau[gix] } else { 0.0 };
                if best != it.cur && pb >= tau && dir_ok && glut_ok && gate_ok {
                    let mut q = qty_of(&c.fracs, best, it.stock);
                    if more {
                        // hard limit: never sell into the committed fertilizer need
                        q = q.min(it.cap.max(sell_qty(a, item)));
                    }
                    if q != sell_qty(a, item) {
                        set_sell(a, item, q);
                        self.overrides += 1;
                    }
                }
                if urge >= c.front && sell_qty(a, item) > 0 {
                    front.push(item);
                }
                prio.push((item, net.map(|n| n.prio).unwrap_or(0.0) + c.prio_urge * urge));
            }
            if c.reorder && prio.len() > 1 {
                // permute our SELLs among their own slots by priority (other orders keep their slots)
                let slots: Vec<usize> = a.market.iter().enumerate().filter(|(_, o)| o.is_sell3() && o.n(2) > 0).map(|(i, _)| i).collect();
                let mut sells: Vec<Cmd> = slots.iter().map(|i| a.market[*i].clone()).collect();
                let p = |o: &Cmd| prio.iter().find(|(it, _)| *it == o.s(1)).map(|x| x.1).unwrap_or(f32::MIN);
                sells.sort_by(|x, y| p(y).total_cmp(&p(x)));
                let before = a.market.clone();
                for (s, cmd) in slots.iter().zip(sells) {
                    a.market[*s] = cmd;
                }
                if a.market != before {
                    self.reorders += 1;
                }
            }
            if !front.is_empty() {
                let (mut head, tail): (Vec<Cmd>, Vec<Cmd>) = a.market.drain(..).partition(|o| o.is_sell3() && front.contains(&o.s(1)) && o.n(2) > 0);
                head.extend(tail);
                a.market = head;
                self.fronts += 1;
            }
        }
        // shell v3 buy head: buy wheat now when the model predicts a margin gain above the gate (never while we
        // sell wheat this step, never into a full queue, at most buy_cash of our money)
        let day = step.div_euclid(24);
        if self.buy_day.0 != day {
            self.buy_day = (day, 0);
        }
        // only on the labelled hours (step % 4 == 1, as buylabel samples) and at most buy_per_day per day
        if c.on && step >= c.buy_from && step <= c.buy_to && step % 4 == 1 && self.buy_day.1 < c.buy_per_day && sell_qty(a, "WHEAT") == 0 && a.market.len() < 10 {
            if let (Some(bn), Some(x)) = (c.buy_net.as_ref(), self.buy_x.as_ref()) {
                let gains = bn.run(x);
                let px = (v.price("WHEAT") as f64).max(1.0);
                let afford = ((v.money() * c.buy_cash as f64) / (px * 1.2)).floor() as i64;
                let best = bn.qs.iter().zip(&gains).filter(|(q, _)| **q <= afford).fold((0i64, c.buy_gate), |acc, (q, g)| if *g > acc.1 { (*q, *g) } else { acc });
                if best.0 > 0 {
                    a.market.push(Cmd::order("BUY_PRODUCT", "WHEAT", best.0));
                    self.buys += 1;
                    self.buy_day.1 += 1;
                }
            }
        }
        self.remember(v, ch, sig, a, &proj);
    }

    fn remember(&mut self, v: &View, ch: &Chassis, sig: &ChainSig, a: &Action, proj: &Qty) {
        let m = &mut self.mem;
        let mut row = [0f64; NI];
        for (k, item) in ITEMS.iter().enumerate() {
            row[k] = v.price(item) as f64;
            m.prev_inv[k] = qget(&v.obs.mkt_inventory, item) as f64;
            m.prev_px[k] = row[k];
        }
        m.px_hist.push(row);
        if m.px_hist.len() > 48 {
            m.px_hist.remove(0);
        }
        let mut sold: Qty = vec![];
        for o in &a.market {
            if o.is_sell3() {
                let Some(it) = ITEMS.iter().copied().find(|p| *p == o.s(1)) else { continue };
                let have = qget(proj, it) - qget(&sold, it);
                qadd(&mut sold, it, o.n(2).max(0).min(have.max(0)));
            }
        }
        let step = v.step;
        let mut row = [0i64; NI];
        for (k, it) in ITEMS.iter().enumerate() {
            row[k] = qget(&sold, it);
        }
        m.ours.push((step, row));
        if m.ours.len() > 240 {
            m.ours.remove(0);
        }
        let mut ahead = [false; NI];
        for (k, it) in ITEMS.iter().enumerate() {
            ahead[k] = tape_sells(ch, sig.route, it, step + 1, step + 25) > 0;
        }
        m.prev = Some(Prev { step, inv: v.obs.mkt_inventory.clone(), shops: v.obs.shops.clone(), our_sold: sold, shed: proj.clone(), ahead });
    }
}
