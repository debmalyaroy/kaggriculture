//! R37 (agent lines 1878-1921) with the R44 cash-response probe (1850-1872): sets the R36
//! reservation horizon (2; 3 after a 6-step layout-similarity streak in 336..647; 4 when the
//! probe matched or in 288..695), then reorders each run of distinct SELLs by the revenue a
//! small rival batch would take off them (`_r37_quote_priority`).
use crate::act::Action;
use crate::chassis::Chassis;
use crate::market;
use crate::obs::{qget, Qty};
use crate::view::View;

#[derive(Clone, Debug)]
struct St {
    step: i64,
    streak: i64,
}
#[derive(Clone, Debug)]
struct Probe {
    step: i64,
    money: Option<(f64, f64)>,
    probe: i64,
    matched: bool,
}

#[derive(Default)]
pub struct R37 {
    players: Vec<(i64, St)>,
    probes: Vec<(i64, Probe)>,
}

/// `_r37_similarity`: share of occupied tiles with the same (crop, animal) on both farms.
pub fn similarity(v: &View) -> f64 {
    let own = v.farm();
    let rival = v.rival();
    if own.quadrants != rival.quadrants {
        return 0.0;
    }
    let (mut matches, mut total) = (0usize, 0usize);
    for (a, b) in own.tiles.iter().zip(rival.tiles.iter()) {
        let sa = if a.is_dict() { (a.crop, a.animal) } else { ("", "") };
        let sb = if b.is_dict() { (b.crop, b.animal) } else { ("", "") };
        if sa != ("", "") || sb != ("", "") {
            total += 1;
            matches += (sa == sb) as usize;
        }
    }
    if total >= 8 {
        matches as f64 / total as f64
    } else {
        0.0
    }
}

impl R37 {
    /// Pre phase: returns `_R37_HORIZONS[player]` for this step.
    pub fn pre(&mut self, v: &View) -> i64 {
        let (player, step) = (v.obs.player, v.step);
        let i = match self.players.iter().position(|(p, _)| *p == player) {
            Some(i) if step > self.players[i].1.step => i,
            Some(i) => {
                self.players[i].1 = St { step: -1, streak: 0 };
                i
            }
            None => {
                self.players.push((player, St { step: -1, streak: 0 }));
                self.players.len() - 1
            }
        };
        self.players[i].1.step = step;
        let mut horizon = 2;
        let matched = self.probe_before(v);
        if step < 648 {
            let s = &mut self.players[i].1;
            s.streak = if similarity(v) >= 0.90 { s.streak + 1 } else { 0 };
            if (336..648).contains(&step) && s.streak >= 6 {
                horizon = 3;
            }
        }
        if horizon == 3 && matched {
            horizon = 4;
        }
        if (288..696).contains(&step) {
            horizon = 4;
        }
        horizon
    }

    /// `_r44_before`; returns `matched`.
    fn probe_before(&mut self, v: &View) -> bool {
        let (player, step) = (v.obs.player, v.step);
        let i = match self.probes.iter().position(|(p, _)| *p == player) {
            Some(i) if step > self.probes[i].1.step => i,
            Some(i) => {
                self.probes[i].1 = Probe { step: -1, money: None, probe: 0, matched: false };
                i
            }
            None => {
                self.probes.push((player, Probe { step: -1, money: None, probe: 0, matched: false }));
                self.probes.len() - 1
            }
        };
        let money = (v.farm().money, v.rival().money);
        let st = &mut self.probes[i].1;
        if let Some(prev) = st.money {
            if st.probe >= 100 && similarity(v) >= 0.90 {
                let own = money.0 - prev.0;
                let rival = money.1 - prev.1;
                if own > 0.0 && rival > 0.0 && (own - rival).abs() <= 5.0f64.max(0.05 * st.probe as f64) {
                    st.matched = true;
                }
            }
        }
        st.step = step;
        st.money = Some(money);
        st.probe = 0;
        st.matched
    }

    pub fn post(&mut self, action: Action, v: &View, ch: &Chassis) -> Action {
        let (player, step) = (v.obs.player, v.step);
        // _r44_after
        if let Some((_, st)) = self.probes.iter_mut().find(|(p, _)| *p == player) {
            if (336..648).contains(&step)
                && !st.matched
                && !action.market.is_empty()
                && !action.market.iter().any(|o| !o.is_empty() && o.op() != "SELL")
            {
                let debts = ch.players.iter().find(|(p, _)| *p == player).map(|(_, s)| &s.sell.r36_debts);
                if let Some(own) = debts.and_then(|d| d.iter().find(|(s, _)| *s == step + 3)).map(|(_, m)| m) {
                    if !own.is_empty() {
                        st.probe = own.iter().map(|(item, n)| (*n).max(0) * v.price(item)).sum();
                    }
                }
            }
        }
        if step >= 288 {
            return reorder_sales(action, v, ch);
        }
        action
    }
}

/// `_r37_quote_priority`.
pub fn quote_priority(v: &View, item: &str, qty_tok: i64, stock: &Qty) -> i64 {
    let quantity = qty_tok.max(0).min(qget(stock, item));
    if quantity == 0 || market::params(item).is_none() {
        return 0;
    }
    let inventory = qget(&v.obs.mkt_inventory, item) as f64;
    let rival = v.rival();
    let crop_item = matches!(item, "WHEAT" | "CARROT" | "TOMATO" | "STRAWBERRY" | "MELON");
    let animal = match item {
        "EGG" => "GOOSE",
        "MILK" => "COW",
        "WOOL" => "SHEEP",
        _ => "",
    };
    let standing: i64 = rival
        .tiles
        .iter()
        .filter(|t| t.is_dict() && ((crop_item && t.crop == item) || (!animal.is_empty() && t.animal == animal)))
        .map(|t| t.yield_units.max(0))
        .sum();
    let batch = standing.clamp(8, 24) as f64;
    let (mut now, mut later) = (0i64, 0i64);
    for j in 0..quantity.max(0) {
        now += market::price(item, inventory + j as f64);
        later += market::price(item, inventory + batch + j as f64);
    }
    now - later
}

/// `_r37_reorder_sales`: sort each contiguous run of SELLs with distinct items by quote
/// priority, descending (stable). Any empty order / short SELL makes Python raise -> unchanged.
fn reorder_sales(action: Action, v: &View, ch: &Chassis) -> Action {
    if action.market.iter().any(|o| o.is_empty() || (o.op() == "SELL" && o.len() < 3)) {
        return action;
    }
    let stock = ch.projected_shed(&action, v);
    let mut orders = action.market.clone();
    let mut start = 0;
    while start < orders.len() {
        if orders[start].op() != "SELL" {
            start += 1;
            continue;
        }
        let mut end = start;
        while end < orders.len() && orders[end].op() == "SELL" {
            end += 1;
        }
        let block = &mut orders[start..end];
        let mut items: Vec<&str> = block.iter().map(|o| o.s(1)).collect();
        items.sort();
        items.dedup();
        if items.len() == block.len() {
            let mut keyed: Vec<(i64, crate::act::Cmd)> =
                block.iter().map(|o| (quote_priority(v, o.s(1), o.n(2), &stock), o.clone())).collect();
            keyed.sort_by(|a, b| b.0.cmp(&a.0));
            for (k, (_, o)) in keyed.into_iter().enumerate() {
                block[k] = o;
            }
        }
        start = end;
    }
    if orders != action.market {
        let mut a = action;
        a.market = orders;
        return a;
    }
    action
}
