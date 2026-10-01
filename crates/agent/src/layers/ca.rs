//! CA "carrot instead of wheat when the carrot book pays" (agent 4134-4388): swap a wheat
//! planting for carrot when the tape's own visits make the carrot worth more, rescue swapped
//! carrots before decay, trim the wheat seed they replaced, and sell the credited carrots.
use crate::act::{Action, Cmd};
use crate::chassis::Chassis;
use crate::obs::qget;
use crate::view::{move_delta, View};

pub const FROM: i64 = 10;
pub const TO: i64 = 28;
pub const MARGIN: f64 = -20.0;
pub const DROP: f64 = 0.0;
pub const BUFFER: i64 = 8;
pub const FEED_DAYS: i64 = 1;
pub const CASH: i64 = 800;
pub const RESCUE: bool = true;
type P = (i64, i64);

fn crop(c: &str) -> (i64, i64) {
    if c == "WHEAT" {
        (4, 6)
    } else {
        (3, 4)
    }
}

#[derive(Clone, Default)]
struct St {
    step: i64,
    tiles: Vec<(P, i64)>,
    spare_wheat: i64,
    spare_carrot: i64,
    credit: i64,
}

#[derive(Default, Clone)]
pub struct Ca {
    players: Vec<(i64, St)>,
    /// profile knob `ca_margin` (None = MARGIN)
    pub margin: Option<f64>,
}

fn ca_tape<'a>(ch: &'a Chassis, player: i64, t: i64) -> Option<&'a Action> {
    let native = ch.players.iter().find(|(p, _)| *p == player).map(|(_, s)| s)?;
    if t > 719 {
        return None;
    }
    let r = if t >= 648 { 2 } else { native.route? };
    ch.route(r).tape.get(t as usize)
}

fn spawn(positions: &[P], board: i64) -> P {
    let half = board / 2;
    let access = [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)];
    let mut best = access[0];
    let mut key = (usize::MAX, usize::MAX);
    for (ai, a) in access.iter().enumerate() {
        let k = (positions.iter().filter(|p| *p == a).count(), ai);
        if k < key {
            key = k;
            best = *a;
        }
    }
    best
}

/// `_ca_visits`: non-move commands issued on `pos` from this step until `t_end`.
pub fn visits(v: &View, ch: &Chassis, action: &Action, pos: P, t_end: i64, start: Option<i64>) -> Vec<(i64, usize, &'static str)> {
    let step = v.step;
    let board = v.farm().rows as i64;
    let half = board / 2;
    let mut positions: Vec<P> = v.positions.clone();
    let mut out = vec![];
    for t in step..=t_end.min(719) {
        let act = if t == step { Some(action) } else { ca_tape(ch, v.obs.player, t) };
        let units: Vec<Cmd> = act.map(|a| a.units()).unwrap_or_else(|| vec![Cmd::pass()]);
        for i in 0..positions.len() {
            let op = units.get(i).filter(|c| !c.is_empty()).map(|c| c.op()).unwrap_or("PASS");
            if let Some((dx, dy)) = move_delta(op) {
                let (nx, ny) = (positions[i].0 + dx, positions[i].1 + dy);
                if (0..board).contains(&nx) && (0..board).contains(&ny) {
                    positions[i] = (nx, ny);
                }
            } else if positions[i] == pos && start.is_none_or(|s| t >= s) {
                out.push((t, i, op));
            }
        }
        if let Some(a) = act {
            for o in &a.market {
                if !o.is_empty() && o.op() == "HIRE" {
                    let s = spawn(&positions, board);
                    positions.push(s);
                }
            }
        }
        if t % 24 == 23 {
            positions = vec![(half - 1, half - 1)];
        }
    }
    out
}

/// `_ca_decays`.
fn decays(mls: i64, a: i64, b: i64) -> i64 {
    let a = a.max(mls);
    if b <= a {
        return 0;
    }
    let first = if (a - mls).rem_euclid(2) == 0 { a } else { a + 1 };
    if first >= b {
        0
    } else {
        (b - 1 - first).div_euclid(2) + 1
    }
}

/// `_ca_yield_path` -> (harvest units, best rescue units).
#[allow(clippy::too_many_arguments)]
pub fn yield_path(c: &str, planted: i64, vis: &[(i64, usize, &str)], y0: i64, fert_until: i64, mut watered_day: i64, now_step: i64) -> (i64, i64) {
    let (myd, cap) = crop(c);
    let lo = (myd + 1) / 2;
    let mls = (planted + myd + 1) * 24;
    let mut y = y0;
    let mut best = 0;
    for &(t, _, op) in vis {
        let day = t / 24;
        let age = day - planted;
        let now = y - decays(mls, now_step, t);
        if now <= 0 && t > mls {
            return (0, best);
        }
        if op == "HARVEST" {
            return (if age >= 2 { now.max(0) } else { 0 }, best);
        }
        if matches!(op, "PLANT" | "DIG" | "BUILD_COOP" | "BUILD_PASTURE") {
            return (0, best);
        }
        if age >= 2 && now > best && t > now_step {
            best = now;
        }
        if op == "WATER" && lo <= age && age <= myd && day != watered_day {
            watered_day = day;
            y = cap.min(y + if fert_until >= day { 2 } else { 1 });
        }
    }
    (0, best)
}

fn wheat_total(v: &View) -> i64 {
    qget(&v.obs.shed, "WHEAT") + v.obs.invs.iter().map(|m| qget(m, "WHEAT")).sum::<i64>()
}

fn feed_need(ch: &Chassis, player: i64, step: i64, days: i64) -> i64 {
    let mut need = 0;
    for t in step..=719.min(step + 24 * days) {
        if let Some(a) = ca_tape(ch, player, t) {
            need += a.units().iter().filter(|c| !c.is_empty() && c.op() == "FEED").count() as i64;
        } else {
            // `{}` -> [["PASS"]]: no feed
        }
    }
    need
}

impl Ca {
    /// E402: lower `spare_carrot` by the seed it cut.
    pub fn cut_spare_carrot(&mut self, player: i64, cut: i64) {
        if let Some((_, s)) = self.players.iter_mut().find(|(p, _)| *p == player) {
            s.spare_carrot = (s.spare_carrot - cut).max(0);
        }
    }

    pub fn post(&mut self, action: Action, v: &View, ch: &Chassis) -> Action {
        let (seat, step) = (v.obs.player, v.step);
        let i = match self.players.iter().position(|(p, _)| *p == seat) {
            Some(i) if step != 0 && step > self.players[i].1.step => i,
            Some(i) => {
                self.players[i].1 = St { step: -1, ..Default::default() };
                i
            }
            None => {
                self.players.push((seat, St { step: -1, ..Default::default() }));
                self.players.len() - 1
            }
        };
        let st = &mut self.players[i].1;
        st.step = step;
        if step > 717 {
            return action;
        }
        let day = step.div_euclid(24);
        let farm = v.farm();
        let (p_c, p_w) = (v.price("CARROT"), v.price("WHEAT"));
        let positions = &v.positions;
        let mut units = action.units();
        let market = action.market.clone();
        let mut changed = false;
        // 1. bookkeeping + rescue of swapped carrots
        for (pos, planted) in st.tiles.clone() {
            let tile = farm.tile(pos.0, pos.1);
            if !(tile.is_dict() && tile.crop == "CARROT" && tile.planted_day == planted) {
                st.tiles.retain(|(p, _)| *p != pos);
                continue;
            }
            let Some(i) = positions.iter().enumerate().position(|(i, p)| *p == pos && i < units.len()) else { continue };
            let cmd = &units[i];
            let yu = tile.yield_units;
            if !cmd.is_empty() && cmd.op() == "HARVEST" {
                if day - planted >= 2 && yu > 0 {
                    st.credit += yu;
                    st.tiles.retain(|(p, _)| *p != pos);
                }
                continue;
            }
            if !RESCUE || (!cmd.is_empty() && move_delta(cmd.op()).is_some()) || day - planted < 2 || yu <= 0 {
                continue;
            }
            let vis = visits(v, ch, &action, pos, (planted + 5) * 24, None);
            let watered = if tile.watered_today { day } else { -1 };
            let (harvest, later) = yield_path("CARROT", planted, &vis, yu, tile.fertilized_until_day, watered, step);
            if yu > harvest.max(later) {
                units[i] = Cmd::new("HARVEST");
                st.credit += yu;
                st.tiles.retain(|(p, _)| *p != pos);
                changed = true;
            }
        }
        // 2. swaps
        let pays_now = 3.0 * (p_c as f64 - DROP) - 20.0 > (4 * p_w - 10) as f64 + self.margin.unwrap_or(MARGIN);
        if (FROM..=TO).contains(&day) && pays_now {
            let planted_c = units.iter().filter(|c| c.len() >= 2 && c.op() == "PLANT" && c.s(1) == "CARROT").count() as i64;
            let mut seeds_c = st.spare_carrot.min(qget(&v.obs.seeds, "CARROT") - planted_c);
            let mut wheat_ok: Option<bool> = None;
            for i in 0..units.len() {
                let c = &units[i];
                if !(c.len() >= 2 && c.op() == "PLANT" && c.s(1) == "WHEAT") || i >= positions.len() || seeds_c <= 0 {
                    continue;
                }
                let pos = positions[i];
                if !farm.tile(pos.0, pos.1).is_none() {
                    continue;
                }
                let ok = *wheat_ok.get_or_insert_with(|| wheat_total(v) >= feed_need(ch, seat, step, FEED_DAYS));
                if !ok {
                    break;
                }
                let vis = visits(v, ch, &action, pos, (day + 6) * 24, Some(step + 1));
                let (wu, _) = yield_path("WHEAT", day, &vis, 1, -1, -1, 0);
                let (chv, cr) = yield_path("CARROT", day, &vis, 1, -1, -1, 0);
                let cu = chv.max(if RESCUE { cr } else { 0 });
                if cu as f64 * (p_c as f64 - DROP) - 20.0 > (wu * p_w - 10) as f64 + self.margin.unwrap_or(MARGIN) {
                    units[i] = Cmd(vec![crate::act::Tok::S("PLANT"), crate::act::Tok::S("CARROT")]);
                    seeds_c -= 1;
                    st.spare_carrot -= 1;
                    match st.tiles.iter_mut().find(|(p, _)| *p == pos) {
                        Some(e) => e.1 = day,
                        None => st.tiles.push((pos, day)),
                    }
                    st.spare_wheat += 1;
                    changed = true;
                }
            }
        }
        // 3. seeds
        let mut new_market = Vec::with_capacity(market.len());
        for o in market.into_iter() {
            if o.len() >= 3 && o.op() == "BUY_SEED" && matches!(o.s(1), "WHEAT" | "CARROT") {
                let spare = if o.s(1) == "WHEAT" { &mut st.spare_wheat } else { &mut st.spare_carrot };
                let cut = o.n(2).min(*spare);
                if cut > 0 {
                    *spare -= cut;
                    changed = true;
                    if o.n(2) - cut <= 0 {
                        continue;
                    }
                    new_market.push(Cmd::order("BUY_SEED", o.s(1), o.n(2) - cut));
                    continue;
                }
            }
            new_market.push(o);
        }
        let mut market = new_market;
        if (FROM..=TO - 1).contains(&day) && pays_now && market.len() < 10 {
            let have = qget(&v.obs.seeds, "CARROT") - units.iter().filter(|c| c.len() >= 2 && c.op() == "PLANT" && c.s(1) == "CARROT").count() as i64;
            let buying: i64 = market.iter().filter(|o| o.len() >= 3 && o.op() == "BUY_SEED" && o.s(1) == "CARROT").map(|o| o.n(2)).sum();
            let q = BUFFER - have - buying;
            if q > 0 && (farm.money.trunc() as i64) >= CASH + 20 * q {
                market.push(Cmd::order("BUY_SEED", "CARROT", q));
                st.spare_carrot += q;
                changed = true;
            }
        }
        // 4. sell credited carrots
        if st.credit > 0 && p_c >= 2 && market.len() < 10 {
            let mut va = Action { farmer: units[0].clone(), hands: units[1..].to_vec(), market: market.clone() };
            va.market = market.clone();
            let stock = qget(&ch.projected_shed(&va, v), "CARROT");
            let selling: i64 = market.iter().filter(|o| o.is_sell3() && o.s(1) == "CARROT").map(|o| o.n(2)).sum();
            let q = st.credit.min(stock - selling);
            if q > 0 {
                market.insert(0, Cmd::order("SELL", "CARROT", q));
                st.credit -= q;
                changed = true;
            }
        }
        if !changed {
            return action;
        }
        let mut r = action;
        r.set_units(units);
        market.truncate(10);
        r.market = market;
        r
    }
}
