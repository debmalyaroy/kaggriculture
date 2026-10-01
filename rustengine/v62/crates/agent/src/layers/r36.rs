//! R36 sale reservation (`_r36_reserve`, agent line 1673, as replaced by V9 RACEGATE at 3652):
//! in steps 192..695 pull the tape's upcoming SELLs of stock already in the shed forward to
//! now, up to the horizon, and book them as debts that the chassis settles on their due step.
//! RACEGATE: products quoted at or below base (+ margin) are left to the tape.
use super::Ctx;
use crate::act::{Action, Cmd};
use crate::chassis::{base_price, Chassis};
use crate::obs::{qadd, qget, Qty};
use crate::view::{animal_structure, View, PRODUCTS};

pub const RACEGATE_MARGIN: i64 = 0;

pub fn reserve(mut action: Action, v: &View, ch: &mut Chassis, cx: &Ctx, racegate_margin: i64) -> Action {
    let step = v.step;
    if !(192..696).contains(&step) {
        return action;
    }
    let glutted: Vec<&str> = PRODUCTS.iter().copied().filter(|i| v.price(i) <= base_price(i) + racegate_margin).collect();
    let player = v.obs.player;
    let Some(pi) = ch.players.iter().position(|(p, _)| *p == player) else { return action };
    let Some(route) = ch.players[pi].1.route else { return action };
    let hz = cx.v9_item_hz.as_ref().filter(|h| !h.is_empty());
    let horizon = match hz {
        Some(h) => h.iter().map(|(_, n)| *n).max().unwrap_or(0),
        // RACE (5504) lifts the R37 horizon to its clone horizon while reserving.
        None => {
            let r37 = cx.r37_horizon.unwrap_or(2);
            if cx.race_horizon > r37 { cx.race_horizon } else { r37 }
        }
    };
    let end = 695.min(step + horizon);
    if end <= step {
        return action;
    }
    let commands = action.units();
    let np = v.positions.len();
    if commands
        .iter()
        .take(np)
        .enumerate()
        .any(|(i, c)| c.len() > 1 && c.op() == "PLACE" && animal_structure(c.s(1)).is_some() && qget(v.inv(i), c.s(1)) > 0)
    {
        return action;
    }
    let stock = ch.projected_shed(&action, v);
    let mut blocked: Vec<&str> = action.market.iter().filter(|o| o.len() > 1 && matches!(o.op(), "SELL" | "BUY_PRODUCT")).map(|o| o.s(1)).collect();
    blocked.extend(commands.iter().filter(|c| c.len() > 1 && c.op() == "PICKUP").map(|c| c.s(1)));
    for (_, q) in &ch.players[pi].1.pending {
        blocked.extend(q.iter().filter(|(_, c)| c.len() > 1 && c.op() == "PICKUP").map(|(_, c)| c.s(1)));
    }
    let tape_len = ch.route(route).tape.len() as i64;
    for item in PRODUCTS {
        if blocked.contains(&item) || glutted.contains(&item) || v.price(item) < 2 {
            continue;
        }
        let mut available = qget(&stock, item).max(0);
        if available == 0 || action.market.len() >= 10 {
            continue;
        }
        let item_end = match hz.and_then(|h| h.iter().find(|(k, _)| *k == item)) {
            Some((_, n)) => end.min(step + n),
            None => end,
        };
        let mut reservations: Vec<(i64, i64)> = vec![];
        let debts = &ch.players[pi].1.sell.r36_debts;
        let mut due_step = step + 1;
        while due_step <= item_end {
            if due_step >= tape_len {
                // Python: tape[due_step] raises IndexError -> the R36 wrapper keeps its action.
                return action;
            }
            let fut = &ch.route(route).tape[due_step as usize];
            if fut.units().iter().any(|c| c.len() > 1 && c.op() == "PICKUP" && c.s(1) == item) {
                break;
            }
            if fut.market.iter().any(|o| o.len() > 1 && o.op() == "BUY_PRODUCT" && o.s(1) == item) {
                break;
            }
            let planned: i64 = fut.market.iter().filter(|o| o.is_sell3() && o.s(1) == item).map(|o| o.n(2).max(0)).sum();
            let owed = debts.iter().find(|(s, _)| *s == due_step).map(|(_, m)| qget(m, item)).unwrap_or(0);
            let amount = available.min((planned - owed).max(0));
            if amount != 0 {
                reservations.push((due_step, amount));
                available -= amount;
            }
            if available == 0 {
                break;
            }
            due_step += 1;
        }
        let qty: i64 = reservations.iter().map(|(_, q)| q).sum();
        if qty != 0 {
            action.market.push(Cmd::order("SELL", item, qty));
            let debts = &mut ch.players[pi].1.sell.r36_debts;
            for (due, q) in reservations {
                match debts.iter_mut().find(|(s, _)| *s == due) {
                    Some((_, m)) => qadd(m, item, q),
                    None => {
                        let mut m: Qty = vec![];
                        qadd(&mut m, item, q);
                        debts.push((due, m));
                    }
                }
            }
        }
    }
    action
}
