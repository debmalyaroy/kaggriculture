//! Small stateless wrappers.
use crate::act::{Action, Cmd};
use crate::obs::{qadd, qget, Qty};
use crate::view::{View, PRODUCTS};

/// Hour-23 day-end storage guard (aurax7 Reactive v2, agent lines 1119-1147): if shed + carried
/// cargo exceeds 99, sell unplanned shed stock, highest price first, into free market slots.
pub fn room_guard(action: Action, v: &View) -> Action {
    if v.step % 24 != 23 {
        return action;
    }
    let carried: i64 = v.invs().iter().flat_map(|m| m.iter().map(|(_, n)| (*n).max(0))).sum();
    let mut needed = v.shed_total() + carried - 99;
    if needed <= 0 {
        return action;
    }
    let mut planned: Qty = vec![];
    for o in &action.market {
        if o.is_sell3() {
            qadd(&mut planned, o.s(1), o.n(2).max(0));
        }
    }
    let mut result = action;
    let mut order: Vec<&'static str> = PRODUCTS.to_vec();
    order.sort_by_key(|it| -v.price(it));
    for item in order {
        let qty = needed.min((v.shed(item) - qget(&planned, item)).max(0));
        if qty <= 0 {
            continue;
        }
        if result.market.len() >= 10 {
            break;
        }
        result.market.push(Cmd::order("SELL", item, qty));
        needed -= qty;
        if needed <= 0 {
            break;
        }
    }
    result
}

/// `_v224_sales_first` (agent line 1491): drop empty / zero-quantity orders (keeping HIRE and
/// BUY_LAND) from the first 10, and move each SELL ahead of the non-SELL orders before it,
/// stopping at a SELL or at a BUY of the same item.
pub fn sales_first(action: Action) -> Action {
    let original: Vec<Cmd> = action.market.iter().take(10).cloned().collect();
    let mut orders: Vec<Cmd> = original
        .iter()
        .filter(|o| !o.is_empty() && (matches!(o.op(), "HIRE" | "BUY_LAND") || (o.len() >= 3 && o.n(2) > 0)))
        .cloned()
        .collect();
    for index in 0..orders.len() {
        if orders[index].op() != "SELL" {
            continue;
        }
        let mut cursor = index;
        while cursor > 0 {
            let prev = &orders[cursor - 1];
            if prev.op() == "SELL" {
                break;
            }
            if matches!(prev.op(), "BUY_PRODUCT" | "BUY_ANIMAL") && prev.s(1) == orders[cursor].s(1) {
                break;
            }
            orders.swap(cursor - 1, cursor);
            cursor -= 1;
        }
    }
    if orders == original {
        return action;
    }
    let mut changed = action;
    changed.market = orders;
    changed
}

/// The PASS action the entry guards return when their inner chain raises.
pub fn pass_action() -> Action {
    Action::pass()
}
