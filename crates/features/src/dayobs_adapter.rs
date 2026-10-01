//! Corpus side of the macro observation: drive a parsed [`Episode`] through the online
//! [`dayobs::Builder`] one step at a time, from one seat's point of view, exactly as the agent
//! does at play time. Also exposes the per-day labels the corpus can supply.
use crate::action::OrderKind;
use crate::consts::MAX_MARKET_ORDERS;
use crate::episode::{snapshot_step, Episode};
use crate::tiles::tile_stats;
use dayobs::{Builder, OwnAct, StepView, TileCount, N};

fn count(s: &crate::tiles::TileStats) -> TileCount {
    let c = |x: u32| x as u16;
    TileCount {
        planted: c(s.planted),
        growing: c(s.growing),
        ripe: c(s.ripe),
        weed: c(s.weed),
        empty: c(s.empty),
        pasture: c(s.pasture),
        coop: c(s.coop),
        by_crop: s.by_crop.map(c),
        animals: s.animals.map(c),
    }
}

/// The view seat `seat` has of step t.
pub fn step_view(ep: &Episode, seat: usize, t: usize) -> StepView {
    let r = &ep.recs[t];
    let o = [seat, 1 - seat];
    let d = (t / dayobs::TURNS_PER_DAY) as i64;
    let tiles = ep.tiles_at(t).map(|g| [count(&tile_stats(&g[seat], d)), count(&tile_stats(&g[1 - seat], d))]);
    StepView {
        step: t,
        money: o.map(|s| r.money[s]),
        hires: o.map(|s| r.hires[s]),
        hands: o.map(|s| r.n_hands[s]),
        quads: o.map(|s| r.n_quads[s]),
        pos_eq: r.pos_eq,
        prices: r.prices,
        inv: r.inv,
        shops: r.shops.iter().copied().filter(|&x| x < 8).collect(),
        own_shed: r.shed[seat].unwrap_or([0.0; 9]),
        tiles,
        layout_sim: ep.layout_sim(t, seat),
    }
}

/// Seat's own SELL / BUY_PRODUCT units applied at step t (first 10 orders, like the engine).
pub fn own_act(ep: &Episode, seat: usize, t: usize) -> OwnAct {
    let mut a = OwnAct::default();
    if let Some(pair) = ep.acts.get(t) {
        for o in pair[seat].orders.iter().take(MAX_MARKET_ORDERS) {
            match (o.kind, o.item, o.qty) {
                (OrderKind::Sell, Some(i), Some(q)) => a.sell[i] += q as f64,
                (OrderKind::BuyProduct, Some(i), Some(q)) => a.buy[i] += q as f64,
                _ => {}
            }
        }
    }
    a
}

/// One decision row: the observation at hour 1 of `day` plus labels known from the episode.
#[derive(Clone, Debug)]
pub struct DayRow {
    pub day: usize,
    pub obs: [f32; N],
    /// Rival's net units sold over the NEXT 24 steps (aux target; None past the end).
    pub rival_next: Option<[f64; 9]>,
}

/// All decision rows of `seat` in `ep`.
pub fn day_rows(ep: &Episode, seat: usize) -> Vec<DayRow> {
    let mut b = Builder::new();
    let mut rows: Vec<DayRow> = Vec::with_capacity(30);
    let mut pending: Vec<(usize, usize)> = Vec::new(); // (row index, step at which next-day flow is complete)
    for t in 0..ep.n {
        b.push(step_view(ep, seat, t), own_act(ep, seat, t));
        // a row's rival_next = the flow window as seen 24 steps after its decision step
        while let Some(&(ri, done_at)) = pending.first() {
            if t == done_at {
                rows[ri].rival_next = Some(b.rival_sold());
                pending.remove(0);
            } else {
                break;
            }
        }
        if b.at_decision() {
            if let Some(v) = b.vector() {
                let day = t / dayobs::TURNS_PER_DAY;
                debug_assert_eq!(t, snapshot_step(day));
                rows.push(DayRow { day, obs: v, rival_next: None });
                pending.push((rows.len() - 1, t + dayobs::TURNS_PER_DAY));
            }
        }
    }
    rows
}
