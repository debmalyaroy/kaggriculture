//! Live side of the macro observation: the agent's own [`View`] and action -> the
//! [`dayobs`] builder's inputs. The corpus side is `features::dayobs_adapter`; both feed the
//! same [`dayobs::Builder`], and `tapeplay --obs-check` verifies they agree on real games.
use crate::act::Action;
use crate::obs::{qget, FarmObs};
use crate::view::View;
use dayobs::{index_of, OwnAct, StepView, TileCount, TileKind, ANIMALS, CROPS, PRODUCTS, SHOPS};

fn tiles(f: &FarmObs, day: i64) -> TileCount {
    let mut c = TileCount::default();
    for t in &f.tiles {
        let kind = match t.kind {
            "" => TileKind::Empty,
            "LOCKED" => TileKind::Locked,
            "PLANT" => TileKind::Plant,
            "WEED" => TileKind::Weed,
            "PASTURE" => TileKind::Pasture,
            "COOP" => TileKind::Coop,
            _ => TileKind::Other,
        };
        let crop = if t.is_dict() { index_of(&CROPS, t.crop) } else { None };
        let animal = if t.is_dict() { index_of(&ANIMALS, t.animal) } else { None };
        c.add(kind, crop, animal, t.yield_units as f64, t.planted_day, day);
    }
    c
}

/// What the seat sees at this step (index 0 = own, 1 = rival). Tile counts and the layout
/// similarity are computed at decision steps only (they are read only there).
pub fn step_view(v: &View) -> StepView {
    let o = &v.obs;
    let (own, riv) = (v.farm(), v.rival());
    let step = o.step().max(0) as usize;
    // tiles and layout similarity are read at decision steps only: hour 1, and hour 13 (mid-day decisions)
    let decision = step % dayobs::TURNS_PER_DAY == 1 || step % dayobs::TURNS_PER_DAY == dayobs::MID_HOUR;
    let day = (step / dayobs::TURNS_PER_DAY) as i64;
    let mut s = StepView {
        step,
        money: [own.money, riv.money],
        hires: [own.hires_today as f64, riv.hires_today as f64],
        hands: [own.hands.len() as u16, riv.hands.len() as u16],
        quads: [own.quadrants.len() as u8, riv.quadrants.len() as u8],
        pos_eq: own.farmer == riv.farmer && own.hands == riv.hands,
        shops: o.shops.iter().filter_map(|n| index_of(&SHOPS, n).map(|i| i as u8)).collect(),
        tiles: decision.then(|| [tiles(own, day), tiles(riv, day)]),
        layout_sim: decision.then(|| crate::layers::r37::similarity(v)),
        ..Default::default()
    };
    for (i, p) in PRODUCTS.iter().enumerate() {
        s.prices[i] = qget(&o.prices, p) as f64;
        s.inv[i] = qget(&o.mkt_inventory, p) as f64;
        s.own_shed[i] = qget(&o.shed, p) as f64;
    }
    s
}

/// The seat's SELL / BUY_PRODUCT units in the first 10 market orders (engine cap).
pub fn own_act(a: &Action) -> OwnAct {
    let mut r = OwnAct::default();
    for c in a.market.iter().take(10) {
        let slot = match c.op() {
            "SELL" => &mut r.sell,
            "BUY_PRODUCT" => &mut r.buy,
            _ => continue,
        };
        if let Some(i) = index_of(&PRODUCTS, c.s(1)) {
            slot[i] += c.n(2) as f64;
        }
    }
    r
}
