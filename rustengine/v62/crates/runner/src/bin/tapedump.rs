//! Replay recorded games exactly and dump, per seat per day, the state at the start of the day and
//! the choices made during it: the input for "what does a strong player do in which situation".
//!
//!     tapedump --tapes DIR [--threads 8] > days.jsonl
//!
//! One JSON line per (tape, seat, day):
//!   key, seat, day, shops (unlocked so far), money, rival_money, crew, quads, land (unlocked cells),
//!   plants {crop: n}, ripe {crop: n} (plants at yield cap), animals {kind: n}, structs {COOP|PASTURE: n}, weeds,
//!   shed {item: n}, seeds {crop: n}, prices {product: $} (quote at the day's first step),
//!   r_plants, r_animals (rival board), copy (share of the day's steps where >=1 rival unit stood on our squares),
//!   ops {"OP ITEM": n} (unit commands, moves excluded), buys {"BUY_SEED WHEAT": units, "HIRE": n, ...},
//!   sells {product: units sold}, sell_px {product: average realized price}, sell_rev {product: revenue}
//!   (exact: counterfactual step without that seat's SELL orders of the product),
//!   bank_end (seat's final bank), rival_end.
//! Recorded actions only, no agent: the replay reproduces the ladder banks exactly (tapeplay --verify).
use agent::act::Action;
use kagg_engine::engine::{self, PlayerAction};
use kagg_engine::json::{self, Json};
use kagg_engine::market;
use kagg_engine::state::{Cell, State, ANIMAL_NAMES, CROP_NAMES, PRODUCTS};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

fn act_of(j: &Json) -> PlayerAction {
    if j.is_null() {
        return runner::to_engine(&Action::pass());
    }
    runner::to_engine(&Action::from_json(j))
}

type M = BTreeMap<String, f64>;

fn obj(m: &M) -> String {
    let parts: Vec<String> = m.iter().filter(|(_, v)| **v != 0.0).map(|(k, v)| format!("\"{k}\":{}", (v * 100.0).round() / 100.0)).collect();
    format!("{{{}}}", parts.join(","))
}

fn board(st: &State, s: usize) -> (M, M, M, M, i64, i64) {
    let (mut plants, mut ripe, mut animals, mut structs) = (M::new(), M::new(), M::new(), M::new());
    let (mut weeds, mut land) = (0, 0);
    for row in &st.farms[s].tiles {
        for c in row {
            if !matches!(c, Cell::Locked) {
                land += 1;
            }
            match c {
                Cell::Weed => weeds += 1,
                Cell::Plant { crop, yield_units, .. } => {
                    *plants.entry(crop.to_string()).or_default() += 1.0;
                    if *yield_units > 0 {
                        *ripe.entry(crop.to_string()).or_default() += 1.0;
                    }
                }
                Cell::Structure { kind, animal } => {
                    *structs.entry(kind.to_string()).or_default() += 1.0;
                    if let Some(a) = animal {
                        *animals.entry(a.animal.to_string()).or_default() += 1.0;
                    }
                }
                _ => {}
            }
        }
    }
    (plants, ripe, animals, structs, weeds, land)
}

fn quote(st: &State, p: &str) -> f64 {
    market::param(p).map(|m| market::price(m, st.market.inventory.get(p) as f64) as f64).unwrap_or(0.0)
}

fn dump(path: &std::path::Path) -> Vec<String> {
    let Some(j) = std::fs::read_to_string(path).ok().and_then(|s| json::parse(&s).ok()) else { return vec![] };
    let key = path.file_stem().unwrap().to_string_lossy().to_string();
    let seed = j.get("seed").i64();
    let acts = j.get("actions").arr();
    let tape = |t: usize, s: usize| acts.get(t).and_then(|p| p.arr().get(s)).map(act_of).unwrap_or_else(|| act_of(&Json::Null));
    let mut st = State::new(seed);
    // per seat, per day accumulators
    let mut days: Vec<[String; 2]> = vec![];
    let mut head: [String; 2] = [String::new(), String::new()];
    let (mut ops, mut buys, mut sells, mut pxw, mut copy): ([M; 2], [M; 2], [M; 2], [M; 2], [f64; 2]) = Default::default();
    let mut spend: [M; 2] = Default::default();
    let mut t = 0usize;
    loop {
        let day = t / 24;
        if t % 24 == 0 {
            for s in 0..2 {
                let o = 1 - s;
                let (plants, ripe, animals, structs, weeds, land) = board(&st, s);
                let (rp, _, ra, _, _, _) = board(&st, o);
                let shed: M = PRODUCTS.iter().chain(ANIMAL_NAMES.iter()).map(|p| (p.to_string(), st.private[s].shed.get(p) as f64)).collect();
                let seeds: M = CROP_NAMES.iter().map(|p| (p.to_string(), st.private[s].seeds.get(p) as f64)).collect();
                let prices: M = PRODUCTS.iter().map(|p| (p.to_string(), quote(&st, p))).collect();
                head[s] = format!(
                    "\"key\":\"{key}\",\"seat\":{s},\"day\":{day},\"shops\":\"{}\",\"money\":{},\"rival_money\":{},\"crew\":{},\"quads\":{},\"land\":{land},\"plants\":{},\"ripe\":{},\"animals\":{},\"structs\":{},\"weeds\":{weeds},\"shed\":{},\"seeds\":{},\"prices\":{},\"r_plants\":{},\"r_animals\":{}",
                    st.town.unlocked_shops.join("|"),
                    st.farms[s].money,
                    st.farms[o].money,
                    1 + st.farms[s].hands.len(),
                    st.farms[s].unlocked_quadrants.len(),
                    obj(&plants),
                    obj(&ripe),
                    obj(&animals),
                    obj(&structs),
                    obj(&shed),
                    obj(&seeds),
                    obj(&prices),
                    obj(&rp),
                    obj(&ra)
                );
            }
        }
        let a = [tape(t, 0), tape(t, 1)];
        for s in 0..2 {
            let mine: Vec<(i64, i64)> = std::iter::once(st.farms[s].farmer).chain(st.farms[s].hands.iter().copied()).collect();
            let o = 1 - s;
            if std::iter::once(st.farms[o].farmer).chain(st.farms[o].hands.iter().copied()).any(|p| mine.contains(&p)) {
                copy[s] += 1.0 / 24.0;
            }
            for u in std::iter::once(&a[s].farmer).chain(a[s].hands.iter()) {
                if !matches!(u.op.as_str(), "NORTH" | "SOUTH" | "EAST" | "WEST" | "PASS" | "") {
                    *ops[s].entry(format!("{} {}", u.op, u.item).trim().to_string()).or_default() += 1.0;
                }
            }
            for m in &a[s].market {
                let op = m.first().map(|x| x.as_str()).unwrap_or("");
                let item = m.get(1).cloned().unwrap_or_default();
                let n: f64 = m.get(2).and_then(|x| x.parse().ok()).unwrap_or(1.0);
                match op {
                    "SELL" => {}
                    "HIRE" | "BUY_LAND" => *buys[s].entry(op.to_string()).or_default() += 1.0,
                    "" => {}
                    _ => *buys[s].entry(format!("{op} {item}")).or_default() += n,
                }
            }
        }
        // realized sales, exactly: rerun the step without seat s's SELL orders of product p (their
        // slots kept empty so every other order keeps its place in the market race); the money
        // difference is s's revenue from p, the market-stock difference the units it sold.
        let actual = {
            let mut x = st.clone();
            engine::step(&mut x, &a);
            x
        };
        for s in 0..2 {
            let prods: std::collections::BTreeSet<String> = a[s].market.iter().filter(|m| m.first().is_some_and(|x| x == "SELL")).filter_map(|m| m.get(1).cloned()).collect();
            // spend, exactly, per buy key ("BUY_SEED WHEAT", "HIRE", ...): the same counterfactual
            let bkeys: std::collections::BTreeSet<String> = a[s]
                .market
                .iter()
                .filter(|m| m.first().is_some_and(|x| x != "SELL" && !x.is_empty()))
                .map(|m| if matches!(m[0].as_str(), "HIRE" | "BUY_LAND") { m[0].clone() } else { format!("{} {}", m[0], m.get(1).cloned().unwrap_or_default()) })
                .collect();
            for bk in bkeys {
                let mut a2 = a.clone();
                for m in a2[s].market.iter_mut() {
                    let k = if m.first().is_some_and(|x| matches!(x.as_str(), "HIRE" | "BUY_LAND")) { m[0].clone() } else { format!("{} {}", m.first().cloned().unwrap_or_default(), m.get(1).cloned().unwrap_or_default()) };
                    if k == bk {
                        m.clear();
                    }
                }
                let mut x = st.clone();
                engine::step(&mut x, &a2);
                let cost = x.farms[s].money - actual.farms[s].money;
                if cost > 0.0 {
                    *spend[s].entry(bk).or_default() += cost;
                }
            }
            for p in prods {
                let mut a2 = a.clone();
                for m in a2[s].market.iter_mut() {
                    if m.first().is_some_and(|x| x == "SELL") && m.get(1) == Some(&p) {
                        m.clear();
                    }
                }
                let mut x = st.clone();
                engine::step(&mut x, &a2);
                let rev = actual.farms[s].money - x.farms[s].money;
                let units = (actual.market.inventory.get(&p) - x.market.inventory.get(&p)) as f64;
                if rev > 0.0 {
                    *sells[s].entry(p.clone()).or_default() += units.max(0.0);
                    *pxw[s].entry(p.clone()).or_default() += rev;
                }
            }
        }
        t += 1;
        let alive = engine::step(&mut st, &a);
        if t % 24 == 0 || !alive {
            let mut row: [String; 2] = [String::new(), String::new()];
            for s in 0..2 {
                let px: M = sells[s].iter().map(|(k, v)| (k.clone(), pxw[s].get(k).copied().unwrap_or(0.0) / v.max(1.0))).collect();
                let rev: M = pxw[s].clone();
                row[s] = format!(
                    "{{{},\"copy\":{:.2},\"ops\":{},\"buys\":{},\"sells\":{},\"sell_px\":{},\"sell_rev\":{},\"spend\":{}",
                    head[s],
                    copy[s],
                    obj(&ops[s]),
                    obj(&buys[s]),
                    obj(&sells[s]),
                    obj(&px),
                    obj(&rev),
                    obj(&spend[s])
                );
                spend[s].clear();
                ops[s].clear();
                buys[s].clear();
                sells[s].clear();
                pxw[s].clear();
                copy[s] = 0.0;
            }
            days.push(row);
        }
        if !alive {
            break;
        }
    }
    let end = [st.farms[0].money, st.farms[1].money];
    days.into_iter().flat_map(|r| (0..2).map(move |s| format!("{},\"bank_end\":{},\"rival_end\":{}}}", r[s], end[s], end[1 - s])).collect::<Vec<_>>()).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).map(|i| args[i + 1].clone());
    let dir = get("--tapes").expect("--tapes DIR");
    let threads: usize = get("--threads").and_then(|s| s.parse().ok()).unwrap_or(4);
    let mut files: Vec<_> = std::fs::read_dir(&dir).expect("dir").filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "json")).collect();
    files.sort();
    let q = Arc::new(Mutex::new(files));
    let out = Arc::new(Mutex::new(std::io::stdout()));
    std::thread::scope(|s| {
        for _ in 0..threads {
            let (q, out) = (q.clone(), out.clone());
            s.spawn(move || loop {
                let Some(p) = q.lock().unwrap().pop() else { break };
                let lines = dump(&p);
                use std::io::Write;
                let mut o = out.lock().unwrap();
                for l in lines {
                    let _ = writeln!(o, "{l}");
                }
            });
        }
    });
}
