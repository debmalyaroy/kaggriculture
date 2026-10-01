//! Per-turn state traces of recorded games, for the top-50 play analysis (python/top50/).
//!
//!     trace --tapes DIR --out FILE.tsv [--seats both|tape]
//!
//! Replays each tape exactly (seed + both recorded action streams, as `tapeplay --verify`) and writes
//! tab-separated rows (one file, several record types, first column = type):
//!   W  id seed seat world1 world2 idle1 idle2 rep0 rep1 sim0 sim1 exact
//!        world = the realized first / first two shops; idle = the world if both players had passed
//!        (world.rs idle_world): world1 != idle1 means the players' own actions moved the first shop
//!   S  id seat step money_me money_rv shed_me[9] shed_rv[9] mkt_inv[9] mkt_px[9]
//!        after every engine step, for each traced seat (`--seats tape` = only the tape's `seat`)
//!   F  id seat day land hands crop_tiles[5] animals[3] weeds empty
//!        at the first step of every day: our farm's composition
//! Items (in this order): WHEAT CARROT TOMATO STRAWBERRY MELON EGG MILK WOOL FERTILIZER.
//! Everything is engine truth; the analysis marks which columns a player can observe.
use kagg_engine::engine::{self, PlayerAction};
use kagg_engine::json::{self, Json};
use kagg_engine::state::{Cell, State};
use kagg_engine::world;
use std::io::Write;

const ITEMS: [&str; 9] = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"];
const CROPS: [&str; 5] = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"];
const ANIMALS: [&str; 3] = ["GOOSE", "COW", "SHEEP"];

fn act_of(j: &Json) -> PlayerAction {
    if j.is_null() {
        return runner::to_engine(&agent::act::Action::pass());
    }
    runner::to_engine(&agent::act::Action::from_json(j))
}

fn farm_row(st: &State, s: usize) -> String {
    let f = &st.farms[s];
    let (mut crops, mut animals, mut weeds, mut empty) = ([0i64; 5], [0i64; 3], 0, 0);
    for row in &f.tiles {
        for c in row {
            match c {
                Cell::Plant { crop, .. } => {
                    if let Some(i) = CROPS.iter().position(|x| x == crop) {
                        crops[i] += 1;
                    }
                }
                Cell::Structure { animal: Some(a), .. } => {
                    if let Some(i) = ANIMALS.iter().position(|x| *x == a.animal) {
                        animals[i] += 1;
                    }
                }
                Cell::Weed => weeds += 1,
                Cell::Empty => empty += 1,
                _ => {}
            }
        }
    }
    let join = |v: &[i64]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join("\t");
    format!("{}\t{}\t{}\t{}\t{}\t{}", f.unlocked_quadrants.len(), f.hands.len(), join(&crops), join(&animals), weeds, empty)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let dir = get("--tapes").expect("--tapes DIR");
    let out = get("--out").expect("--out FILE");
    let only_tape_seat = get("--seats").as_deref() == Some("tape");
    let mut w = std::io::BufWriter::new(std::fs::File::create(&out).expect("out"));
    let mut files: Vec<_> = std::fs::read_dir(&dir).expect("dir").filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "json")).collect();
    files.sort();
    let (mut n, mut exact) = (0usize, 0usize);
    for f in files {
        let j = match json::parse(&std::fs::read_to_string(&f).unwrap()) {
            Ok(j) => j,
            Err(_) => continue,
        };
        let id = f.file_stem().unwrap().to_string_lossy().to_string();
        let seed = j.get("seed").i64();
        let tseat = j.get("seat").i64() as usize;
        let rewards: Vec<f64> = j.get("rewards").arr().iter().map(|x| x.f64()).collect();
        let acts = j.get("actions").arr();
        let seats: Vec<usize> = if only_tape_seat { vec![tseat] } else { vec![0, 1] };
        let stream = |s: usize| -> Vec<PlayerAction> { acts.iter().map(|p| p.arr().get(s).map(act_of).unwrap_or_else(|| act_of(&Json::Null))).collect() };
        let (a0, a1) = (stream(0), stream(1));
        let idle1 = world::idle_world(seed, 1).unwrap_or_default();
        let idle2 = world::idle_world(seed, 2).unwrap_or_default();
        let mut st = State::new(seed);
        let mut t = 0usize;
        loop {
            if st.step % 24 == 0 {
                for &s in &seats {
                    let _ = writeln!(w, "F\t{id}\t{s}\t{}\t{}", st.step / 24, farm_row(&st, s));
                }
            }
            let pair = [a0.get(t).cloned().unwrap_or_default(), a1.get(t).cloned().unwrap_or_default()];
            t += 1;
            let alive = engine::step(&mut st, &pair);
            for &s in &seats {
                let r = 1 - s;
                let mut line = format!("S\t{id}\t{s}\t{}\t{}\t{}", st.step, st.farms[s].money, st.farms[r].money);
                for who in [s, r] {
                    for it in ITEMS {
                        line.push_str(&format!("\t{}", st.private[who].shed.get(it)));
                    }
                }
                for it in ITEMS {
                    line.push_str(&format!("\t{}", st.market.inventory.get(it)));
                }
                for it in ITEMS {
                    line.push_str(&format!("\t{}", st.market.prices.get(it)));
                }
                let _ = writeln!(w, "{line}");
            }
            if !alive {
                break;
            }
        }
        let shops = &st.town.unlocked_shops;
        let w1 = world::world_key(shops, 1).unwrap_or_default();
        let w2 = world::world_key(shops, 2).unwrap_or_default();
        let (s0, s1) = (st.farms[0].money, st.farms[1].money);
        let (r0, r1) = (rewards.first().copied().unwrap_or(f64::NAN), rewards.get(1).copied().unwrap_or(f64::NAN));
        let ok = s0 == r0 && s1 == r1;
        exact += ok as usize;
        n += 1;
        let _ = writeln!(w, "W\t{id}\t{seed}\t{tseat}\t{w1}\t{w2}\t{idle1}\t{idle2}\t{r0}\t{r1}\t{s0}\t{s1}\t{}", ok as u8);
    }
    eprintln!("[trace] {n} tapes, {exact} exact -> {out}");
}
