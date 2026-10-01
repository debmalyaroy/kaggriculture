//! Our agent vs a REACTIVE top-player model built from that player's own ladder tapes.
//!
//! `tapeplay` replays one recorded opponent tape open-loop: once our play differs from the
//! recorded game, the opponent keeps doing what it did against someone else. Here the opponent
//! (the "top player" X) has a LIBRARY: every recorded game of X (tapes from
//! `python -m kaggriculture.bandit.top_field`). Each library game is simulated once to get X's
//! situation at every step. During play, X starts on the game's own tape and, each step, may switch
//! to the library game whose recorded situation best matches the live one:
//!   * eligible: same shops unlocked so far (same world so far) AND X's units on exactly the same
//!     squares (so X's recorded moves stay valid);
//!   * distance over the last `--win` steps: money (X and rival), X's shed, both boards' crop and
//!     animal counts, and how many rival units stand on X's squares (the copy signal);
//!   * switch only when the best is clearly better than the current tape (`--gain`).
//!
//!     fieldplay --tapes DIR --libs libs.tsv --profiles P.json --pa 71 [--base B] [--threads 8] [--open]
//!
//! `--seller perfect|lag` (default off) adds the REACTIVE SELLER: X keeps its recorded farm moves and
//! buys, and front-runs our sales. Each product we sell that X holds (not WHEAT = feed, not
//! FERTILIZER) is sold by X at the head of its order queue, so X's units clear before ours in the
//! per-slot market race. `perfect` sees our orders this step (a flawless forecaster, worst case for
//! us); `lag` reacts to the sales it saw us make last step (market stock rise minus its own sales).
//!
//! `--worlds S1,S2,..` = the LAYER mode: every library (player) plays every listed world seed from
//! both seats (not only its recorded games), starting on its most common opening tape. Switching
//! then only needs X's units on the same squares; a tape recorded in a different world (other shops
//! unlocked so far) pays a distance penalty per step instead of being barred.
//! Output (worlds mode): `library<TAB>seed<TAB>our_seat<TAB>us<TAB>them<TAB>switches<TAB>own_steps<TAB>lib_size<TAB>fronts`.
//!
//! libs.tsv: `key<TAB>library` (key = tape file stem; same library = same player).
//! `--open` = no switching (must equal tapeplay). Output per tape:
//! `key<TAB>our_seat<TAB>us<TAB>them<TAB>switches<TAB>own_steps<TAB>lib_size`.
use agent::act::Action;
use kagg_engine::engine::{self, PlayerAction};
use kagg_engine::json::{self, Json};
use kagg_engine::obsjson::seat_obs_json;
use kagg_engine::state::{Cell, State, ANIMAL_NAMES, CROP_NAMES, PRODUCTS};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// distance added per window step when a tape's world (shops unlocked so far) differs from the live one
const WORLD_PEN: f32 = 20.0;
const NF: usize = 2 + 9 + 5 + 3 + 5 + 3 + 1;

fn act_of(j: &Json) -> PlayerAction {
    if j.is_null() {
        return runner::to_engine(&Action::pass());
    }
    runner::to_engine(&Action::from_json(j))
}

/// X's situation at the current state (X = seat `x`).
fn feat(st: &State, x: usize) -> ([f32; NF], u64, u64) {
    let mut f = [0f32; NF];
    let o = 1 - x;
    f[0] = (st.farms[x].money / 1000.0) as f32;
    f[1] = (st.farms[o].money / 1000.0) as f32;
    for (i, p) in PRODUCTS.iter().enumerate() {
        f[2 + i] = st.private[x].shed.get(p) as f32 * 0.25;
    }
    for (side, off) in [(x, 11usize), (o, 19usize)] {
        for row in &st.farms[side].tiles {
            for c in row {
                match c {
                    Cell::Plant { crop, .. } => {
                        if let Some(i) = CROP_NAMES.iter().position(|n| n == crop) {
                            f[off + i] += 0.5;
                        }
                    }
                    Cell::Structure { animal: Some(a), .. } => {
                        if let Some(i) = ANIMAL_NAMES.iter().position(|n| *n == a.animal) {
                            f[off + 5 + i] += 1.0;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    let mine: Vec<(i64, i64)> = std::iter::once(st.farms[x].farmer).chain(st.farms[x].hands.iter().copied()).collect();
    let theirs: Vec<(i64, i64)> = std::iter::once(st.farms[o].farmer).chain(st.farms[o].hands.iter().copied()).collect();
    f[NF - 1] = theirs.iter().filter(|p| mine.contains(p)).count() as f32 * 2.0;
    let mut h: u64 = 1469598103934665603;
    for (a, b) in &mine {
        h = (h ^ (*a as u64 * 31 + *b as u64 + 7)).wrapping_mul(1099511628211);
    }
    let mut w: u64 = 1469598103934665603;
    for s in &st.town.unlocked_shops {
        for c in s.bytes() {
            w = (w ^ c as u64).wrapping_mul(1099511628211);
        }
        w = (w ^ 0xff).wrapping_mul(1099511628211);
    }
    (f, h, w)
}

struct Game {
    key: String,
    seed: i64,
    our: usize,
    x_acts: Vec<PlayerAction>,
    /// per step (before the step's actions): X's features, position hash, world hash
    f: Vec<[f32; NF]>,
    pos: Vec<u64>,
    world: Vec<u64>,
}

fn load(path: &std::path::Path) -> Option<Game> {
    let j = json::parse(&std::fs::read_to_string(path).ok()?).ok()?;
    let seed = j.get("seed").i64();
    let our = j.get("seat").i64() as usize;
    let x = 1 - our;
    let acts = j.get("actions").arr();
    let tape = |t: usize, s: usize| acts.get(t).and_then(|p| p.arr().get(s)).map(act_of).unwrap_or_else(|| act_of(&Json::Null));
    let mut st = State::new(seed);
    let (mut f, mut pos, mut world, mut x_acts) = (vec![], vec![], vec![], vec![]);
    let mut t = 0;
    loop {
        let (ff, h, w) = feat(&st, x);
        f.push(ff);
        pos.push(h);
        world.push(w);
        let a = [tape(t, 0), tape(t, 1)];
        x_acts.push(a[x].clone());
        t += 1;
        if !engine::step(&mut st, &a) {
            break;
        }
    }
    Some(Game { key: path.file_stem()?.to_string_lossy().into(), seed, our, x_acts, f, pos, world })
}

fn dist(a: &[f32; NF], b: &[f32; NF]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).map(|i| args[i + 1].clone());
    let dir = get("--tapes").expect("--tapes DIR");
    let threads: usize = get("--threads").and_then(|s| s.parse().ok()).unwrap_or(4);
    let win: usize = get("--win").and_then(|s| s.parse().ok()).unwrap_or(12);
    let gain: f32 = get("--gain").and_then(|s| s.parse().ok()).unwrap_or(0.7);
    let open = args.iter().any(|a| a == "--open");
    let seller = get("--seller").unwrap_or_default();
    assert!(matches!(seller.as_str(), "" | "perfect" | "lag"), "--seller perfect|lag");
    let only: Option<Vec<String>> = get("--only").map(|s| s.split(',').map(|x| x.to_string()).collect());
    let mut base = agent::base::Base::load(&get("--base").unwrap_or_else(|| "configs/bandit/bases/v61.1".into())).expect("base");
    if let Some(p) = get("--profiles") {
        base.set_profiles(&p, get("--pa").and_then(|s| s.parse().ok())).expect("profiles");
    }
    let lib_of: HashMap<String, String> = get("--libs")
        .map(|p| {
            std::fs::read_to_string(p)
                .expect("libs")
                .lines()
                .filter_map(|l| l.split_once('\t').map(|(k, v)| (k.to_string(), v.to_string())))
                .collect()
        })
        .unwrap_or_default();
    let mut files: Vec<_> = std::fs::read_dir(&dir).expect("dir").filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "json")).collect();
    files.sort();
    // load + pre-simulate every tape (parallel)
    let q = Arc::new(Mutex::new(files));
    let games = Arc::new(Mutex::new(Vec::new()));
    std::thread::scope(|s| {
        for _ in 0..threads {
            let (q, games) = (q.clone(), games.clone());
            s.spawn(move || loop {
                let Some(p) = q.lock().unwrap().pop() else { break };
                if let Some(g) = load(&p) {
                    games.lock().unwrap().push(g);
                }
            });
        }
    });
    let mut games = Arc::try_unwrap(games).ok().unwrap().into_inner().unwrap();
    games.sort_by(|a, b| a.key.cmp(&b.key));
    let games = Arc::new(games);
    let mut libs: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, g) in games.iter().enumerate() {
        libs.entry(lib_of.get(&g.key).cloned().unwrap_or_else(|| g.key.clone())).or_default().push(i);
    }
    let libs = Arc::new(libs);
    eprintln!("[fieldplay] {} tapes, {} libraries; win {win} gain {gain} open {open}", games.len(), libs.len());
    // jobs: (label, library, start tape, seed, our seat)
    let worlds: Option<Vec<i64>> = get("--worlds").map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect());
    let lib_name = |k: &str| lib_of.get(k).cloned().unwrap_or_else(|| k.to_string());
    let todo: Vec<(String, String, usize, i64, usize)> = match &worlds {
        None => (0..games.len())
            .filter(|&i| only.as_ref().is_none_or(|o| o.contains(&games[i].key)))
            .map(|i| (games[i].key.clone(), lib_name(&games[i].key), i, games[i].seed, games[i].our))
            .collect(),
        Some(ws) => {
            let mut v = vec![];
            let mut names: Vec<&String> = libs.keys().collect();
            names.sort();
            for name in names {
                let lib = &libs[name];
                // most common opening: the tape sharing its day-1 (step 24) unit squares with the most library-mates
                let at = |k: usize| games[k].pos.get(24).copied().unwrap_or(0);
                let start = *lib.iter().max_by_key(|&&k| (lib.iter().filter(|&&j| at(j) == at(k)).count(), std::cmp::Reverse(k))).unwrap();
                for &w in ws {
                    for our in 0..2 {
                        v.push((name.clone(), name.clone(), start, w, our));
                    }
                }
            }
            v
        }
    };
    let q = Arc::new(Mutex::new(todo));
    let out = Arc::new(Mutex::new(Vec::new()));
    std::thread::scope(|s| {
        for _ in 0..threads {
            let (q, out, games, libs) = (q.clone(), out.clone(), games.clone(), libs.clone());
            let worlds_mode = worlds.is_some();
            let base = &base;
            let seller = &seller;
            s.spawn(move || loop {
                let Some((label, lname, gi, seed, our)) = q.lock().unwrap().pop() else { break };
                let lib = &libs[&lname];
                let x = 1 - our;
                let mut me = base.fresh();
                let mut st = State::new(seed);
                let (mut cur, mut switches, mut own, mut t) = (gi, 0usize, 0usize, 0usize);
                // tape position: library game `cur` is followed at its own step index `t` (same clock)
                let mut hist: Vec<[f32; NF]> = Vec::with_capacity(720);
                // lag seller: what we sold last step, per product
                let mut our_sold_prev = [0i64; 9];
                let mut fronts = 0usize;
                loop {
                    let (f, h, w) = feat(&st, x);
                    hist.push(f);
                    if !open && t > 0 {
                        let lo = t.saturating_sub(win - 1);
                        let d = |k: usize| -> f32 {
                            let gk = &games[k];
                            (lo..=t).map(|u| gk.f.get(u).map_or(1e6, |r| dist(r, &hist[u]))).sum::<f32>()
                                + if gk.world.get(t) == Some(&w) { 0.0 } else { WORLD_PEN * (t + 1 - lo) as f32 }
                        };
                        let ok = |k: usize| games[k].pos.get(t) == Some(&h) && (worlds_mode || games[k].world.get(t) == Some(&w));
                        let dc = if ok(cur) { d(cur) } else { f32::INFINITY };
                        let mut best = (dc, cur);
                        for &k in lib {
                            if k != cur && ok(k) {
                                let dk = d(k);
                                if dk < best.0 {
                                    best = (dk, k);
                                }
                            }
                        }
                        if best.1 != cur && (dc.is_infinite() || best.0 < dc * gain) {
                            cur = best.1;
                            switches += 1;
                        }
                    }
                    own += (cur == gi) as usize;
                    let mut theirs = games[cur].x_acts.get(t).cloned().unwrap_or_else(|| act_of(&Json::Null));
                    let text = seat_obs_json(&st, our);
                    let ours = match agent::obs::Obs::parse(&text) {
                        Ok(o) => runner::to_engine(&me.act(o).0),
                        Err(_) => runner::to_engine(&Action::pass()),
                    };
                    if !seller.is_empty() {
                        let mut want = [false; 9];
                        if seller == "perfect" {
                            for o in &ours.market {
                                if o.first().is_some_and(|x| x == "SELL") {
                                    if let Some(i) = o.get(1).and_then(|p| PRODUCTS.iter().position(|q| q == p)) {
                                        want[i] = true;
                                    }
                                }
                            }
                        } else {
                            for i in 0..9 {
                                want[i] = our_sold_prev[i] > 0;
                            }
                        }
                        let mut front: Vec<Vec<String>> = vec![];
                        for (i, p) in PRODUCTS.iter().enumerate() {
                            let have = st.private[x].shed.get(p);
                            if want[i] && have > 0 && *p != "WHEAT" && *p != "FERTILIZER" {
                                front.push(vec!["SELL".into(), p.to_string(), have.to_string()]);
                            }
                        }
                        if !front.is_empty() {
                            fronts += 1;
                            let sold: Vec<String> = front.iter().map(|o| o[1].clone()).collect();
                            let rest = theirs.market.drain(..).filter(|o| !(o.first().is_some_and(|x| x == "SELL") && o.get(1).is_some_and(|p| sold.contains(p))));
                            front.extend(rest);
                            front.truncate(10);
                            theirs.market = front;
                        }
                    }
                    let (inv0, shed0): (Vec<i64>, Vec<i64>) = (PRODUCTS.iter().map(|p| st.market.inventory.get(p)).collect(), PRODUCTS.iter().map(|p| st.private[x].shed.get(p)).collect());
                    let a: [PlayerAction; 2] = if our == 0 { [ours, theirs] } else { [theirs, ours] };
                    t += 1;
                    let alive = engine::step(&mut st, &a);
                    // market stock rise = units sold at > $1 by both seats; minus X's own shed drop from selling
                    for (i, p) in PRODUCTS.iter().enumerate() {
                        let rise = st.market.inventory.get(p) - inv0[i];
                        let x_sold = (shed0[i] - st.private[x].shed.get(p)).max(0);
                        our_sold_prev[i] = (rise - x_sold).max(0);
                    }
                    if !alive {
                        break;
                    }
                }
                let label = if worlds_mode { format!("{label}\t{seed}") } else { label };
                let line = format!("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}", label, our, st.farms[our].money, st.farms[x].money, switches, own, lib.len(), fronts);
                out.lock().unwrap().push(line);
            });
        }
    });
    let mut out = Arc::try_unwrap(out).ok().unwrap().into_inner().unwrap();
    out.sort();
    for l in &out {
        println!("{l}");
    }
}
