//! Build a TEAM BASE (a bandit whose routes are one player's recorded tapes, routed by the trie
//! router, `agent/src/trie.rs`) from GM tapes (`python -m kaggriculture.bandit.top_field`).
//!
//!     teambase --tapes DIR --keys keys.txt --out DIR [--id0 1000] [--folds N --fold I] [--threads 8]
//!
//! keys.txt: one tape key per line (`<episode>_<seat>`, the seat the player sat in). Each game is
//! replayed exactly (both recorded sides) to record the player's situation (`trie::feats`, from its
//! own observation) at every step, its realized world (first two shops at step 144) and result.
//! The branch steps of every route are found by walking the trie along the route's own path, and
//! the situation is stored only there. Writes `<out>/S0` (tapes only: cut "chassis", weed repair
//! and hand alignment on) and `<out>/S1` (the same tapes under the full shell: cut "full", the
//! v61.1 chassis settings). With `--folds N --fold I`, keys with index % N == I are held out
//! (listed in `<out>/heldout.txt`).
use agent::act::Action;
use agent::trie::{feats, unit_key, K};
use kagg_engine::engine::{self, PlayerAction};
use kagg_engine::json;
use kagg_engine::obsjson::seat_obs_json;
use kagg_engine::state::State;
use std::sync::{Arc, Mutex};

struct Rec {
    key: String,
    seed: i64,
    acts: Vec<Action>,
    keys: Vec<u64>,
    feats: Vec<[f32; K]>,
    world: String,
    win: f32,
}

fn eng(a: &Action) -> PlayerAction {
    runner::to_engine(a)
}

fn record(dir: &str, key: &str) -> Option<Rec> {
    let j = json::parse(&std::fs::read_to_string(format!("{dir}/{key}.json")).ok()?).ok()?;
    let seat: usize = key.rsplit('_').next()?.parse().ok()?;
    let seed = j.get("seed").i64();
    let raw = j.get("actions").arr();
    let side = |t: usize, s: usize| -> Action {
        match raw.get(t).and_then(|p| p.arr().get(s)) {
            Some(a) if a.is_obj() => Action::from_json(a),
            _ => Action::pass(),
        }
    };
    let mut st = State::new(seed);
    let (mut acts, mut keys, mut fs) = (vec![], vec![], vec![]);
    let mut world = String::new();
    let mut t = 0usize;
    loop {
        let f = agent::obs::Obs::parse(&seat_obs_json(&st, seat)).ok().and_then(|o| agent::view::View::new(o).ok()).map(|v| feats(&v)).unwrap_or([0.0; K]);
        fs.push(f);
        if t == 145 {
            world = st.town.unlocked_shops.iter().take(2).cloned().collect::<Vec<_>>().join("|");
        }
        let (a0, a1) = (side(t, 0), side(t, 1));
        let mine = if seat == 0 { a0.clone() } else { a1.clone() };
        keys.push(unit_key(&mine));
        acts.push(mine);
        t += 1;
        if !engine::step(&mut st, &[eng(&a0), eng(&a1)]) {
            break;
        }
    }
    let (us, them) = (st.farms[seat].money, st.farms[1 - seat].money);
    Some(Rec { key: key.to_string(), seed, acts, keys, feats: fs, world, win: if us > them { 1.0 } else if us < them { 0.0 } else { 0.5 } })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).map(|i| args[i + 1].clone());
    let dir = get("--tapes").expect("--tapes DIR");
    let out = get("--out").expect("--out DIR");
    let id0: i64 = get("--id0").and_then(|s| s.parse().ok()).unwrap_or(1000);
    let threads: usize = get("--threads").and_then(|s| s.parse().ok()).unwrap_or(8);
    let folds: usize = get("--folds").and_then(|s| s.parse().ok()).unwrap_or(0);
    let fold: usize = get("--fold").and_then(|s| s.parse().ok()).unwrap_or(0);
    let all: Vec<String> = std::fs::read_to_string(get("--keys").expect("--keys FILE")).expect("keys").lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect();
    let (keep, held): (Vec<(usize, String)>, Vec<(usize, String)>) = all.into_iter().enumerate().partition(|(i, _)| folds == 0 || i % folds != fold);
    let q = Arc::new(Mutex::new(keep.into_iter().map(|x| x.1).collect::<Vec<_>>()));
    let recs = Arc::new(Mutex::new(vec![]));
    std::thread::scope(|s| {
        for _ in 0..threads {
            let (q, recs, dir) = (q.clone(), recs.clone(), &dir);
            s.spawn(move || loop {
                let Some(k) = q.lock().unwrap().pop() else { break };
                if let Some(r) = record(dir, &k) {
                    recs.lock().unwrap().push(r);
                }
            });
        }
    });
    let mut recs = Arc::try_unwrap(recs).ok().unwrap().into_inner().unwrap();
    recs.sort_by(|a, b| a.key.cmp(&b.key));
    let n = recs.len();
    // branch steps along each route's own path
    let mut splits: Vec<Vec<usize>> = vec![vec![]; n];
    let mut nbranch = 0usize;
    for r in 0..n {
        let mut cands: Vec<usize> = (0..n).collect();
        for t in 0..720 {
            let kr = recs[r].keys.get(t).copied().unwrap_or(0);
            let distinct = cands.iter().map(|&k| recs[k].keys.get(t).copied().unwrap_or(0)).collect::<std::collections::BTreeSet<_>>();
            if distinct.len() > 1 {
                splits[r].push(t);
                cands.retain(|&k| recs[k].keys.get(t).copied().unwrap_or(0) == kr);
            }
        }
        nbranch += splits[r].len();
    }
    let mut routes = String::from("{");
    let mut troutes = vec![];
    for (i, r) in recs.iter().enumerate() {
        let id = id0 + i as i64;
        if i > 0 {
            routes.push_str(",\n");
        }
        routes.push_str(&format!("\"{id}\": ["));
        for (t, a) in r.acts.iter().take(719).enumerate() {
            if t > 0 {
                routes.push(',');
            }
            routes.push_str(&a.dump());
        }
        routes.push(']');
        let sp: Vec<String> = splits[i]
            .iter()
            .map(|&t| format!("\"{t}\": [{}]", r.feats[t].iter().map(|x| format!("{:.3}", x)).collect::<Vec<_>>().join(",")))
            .collect();
        troutes.push(format!("{{\"id\": {id}, \"key\": \"{}\", \"seed\": {}, \"world\": \"{}\", \"win\": {}, \"splits\": {{{}}}}}", r.key, r.seed, r.world, r.win, sp.join(", ")));
    }
    routes.push('}');
    let trie = format!("{{\"world_pen\": 20.0, \"routes\": [\n{}\n]}}", troutes.join(",\n"));
    let s0 = r#"{"hand_align": true, "weed_repair": true, "sell_lead": false, "budget_guard": false, "room_guard": false, "clamp_sells": false, "dead_stock": false, "terminal_liquidation": false, "front_run": false, "r36": false, "racepx": false}"#;
    let s1 = r#"{"hand_align": true, "weed_repair": true, "sell_lead": true, "budget_guard": false, "room_guard": false, "clamp_sells": false, "dead_stock": false, "terminal_liquidation": false, "front_run": false, "r36": true, "r36_lead_from": 288, "r36_lead_to": 696, "racepx": true, "racepx_margin": 0}"#;
    for (name, cut, settings) in [("S0", "chassis", s0), ("S1", "full", s1)] {
        let d = format!("{out}/{name}");
        std::fs::create_dir_all(&d).expect("mkdir");
        std::fs::write(format!("{d}/routes.json"), &routes).expect("routes");
        let router = format!("{{\"mode\": \"trie\", \"cut\": \"{cut}\", \"settings\": {settings}, \"endgame_step\": 99999, \"trie\": {trie}}}");
        std::fs::write(format!("{d}/router.json"), router).expect("router");
    }
    std::fs::write(format!("{out}/heldout.txt"), held.iter().map(|x| format!("{}\n", x.1)).collect::<String>()).expect("heldout");
    let wins = recs.iter().map(|r| r.win).sum::<f32>();
    let worlds = recs.iter().map(|r| r.world.as_str()).collect::<std::collections::BTreeSet<_>>().len();
    eprintln!("[teambase] {out}: {n} routes, {worlds} worlds, win {:.3}, branch points per route {:.1}, held out {}", wins / n.max(1) as f32, nbranch as f32 / n.max(1) as f32, held.len());
}
