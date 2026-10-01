//! Lineage table for reactive shell v2: every route of our library played on the engine, its farm
//! layout (occupied tiles with a product code, agent::rshell::layout) recorded at hour 12 of each day.
//! The shell matches the rival's farm against it to tell a same-lineage opponent (one of our public
//! routes, maybe a different one from ours) and forecast its sales from that route's tape.
//!
//!     lineage-table --base configs/bases/v61.1 --seeds 1,2 --out configs/lineage/v61.1.json
//!
//! Per route the layout of the FIRST seed is written; the other seeds report how stable it is.
use agent::base::Base;
use agent::obs::Obs;
use kagg_engine::engine;
use kagg_engine::state::State;

fn run(b: &Base, route: i64, seed: i64) -> Vec<Vec<(u16, u8)>> {
    let mut a = [b.fresh(), b.fresh()];
    a[0].chassis.router.force(route);
    let mut st = State::new(seed);
    let mut days: Vec<Vec<(u16, u8)>> = vec![vec![]; 30];
    loop {
        let mut acts = vec![];
        for (s, ag) in a.iter_mut().enumerate() {
            let o = Obs::from_state(&st, s);
            let step = o.step();
            if s == 0 && step % 24 == 12 {
                if let Ok(v) = agent::view::View::new(o.clone()) {
                    days[(step / 24) as usize] = agent::rshell::layout(v.farm());
                }
            }
            acts.push(runner::to_engine(&ag.act(o).0));
        }
        let pair = [acts.remove(0), acts.remove(0)];
        if !engine::step(&mut st, &pair) {
            break;
        }
    }
    days
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let base_dir = get("--base").unwrap_or_else(|| "configs/bases/v61.1".into());
    let out = get("--out").expect("--out FILE");
    let seeds: Vec<i64> = get("--seeds").unwrap_or_else(|| "1,2".into()).split(',').filter_map(|x| x.trim().parse().ok()).collect();
    let b = Base::load(&base_dir).expect("base");
    let ids: Vec<i64> = b.chassis.routes.iter().map(|r| r.id).collect();
    let hs: Vec<_> = ids
        .iter()
        .map(|&r| {
            let (b, seeds) = (b.fresh(), seeds.clone());
            std::thread::spawn(move || {
                let runs: Vec<Vec<Vec<(u16, u8)>>> = seeds.iter().map(|s| run(&b, r, *s)).collect();
                (r, runs)
            })
        })
        .collect();
    let mut body = vec![];
    for h in hs {
        let (r, runs) = h.join().unwrap();
        let first = &runs[0];
        // stability: share of (day, tile, code) of the first seed present in every other seed
        let mut same = 0usize;
        let mut tot = 0usize;
        for (d, lay) in first.iter().enumerate() {
            for e in lay {
                tot += 1;
                if runs[1..].iter().all(|o| o[d].contains(e)) {
                    same += 1;
                }
            }
        }
        eprintln!("[lineage] route {r}: {} tiles by day 29, stable {:.3}", first[29].len(), if tot > 0 { same as f64 / tot as f64 } else { 1.0 });
        let days: Vec<String> = first.iter().map(|lay| format!("[{}]", lay.iter().map(|(t, c)| format!("[{t},{c}]")).collect::<Vec<_>>().join(","))).collect();
        body.push(format!("\"{r}\": [{}]", days.join(",")));
    }
    if let Some(dir) = std::path::Path::new(&out).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(&out, format!("{{\"base\": \"{base_dir}\", \"seeds\": {seeds:?}, \"routes\": {{{}}}}}", body.join(", "))).expect("write");
    eprintln!("[lineage] {} routes -> {out}", ids.len());
}
