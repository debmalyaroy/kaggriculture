//! Top-player turn dataset for the TPP policy (crates/agent/src/tpp.rs): replays every top-player game
//! EXACTLY (both recorded streams on the engine seed) and writes the top player's encoded observation and
//! full action (every unit + every market order) at a sample of turns.
//!
//!     bcdump --tapes DIR[,DIR...] --out DIR [--every 6] [--threads 8]
//!
//! Tapes: python/gm_top_tapes.py / data/tapes/leaders (`seat` = the OTHER seat; the top player sits at
//! 1 - seat). Out: part_<thread>.bin (fixed-size records, layout below) + games_<thread>.tsv
//! (game index, tape id, top seat, top result 1/0.5/0, top margin, opp_rating).
//! Record (little endian): u32 game | u16 step | u8 n_units | u8 n_mkt | f32 result | f32 margin/1e4 |
//!   board C*B*B u8 | glob G f32 | units MAXU*UF u8 | unit labels MAXU*(cls u8, qb u8) | market MAXM*(id u8, qb u8)
use agent::act::Action;
use agent::tpp;
use kagg_engine::engine::{self, PlayerAction};
use kagg_engine::json;
use kagg_engine::state::State;
use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).map(|i| args[i + 1].clone());
    let dirs = get("--tapes").expect("--tapes DIR[,DIR]");
    let out = get("--out").expect("--out DIR");
    let every: usize = get("--every").and_then(|s| s.parse().ok()).unwrap_or(6);
    let threads: usize = get("--threads").and_then(|s| s.parse().ok()).unwrap_or(8);
    std::fs::create_dir_all(&out).unwrap();
    fn walk(p: &std::path::Path, v: &mut Vec<std::path::PathBuf>) {
        if let Ok(rd) = std::fs::read_dir(p) {
            for e in rd.flatten() {
                let q = e.path();
                if q.is_dir() {
                    walk(&q, v);
                } else if q.extension().is_some_and(|x| x == "json") {
                    v.push(q);
                }
            }
        }
    }
    let mut files = vec![];
    for d in dirs.split(',').filter(|d| !d.is_empty()) {
        walk(std::path::Path::new(d), &mut files);
    }
    files.sort();
    eprintln!("[bcdump] {} tapes, every {every} turns, {threads} threads", files.len());
    let files = std::sync::Arc::new(files);
    let hs: Vec<_> = (0..threads)
        .map(|w| {
            let files = files.clone();
            let out = out.clone();
            std::thread::spawn(move || {
                let mut bin = std::io::BufWriter::new(std::fs::File::create(format!("{out}/part_{w:02}.bin")).unwrap());
                let mut idx = std::io::BufWriter::new(std::fs::File::create(format!("{out}/games_{w:02}.tsv")).unwrap());
                let (mut ng, mut nr) = (0usize, 0usize);
                for (gi, f) in files.iter().enumerate().filter(|(i, _)| i % threads == w) {
                    let Ok(txt) = std::fs::read_to_string(f) else { continue };
                    let Ok(j) = json::parse(&txt) else { continue };
                    let seed = j.get("seed").i64();
                    let top = 1 - j.get("seat").i64() as usize;
                    let acts = j.get("actions").arr();
                    if acts.len() < 700 {
                        continue;
                    }
                    let rw: Vec<f64> = j.get("rewards").arr().iter().map(|x| x.f64()).collect();
                    let (a, b) = (rw.get(top).copied().unwrap_or(0.0), rw.get(1 - top).copied().unwrap_or(0.0));
                    let res: f32 = if a > b { 1.0 } else if a == b { 0.5 } else { 0.0 };
                    let margin = ((a - b) / 1e4) as f32;
                    let orat = j.get("opp_rating");
                    let orat = if orat.is_null() { 0.0 } else { orat.f64() };
                    let id = f.file_stem().unwrap().to_string_lossy().to_string();
                    let mut st = State::new(seed);
                    let off = (gi * 7) % every;
                    let mut t = 0usize;
                    loop {
                        let pair = acts.get(t).map(|p| p.arr()).unwrap_or(&[]);
                        let act = |s: usize| pair.get(s).filter(|x| !x.is_null()).map(Action::from_json).unwrap_or_else(Action::pass);
                        if t % every == off && t < 719 {
                            let o = agent::obs::Obs::from_state(&st, top);
                            let e = tpp::encode(&o);
                            let (ul, ml) = tpp::labels(&o, &act(top));
                            let mut r: Vec<u8> = Vec::with_capacity(4096);
                            r.extend_from_slice(&(gi as u32).to_le_bytes());
                            r.extend_from_slice(&(t as u16).to_le_bytes());
                            r.push(e.units.len() as u8);
                            r.push(ml.len() as u8);
                            r.extend_from_slice(&res.to_le_bytes());
                            r.extend_from_slice(&margin.to_le_bytes());
                            r.extend_from_slice(&e.board);
                            for v in e.glob {
                                r.extend_from_slice(&v.to_le_bytes());
                            }
                            for i in 0..tpp::MAXU {
                                r.extend_from_slice(e.units.get(i).unwrap_or(&[0u8; tpp::UF]));
                            }
                            for i in 0..tpp::MAXU {
                                let (c, q) = ul.get(i).copied().unwrap_or((255, 0));
                                r.push(c);
                                r.push(q);
                            }
                            for i in 0..tpp::MAXM {
                                let (c, q) = ml.get(i).copied().unwrap_or((255, 0));
                                r.push(c);
                                r.push(q);
                            }
                            bin.write_all(&r).unwrap();
                            nr += 1;
                        }
                        let pa: [PlayerAction; 2] = [runner::to_engine(&act(0)), runner::to_engine(&act(1))];
                        t += 1;
                        if !engine::step(&mut st, &pa) {
                            break;
                        }
                    }
                    let exact = st.farms[top].money == a && st.farms[1 - top].money == b;
                    writeln!(idx, "{gi}\t{id}\t{top}\t{res}\t{margin}\t{orat}\t{}", exact as u8).unwrap();
                    ng += 1;
                }
                (ng, nr)
            })
        })
        .collect();
    let (mut ng, mut nr) = (0, 0);
    for h in hs {
        let (a, b) = h.join().unwrap();
        ng += a;
        nr += b;
    }
    let rec = 16 + tpp::C * tpp::B * tpp::B + tpp::G * 4 + tpp::MAXU * tpp::UF + tpp::MAXU * 2 + tpp::MAXM * 2;
    eprintln!("[bcdump] {ng} games, {nr} records of {rec} bytes -> {out}");
}
