//! chassis-pick: from the chassis-scan table, the opening families and the most robust full farm tapes per world.
//!
//!     chassis-pick --scan data/chassis/c1_scan.tsv --out-dir data/chassis [--min-games 4] [--min-worlds 20]
//!
//! Columns read (by position): eid seat rating source date n world b6 b12 b18 b24 b30 opp30 farm24 farm72 farm144 farm_all.
//! Writes:
//!   families.tsv  per opening (farm144): games, worlds covered, mean / p25 final bank, win rate
//!   tapes.tsv     per (world, full farm tape): games, win rate, p25 / median final bank, mean d12, its opening
//!   chassis.tsv   for the top opening families: per world the full tape that shares the opening with the best
//!                 (p25 bank, win rate) -- the synced chassis candidate, one route per covered world
//! A "tape" here is a whole-game farm stream (farmer + hands) seen in several games against different opponents, so its
//! spread of banks measures how well it holds up (does not desync) against the field.
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, Write};

#[derive(Default, Clone)]
struct Agg {
    banks: Vec<f64>,
    wins: f64,
    d12: f64,
    worlds: HashSet<String>,
    open: String,
    ex: String,
}

impl Agg {
    fn add(&mut self, bank: f64, opp: f64, d12: f64, world: &str, open: &str, ex: &str) {
        self.banks.push(bank);
        self.wins += if bank > opp { 1.0 } else if bank == opp { 0.5 } else { 0.0 };
        self.d12 += d12;
        self.worlds.insert(world.to_string());
        if self.open.is_empty() {
            self.open = open.to_string();
            self.ex = ex.to_string();
        }
    }
    fn q(&self, p: f64) -> f64 {
        let mut v = self.banks.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[((v.len() - 1) as f64 * p).round() as usize]
    }
    fn n(&self) -> usize {
        self.banks.len()
    }
    fn mean(&self) -> f64 {
        self.banks.iter().sum::<f64>() / self.n() as f64
    }
    fn wr(&self) -> f64 {
        self.wins / self.n() as f64
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let scan = get("--scan").unwrap_or_else(|| "data/chassis/c1_scan.tsv".into());
    let dir = get("--out-dir").unwrap_or_else(|| "data/chassis".into());
    let min_games: usize = get("--min-games").and_then(|s| s.parse().ok()).unwrap_or(4);
    let min_worlds: usize = get("--min-worlds").and_then(|s| s.parse().ok()).unwrap_or(20);
    let top_fam: usize = get("--families").and_then(|s| s.parse().ok()).unwrap_or(5);
    let f = std::io::BufReader::new(std::fs::File::open(&scan).expect("scan"));
    let mut seen: HashSet<(String, String)> = HashSet::new();
    let mut fam: HashMap<String, Agg> = HashMap::new();
    let mut tape: HashMap<(String, String), Agg> = HashMap::new();
    let mut worlds: HashSet<String> = HashSet::new();
    for (i, l) in f.lines().enumerate() {
        let Ok(l) = l else { continue };
        if i == 0 {
            continue;
        }
        let x: Vec<&str> = l.split('\t').collect();
        if x.len() < 17 || x[6].contains('-') && x[6].starts_with('-') {
            continue;
        }
        if !seen.insert((x[0].to_string(), x[1].to_string())) {
            continue;
        }
        let (Ok(b30), Ok(o30), Ok(d12)) = (x[11].parse::<f64>(), x[12].parse::<f64>(), x[8].parse::<f64>()) else { continue };
        if !b30.is_finite() || !o30.is_finite() {
            continue;
        }
        let (world, open, full) = (x[6], x[15], x[16]);
        let ex = format!("{}_{}", x[0], x[1]);
        worlds.insert(world.to_string());
        fam.entry(open.to_string()).or_default().add(b30, o30, d12, world, open, &ex);
        tape.entry((world.to_string(), full.to_string())).or_default().add(b30, o30, d12, world, open, &ex);
    }
    std::fs::create_dir_all(&dir).ok();
    // opening families
    let mut fams: Vec<(&String, &Agg)> = fam.iter().filter(|(_, a)| a.n() >= min_games && a.worlds.len() >= min_worlds).collect();
    fams.sort_by(|a, b| b.1.q(0.25).partial_cmp(&a.1.q(0.25)).unwrap());
    let mut w = std::io::BufWriter::new(std::fs::File::create(format!("{dir}/families.tsv")).unwrap());
    writeln!(w, "farm144\tgames\tworlds\tmean_bank\tp25_bank\twin_rate\texample").unwrap();
    for (k, a) in &fams {
        writeln!(w, "{k}\t{}\t{}\t{:.0}\t{:.0}\t{:.3}\t{}", a.n(), a.worlds.len(), a.mean(), a.q(0.25), a.wr(), a.ex).unwrap();
    }
    // full tapes per world
    let mut tapes: Vec<(&(String, String), &Agg)> = tape.iter().filter(|(_, a)| a.n() >= min_games).collect();
    tapes.sort_by(|a, b| a.0 .0.cmp(&b.0 .0).then(b.1.q(0.25).partial_cmp(&a.1.q(0.25)).unwrap()));
    let mut w2 = std::io::BufWriter::new(std::fs::File::create(format!("{dir}/tapes.tsv")).unwrap());
    writeln!(w2, "world\tfarm_all\tgames\twin_rate\tp25_bank\tmedian_bank\tmean_d12\tfarm144\texample").unwrap();
    for ((wd, k), a) in &tapes {
        writeln!(w2, "{wd}\t{k}\t{}\t{:.3}\t{:.0}\t{:.0}\t{:.0}\t{}\t{}", a.n(), a.wr(), a.q(0.25), a.q(0.5), a.d12 / a.n() as f64, a.open, a.ex).unwrap();
    }
    // per top family and world: every game with that opening (count, median bank, win rate, the best game)
    {
        let mut per: HashMap<(String, String), Agg> = HashMap::new();
        let mut best: HashMap<(String, String), (f64, String)> = HashMap::new();
        let top: HashSet<&String> = fams.iter().take(top_fam * 4).map(|(k, _)| *k).collect();
        let f = std::io::BufReader::new(std::fs::File::open(&scan).expect("scan"));
        let mut seen2: HashSet<(String, String)> = HashSet::new();
        for (i, l) in f.lines().enumerate() {
            let Ok(l) = l else { continue };
            let x: Vec<&str> = l.split('\t').collect();
            if i == 0 || x.len() < 17 || !top.contains(&x[15].to_string()) || !seen2.insert((x[0].to_string(), x[1].to_string())) {
                continue;
            }
            let (Ok(b30), Ok(o30), Ok(d12)) = (x[11].parse::<f64>(), x[12].parse::<f64>(), x[8].parse::<f64>()) else { continue };
            let key = (x[15].to_string(), x[6].to_string());
            per.entry(key.clone()).or_default().add(b30, o30, d12, x[6], x[15], "");
            let e = best.entry(key).or_insert((f64::MIN, String::new()));
            if b30 > o30 && b30 > e.0 {
                *e = (b30, format!("{}_{}", x[0], x[1]));
            }
        }
        let mut w4 = std::io::BufWriter::new(std::fs::File::create(format!("{dir}/family_worlds.tsv")).unwrap());
        writeln!(w4, "family\tworld\tgames\twin_rate\tmedian_bank\tbest_win_bank\tbest_game").unwrap();
        let mut keys: Vec<&(String, String)> = per.keys().collect();
        keys.sort();
        for k in keys {
            let a = &per[k];
            let (bb, bg) = best.get(k).cloned().unwrap_or((f64::NAN, String::new()));
            writeln!(w4, "{}\t{}\t{}\t{:.3}\t{:.0}\t{:.0}\t{}", k.0, k.1, a.n(), a.wr(), a.q(0.5), bb, bg).unwrap();
        }
    }
    // synced chassis candidates: per top family, per world, the best full tape with that opening
    let mut w3 = std::io::BufWriter::new(std::fs::File::create(format!("{dir}/chassis.tsv")).unwrap());
    writeln!(w3, "family\tworld\tfarm_all\tgames\twin_rate\tp25_bank\tmedian_bank\texample").unwrap();
    eprintln!("[chassis-pick] {} worlds, {} opening families (>= {min_games} games, >= {min_worlds} worlds), {} robust tapes", worlds.len(), fams.len(), tapes.len());
    for (fk, fa) in fams.iter().take(top_fam) {
        let mut covered = 0;
        let mut wl: Vec<&String> = worlds.iter().collect();
        wl.sort();
        for wd in wl {
            let best = tapes
                .iter()
                .filter(|((w_, _), a)| w_ == wd && a.open == **fk)
                .max_by(|a, b| a.1.q(0.25).partial_cmp(&b.1.q(0.25)).unwrap().then(a.1.wr().partial_cmp(&b.1.wr()).unwrap()));
            if let Some(((_, k), a)) = best {
                covered += 1;
                writeln!(w3, "{fk}\t{wd}\t{k}\t{}\t{:.3}\t{:.0}\t{:.0}\t{}", a.n(), a.wr(), a.q(0.25), a.q(0.5), a.ex).unwrap();
            }
        }
        eprintln!("[chassis-pick] family {fk}: {} games, {} worlds, p25 {:.0}, win {:.3}; robust tapes cover {covered}/{} worlds", fa.n(), fa.worlds.len(), fa.q(0.25), fa.wr(), worlds.len());
    }
}
