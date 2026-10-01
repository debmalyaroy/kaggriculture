//! Tournament between EXTERNAL agent binaries (any `agent-stdio` build, ours or the main repo's) on our exact
//! engine: each game spawns one process per seat, feeds it the official per-seat observation JSON one line per
//! step (what a submission's main.py sends) and reads one action JSON line back.
//!
//!     stdio-match --agent "A=DIR::BIN ARGS..." --agent "B=DIR::BIN ARGS..." [--pairs all|A:B,A:A,...]
//!                 --bank data/worlds/w64_bank.json --split heldout --per-world 3 [--threads 24] --out F
//!
//!   DIR = the agent's working folder (its unpacked tarball stage: base/, profiles.json, policy.bin ...),
//!   BIN ARGS = the command main.py would run. --pairs all = every unordered pair including each agent vs its own
//!   clone. Every pair plays every seed from BOTH seats. Row: pair seed world seat_of_first bank_first bank_second.
use agent::act::Action;
use kagg_engine::engine::{self, PlayerAction};
use kagg_engine::json;
use kagg_engine::obsjson::seat_obs_json;
use kagg_engine::state::State;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};

struct Proc {
    child: Child,
    inp: ChildStdin,
    out: BufReader<ChildStdout>,
}

fn spawn(spec: &(String, String, Vec<String>)) -> Proc {
    let (dir, bin, args) = spec;
    let mut child = Command::new(bin).args(args).current_dir(dir).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().expect("spawn agent");
    let inp = child.stdin.take().unwrap();
    let out = BufReader::new(child.stdout.take().unwrap());
    Proc { child, inp, out }
}

fn ask(p: &mut Proc, obs: &str) -> PlayerAction {
    if writeln!(p.inp, "{obs}").and_then(|_| p.inp.flush()).is_err() {
        return runner::to_engine(&Action::pass());
    }
    let mut line = String::new();
    if p.out.read_line(&mut line).unwrap_or(0) == 0 {
        return runner::to_engine(&Action::pass());
    }
    match json::parse(line.trim()) {
        Ok(j) => runner::to_engine(&Action::from_json(&j)),
        Err(_) => runner::to_engine(&Action::pass()),
    }
}

fn game(seed: i64, a: &(String, String, Vec<String>), b: &(String, String, Vec<String>)) -> ([f64; 2], String) {
    let mut ps = [spawn(a), spawn(b)];
    let mut st = State::new(seed);
    loop {
        let acts: Vec<PlayerAction> = (0..2).map(|s| ask(&mut ps[s], &seat_obs_json(&st, s))).collect();
        let pair = [acts[0].clone(), acts[1].clone()];
        if !engine::step(&mut st, &pair) {
            break;
        }
    }
    for p in ps.iter_mut() {
        let _ = p.child.kill();
        let _ = p.child.wait();
    }
    let sh = &st.town.unlocked_shops;
    let world = format!("{}|{}", sh.first().map(|s| s.as_str()).unwrap_or("-"), sh.get(1).map(|s| s.as_str()).unwrap_or("-"));
    ([st.farms[0].money, st.farms[1].money], world)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let mut agents: Vec<(String, (String, String, Vec<String>))> = vec![];
    for (i, a) in args.iter().enumerate() {
        if a == "--agent" {
            let s = &args[i + 1];
            let (name, rest) = s.split_once('=').expect("--agent NAME=DIR::BIN ARGS");
            let (dir, cmd) = rest.split_once("::").expect("--agent NAME=DIR::BIN ARGS");
            let mut parts = cmd.split_whitespace().map(|x| x.to_string());
            let bin = parts.next().expect("binary");
            agents.push((name.to_string(), (dir.to_string(), bin, parts.collect())));
        }
    }
    let idx = |n: &str| agents.iter().position(|(m, _)| m == n).unwrap_or_else(|| panic!("unknown agent {n}"));
    let pairs: Vec<(usize, usize)> = match get("--pairs").as_deref() {
        None | Some("all") => (0..agents.len()).flat_map(|i| (i..agents.len()).map(move |j| (i, j))).collect(),
        Some(s) => s.split(',').map(|p| {
            let (x, y) = p.split_once(':').expect("A:B");
            (idx(x), idx(y))
        }).collect(),
    };
    let bank = json::parse(&std::fs::read_to_string(get("--bank").expect("--bank")).expect("bank")).expect("bank json");
    let per: usize = get("--per-world").and_then(|s| s.parse().ok()).unwrap_or(3);
    let mut seeds: Vec<(String, i64)> = vec![];
    for (w, ss) in bank.get(&get("--split").unwrap_or_else(|| "heldout".into())).obj() {
        for s in ss.arr().iter().take(per) {
            seeds.push((w.clone(), s.i64()));
        }
    }
    let threads: usize = get("--threads").and_then(|s| s.parse().ok()).unwrap_or(8);
    let mut jobs = vec![];
    for &(i, j) in &pairs {
        for (w, s) in &seeds {
            for first_seat in [0usize, 1] {
                jobs.push((i, j, *s, w.clone(), first_seat));
            }
        }
    }
    eprintln!("[stdio-match] {} agents, {} pairs, {} seeds, {} games", agents.len(), pairs.len(), seeds.len(), jobs.len());
    let q = Arc::new(Mutex::new(jobs));
    let agents = Arc::new(agents);
    let out = Arc::new(Mutex::new(std::io::BufWriter::new(std::fs::File::create(get("--out").expect("--out")).expect("out"))));
    let t0 = std::time::Instant::now();
    let done = Arc::new(Mutex::new(0usize));
    let hs: Vec<_> = (0..threads.max(1))
        .map(|_| {
            let (q, agents, out, done) = (q.clone(), agents.clone(), out.clone(), done.clone());
            std::thread::spawn(move || loop {
                let Some((i, j, seed, w, fs)) = q.lock().unwrap().pop() else { break };
                let (a, b) = (&agents[i], &agents[j]);
                let (seat0, seat1) = if fs == 0 { (&a.1, &b.1) } else { (&b.1, &a.1) };
                let (banks, world) = game(seed, seat0, seat1);
                let (bf, bs) = if fs == 0 { (banks[0], banks[1]) } else { (banks[1], banks[0]) };
                let mism = if world != w { "\tWORLD_MISMATCH" } else { "" };
                writeln!(out.lock().unwrap(), "{}:{}\t{seed}\t{world}\t{fs}\t{bf}\t{bs}{mism}", a.0, b.0).unwrap();
                let mut n = done.lock().unwrap();
                *n += 1;
                if *n % 200 == 0 {
                    eprintln!("[stdio-match] {} games ({:.0}s)", *n, t0.elapsed().as_secs_f64());
                }
            })
        })
        .collect();
    for h in hs {
        h.join().unwrap();
    }
    out.lock().unwrap().flush().unwrap();
    eprintln!("[stdio-match] done in {:.0}s", t0.elapsed().as_secs_f64());
}
