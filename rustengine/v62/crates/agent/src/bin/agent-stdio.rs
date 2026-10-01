//! Line protocol: one observation JSON per stdin line -> one action JSON per stdout line.
//! A line `{"cmd": "swap_base", "dir": "..."}` hot-swaps the base (answers `{"ok": true}`);
//! `{"cmd": "swap_profiles", "path": "...", "profile": N}` hot-swaps the profile table + dispatch sets.
//! With `--timing`, the per-turn latency summary is printed to stderr at EOF.
//!
//!     agent-stdio --base configs/bases/v61.1 [--timing]
//!     agent-stdio --base ... --profiles P --sched 19,19,...,10   (per-day profile list; the last repeats)
//!     agent-stdio --base ... --profiles P --group 19,19,19 --jitter 0,0,1   (opponent-group controller)
use std::io::{BufRead, BufWriter, Write};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut base = "configs/bases/v61.1".to_string();
    let mut timing = false;
    let mut cut: Option<String> = None;
    let mut profiles: Option<String> = None;
    let mut profile: Option<usize> = None;
    let mut clone_profile: Option<usize> = None;
    let mut clone_strict = false;
    let mut sched: Option<String> = None;
    let mut group: Option<String> = None;
    let mut jitter: Option<String> = None;
    let mut endgame: Option<String> = None;
    let mut jitter_end: Option<String> = None;
    let mut mirror_tol: Option<f64> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--base" => {
                base = args[i + 1].clone();
                i += 1;
            }
            "--timing" => timing = true,
            "--clone-strict" => clone_strict = true,
            "--profiles" => {
                profiles = Some(args[i + 1].clone());
                i += 1;
            }
            "--clone-profile" => {
                clone_profile = args[i + 1].parse().ok();
                i += 1;
            }
            "--profile" => {
                profile = args[i + 1].parse().ok();
                i += 1;
            }
            "--sched" => {
                sched = Some(args[i + 1].clone());
                i += 1;
            }
            "--group" => {
                group = Some(args[i + 1].clone());
                i += 1;
            }
            "--jitter" => {
                jitter = Some(args[i + 1].clone());
                i += 1;
            }
            "--endgame" => {
                endgame = Some(args[i + 1].clone());
                i += 1;
            }
            "--jitter-end" => {
                jitter_end = Some(args[i + 1].clone());
                i += 1;
            }
            "--mirror-tol" => {
                mirror_tol = args[i + 1].parse().ok();
                i += 1;
            }
            "--cut" => {
                cut = Some(args[i + 1].clone());
                i += 1;
            }
            a => {
                eprintln!("unknown arg {a}");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    let t0 = Instant::now();
    let mut agent = match agent::Agent::load(&base) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("load {base}: {e}");
            std::process::exit(1);
        }
    };
    if let Some(c) = cut {
        match agent::base::cut_index(&c) {
            Some(k) => agent.base.cut = k,
            None => {
                eprintln!("unknown cut {c}");
                std::process::exit(2);
            }
        }
    }
    if let Some(p) = profiles {
        if let Err(e) = agent.base.set_profiles(&p, profile) {
            eprintln!("profiles: {e}");
            std::process::exit(2);
        }
        agent.base.chain.clone_profile = clone_profile;
        agent.base.chain.clone_strict = clone_strict;
    }
    if let Some(sc) = sched {
        let mut v: Vec<usize> = sc.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        while !v.is_empty() && v.len() < 30 {
            v.push(*v.last().unwrap());
        }
        if v.iter().any(|&k| k >= agent.base.chain.profiles.len().max(1)) {
            eprintln!("sched: profile id out of range for the --profiles table");
            std::process::exit(2);
        }
        agent.base.chain.schedule = Some(v);
    }
    if let Some(g) = group {
        let tri = |s: &str| -> Vec<i64> { s.split(',').filter_map(|x| x.trim().parse().ok()).collect() };
        let (p, j) = (tri(&g), tri(jitter.as_deref().unwrap_or("0,0,0")));
        if p.len() != 3 || j.len() != 3 {
            eprintln!("--group / --jitter take 3 comma-separated values (DIFFERENT, PARTIAL, COPY)");
            std::process::exit(2);
        }
        let e = tri(endgame.as_deref().unwrap_or("-1,-1,-1"));
        let je = tri(jitter_end.as_deref().unwrap_or(jitter.as_deref().unwrap_or("0,0,0")));
        if e.len() != 3 || je.len() != 3 {
            eprintln!("--endgame / --jitter-end take 3 comma-separated values");
            std::process::exit(2);
        }
        let eg = [0, 1, 2].map(|i| (e[i] >= 0).then_some(e[i] as usize));
        agent.base.chain.group_ctl = Some(
            agent::layers::group::GroupCtl::new([p[0] as usize, p[1] as usize, p[2] as usize], [j[0], j[1], j[2]])
                .with_endgame(eg, [je[0], je[1], je[2]]),
        );
        if let Some(g) = agent.base.chain.group_ctl.as_mut() {
            g.mirror_tol = mirror_tol;
        }
    }
    if timing {
        eprintln!("[agent] loaded {base} in {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
    }
    let stdin = std::io::stdin();
    let mut out = BufWriter::new(std::io::stdout());
    let mut lat: Vec<f64> = Vec::new();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        if line.starts_with("{\"cmd\"") {
            let j = kagg_engine::json::parse(&line).unwrap_or(kagg_engine::json::Json::Null);
            let r = match j.get("cmd").str() {
                "swap_base" => agent.swap_base(j.get("dir").str()).map(|_| "{\"ok\": true}".to_string()),
                "swap_profiles" => {
                    let p = if j.get("profile").is_null() { None } else { Some(j.get("profile").i64() as usize) };
                    agent.swap_profiles(j.get("path").str(), p).map(|_| "{\"ok\": true}".to_string())
                }
                c => Err(format!("unknown cmd {c}")),
            };
            let _ = writeln!(out, "{}", r.unwrap_or_else(|e| format!("{{\"ok\": false, \"error\": {}}}", kagg_engine::json::quote(&e))));
            let _ = out.flush();
            continue;
        }
        let t = Instant::now();
        if std::env::var("AGENT_PROFILE").is_ok() {
            let p0 = Instant::now();
            let j = agent::obs::Obs::parse(&line).unwrap();
            let p1 = Instant::now();
            let (a, _) = agent.base.act(j);
            let p2 = Instant::now();
            let s = a.dump();
            let p3 = Instant::now();
            drop(s);
            eprintln!("[prof] parse {:.1}us act {:.1}us dump {:.1}us bytes {}", (p1 - p0).as_secs_f64() * 1e6,
                      (p2 - p1).as_secs_f64() * 1e6, (p3 - p2).as_secs_f64() * 1e6, line.len());
            let _ = writeln!(out, "{}", a.dump());
            continue;
        }
        let t = if std::env::var("AGENT_PROFILE").is_ok() { Instant::now() } else { t };
        let a = agent.act_json(&line);
        lat.push(t.elapsed().as_secs_f64() * 1e6);
        let _ = writeln!(out, "{a}");
        let _ = out.flush();
    }
    if timing {
        let f: Vec<String> = agent::layers::CUTS.iter().zip(agent.base.chain.fired.iter()).skip(1).filter(|(_, n)| **n > 0).map(|(c, n)| format!("{c}={n}")).collect();
        eprintln!("[agent] fired {}", f.join(" "));
        let s: Vec<String> = agent::layers::CUTS.iter().zip(agent.base.chain.stage_us.iter()).skip(1).filter(|(_, u)| u.1 > 20.0).map(|(c, u)| format!("{c}={:.0}/{:.0}", u.0 / lat.len().max(1) as f64, u.1)).collect();
        eprintln!("[agent] stage mean/max us {}", s.join(" "));
    }
    if timing && !lat.is_empty() {
        let mut s = lat.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |p: f64| s[((s.len() - 1) as f64 * p) as usize];
        eprintln!(
            "[agent] turns {} mean {:.1}us p50 {:.1}us p99 {:.1}us max {:.1}us",
            s.len(),
            s.iter().sum::<f64>() / s.len() as f64,
            q(0.5),
            q(0.99),
            s[s.len() - 1]
        );
    }
}
