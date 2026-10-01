//! Line protocol: one observation JSON per stdin line -> one action JSON per stdout line.
//! A line `{"cmd": "swap_base", "dir": "..."}` hot-swaps the base (answers `{"ok": true}`).
//! With `--timing`, the per-turn latency summary is printed to stderr at EOF.
//!
//!     agent-stdio --base configs/bases/v61.1 [--timing]
//!     agent-stdio --base ... --profiles configs/profiles/v2.json --policy weights.bin   (learned macro policy:
//!         the GRU picks each day's profile greedily at hour 1; needs --profiles with the same action count)
//!     agent-stdio --base ... --profiles P --sched 19,19,...,10   (per-day profile list; the last repeats)
//!     agent-stdio --base ... --profiles P --group 19,19,19 --jitter 0,0,1   (opponent-group controller)
//!     agent-stdio --base ... --profiles P --policy W.bin --shield configs/shield/v1.json [--jitter 0,0,1]
//!         (learned policy under the opponent-group shield; --group, if given, is the no-policy fallback)
use std::io::{BufRead, BufWriter, Write};
use std::time::Instant;

fn main() {
    let t0 = Instant::now();
    let built = agent::cli::build(std::env::args().collect());
    let (mut agent, timing, dump_knobs, base) = (built.agent, built.timing, built.dump_knobs, built.base);
    if dump_knobs {
        println!("{}", agent::managers::dump(&agent.base));
        return;
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
