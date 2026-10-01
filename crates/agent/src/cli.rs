//! The agent's command line (`agent-stdio` flags or `--config agent.json`) -> a ready Agent. Shared by agent-stdio and
//! the Rust tournament runners (chassis-vs-tapes --agent), so every runner sets the agent up exactly as the
//! submission does. Exits the process on a bad argument (it is a CLI builder).
pub struct Built {
    pub agent: crate::Agent,
    pub timing: bool,
    pub dump_knobs: bool,
    pub base: String,
}

pub fn build(args: Vec<String>) -> Built {
    let mut args: Vec<String> = args;
    // --config agent.json: the managers config (crate::managers), expanded into the equivalent flags + a post step
    let mut post: Option<crate::managers::Post> = None;
    if let Some(i) = args.iter().position(|a| a == "--config") {
        match args.get(i + 1).ok_or("--config needs a file".to_string()).and_then(|f| crate::managers::expand(f)) {
            Ok((ex, p)) => {
                args.splice(i..i + 2, ex);
                post = Some(p);
            }
            Err(e) => {
                eprintln!("config: {e}");
                std::process::exit(2);
            }
        }
    }
    let dump_knobs = args.iter().any(|a| a == "--dump-knobs");
    args.retain(|a| a != "--dump-knobs");
    let knobs_lineage = args.iter().any(|a| a == "--group-knobs-lineage");
    let mut base = "configs/bases/v61.1".to_string();
    let mut timing = false;
    let mut cut: Option<String> = None;
    let mut profiles: Option<String> = None;
    let mut profile: Option<usize> = None;
    let mut clone_profile: Option<usize> = None;
    let mut clone_strict = false;
    let mut policy: Option<String> = None;
    let mut sched: Option<String> = None;
    let mut group: Option<String> = None;
    let mut jitter: Option<String> = None;
    let mut endgame: Option<String> = None;
    let mut jitter_end: Option<String> = None;
    let mut mirror_tol: Option<f64> = None;
    let mut shield: Option<String> = None;
    let mut shell: Option<String> = None;
    let mut disguise = false;
    let mut route_table: Option<String> = None;
    let mut extras: Vec<(String, String)> = vec![];
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
            "--policy" => {
                policy = Some(args[i + 1].clone());
                i += 1;
            }
            "--sched" => {
                sched = Some(args[i + 1].clone());
                i += 1;
            }
            "--route-table" => {
                route_table = Some(args[i + 1].clone());
                i += 1;
            }
            "--disguise" => disguise = true,
            "--shell" => {
                shell = Some(args[i + 1].clone());
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
            "--shield" => {
                shield = Some(args[i + 1].clone());
                i += 1;
            }
            "--rshell" | "--chain-off" | "--knob-over" | "--endg" | "--group-knobs" | "--group-knobs-for" | "--preempt" | "--gt" | "--dispatch" => {
                extras.push((args[i].clone(), args[i + 1].clone()));
                i += 1;
            }
            "--group-knobs-lineage" => {} // flag (no value): read from std::env::args() where the overlay is installed
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
    let mut agent = match crate::Agent::load(&base) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("load {base}: {e}");
            std::process::exit(1);
        }
    };
    if let Some(c) = cut {
        match crate::base::cut_index(&c) {
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
            crate::layers::group::GroupCtl::new([p[0] as usize, p[1] as usize, p[2] as usize], [j[0], j[1], j[2]])
                .with_endgame(eg, [je[0], je[1], je[2]]),
        );
        if let Some(g) = agent.base.chain.group_ctl.as_mut() {
            g.mirror_tol = mirror_tol;
        }
    }
    if let Some(f) = shield {
        let r = std::fs::read_to_string(&f).map_err(|e| e.to_string()).and_then(|t| kagg_engine::json::parse(&t)).and_then(|j| crate::layers::group::Shield::from_json(&j));
        let (sh, tol) = match r {
            Ok(x) => x,
            Err(e) => {
                eprintln!("shield {f}: {e}");
                std::process::exit(2);
            }
        };
        // the shield rides on the group controller; without --group it has no fallback profiles
        let mut j3: Vec<i64> = jitter.as_deref().unwrap_or("0,0,0").split(',').filter_map(|x| x.trim().parse().ok()).collect();
        j3.resize(3, 0);
        let g = agent.base.chain.group_ctl.take().unwrap_or_else(|| {
            let mut c = crate::layers::group::GroupCtl::new([0, 0, 0], [j3[0], j3[1], j3[2]]);
            c.fallback = false; // shield-only: masks the policy, never plays a profile of its own
            c
        });
        let tol = mirror_tol.or(tol).or(g.mirror_tol);
        agent.base.chain.group_ctl = Some(g.with_mirror(tol).with_shield(Some(sh)));
    }
    if disguise {
        agent.base.disguise = Some(crate::disguise::Disguise::default());
    }
    if let Some(f) = route_table {
        match std::fs::read_to_string(&f).map_err(|e| e.to_string()).and_then(|t| kagg_engine::json::parse(&t)) {
            Ok(j) => agent.base.chassis.router.override_worlds(&j),
            Err(e) => {
                eprintln!("route table {f}: {e}");
                std::process::exit(2);
            }
        }
    }
    {
        let get = |k: &str| extras.iter().find(|(a, _)| a == k).map(|(_, v)| v.as_str());
        if let Err(e) = agent.base.apply_extras2(get("--rshell"), get("--chain-off"), get("--knob-over"), get("--endg"), None) {
            eprintln!("rshell / chain-off / knob-over / endg: {e}");
            std::process::exit(2);
        }
        // --preempt F: the sale-control layer (crates/agent/src/preempt.rs: lineage fingerprint + pre-emption); a relative
        // "clusters" path in F resolves against the current directory (main.py starts us in the tarball root)
        if let Some(f) = get("--preempt") {
            match crate::preempt::PreCfg::load(f) {
                Ok(c) => agent.base.preempt = Some(crate::preempt::Preempt::new(std::sync::Arc::new(c))),
                Err(e) => {
                    eprintln!("preempt {f}: {e}");
                    std::process::exit(2);
                }
            }
        }
        // --dispatch F: specialist dispatcher (crates/agent/src/dispatch.rs, v63.12)
        if let Some(f) = get("--dispatch") {
            match crate::dispatch::DispatchCfg::load(f) {
                Ok(c) => agent.base.dispatch = Some(crate::dispatch::Dispatch::new(std::sync::Arc::new(c))),
                Err(e) => {
                    eprintln!("dispatch {f}: {e}");
                    std::process::exit(2);
                }
            }
        }
        // --gt F: game-theoretic liquidation layer (crates/agent/src/gt.rs, v63.12)
        if let Some(f) = get("--gt") {
            match crate::gt::GtCfg::load(f) {
                Ok(c) => agent.base.gtl = Some(crate::gt::GtLayer::new(std::sync::Arc::new(c))),
                Err(e) => {
                    eprintln!("gt {f}: {e}");
                    std::process::exit(2);
                }
            }
        }
        // --group-knobs F [--group-knobs-for 0,1]: knob overlay only against these rival groups (layers::Chain::group_over)
        if let Some(f) = get("--group-knobs") {
            match std::fs::read_to_string(f).map_err(|e| e.to_string()).and_then(|t| kagg_engine::json::parse(&t)) {
                Ok(j) => {
                    let gs: Vec<usize> = get("--group-knobs-for").unwrap_or("0,1").split(',').filter_map(|x| x.trim().parse().ok()).collect();
                    agent.base.chain.group_over = Some((gs, j));
                    agent.base.chain.group_over_lineage = knobs_lineage;
                }
                Err(e) => {
                    eprintln!("group-knobs {f}: {e}");
                    std::process::exit(2);
                }
            }
        }
    }
    if let Some(f) = shell {
        match crate::shell::ShellModel::load(&f) {
            Ok(m) => agent.base.shell = Some(crate::shell::ShellCtl::new(std::sync::Arc::new(m))),
            Err(e) => {
                eprintln!("shell: {e}");
                std::process::exit(2);
            }
        }
    }
    if let Some(w) = policy {
        let net = match policy::Net::load(&w) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("policy: {e}");
                std::process::exit(2);
            }
        };
        let n_prof = agent.base.chain.profiles.len();
        if net.n_act != n_prof {
            eprintln!("policy: {} actions but the profile table has {n_prof} (pass the matching --profiles)", net.n_act);
            std::process::exit(2);
        }
        agent.base.obs_track = Some(dayobs::Builder::new());
        agent.base.policy = Some(crate::base::PolicyCtl::new(std::sync::Arc::new(net), vec![true; n_prof], None));
    }
    if let Some(p) = post.as_ref() {
        if let Err(e) = crate::managers::finish(&mut agent.base, p) {
            eprintln!("config: {e}");
            std::process::exit(2);
        }
    }
    Built { agent, timing, dump_knobs, base }
}
