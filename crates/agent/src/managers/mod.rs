//! The layer MANAGERS (operator 30 Sep): the agent's layers grouped into six plug-and-play managers under one generic
//! CONTROLLER, every knob exposed, one config file (`agent-stdio --config agent.json`).
//!
//! | manager        | owns                                                                                          |
//! |----------------|-----------------------------------------------------------------------------------------------|
//! | market_guard   | the lead-sell / race guards moved out of the chassis (hooks A/B in the chassis step: sell_lead |
//! |                | + R36 window + RACEPX, front_run, dead_stock, terminal_liquidation), the hygiene stages (room, |
//! |                | v28, v31, overflow, t62a, mg, ig) and the final check                                          |
//! | sale           | the chain's sale-timing stages, the reactive shell (price calculator), the sales shell and the |
//! |                | demand batcher + small-batch front-runner (managers::sale)                                     |
//! | endgame        | terminal planner knobs, R46, TSELL, the endgame model (endg) and the standoff layer (gt)       |
//! | rival          | rival identification and front running: preempt (lineage fingerprint), AFR, V92, the group    |
//! |                | knob overlay                                                                                   |
//! | clone          | clone / lineage handling: race + R37 horizons, clone profile, per-game jitter, the disguise   |
//! | economy        | the economy projects (V219, V231, V233, R51, R85/R95/R97, courier, carrot, CA, CH, herd, ...)  |
//!
//! The CONTROLLER (config "controller") picks the day's knob profile: profiles, PPO policy, shield, group profiles,
//! schedule, the specialist dispatcher, and the precedence of those sources (layers::PRECEDENCE by default).
//!
//! agent.json (paths relative to the file; every section optional; a manager with "on": false is removed whole):
//! {"base": "base", "cut": "full",
//!  "controller": {"profiles": "profiles.json", "profile": 3, "policy": "policy.bin", "shield": "shield.json",
//!                 "group": [35,35,35], "endgame": [-1,-1,-1], "jitter_end": [0,0,0], "mirror_tol": null,
//!                 "sched": [..], "knob_over": "knobs.json", "dispatch": "dispatch/dispatch.json",
//!                 "precedence": ["request","afr","clone","group","schedule"]},
//!  "managers": {
//!   "market_guard": {"on": true, "settings": {"sell_lead": true, "r36": true, ...}, "knobs": {"racepx_margin": 0},
//!                    "stages_off": [], "final_check": {"on": false, "clamp": true, "min_price": 0}},
//!   "sale":    {"on": true, "stages_off": ["r127","sm"], "knobs": {...}, "rshell": "rshell/rshell.json",
//!               "shell": "shell.json", "batch": {"on": false, ...managers::sale::SaleCfg}},
//!   "endgame": {"on": true, "stages_off": [], "knobs": {"term_on": true, ...}, "endg": "endg.json",
//!               "gt": "gt.json", "gt_over": {...}},
//!   "rival":   {"on": true, "stages_off": [], "knobs": {"afr_on": false, ...}, "preempt": "preempt.json",
//!               "preempt_over": {"to": 711}, "group_knobs": "group_knobs.json", "group_knobs_for": [0,1],
//!               "group_knobs_lineage": false},
//!   "clone":   {"on": true, "knobs": {"race_clone": 9, ...}, "jitter": [0,0,1], "disguise": true,
//!               "clone_profile": null, "clone_strict": false},
//!   "economy": {"on": true, "stages_off": ["r95"], "knobs": {"v233_on": true, ...}}}}
//! Manager "knobs" are chain knobs (layers::knobs) applied on top of EVERY profile after knob_over; a knob may only be
//! set by the manager that owns it (KNOBS). `agent-stdio --dump-knobs` prints every manager, stage and knob.
pub mod market_guard;
pub mod sale;

use kagg_engine::json::Json;

pub const MANAGERS: [&str; 6] = ["market_guard", "sale", "endgame", "rival", "clone", "economy"];

/// Chain stage (layers::CUTS index) -> (unique name, owning manager).
pub const STAGES: [(&str, &str); 65] = [
    ("chassis", "chassis"), ("terminal", "endgame"), ("room", "market_guard"), ("v28", "market_guard"),
    ("v219", "economy"), ("experiment", "sale"), ("order", "sale"), ("v31", "market_guard"), ("v231", "economy"),
    ("r36", "sale"), ("r37", "sale"), ("release", "sale"), ("v233", "economy"), ("r46", "endgame"),
    ("r51_input", "economy"), ("r51_warehouse", "economy"), ("r53", "economy"), ("r70", "economy"), ("r85", "economy"),
    ("r95", "economy"), ("r97", "economy"), ("courier", "economy"), ("carrot", "economy"), ("herd", "economy"),
    ("fert", "economy"), ("opening", "economy"), ("v9_race", "sale"), ("racepx", "sale"), ("racegate", "sale"),
    ("ctrtable", "sale"), ("overflow", "market_guard"), ("ca", "economy"), ("or2", "sale"), ("ch", "economy"),
    ("sr", "economy"), ("hd2", "economy"), ("cs", "economy"), ("race", "sale"), ("r127", "sale"), ("preguard", "sale"),
    ("v44y", "sale"), ("y", "economy"), ("e335", "sale"), ("v11", "sale"), ("v13v", "sale"), ("wl", "economy"),
    ("adv", "sale"), ("t62a", "market_guard"), ("pipe", "economy"), ("ma", "economy"), ("wb3", "sale"), ("fx", "sale"),
    ("dp", "sale"), ("mp", "sale"), ("bd", "sale"), ("mpx", "sale"), ("sm", "sale"), ("cxd", "sale"), ("e410", "economy"),
    ("e402", "economy"), ("mg", "market_guard"), ("ig", "market_guard"), ("rsa", "sale"), ("afr", "rival"),
    ("tsell", "endgame"),
];

/// Chain knob (layers::knobs::Knobs) -> owning manager.
pub const KNOBS: [(&str, &str); 80] = [
    ("rsa_on", "sale"), ("rsa_look", "sale"), ("rsa_min_frac", "sale"), ("ev_on", "sale"), ("ev_h", "sale"),
    ("dp_on", "sale"), ("dp_h", "sale"), ("mp_on", "sale"), ("mp_h", "sale"), ("mpx_on", "sale"), ("cxd_on", "sale"),
    ("v44y_on", "sale"), ("adv_on", "sale"), ("adv_look", "sale"), ("race_clone", "clone"), ("race_escalated", "clone"),
    ("race_mirror", "clone"), ("v9_race_default", "sale"), ("v9_race_max", "sale"), ("v9_race_margin", "sale"),
    ("racepx_margin", "market_guard"), ("racegate_margin", "sale"), ("r85_feed_on", "economy"), ("e410_on", "economy"),
    ("r51_input_on", "economy"), ("v9_fert_on", "economy"), ("courier_on", "economy"), ("carrot_on", "economy"),
    ("e402_on", "economy"), ("afr_on", "rival"), ("afr_extra", "rival"), ("afr_jitter", "rival"), ("afr_hold", "rival"),
    ("afr_look_max", "rival"), ("afr_min_frac", "rival"), ("ca_margin", "economy"), ("or2_slot_margin", "sale"),
    ("term_on", "endgame"), ("term_start", "endgame"), ("term_sims", "endgame"), ("term_passes", "endgame"),
    ("term_props", "endgame"), ("v92_on", "rival"), ("v92_h", "rival"), ("v92_k", "rival"), ("v92_every", "rival"),
    ("v92_top", "rival"), ("v92_ext_window", "rival"), ("cxd_model", "rival"), ("press_on", "sale"), ("wb3_mode", "sale"),
    ("press_from", "sale"), ("press_trigger", "sale"), ("press_h", "sale"), ("press_tranche", "sale"),
    ("press_max", "sale"), ("tsell_on", "endgame"), ("tsell_from", "endgame"), ("tsell_window", "endgame"),
    ("tsell_model", "endgame"), ("tsell_min", "endgame"), ("r36_from", "sale"), ("v219_min_shops", "economy"),
    ("v233_on", "economy"), ("v231_on", "economy"), ("hd2_on", "economy"), ("cs_on", "economy"), ("y_on", "economy"),
    ("r37_base", "clone"), ("r37_streak", "clone"), ("r37_late", "clone"), ("r37_late_from", "clone"),
    ("r37_sim", "clone"), ("race_from", "clone"), ("lead_frac", "sale"), ("lead_from", "sale"), ("ev_hours", "sale"),
    ("dp_hours", "sale"), ("mp_hours", "sale"), ("off", "sale"),
];

/// Market-guard settings (chassis::Settings keys) owned by the market_guard manager; OFF in every chassis.
pub const GUARD_SETTINGS: [&str; 26] = [
    "animal_guard", "animal_guard_max", "animal_guard_steps", "land_repair",
    "verify_fills", "lead_signal", "lead_signal_look", "lead_signal_p", "lead_signal_stock", "lead_price", "lead_price_h", "lead_price_rival_h", "lead_price_tol",
    "sell_lead", "lead_every", "r36", "r36_lead_from", "r36_lead_to", "racepx", "racepx_margin", "front_run",
    "dead_stock", "dead_stock_day", "dead_stock_min_price", "terminal_liquidation", "terminal_from",
];

/// Knobs a switched-off manager forces (its layers that live outside the chain stages).
const OFF_KNOBS: [(&str, &str); 6] = [
    ("endgame", "{\"term_on\":false,\"tsell_on\":false}"),
    ("rival", "{\"afr_on\":false,\"v92_on\":false}"),
    ("sale", "{\"press_on\":false}"),
    ("market_guard", "{}"),
    ("clone", "{}"),
    ("economy", "{}"),
];

/// Replace/add the keys of `over` in `j` (both objects; anything else leaves `j` as is).
pub fn merge(j: &mut Json, over: &Json) {
    if let (Json::Obj(a), Json::Obj(b)) = (j, over) {
        for (k, v) in b {
            match a.iter_mut().find(|(x, _)| x == k) {
                Some(e) => e.1 = v.clone(),
                None => a.push((k.clone(), v.clone())),
            }
        }
    }
}

pub fn stage_index(name: &str) -> Option<usize> {
    STAGES.iter().position(|(n, _)| *n == name)
}

/// What `--config` sets after the flag-equivalent setup (see `expand`).
#[derive(Default, Debug, Clone)]
pub struct Post {
    pub game_off: u64,
    pub knobs: Vec<Json>,
    pub guard_settings: Option<Json>,
    pub sale: Option<sale::SaleCfg>,
    pub final_check: Option<market_guard::FinalCheck>,
    pub cash_floor: Option<(f64, i64)>,
    pub precedence: Vec<String>,
    pub preempt: Option<(String, Json)>,
    pub gt: Option<(String, Json)>,
    pub managers_off: Vec<String>,
}

fn arr3(j: &Json) -> String {
    j.arr().iter().map(|x| x.i64().to_string()).collect::<Vec<_>>().join(",")
}

/// agent.json -> the equivalent agent-stdio arguments (so the setup code path is exactly the flags' one) + the
/// settings that have no flag (`Post`, applied by `finish`).
pub fn expand(path: &str) -> Result<(Vec<String>, Post), String> {
    let j = kagg_engine::json::parse(&std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?)?;
    let dir = std::path::Path::new(path).parent().map(|p| p.to_path_buf()).unwrap_or_default();
    let rel = |x: &Json| -> String {
        let p = std::path::Path::new(x.str());
        if p.is_absolute() { x.str().to_string() } else { dir.join(p).to_string_lossy().to_string() }
    };
    let mut a: Vec<String> = vec![];
    let mut post = Post::default();
    let mut push = |k: &str, v: String| {
        a.push(k.to_string());
        if !v.is_empty() {
            a.push(v);
        }
    };
    for (k, _) in j.obj() {
        if !["base", "cut", "controller", "managers", "note", "name"].contains(&k.as_str()) {
            return Err(format!("config: unknown section {k:?}"));
        }
    }
    if !j.get("base").is_null() {
        push("--base", rel(j.get("base")));
    }
    if !j.get("cut").is_null() {
        push("--cut", j.get("cut").str().to_string());
    }
    let c = j.get("controller");
    for (k, v) in c.obj() {
        if v.is_null() {
            continue;
        }
        match k.as_str() {
            "profiles" => push("--profiles", rel(v)),
            "profile" => push("--profile", v.i64().to_string()),
            "policy" => push("--policy", rel(v)),
            "shield" => push("--shield", rel(v)),
            "group" => push("--group", arr3(v)),
            "endgame" => push("--endgame", arr3(v)),
            "jitter_end" => push("--jitter-end", arr3(v)),
            "mirror_tol" => push("--mirror-tol", v.f64().to_string()),
            "sched" => push("--sched", arr3(v)),
            "knob_over" => push("--knob-over", rel(v)),
            "dispatch" => push("--dispatch", rel(v)),
            "precedence" => {
                for s in v.arr() {
                    if !crate::layers::PRECEDENCE.contains(&s.str()) {
                        return Err(format!("controller.precedence: unknown source {:?}", s.str()));
                    }
                    post.precedence.push(s.str().to_string());
                }
            }
            "note" => {}
            o => return Err(format!("controller: unknown key {o:?}")),
        }
    }
    let m = j.get("managers");
    for (k, _) in m.obj() {
        if !MANAGERS.contains(&k.as_str()) {
            return Err(format!("managers: unknown manager {k:?}"));
        }
    }
    for name in MANAGERS {
        let s = m.get(name);
        let on = s.get("on").is_null() || s.get("on").bool();
        if !on {
            post.managers_off.push(name.to_string());
            for (i, (_, owner)) in STAGES.iter().enumerate() {
                if *owner == name {
                    post.game_off |= 1u64 << i;
                }
            }
            let f = OFF_KNOBS.iter().find(|(n, _)| *n == name).map(|(_, j)| *j).unwrap_or("{}");
            post.knobs.push(kagg_engine::json::parse(f)?);
            if name == "market_guard" {
                let off: Vec<String> = ["sell_lead", "front_run", "r36", "racepx", "dead_stock", "terminal_liquidation"].iter().map(|k| format!("\"{k}\":false")).collect();
                post.guard_settings = Some(kagg_engine::json::parse(&format!("{{{}}}", off.join(",")))?);
            }
            continue;
        }
        for st in s.get("stages_off").arr() {
            let i = stage_index(st.str()).ok_or(format!("{name}.stages_off: unknown stage {:?}", st.str()))?;
            if STAGES[i].1 != name {
                return Err(format!("{name}.stages_off: stage {:?} belongs to {}", st.str(), STAGES[i].1));
            }
            post.game_off |= 1u64 << i;
        }
        let kn = s.get("knobs");
        for (k, _) in kn.obj() {
            match KNOBS.iter().find(|(n, _)| n == k) {
                Some((_, owner)) if *owner == name => {}
                Some((_, owner)) => return Err(format!("{name}.knobs: knob {k:?} belongs to {owner}")),
                None => return Err(format!("{name}.knobs: unknown knob {k:?}")),
            }
        }
        if !kn.obj().is_empty() {
            post.knobs.push(kn.clone());
        }
        for (k, v) in s.obj() {
            if v.is_null() {
                continue;
            }
            match (name, k.as_str()) {
                (_, "on" | "stages_off" | "knobs" | "note") => {}
                ("market_guard", "settings") => {
                    for (g, _) in v.obj() {
                        if !GUARD_SETTINGS.contains(&g.as_str()) {
                            return Err(format!("market_guard.settings: {g:?} is not a market guard"));
                        }
                    }
                    post.guard_settings = Some(v.clone());
                }
                ("market_guard", "final_check") => {
                    if v.get("on").is_null() || v.get("on").bool() {
                        post.final_check = Some(market_guard::FinalCheck::default().with(v)?);
                    }
                }
                ("sale", "rshell") => push("--rshell", rel(v)),
                ("sale", "shell") => push("--shell", rel(v)),
                ("sale", "cash_floor") => {
                    if v.get("on").is_null() || v.get("on").bool() {
                        post.cash_floor = Some((v.get("money").f64(), v.get("until").i64()));
                    }
                }
                ("sale", "batch") => {
                    let c = sale::SaleCfg::default().with(v)?;
                    if c.on {
                        post.sale = Some(c);
                    }
                }
                ("endgame", "endg") => push("--endg", rel(v)),
                ("endgame", "gt") => post.gt = Some((rel(v), s.get("gt_over").clone())),
                ("endgame", "gt_over") => {}
                ("rival", "preempt") => post.preempt = Some((rel(v), s.get("preempt_over").clone())),
                ("rival", "preempt_over") => {}
                ("rival", "group_knobs") => push("--group-knobs", rel(v)),
                ("rival", "group_knobs_for") => push("--group-knobs-for", arr3(v)),
                ("rival", "group_knobs_lineage") => {
                    if v.bool() {
                        push("--group-knobs-lineage", String::new());
                    }
                }
                ("clone", "jitter") => push("--jitter", arr3(v)),
                ("clone", "disguise") => {
                    if v.bool() {
                        push("--disguise", String::new());
                    }
                }
                ("clone", "clone_profile") => push("--clone-profile", v.i64().to_string()),
                ("clone", "clone_strict") => {
                    if v.bool() {
                        push("--clone-strict", String::new());
                    }
                }
                (n, o) => return Err(format!("managers.{n}: unknown key {o:?}")),
            }
        }
    }
    Ok((a, post))
}

/// Apply the no-flag part of a config to a loaded base (after the flag-equivalent setup).
pub fn finish(b: &mut crate::base::Base, p: &Post) -> Result<(), String> {
    let chassis_only = b.cut == 0;
    if let Some(g) = p.guard_settings.as_ref() {
        if chassis_only {
            // the market guards are never part of a chassis-only run
            eprintln!("[config] cut=chassis: market_guard settings ignored");
        } else {
            b.chassis.cfg.apply(g);
        }
    }
    b.chain.game_off |= p.game_off;
    if !p.knobs.is_empty() {
        if b.chain.profiles.is_empty() {
            b.chain.profiles = vec![("v61.1".into(), Default::default())];
        }
        for (_, k) in b.chain.profiles.iter_mut() {
            for j in &p.knobs {
                *k = k.with(j)?;
            }
        }
    }
    b.chain.precedence = p.precedence.clone();
    if let Some(c) = p.sale.as_ref() {
        b.sale = Some(sale::Sale::new(std::sync::Arc::new(c.clone())));
    }
    b.final_check = p.final_check.clone();
    b.cash_floor = p.cash_floor;
    if let Some((f, over)) = p.preempt.as_ref() {
        let c = crate::preempt::PreCfg::load_over(f, over).map_err(|e| format!("preempt {f}: {e}"))?;
        b.preempt = Some(crate::preempt::Preempt::new(std::sync::Arc::new(c)));
    }
    if let Some((f, over)) = p.gt.as_ref() {
        let c = crate::gt::GtCfg::load_over(f, over).map_err(|e| format!("gt {f}: {e}"))?;
        b.gtl = Some(crate::gt::GtLayer::new(std::sync::Arc::new(c)));
    }
    for name in &p.managers_off {
        match name.as_str() {
            "sale" => {
                b.sale = None;
            }
            "rival" => {
                b.preempt = None;
                b.chain.group_over = None;
            }
            "endgame" => {
                b.endg = None;
                b.gtl = None;
            }
            "clone" => {
                b.disguise = None;
                b.chain.clone_profile = None;
            }
            "market_guard" => {
                b.final_check = None;
            }
            _ => {}
        }
    }
    Ok(())
}

/// `--dump-knobs`: every manager with its stages, knobs (current profile-0 values of `b`) and components.
pub fn dump(b: &crate::base::Base) -> String {
    let k = b.chain.profiles.first().map(|(_, k)| k.clone()).unwrap_or_default();
    let dbg = format!("{k:?}");
    let body = dbg.trim_start_matches("Knobs {").trim_end_matches('}');
    let vals: Vec<(String, String)> = body.split(", ").filter_map(|kv| kv.split_once(": ").map(|(a, v)| (a.trim().to_string(), v.trim().to_string()))).collect();
    let val = |n: &str| vals.iter().find(|(a, _)| a == n).map(|(_, v)| v.clone()).unwrap_or_else(|| "null".into());
    let cfg = format!("{:?}", b.chassis.cfg);
    let cbody = cfg.trim_start_matches("Settings {").trim_end_matches('}');
    let cvals: Vec<(String, String)> = cbody.split(", ").filter_map(|kv| kv.split_once(": ").map(|(a, v)| (a.trim().to_string(), v.trim().to_string()))).collect();
    let cval = |n: &str| cvals.iter().find(|(a, _)| a == n).map(|(_, v)| v.clone()).unwrap_or_else(|| "null".into());
    let mut out = vec![];
    for name in MANAGERS {
        let st: Vec<String> = STAGES.iter().enumerate().filter(|(_, (_, o))| *o == name)
            .map(|(i, (n, _))| format!("{{\"stage\":\"{n}\",\"index\":{i},\"on\":{}}}", (b.chain.game_off >> i) & 1 == 0)).collect();
        let kn: Vec<String> = KNOBS.iter().filter(|(_, o)| *o == name).map(|(n, _)| format!("\"{n}\":{}", val(n))).collect();
        let mut extra = String::new();
        match name {
            "market_guard" => {
                let g: Vec<String> = GUARD_SETTINGS.iter().map(|n| format!("\"{n}\":{}", cval(n))).collect();
                extra = format!(",\"settings\":{{{}}},\"final_check\":{}", g.join(","), b.final_check.as_ref().map(|f| f.dump()).unwrap_or("null".into()));
            }
            "sale" => {
                extra = format!(",\"rshell\":{},\"shell\":{},\"batch\":{}", b.rshell.is_some(), b.shell.is_some(), b.sale.as_ref().map(|s| s.cfg.dump()).unwrap_or_else(|| sale::SaleCfg::default().dump()));
            }
            "endgame" => extra = format!(",\"endg\":{},\"gt\":{}", b.endg.is_some(), b.gtl.is_some()),
            "rival" => extra = format!(",\"preempt\":{},\"group_knobs\":{}", b.preempt.is_some(), b.chain.group_over.is_some()),
            "clone" => extra = format!(",\"disguise\":{},\"clone_profile\":{}", b.disguise.is_some(), b.chain.clone_profile.map(|x| x.to_string()).unwrap_or("null".into())),
            _ => {}
        }
        out.push(format!("\"{name}\":{{\"stages\":[{}],\"knobs\":{{{}}}{extra}}}", st.join(","), kn.join(",")));
    }
    let prec: Vec<String> = if b.chain.precedence.is_empty() { crate::layers::PRECEDENCE.iter().map(|s| format!("\"{s}\"")).collect() } else { b.chain.precedence.iter().map(|s| format!("\"{s}\"")).collect() };
    format!(
        "{{\"controller\":{{\"profiles\":{},\"policy\":{},\"group\":{},\"dispatch\":{},\"precedence\":[{}]}},\"managers\":{{{}}}}}",
        b.chain.profiles.len(), b.policy.is_some(), b.chain.group_ctl.is_some(), b.dispatch.is_some(), prec.join(","), out.join(",")
    )
}
