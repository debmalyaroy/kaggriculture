//! One replay JSON -> episode row + per-day rows.
use crate::gm::GmEp;
use features::consts::HASH_CUTS;
use features::{Episode, Replay, Row};
use serde_json::Value;

/// Everything known about an episode before its replay is opened.
#[derive(Clone, Debug, Default)]
pub struct Meta {
    pub eid: i64,
    pub source: &'static str,
    pub source_file: String,
    pub file_sha: String,
    pub end_time: String,
    pub end_date: String,
    pub gm: Option<GmEp>,
    pub rating_min: Option<f64>,
    pub rating_max: Option<f64>,
    pub gm_hash: Option<[[Option<u64>; 4]; 2]>,
    pub coverage: [Option<f64>; 2],
}

pub struct Out {
    pub meta: Meta,
    pub engine: String,
    pub status: String,
    pub episode: Option<Row>,
    pub states: Vec<Row>,
    pub behaviour: Vec<Row>,
    /// Computed stream-hash cuts equal the GM csv (None when GM has none).
    pub hash_match: Option<bool>,
    /// `--mode slim`: the slim record row (see slim.rs).
    pub slim: Option<Row>,
    /// `--mode obs`: one row per (seat, decision day), see `extract_obs`.
    pub obs: Vec<Row>,
}

impl Out {
    pub fn status_only(meta: Meta, engine: String, status: String) -> Out {
        Out { meta, engine, status, episode: None, states: vec![], behaviour: vec![], hash_match: None, slim: None, obs: vec![] }
    }
}

fn json_i64(v: &Option<Value>) -> Option<i64> {
    v.as_ref().and_then(|v| v.as_i64().or_else(|| v.as_f64().map(|f| f as i64)).or_else(|| {
        v.as_str().and_then(|s| s.parse().ok())
    }))
}

fn head(meta: &Meta, seat: usize, day: usize) -> Row {
    let mut h = Row::with_capacity(4);
    h.i("episode_id", "Kaggle episode id (negative = hashed file stem)", Some(meta.eid));
    h.i("seat", "player index 0/1 this row describes", Some(seat as i64));
    h.i("day", "game day 0..29", Some(day as i64));
    h.s("end_date", "episode end date (UTC, YYYY-MM-DD)", Some(meta.end_date.clone()));
    h
}

pub fn extract(json: &str, meta: Meta, keep_engine: &str, extractor_version: &str) -> Out {
    let rep = match Replay::parse(json) {
        Ok(r) => r,
        Err(e) => return Out::status_only(meta, String::new(), format!("error:parse {e}")),
    };
    let engine = rep.module_version.clone().unwrap_or_default();
    if engine != keep_engine {
        let st = format!("skipped:engine={}", if engine.is_empty() { "?" } else { &engine });
        return Out::status_only(meta, engine, st);
    }
    if rep.steps.len() < 25 {
        let st = format!("skipped:short={}", rep.steps.len());
        return Out::status_only(meta, engine, st);
    }
    let ep = match Episode::build(&rep) {
        Ok(e) => e,
        Err(e) => return Out::status_only(meta, engine, format!("error:{e}")),
    };
    let rewards = rep.rewards.clone().unwrap_or_default();
    rows(ep, meta, engine, rewards, rep.info.as_ref(), extractor_version)
}

/// Features from a slim record (`--mode features --slim DIR`): same rows as `extract`.
pub fn extract_slim(json: &str, meta: Meta, keep_engine: &str, extractor_version: &str) -> Out {
    let doc = match features::obs::SlimDoc::parse(json) {
        Ok(r) => r,
        Err(e) => return Out::status_only(meta, String::new(), format!("error:parse {e}")),
    };
    let engine = doc.module_version.clone().unwrap_or_default();
    if engine != keep_engine {
        let st = format!("skipped:engine={}", if engine.is_empty() { "?" } else { &engine });
        return Out::status_only(meta, engine, st);
    }
    if doc.steps.len() < 25 {
        let st = format!("skipped:short={}", doc.steps.len());
        return Out::status_only(meta, engine, st);
    }
    let ep = match Episode::build_slim(&doc) {
        Ok(e) => e,
        Err(e) => return Out::status_only(meta, engine, format!("error:{e}")),
    };
    let rewards = doc.rewards.clone().unwrap_or_default();
    rows(ep, meta, engine, rewards, doc.info.as_ref(), extractor_version)
}

/// Shared-column mismatches between the online observation and `daily_state` (all episodes of
/// the run); printed by main at the end. Must stay 0: the policy trains on exactly what it sees.
pub static OBS_CHECK_DIFFS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static OBS_CHECK_ROWS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// `--mode obs`: one `day_obs` row per (seat, decision day): the dayobs vector (crates/dayobs,
/// the same builder the agent runs), the rival's next-day sales (aux target) and the outcome.
/// Every row is cross-checked against the corpus `daily_state` columns the two share.
pub fn extract_obs(json: &str, meta: Meta, keep_engine: &str, extractor_version: &str) -> Out {
    use std::sync::atomic::Ordering::Relaxed;
    let doc = match features::obs::SlimDoc::parse(json) {
        Ok(r) => r,
        Err(e) => return Out::status_only(meta, String::new(), format!("error:parse {e}")),
    };
    let engine = doc.module_version.clone().unwrap_or_default();
    if engine != keep_engine {
        return Out::status_only(meta, engine.clone(), format!("skipped:engine={}", if engine.is_empty() { "?" } else { &engine }));
    }
    if doc.steps.len() < 25 {
        return Out::status_only(meta, engine, format!("skipped:short={}", doc.steps.len()));
    }
    let ep = match Episode::build_slim(&doc) {
        Ok(e) => e,
        Err(e) => return Out::status_only(meta, engine, format!("error:{e}")),
    };
    let rewards = doc.rewards.clone().unwrap_or_default();
    let last = &ep.recs[ep.n - 1];
    let bank = |s: usize| rewards.get(s).copied().flatten().unwrap_or(last.money[s]);
    let names = dayobs::names();
    let mut obs = Vec::with_capacity(60);
    for seat in 0..2 {
        let (own, riv) = (bank(seat), bank(1 - seat));
        let score = if own > riv { 1.0 } else if own < riv { 0.0 } else { 0.5 };
        for dr in features::dayobs_adapter::day_rows(&ep, seat) {
            // cross-check the shared columns against the corpus daily_state row
            if let Some(ds) = ep.daily_state(seat, dr.day) {
                let get = |k: &str| match ds.get(k) {
                    Some(features::Val::F(v)) => *v,
                    Some(features::Val::I(v)) => v.map(|x| x as f64),
                    _ => None,
                };
                let mut diffs = 0;
                let close = |a: f32, b: f64| (a as f64 - b).abs() <= 1e-4 * (1.0 + b.abs());
                if !close(dr.obs[1], get("own_money").unwrap_or(0.0) / 1e5) {
                    diffs += 1;
                }
                for (i, p) in features::consts::PRODUCTS.iter().enumerate() {
                    let want = get(&format!("rival_sold_{}", p.to_ascii_lowercase())).unwrap_or(0.0);
                    let at = names.iter().position(|n| n == &format!("rival_sold_{}", p.to_ascii_lowercase())).unwrap();
                    let v = dr.obs[at] as f64;
                    let back = v.signum() * (v.abs().exp() - 1.0);
                    if (back - want).abs() > 1e-2 * (1.0 + want.abs()) {
                        diffs += 1;
                    }
                    let _ = i;
                }
                let pe = names.iter().position(|n| n == "pos_equal_frac").unwrap();
                if !close(dr.obs[pe], get("pos_equal_frac").unwrap_or(0.0)) {
                    diffs += 1;
                }
                let ls = names.iter().position(|n| n == "layout_sim").unwrap();
                if !close(dr.obs[ls], get("layout_sim").unwrap_or(0.0)) {
                    diffs += 1;
                }
                OBS_CHECK_ROWS.fetch_add(1, Relaxed);
                if diffs > 0 && OBS_CHECK_DIFFS.fetch_add(diffs, Relaxed) < 5 {
                    eprintln!("[obs-check] episode {} seat {seat} day {}: {diffs} shared columns differ from daily_state", meta.eid, dr.day);
                }
            }
            let mut r = head(&meta, seat, dr.day);
            r.f("final_own", "own final bank", Some(own));
            r.f("final_rival", "rival final bank", Some(riv));
            r.f("score", "1 win / 0.5 draw / 0 loss for this seat", Some(score));
            for (k, v) in names.iter().zip(dr.obs.iter()) {
                r.f(format!("o_{k}"), "dayobs feature (crates/dayobs, FEAT_VERSION in obs_version)", Some(*v as f64));
            }
            for (i, p) in features::consts::PRODUCTS.iter().enumerate() {
                r.f(format!("y_rival_next_{}", p.to_ascii_lowercase()),
                    "aux target: rival net units sold over the next 24 steps (null on the last day)",
                    dr.rival_next.map(|x| x[i]));
            }
            r.i("obs_version", "dayobs::FEAT_VERSION", Some(dayobs::FEAT_VERSION as i64));
            r.s("extractor_version", "corpus-extract version that wrote the row", Some(extractor_version.to_string()));
            obs.push(r);
        }
    }
    let mut o = Out::status_only(meta, engine, "ok".into());
    o.obs = obs;
    o
}

fn rows(ep: Episode, meta: Meta, engine: String, rewards: Vec<Option<f64>>, info: Option<&features::obs::Info>,
        extractor_version: &str) -> Out {
    let mut states = Vec::with_capacity(60);
    let mut behaviour = Vec::with_capacity(60);
    for seat in 0..2 {
        for d in 0..ep.n_state_days() {
            if let Some(mut r) = ep.daily_state(seat, d) {
                r.prepend(head(&meta, seat, d));
                states.push(r);
            }
        }
        for d in 0..ep.n_behaviour_days() {
            if let Some(mut r) = ep.daily_behaviour(seat, d) {
                r.prepend(head(&meta, seat, d));
                behaviour.push(r);
            }
        }
    }

    // --- episode row ------------------------------------------------------------------
    let last = &ep.recs[ep.n - 1];
    let bank = |s: usize| rewards.get(s).copied().flatten().or(Some(last.money[s]));
    let (b0, b1) = (bank(0).unwrap_or(0.0), bank(1).unwrap_or(0.0));
    let names = info.and_then(|i| i.team_names.clone()).unwrap_or_default();
    let gm = meta.gm.clone().unwrap_or_default();
    let mut hash_match = None;
    if let Some(g) = &meta.gm_hash {
        let mut all = true;
        let mut any = false;
        for seat in 0..2 {
            for k in 0..HASH_CUTS.len() {
                if let (Some(want), Some(have)) = (g[seat][k], ep.stream_hash[seat][k].as_ref()) {
                    any = true;
                    all &= u64::from_str_radix(have, 16).ok() == Some(want);
                }
            }
        }
        if any {
            hash_match = Some(all);
        }
    }

    let mut e = Row::with_capacity(64);
    e.i("episode_id", "Kaggle episode id (negative = hashed file stem)", Some(meta.eid));
    e.s("end_date", "episode end date (UTC, YYYY-MM-DD)", Some(meta.end_date.clone()));
    e.s("end_time", "episode end time as given by the source (GM end_time / manifest time / file mtime)",
        Some(meta.end_time.clone()));
    e.s("source", "gm | daily | replays", Some(meta.source.to_string()));
    e.s("source_file", "shard / JSON file the replay came from", Some(meta.source_file.clone()));
    e.s("engine_version", "replay module_version", Some(engine.clone()));
    e.s("episode_type", "GM episodes.csv type (EPISODE_TYPE_PUBLIC / _VALIDATION); null elsewhere",
        (!gm.episode_type.is_empty()).then(|| gm.episode_type.clone()));
    e.i("seed", "info.seed (engine RNG seed)", info.and_then(|i| json_i64(&i.seed)));
    e.i("n_steps", "len(steps) (720 for a full game)", Some(ep.n as i64));
    for s in 0..2 {
        e.s(format!("team_name_{s}"), "info.TeamNames[seat]", names.get(s).cloned().flatten());
    }
    for s in 0..2 {
        e.i(format!("team_id_{s}"), "GM team id", gm.team[s]);
    }
    for s in 0..2 {
        e.i(format!("submission_id_{s}"), "GM submission id", gm.sub[s]);
    }
    for s in 0..2 {
        e.f(format!("rating_after_{s}"), "GM rating right after the game", gm.rating[s]);
    }
    e.f("rating_min", "daily manifest min_score (post-game rating of the weaker seat; seat unknown)", meta.rating_min);
    e.f("rating_max", "daily manifest 2*avg_score - min_score (the other seat's rating)", meta.rating_max);
    for s in 0..2 {
        e.f(format!("coverage_{s}"), "GM per_submission_coverage.coverage of this seat's submission (reweighting factor)",
            meta.coverage[s]);
    }
    e.f("bank_0", "final score seat 0 (rewards, else final money)", Some(b0));
    e.f("bank_1", "final score seat 1", Some(b1));
    for s in 0..2 {
        e.f(format!("final_money_{s}"), "farm money at the last observation", Some(last.money[s]));
    }
    e.f("margin", "bank_0 - bank_1", Some(b0 - b1));
    e.i("winner", "0 / 1, or -1 for a draw", Some(if b0 > b1 { 0 } else if b1 > b0 { 1 } else { -1 }));
    let sh = &ep.final_shops;
    e.s("shop_1", "first unlocked shop (day 3) = realized world part 1", sh.first().cloned());
    e.s("shop_2", "second unlocked shop (day 6) = realized world part 2", sh.get(1).cloned());
    let world = if sh.len() >= 2 {
        let mut w = [sh[0].clone(), sh[1].clone()];
        w.sort();
        Some(w.join("|"))
    } else {
        sh.first().cloned()
    };
    e.s("world", "sorted 'SHOP_A|SHOP_B' of the first two unlocked shops", world);
    e.s("shops_final", "all unlocked shop instances at the end, unlock order, '|'-joined", Some(sh.join("|")));
    for s in 0..2 {
        for (k, c) in HASH_CUTS.iter().enumerate() {
            e.s(format!("stream_h{c}_{s}"),
                "sha256 over the seat's canonical action stream through turn N, first 16 hex (GM stream_hashes convention)",
                ep.stream_hash[s][k].clone());
        }
    }
    e.i("stream_hash_gm_match", "1 if the computed cuts equal GM stream_hashes.csv, 0 if not, null if GM has none",
        hash_match.map(|b| b as i64));
    e.i("is_mirror", "1 if both seats submitted identical action streams through the last turn", Some(ep.is_mirror() as i64));
    e.f("layout_sim_d15", "layout similarity at step 361 (day 15 hour 1), seat-0 view", ep.layout_sim(features::episode::LAYOUT_SIM_STEP, 0));
    e.i("cash_equal_step1", "1 if both farms had identical money at step 1", Some(ep.cash_equal_step1() as i64));
    e.s("extractor_version", "corpus-extract version that wrote the row", Some(extractor_version.to_string()));

    Out { meta, engine, status: "ok".into(), episode: Some(e), states, behaviour, hash_match, slim: None, obs: vec![] }
}
