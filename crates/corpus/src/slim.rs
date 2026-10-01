//! Slim record: one replay JSON -> one compact, lossless-for-our-features JSON document.
//!
//! Kept for every step t (the replay's `steps[t]`):
//!   a   : [action seat0, action seat1]      (steps[t][s].action, as served: the action that led to state t)
//!   m   : market {inventory, prices[, params]} (seat-0 observation; shared)
//!   tw  : town.unlocked_shops, only when it changed since the previous step
//!   f   : [farm0, farm1] without `tiles` (money, farmer, hands, hires_today, unlocked_quadrants)
//!   p   : [private seat0, private seat1]    (shed, inventories, seeds; each seat's own observation)
//!   tl  : [tiles farm0, tiles farm1] only at hour 1 of every day (t % 24 == 1) and at the last step
//! Top level keeps id, info {EpisodeId, TeamNames, seed}, configuration, rewards, statuses,
//! module_version and n (number of steps).
//!
//! Memory: the replay is deserialized into BORROWED raw slices (`&RawValue`) -- fields we do
//! not keep are skipped without allocation, kept values are copied verbatim into the output
//! string. No JSON tree of the 33 MB replay is ever built.
use crate::extract::{Meta, Out};
use features::Row;
use serde::Deserialize;
use serde_json::value::RawValue;

pub const SLIM_FORMAT: &str = "slim-1";

#[derive(Deserialize)]
struct Rep<'a> {
    #[serde(borrow, default)]
    id: Option<&'a RawValue>,
    #[serde(borrow, default)]
    info: Option<&'a RawValue>,
    #[serde(borrow, default)]
    configuration: Option<&'a RawValue>,
    #[serde(borrow, default)]
    rewards: Option<&'a RawValue>,
    #[serde(borrow, default)]
    statuses: Option<&'a RawValue>,
    #[serde(default)]
    module_version: Option<String>,
    #[serde(borrow, default)]
    steps: Vec<Vec<SeatStep<'a>>>,
}

#[derive(Deserialize)]
struct SeatStep<'a> {
    #[serde(borrow, default)]
    action: Option<&'a RawValue>,
    #[serde(borrow, default)]
    observation: Option<Obs<'a>>,
}

#[derive(Deserialize)]
struct Obs<'a> {
    #[serde(borrow, default)]
    market: Option<&'a RawValue>,
    #[serde(borrow, default)]
    town: Option<Town<'a>>,
    #[serde(borrow, default)]
    farms: Option<Vec<Farm<'a>>>,
    #[serde(borrow, default)]
    private: Option<&'a RawValue>,
}

#[derive(Deserialize)]
struct Town<'a> {
    #[serde(borrow, default)]
    unlocked_shops: Option<&'a RawValue>,
}

#[derive(Deserialize)]
struct Farm<'a> {
    #[serde(borrow, default)]
    money: Option<&'a RawValue>,
    #[serde(borrow, default)]
    farmer: Option<&'a RawValue>,
    #[serde(borrow, default)]
    hands: Option<&'a RawValue>,
    #[serde(borrow, default)]
    hires_today: Option<&'a RawValue>,
    #[serde(borrow, default)]
    unlocked_quadrants: Option<&'a RawValue>,
    #[serde(borrow, default)]
    tiles: Option<&'a RawValue>,
}

#[derive(Deserialize, Default)]
struct Info {
    #[serde(rename = "TeamNames", default)]
    team_names: Option<Vec<Option<String>>>,
    #[serde(default)]
    seed: Option<serde_json::Value>,
}

fn raw(v: Option<&RawValue>) -> &str {
    v.map(|r| r.get()).unwrap_or("null")
}

fn farm_json(out: &mut String, f: Option<&Farm>) {
    match f {
        None => out.push_str("null"),
        Some(f) => {
            out.push_str("{\"money\":");
            out.push_str(raw(f.money));
            out.push_str(",\"farmer\":");
            out.push_str(raw(f.farmer));
            out.push_str(",\"hands\":");
            out.push_str(raw(f.hands));
            out.push_str(",\"hires_today\":");
            out.push_str(raw(f.hires_today));
            out.push_str(",\"unlocked_quadrants\":");
            out.push_str(raw(f.unlocked_quadrants));
            out.push('}');
        }
    }
}

pub fn slim(json_text: &str, meta: Meta, keep_engine: &str, extractor_version: &str) -> Out {
    let rep: Rep = match serde_json::from_str(json_text) {
        Ok(v) => v,
        Err(e) => return Out::status_only(meta, String::new(), format!("error:parse {e}")),
    };
    let engine = rep.module_version.clone().unwrap_or_default();
    if engine != keep_engine {
        let st = format!("skipped:engine={}", if engine.is_empty() { "?" } else { &engine });
        return Out::status_only(meta, engine, st);
    }
    let n = rep.steps.len();
    if n < 25 {
        return Out::status_only(meta, engine, format!("skipped:short={n}"));
    }
    let info: Info = rep.info.and_then(|r| serde_json::from_str(r.get()).ok()).unwrap_or_default();
    let names = info.team_names.clone().unwrap_or_default();
    let seed = info.seed.as_ref().and_then(|s| s.as_i64().or_else(|| s.as_str().and_then(|x| x.parse().ok())));

    let mut out = String::with_capacity(json_text.len() / 20);
    out.push_str("{\"format\":\"");
    out.push_str(SLIM_FORMAT);
    out.push_str("\",\"id\":");
    out.push_str(raw(rep.id));
    out.push_str(",\"info\":{\"TeamNames\":");
    out.push_str(&serde_json::to_string(&names).unwrap_or_else(|_| "null".into()));
    out.push_str(",\"seed\":");
    out.push_str(&seed.map(|s| s.to_string()).unwrap_or_else(|| "null".into()));
    out.push_str("},\"configuration\":");
    out.push_str(raw(rep.configuration));
    out.push_str(",\"rewards\":");
    out.push_str(raw(rep.rewards));
    out.push_str(",\"statuses\":");
    out.push_str(raw(rep.statuses));
    out.push_str(",\"module_version\":");
    out.push_str(&serde_json::to_string(&engine).unwrap_or_default());
    out.push_str(",\"n\":");
    out.push_str(&n.to_string());
    out.push_str(",\"steps\":[");
    let mut prev_shops: Option<&str> = None;
    for (t, st) in rep.steps.iter().enumerate() {
        if t > 0 {
            out.push(',');
        }
        let s0 = st.first();
        let s1 = st.get(1);
        let o0 = s0.and_then(|s| s.observation.as_ref());
        let o1 = s1.and_then(|s| s.observation.as_ref());
        out.push_str("{\"a\":[");
        out.push_str(raw(s0.and_then(|s| s.action)));
        out.push(',');
        out.push_str(raw(s1.and_then(|s| s.action)));
        out.push_str("],\"m\":");
        out.push_str(raw(o0.and_then(|o| o.market)));
        let shops = raw(o0.and_then(|o| o.town.as_ref()).and_then(|tw| tw.unlocked_shops));
        if prev_shops != Some(shops) {
            out.push_str(",\"tw\":");
            out.push_str(shops);
            prev_shops = Some(shops);
        }
        let farms = o0.and_then(|o| o.farms.as_ref());
        let farm = |i: usize| farms.and_then(|f| f.get(i));
        out.push_str(",\"f\":[");
        farm_json(&mut out, farm(0));
        out.push(',');
        farm_json(&mut out, farm(1));
        out.push_str("],\"p\":[");
        out.push_str(raw(o0.and_then(|o| o.private)));
        out.push(',');
        out.push_str(raw(o1.and_then(|o| o.private)));
        out.push(']');
        if t % 24 == 1 || t + 1 == n {
            out.push_str(",\"tl\":[");
            out.push_str(raw(farm(0).and_then(|f| f.tiles)));
            out.push(',');
            out.push_str(raw(farm(1).and_then(|f| f.tiles)));
            out.push(']');
        }
        out.push('}');
    }
    out.push_str("]}");

    let rewards: Vec<Option<f64>> = rep.rewards.and_then(|r| serde_json::from_str(r.get()).ok()).unwrap_or_default();
    let gm = meta.gm.clone().unwrap_or_default();
    let mut r = Row::with_capacity(28);
    r.i("episode_id", "Kaggle episode id (negative = hashed file stem)", Some(meta.eid));
    r.s("end_date", "episode end date (UTC, YYYY-MM-DD)", Some(meta.end_date.clone()));
    r.s("end_time", "episode end time as given by the source", Some(meta.end_time.clone()));
    r.s("source", "gm | daily | replays", Some(meta.source.to_string()));
    r.s("source_file", "shard / JSON file the replay came from", Some(meta.source_file.clone()));
    r.s("engine_version", "replay module_version", Some(engine.clone()));
    r.s("episode_type", "GM episodes.csv type (EPISODE_TYPE_PUBLIC / _VALIDATION); null elsewhere",
        (!gm.episode_type.is_empty()).then(|| gm.episode_type.clone()));
    r.i("seed", "info.seed", seed);
    r.i("n_steps", "len(steps)", Some(n as i64));
    for s in 0..2 {
        r.s(format!("team_name_{s}"), "info.TeamNames[seat]", names.get(s).cloned().flatten());
    }
    for s in 0..2 {
        r.i(format!("submission_id_{s}"), "GM submission id", gm.sub[s]);
    }
    for s in 0..2 {
        r.f(format!("rating_after_{s}"), "GM rating right after the game", gm.rating[s]);
    }
    r.f("rating_min", "daily manifest min_score (weaker seat's post-game rating; seat unknown)", meta.rating_min);
    r.f("rating_max", "daily manifest 2*avg - min (the other seat's rating)", meta.rating_max);
    for s in 0..2 {
        r.f(format!("coverage_{s}"), "GM per_submission_coverage of this seat's submission", meta.coverage[s]);
    }
    for s in 0..2 {
        r.f(format!("bank_{s}"), "final score (rewards)", rewards.get(s).copied().flatten());
    }
    r.i("slim_bytes", "length of the slim JSON text", Some(out.len() as i64));
    r.s("slim_format", "slim record format version", Some(SLIM_FORMAT.to_string()));
    r.s("extractor_version", "corpus-extract version that wrote the row", Some(extractor_version.to_string()));
    r.s("slim", "the slim record (JSON text; see crates/corpus/src/slim.rs)", Some(out));
    let mut o = Out::status_only(meta, engine, "ok".into());
    o.slim = Some(r);
    o
}
