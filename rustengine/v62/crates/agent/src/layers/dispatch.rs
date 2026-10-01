//! D6 dispatch (v64): classify the opponent by day 6, then play the base tape, end-game tape and
//! knob overrides that config says suit (shop pair, cluster) best.
//!
//! Clusters (decided at step 144, when both first shops are known):
//!   `<G>_<M|S>` with G from the day-0 group rule of layers/group.rs
//!     DIFF     our units stood on the rival's squares on < 10% of day 0's turns;
//!     PARTIAL  same squares, different money at step 1;
//!     COPY     same squares and identical money at step 1;
//!   and M = still on our squares on >= 90% of day 5's turns (steps 120..143), S = split by day 5.
//!
//! Rules live in the profile table (`{"profiles": [...], "dispatch": [{"name": .., "rules": [..]}]}`),
//! so a new table is a config swap, not a rebuild; the knob `dispatch_set` (-1 = off, v61.1) picks the
//! set. A rule: `{"pair": "A|B" | "A|*" | "*" | [..], "cluster": "COPY" | "COPY_M" | "*" | [..],
//! "route": id, "endgame": id | -1, "knobs": {..}, "from_day": 6}`; the first rule that matches wins.
//!   table    "new" | "old": the D6 route is that router table's entry for the pair (pair-general);
//!            "new_swap" | "new_aa" | "new_bb": the new table's route for B|A, A|A or B|B (same shops);
//!   route    the D6 base tape (overrides table). Applied only if it is SYNC-compatible: its farm actions on steps
//!            1..143 equal route 0's (what every route plays before day 6), else ignored and counted.
//!   endgame  the tape from step 648 (default route 2); -1 keeps the D6 route. Implemented by
//!            installing that tape as route 2 for this game (every layer reads route 2 from 648 on);
//!            the original is restored at the next game's step 0.
//!   knobs    overrides layered on the day's profile at every day boundary from `from_day` (>= 6).
//! With `KAGG_DISPATCH_LOG=<file>` every decision (also with dispatch off) is appended as one JSON
//! line: player, pair, cluster, set, rule, route, endgame, sync.
use crate::chassis::{Chassis, Route};
use crate::layers::knobs::Knobs;
use crate::view::View;
use kagg_engine::json::{self, Json};

pub const DECIDE_STEP: i64 = 144;
pub const ENDGAME_ROUTE: i64 = 2;

#[derive(Clone, Debug)]
pub struct Rule {
    pub pair: Vec<String>,
    pub cluster: Vec<String>,
    pub route: Option<i64>,
    /// "new" | "old": the D6 route is that router table's route for this pair (pair-general).
    pub table: Option<String>,
    pub endgame: Option<i64>,
    pub knobs: Option<Json>,
    pub from_day: i64,
}

#[derive(Clone, Debug)]
pub struct Set {
    pub name: String,
    pub rules: Vec<Rule>,
}

fn pats(j: &Json) -> Vec<String> {
    if j.is_null() {
        vec!["*".into()]
    } else if j.is_arr() {
        j.arr().iter().map(|x| x.str().to_string()).collect()
    } else {
        vec![j.str().to_string()]
    }
}

/// Parse the `dispatch` list of a profile table (knob overrides are validated here).
pub fn load_sets(j: &Json) -> Result<Vec<Set>, String> {
    let mut out = vec![];
    for s in j.arr() {
        let mut rules = vec![];
        for r in s.get("rules").arr() {
            let knobs = if r.get("knobs").is_obj() { Some(r.get("knobs").clone()) } else { None };
            if let Some(k) = knobs.as_ref() {
                Knobs::default().with(k).map_err(|e| format!("dispatch {:?}: {e}", s.get("name").str()))?;
                if k.obj().iter().any(|(n, _)| n == "dispatch_set") {
                    return Err("dispatch rules may not set dispatch_set".into());
                }
            }
            let opt = |k: &str| if r.get(k).is_null() { None } else { Some(r.get(k).i64()) };
            rules.push(Rule {
                pair: pats(r.get("pair")),
                cluster: pats(r.get("cluster")),
                route: opt("route"),
                table: match r.get("table").str() {
                    "" => None,
                    t @ ("new" | "old" | "new_swap" | "new_aa" | "new_bb") => Some(t.to_string()),
                    t => return Err(format!("dispatch table {t:?} (new | old | new_swap | new_aa | new_bb)")),
                },
                endgame: opt("endgame"),
                knobs,
                from_day: opt("from_day").unwrap_or(6).max(6),
            });
        }
        out.push(Set { name: s.get("name").str().to_string(), rules });
    }
    Ok(out)
}

fn pair_match(p: &str, pair: &str) -> bool {
    if p == "*" || p == pair {
        return true;
    }
    let (Some((a, b)), Some((x, y))) = (p.split_once('|'), pair.split_once('|')) else { return false };
    (a == "*" || a == x) && (b == "*" || b == y)
}

fn cluster_match(p: &str, c: &str) -> bool {
    p == "*" || c == p || c.starts_with(&format!("{p}_"))
}

#[derive(Clone, Default)]
pub struct Dispatch {
    pub sets: Vec<Set>,
    last_step: i64,
    pos_eq0: Vec<bool>,
    cash_eq1: Option<bool>,
    pos_eq5: Vec<bool>,
    pub cluster: Option<&'static str>,
    pub pair: Option<String>,
    /// (set, rule) chosen at step 144.
    pub rule: Option<(usize, usize)>,
    /// The original route 2 while an end-game override is installed.
    saved_endgame: Option<Route>,
    /// Rules whose route failed the sync check (diagnostic).
    pub sync_rejects: u64,
    /// Routes that may replace route 0 at step 144, computed from the PRISTINE tapes at load (PIPE
    /// rewrites route 0 in place at steps 57/91, like it does under the table router).
    pub sync_routes: Vec<i64>,
}

impl Dispatch {
    /// `ch` must hold the pristine (freshly loaded) routes.
    pub fn with_sets(sets: Vec<Set>, ch: &Chassis) -> Dispatch {
        let sync_routes = ch.routes.iter().map(|r| r.id).filter(|&r| sync_ok(ch, r)).collect();
        Dispatch { sets, last_step: -1, sync_routes, ..Default::default() }
    }

    /// A copy for a new game (same sets and sync list, fresh per-game state).
    pub fn fresh(&self) -> Dispatch {
        Dispatch { sets: self.sets.clone(), last_step: -1, sync_routes: self.sync_routes.clone(), ..Default::default() }
    }

    fn reset(&mut self, ch: &mut Chassis) {
        if let Some(r) = self.saved_endgame.take() {
            if let Some(i) = ch.route_idx(ENDGAME_ROUTE) {
                ch.routes[i] = r;
            }
        }
        *self = Dispatch {
            sets: std::mem::take(&mut self.sets),
            sync_rejects: self.sync_rejects,
            sync_routes: std::mem::take(&mut self.sync_routes),
            ..Default::default()
        };
    }

    /// Every step, before the chassis. `set` = the active `dispatch_set` knob (-1 = off).
    pub fn pre(&mut self, v: &View, ch: &mut Chassis, set: i64) {
        let step = v.step;
        if step == 0 || step <= self.last_step {
            self.reset(ch);
        }
        self.last_step = step;
        let (own, rival) = (v.farm(), v.rival());
        let eq = own.farmer == rival.farmer && own.hands == rival.hands;
        if (0..24).contains(&step) {
            self.pos_eq0.push(eq);
        }
        if step == 1 {
            self.cash_eq1 = Some(own.money == rival.money);
        }
        if (120..DECIDE_STEP).contains(&step) {
            self.pos_eq5.push(eq);
        }
        if step >= DECIDE_STEP && self.cluster.is_none() {
            self.decide(v, ch, set);
        }
    }

    fn decide(&mut self, v: &View, ch: &mut Chassis, set: i64) {
        let n0 = self.pos_eq0.iter().filter(|b| **b).count();
        let same = !self.pos_eq0.is_empty() && n0 * 10 >= self.pos_eq0.len();
        let g = if !same { "DIFF" } else if self.cash_eq1 == Some(true) { "COPY" } else { "PARTIAL" };
        let n5 = self.pos_eq5.iter().filter(|b| **b).count();
        let m = !self.pos_eq5.is_empty() && n5 * 10 >= self.pos_eq5.len() * 9;
        let cluster: &'static str = match (g, m) {
            ("DIFF", true) => "DIFF_M",
            ("DIFF", false) => "DIFF_S",
            ("PARTIAL", true) => "PARTIAL_M",
            ("PARTIAL", false) => "PARTIAL_S",
            ("COPY", true) => "COPY_M",
            _ => "COPY_S",
        };
        let pair = v.shops().iter().take(2).copied().collect::<Vec<_>>().join("|");
        self.cluster = Some(cluster);
        self.pair = Some(pair.clone());
        let p = v.obs.player;
        let mut applied_route = None;
        let mut applied_end = None;
        let mut sync = true;
        if set >= 0 {
            if let Some(s) = self.sets.get(set as usize) {
                let hit = s.rules.iter().position(|r| {
                    r.pair.iter().any(|x| pair_match(x, &pair)) && r.cluster.iter().any(|x| cluster_match(x, cluster))
                });
                if let Some(ri) = hit {
                    self.rule = Some((set as usize, ri));
                    let r = s.rules[ri].clone();
                    let route = r.route.or_else(|| {
                        let t = r.table.as_deref()?;
                        let (tab, d) = if t == "old" { (&ch.router.old, ch.router.default_old) } else { (&ch.router.new, ch.router.default_new) };
                        let (a, b) = pair.split_once('|').unwrap_or((&pair, &pair));
                        let key = match t {
                            "new_swap" => format!("{b}|{a}"),
                            "new_aa" => format!("{a}|{a}"),
                            "new_bb" => format!("{b}|{b}"),
                            _ => pair.clone(),
                        };
                        Some(tab.iter().find(|(k, _)| *k == key).map(|(_, v)| *v).unwrap_or(d))
                    });
                    if let Some(rt) = route {
                        sync = self.sync_routes.contains(&rt);
                        if sync {
                            if let Some(st) = ch.player(p) {
                                st.router.dispatch_route = Some(rt);
                            }
                            applied_route = Some(rt);
                        } else {
                            self.sync_rejects += 1;
                        }
                    }
                    if let Some(e) = r.endgame {
                        // the D6 route the router will pick this step (dispatch included)
                        let d6 = ch.players.iter().find(|(k, _)| *k == p).map(|(_, s)| {
                            let mut rs = s.router.clone();
                            ch.router.route(v, DECIDE_STEP, &mut rs)
                        });
                        let target = if e < 0 { d6 } else { Some(e) };
                        if let (Some(t), Some(i2)) = (target, ch.route_idx(ENDGAME_ROUTE)) {
                            if t != ENDGAME_ROUTE && ch.route_idx(t).is_some() {
                                let tape = ch.route(t).tape.clone();
                                self.saved_endgame = Some(std::mem::replace(&mut ch.routes[i2], Route::new(ENDGAME_ROUTE, tape)));
                            }
                            applied_end = Some(t);
                        }
                    }
                }
            }
        }
        if let Ok(path) = std::env::var("KAGG_DISPATCH_LOG") {
            let line = format!(
                "{{\"player\": {p}, \"step\": {}, \"pair\": {}, \"cluster\": {}, \"pos_eq0\": {}, \"pos_eq5\": {}, \"cash_eq1\": {}, \"set\": {set}, \"rule\": {}, \"route\": {}, \"endgame\": {}, \"sync\": {sync}}}",
                v.step,
                json::quote(&pair),
                json::quote(cluster),
                n0,
                n5,
                self.cash_eq1.unwrap_or(false),
                self.rule.map(|r| r.1 as i64).unwrap_or(-1),
                applied_route.map(|r| r.to_string()).unwrap_or("null".into()),
                applied_end.map(|r| r.to_string()).unwrap_or("null".into()),
            );
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                let _ = writeln!(f, "{line}");
            }
        }
    }

    /// Knob overrides for `day` (None = none).
    pub fn knobs(&self, day: i64) -> Option<&Json> {
        let (s, r) = self.rule?;
        let rule = &self.sets[s].rules[r];
        if day >= rule.from_day { rule.knobs.as_ref() } else { None }
    }
}

/// A route may replace route 0 at step 144 only if its farm actions on steps 1..143 are route 0's.
pub fn sync_ok(ch: &Chassis, route: i64) -> bool {
    let (Some(a), Some(b)) = (ch.route_idx(0), ch.route_idx(route)) else { return false };
    let (a, b) = (&ch.routes[a].tape, &ch.routes[b].tape);
    (1..DECIDE_STEP as usize).all(|t| match (a.get(t), b.get(t)) {
        (Some(x), Some(y)) => x.farmer == y.farmer && x.hands == y.hands,
        _ => false,
    })
}
