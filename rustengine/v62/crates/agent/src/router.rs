//! The v61.1 route selector (`_router`, agent lines 958-975), driven by `router.json`.
//!
//! At step >= `select_step` (144, day 6: both first shops known) pick the route from the
//! first two unlocked shops: worlds with a YARN_STORE use the old (V39) table, the others
//! the new (EXP240) table, then the V92 table overrides both. From `endgame_step` (648)
//! the endgame route. Before step 144 the route is 0.
use crate::view::View;
use kagg_engine::json::Json;

#[derive(Clone, Debug, Default)]
pub struct RouterState {
    /// Step-2 opponent fingerprint `(round(rival money, 3), market WHEAT)` (read by later layers).
    pub rkey: Option<(f64, i64)>,
    pub rkey_set: bool,
    pub expert: Option<&'static str>,
    pub route: Option<i64>,
    pub day6: bool,
    pub day27: bool,
    /// D6 dispatch (layers/dispatch.rs): replaces the table route at `select_step` (sync-checked there).
    pub dispatch_route: Option<i64>,
    /// Trie mode (team bases): candidate tapes and the one followed.
    pub trie: crate::trie::TrieState,
}

#[derive(Clone, Debug)]
pub struct ShopRouter {
    pub new: Vec<(String, i64)>,
    pub old: Vec<(String, i64)>,
    pub v92: Vec<(String, i64)>,
    pub yarn_uses_old: bool,
    pub default_new: i64,
    pub default_old: i64,
    pub select_step: i64,
    pub endgame_step: i64,
    pub endgame_route: i64,
    /// `"mode": "trie"` (a team base): route by the tape trie instead of the shop tables.
    pub trie: Option<std::sync::Arc<crate::trie::Trie>>,
}

fn table(j: &Json) -> Vec<(String, i64)> {
    j.obj().iter().map(|(k, v)| (k.clone(), v.i64())).collect()
}
fn lookup(t: &[(String, i64)], k: &str) -> Option<i64> {
    t.iter().find(|(n, _)| n == k).map(|(_, v)| *v)
}

/// Python `round(x, 3)` for a money value (exact for the cents-granular values the engine
/// produces; ties go to even like CPython).
pub fn round3(x: f64) -> f64 {
    let s = x * 1000.0;
    let r = s.round();
    let r = if (s - s.trunc()).abs() == 0.5 && r % 2.0 != 0.0 { r - s.signum() } else { r };
    r / 1000.0
}

impl ShopRouter {
    pub fn from_json(j: &Json) -> ShopRouter {
        let num = |k: &str, d: i64| if j.get(k).is_null() { d } else { j.get(k).i64() };
        ShopRouter {
            new: table(j.get("shop_routes_new")),
            old: table(j.get("shop_routes_old")),
            v92: table(j.get("shop_routes_v92")),
            yarn_uses_old: j.get("yarn_uses_old").is_null() || j.get("yarn_uses_old").bool(),
            default_new: num("default_new", 100),
            default_old: num("default_old", 0),
            select_step: num("select_step", 144),
            endgame_step: num("endgame_step", 648),
            endgame_route: num("endgame_route", 2),
            trie: None,
        }
    }

    pub fn route(&self, v: &View, step: i64, s: &mut RouterState) -> i64 {
        if step == 2 {
            let wheat = v.obs.mkt_inventory.iter().find(|(k, _)| *k == "WHEAT");
            s.rkey = wheat.map(|(_, w)| (round3(v.rival().money), *w));
            s.rkey_set = true;
        }
        if let Some(t) = &self.trie {
            let r = t.route(v, step, &mut s.trie);
            s.route = Some(r);
            return r;
        }
        if step >= self.select_step && !s.day6 {
            let shops: Vec<&str> = v.shops().iter().take(2).copied().collect();
            let key = shops.join("|");
            let use_new = !(self.yarn_uses_old && shops.contains(&"YARN_STORE"));
            s.expert = Some(if use_new { "EXP240" } else { "V39" });
            let mut r = if use_new {
                lookup(&self.new, &key).unwrap_or(self.default_new)
            } else {
                lookup(&self.old, &key).unwrap_or(self.default_old)
            };
            if let Some(o) = lookup(&self.v92, &key) {
                r = o;
            }
            if let Some(o) = s.dispatch_route {
                r = o;
            }
            s.route = Some(r);
            s.day6 = true;
        }
        if step >= self.endgame_step && !s.day27 {
            s.route = Some(self.endgame_route);
            s.day27 = true;
        }
        s.route.unwrap_or(0)
    }
}
