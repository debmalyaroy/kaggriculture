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
    /// The rival cluster key read at `cluster_step` (None before / when the router has no cluster table).
    pub cluster: Option<u64>,
    pub clustered: bool,
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
    /// Route played before `select_step` (the opening); 0 = the base's own opening route.
    pub opening: i64,
    /// Per (world, rival cluster) overrides {"SHOP1|SHOP2#<cluster hex>": route}, applied once at `cluster_step`
    /// (cluster::econ_key of the rival's farm); a pair not in the table keeps the world route.
    pub cluster_routes: Vec<(String, i64)>,
    pub cluster_step: i64,
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
    /// Play route `r` in every world (route screening).
    pub fn force(&mut self, r: i64) {
        self.new.iter_mut().for_each(|(_, v)| *v = r);
        self.old.iter_mut().for_each(|(_, v)| *v = r);
        self.v92.clear();
        self.cluster_routes.clear();
        self.default_new = r;
        self.default_old = r;
    }

    /// Per-world overrides {"SHOP1|SHOP2": route} on top of every table (a refitted world table).
    pub fn override_worlds(&mut self, j: &Json) {
        for (k, v) in j.obj() {
            let r = v.i64();
            match self.v92.iter_mut().find(|(n, _)| n == k) {
                Some(e) => e.1 = r,
                None => self.v92.push((k.clone(), r)),
            }
        }
    }

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
            opening: num("opening_route", 0),
            cluster_routes: table(j.get("cluster_routes")),
            cluster_step: num("cluster_step", 145),
        }
    }

    pub fn route(&self, v: &View, step: i64, s: &mut RouterState) -> i64 {
        if step == 2 {
            let wheat = v.obs.mkt_inventory.iter().find(|(k, _)| *k == "WHEAT");
            s.rkey = wheat.map(|(_, w)| (round3(v.rival().money), *w));
            s.rkey_set = true;
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
            s.route = Some(r);
            s.day6 = true;
        }
        if !self.cluster_routes.is_empty() && step >= self.cluster_step && !s.clustered && s.day6 {
            s.clustered = true;
            let k = crate::cluster::econ_key(v.rival());
            s.cluster = Some(k);
            // most specific key first: exact layout, exact counts, bucketed counts
            let w = v.shops().iter().take(2).copied().collect::<Vec<_>>().join("|");
            for key in crate::cluster::keys(v.rival()) {
                if let Some(r) = lookup(&self.cluster_routes, &format!("{w}#{key}")) {
                    s.route = Some(r);
                    break;
                }
            }
        }
        if step >= self.endgame_step && !s.day27 {
            s.route = Some(self.endgame_route);
            s.day27 = true;
        }
        s.route.unwrap_or(self.opening)
    }
}
