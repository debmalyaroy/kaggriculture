//! Trie router: an agent whose routes are one player's recorded tapes (a "team base", built by
//! `runner/bin/teambase.rs` from GM replays), dispatched desync-free.
//!
//! The candidates start as every route. At step t the candidates are grouped by their recorded
//! ACTION at t (farmer + hands + market); when they disagree the tapes BRANCH, and the router keeps the
//! group whose recorded situation at that step is nearest the live one (`feats`: money, crews,
//! both boards, copy signal, shops), with tapes from another world (shops seen so far) penalised.
//! Within a group every action so far is identical, so the router may follow any member without
//! desync; it follows the member from the right world, nearest at the last branch, then the one
//! whose game was won. Off unless router.json has `"mode": "trie"`.
use crate::chassis::Route;
use crate::view::View;
use kagg_engine::json::Json;

pub const K: usize = 16;

/// The live situation, from the seat's observation only (both boards and moneys are public).
pub fn feats(v: &View) -> [f32; K] {
    let mut f = [0f32; K];
    let (me, rv) = (v.farm(), v.rival());
    let board = |fo: &crate::obs::FarmObs| {
        let (mut plants, mut animals, mut structs) = (0f32, 0f32, 0f32);
        for t in &fo.tiles {
            if !t.crop.is_empty() {
                plants += 1.0;
            }
            if !t.animal.is_empty() {
                animals += 1.0;
            }
            if t.kind == "COOP" || t.kind == "PASTURE" {
                structs += 1.0;
            }
        }
        (plants, animals, structs)
    };
    let (mp, ma, ms) = board(me);
    let (rp, ra, rs) = board(rv);
    let mine: Vec<(i64, i64)> = me.farmer.into_iter().chain(me.hands.iter().copied()).collect();
    let theirs: Vec<(i64, i64)> = rv.farmer.into_iter().chain(rv.hands.iter().copied()).collect();
    f[0] = (me.money / 1000.0) as f32;
    f[1] = (rv.money / 1000.0) as f32;
    f[2] = mine.len() as f32;
    f[3] = theirs.len() as f32;
    f[4] = rv.hires_today as f32;
    f[5] = me.quadrants.len() as f32;
    f[6] = rv.quadrants.len() as f32;
    f[7] = mp / 10.0;
    f[8] = ma;
    f[9] = ms;
    f[10] = rp / 10.0;
    f[11] = ra;
    f[12] = rs;
    f[13] = theirs.iter().filter(|p| mine.contains(p)).count() as f32 * 2.0;
    f[14] = v.obs.mkt_inventory.iter().find(|(k, _)| *k == "WHEAT").map_or(0.0, |(_, n)| *n as f32 / 10.0);
    f[15] = v.shops().len() as f32;
    f
}

fn dist(a: &[f32; K], b: &[f32; K]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
}

/// FNV-1a over one step's FULL action (farmer + hands + market orders). Two tapes are in sync only
/// while every order so far is identical: equal unit moves with different market histories (a cow
/// bought on step 40 by one tape, on step 50 by the other) are not interchangeable.
pub fn unit_key(a: &crate::act::Action) -> u64 {
    let s = a.dump();
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
    }
    h
}

#[derive(Clone, Debug)]
pub struct TrieRoute {
    pub id: i64,
    /// realized first two shops of the recorded game, "A|B"
    pub world: Vec<String>,
    pub win: f32,
    pub keys: Vec<u64>,
    /// recorded situation at this route's branch steps: (step, feats)
    pub splits: Vec<(i64, [f32; K])>,
}

#[derive(Clone, Debug)]
pub struct Trie {
    pub routes: Vec<TrieRoute>,
    pub world_pen: f32,
    /// groups whose nearest member is within this of the best count as tied (then support decides)
    pub tie_eps: f32,
    /// support per world-consistent member = base_support + result (0.5 counts every tape, 0 only wins)
    pub base_support: f32,
}

#[derive(Clone, Debug, Default)]
pub struct TrieState {
    pub cands: Vec<usize>,
    pub cur: usize,
    pub last_split: i64,
    pub last_dist: Vec<f32>,
    pub branches: u32,
    pub shops_seen: usize,
}

impl Trie {
    /// `j` = router.json `"trie"`; `routes` = the base's parsed routes (keys are computed from them).
    pub fn from_json(j: &Json, routes: &[Route]) -> Option<Trie> {
        if !j.is_obj() {
            return None;
        }
        let mut out = vec![];
        for r in j.get("routes").arr() {
            let id = r.get("id").i64();
            let route = routes.iter().find(|x| x.id == id)?;
            let world: Vec<String> = r.get("world").str().split('|').filter(|s| !s.is_empty()).map(|s| s.to_string()).collect();
            let keys = (0..720).map(|t| route.at(t).map(unit_key).unwrap_or(0)).collect();
            let mut splits = vec![];
            for (k, v) in r.get("splits").obj() {
                let mut f = [0f32; K];
                for (i, x) in v.arr().iter().enumerate().take(K) {
                    f[i] = x.f64() as f32;
                }
                splits.push((k.parse().ok()?, f));
            }
            splits.sort_by_key(|s| s.0);
            out.push(TrieRoute { id, world, win: r.get("win").f64() as f32, keys, splits });
        }
        let world_pen = if j.get("world_pen").is_null() { 20.0 } else { j.get("world_pen").f64() as f32 };
        let tie_eps = if j.get("tie_eps").is_null() { 1.0 } else { j.get("tie_eps").f64() as f32 };
        let base_support = if j.get("base_support").is_null() { 0.5 } else { j.get("base_support").f64() as f32 };
        (!out.is_empty()).then_some(Trie { routes: out, world_pen, tie_eps, base_support })
    }

    fn world_ok(&self, k: usize, shops: &[&str]) -> bool {
        let w = &self.routes[k].world;
        shops.iter().take(2).enumerate().all(|(i, s)| w.get(i).is_some_and(|x| x == s))
    }

    fn feat_at(&self, k: usize, step: i64) -> Option<&[f32; K]> {
        self.routes[k].splits.iter().find(|s| s.0 == step).map(|s| &s.1)
    }

    fn pick(&self, s: &TrieState, shops: &[&str]) -> usize {
        let mut best = (true, f32::INFINITY, f32::NEG_INFINITY, s.cands[0]);
        for (i, &k) in s.cands.iter().enumerate() {
            let d = s.last_dist.get(i).copied().unwrap_or(0.0);
            let cand = (!self.world_ok(k, shops), d, self.routes[k].win, k);
            if (cand.0, cand.1, -cand.2) < (best.0, best.1, -best.2) {
                best = cand;
            }
        }
        best.3
    }

    /// Route id to follow at `step`.
    pub fn route(&self, v: &View, step: i64, s: &mut TrieState) -> i64 {
        let shops: Vec<&str> = v.shops().iter().take(2).copied().collect();
        if s.cands.is_empty() || step == 0 {
            *s = TrieState { cands: (0..self.routes.len()).collect(), ..Default::default() };
            s.last_dist = vec![0.0; s.cands.len()];
            s.cur = self.pick(s, &shops);
        }
        let t = step as usize;
        // group the candidates by their recorded unit moves at this step
        let mut groups: Vec<(u64, Vec<usize>)> = vec![];
        for &k in &s.cands {
            let key = self.routes[k].keys.get(t).copied().unwrap_or(0);
            match groups.iter_mut().find(|g| g.0 == key) {
                Some(g) => g.1.push(k),
                None => groups.push((key, vec![k])),
            }
        }
        if groups.len() > 1 {
            let live = feats(v);
            // per group: distances of its members (world-penalised) and its SUPPORT = sum over the
            // world-consistent members of (0.5 + result). Among the groups whose nearest member is within
            // `tie_eps` of the best, the best-supported wins: a branch the situation cannot separate
            // (another submission version, a hidden cause) keeps the most options open.
            let scored: Vec<(f32, f32, Vec<f32>)> = groups
                .iter()
                .map(|(_, members)| {
                    let ds: Vec<f32> = members
                        .iter()
                        .map(|&k| self.feat_at(k, step).map_or(1e4, |f| dist(f, &live)) + if self.world_ok(k, &shops) { 0.0 } else { self.world_pen })
                        .collect();
                    let support: f32 = members.iter().filter(|&&k| self.world_ok(k, &shops)).map(|&k| self.base_support + self.routes[k].win).sum();
                    (ds.iter().copied().fold(f32::INFINITY, f32::min), support, ds)
                })
                .collect();
            let m = scored.iter().map(|x| x.0).fold(f32::INFINITY, f32::min);
            // tie_eps < 0: nearest group only, ties to the first group (the recorded library order)
            let gi = if self.tie_eps < 0.0 {
                (0..scored.len()).find(|&i| scored[i].0 <= m).unwrap_or(0)
            } else {
                (0..scored.len()).filter(|&i| scored[i].0 <= m + self.tie_eps).max_by(|&a, &b| scored[a].1.total_cmp(&scored[b].1)).unwrap_or(0)
            };
            let (bd, _, ds) = (scored[gi].0, scored[gi].1, scored[gi].2.clone());
            if std::env::var_os("KAGG_TRIE_DEBUG").is_some() {
                let mins: Vec<String> = groups
                    .iter()
                    .map(|(_, m)| {
                        let d = m.iter().map(|&k| self.feat_at(k, step).map_or(1e4, |f| dist(f, &live))).fold(f32::INFINITY, f32::min);
                        format!("{}:{:.2}", m.len(), d)
                    })
                    .collect();
                eprintln!("TRIE step {step} groups [{}] chose #{gi} d {bd:.2} shops {:?}", mins.join(" "), shops);
            }
            s.cands = groups.swap_remove(gi).1;
            s.last_dist = ds;
            s.last_split = step;
            s.branches += 1;
            s.cur = self.pick(s, &shops);
        } else if shops.len() != s.shops_seen || !s.cands.contains(&s.cur) {
            s.cur = self.pick(s, &shops);
        }
        s.shops_seen = shops.len();
        self.routes[s.cur].id
    }
}
