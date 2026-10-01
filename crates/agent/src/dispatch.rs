//! Specialist dispatcher (v63.12 A1 + A3): at D6 (step `at`, default 144) the rival class and the world pick ONE
//! package of market-side / policy configs for the rest of the game. Only configs are swapped (the tape, the farm plan
//! and every layer's memory stay), so each package stays in sync with the base exactly like a group profile does.
//!
//!   class (A2): DIFF / PARTIAL / COPY from the group controller (crate::layers::group, fixed at step 25); a COPY rival
//!               whose money equalled ours on >= `lin_cash` of the steps 1..144 is LINEAGE (a build of our own stack),
//!               else CLONE (a public copy of the meta route).
//!   world (A3): the first two unlocked shops, "A|B", as observed at step `at`.
//! dispatch.json: {"at":144, "lin_cash":0.9,
//!   "packages": {"clone": {"gt":"gtS.json", "policy":"clone.bin", "rshell":"rs.json", "preempt":"pre.json"}, ...},
//!   "rules": [{"class":"CLONE", "world":"*", "pkg":"clone"}, ...]}      first matching rule wins; paths are relative to
//! the dispatch file. A class rule "COPY" matches LINEAGE and CLONE.
use crate::view::View;
use std::sync::Arc;

#[derive(Clone, Debug, Default)]
pub struct Package {
    pub name: String,
    pub gt: Option<Arc<crate::gt::GtCfg>>,
    pub policy: Option<Arc<policy::Net>>,
    pub rshell: Option<Arc<crate::rshell::RConfig>>,
    pub preempt: Option<Arc<crate::preempt::PreCfg>>,
    /// endgame controller config (e.g. a forced proposal for a world specialist, F7)
    pub endg: Option<Arc<crate::endg::EndgCfg>>,
}

#[derive(Clone, Debug, Default)]
pub struct DispatchCfg {
    pub at: i64,
    pub lin_cash: f64,
    pub pkgs: Vec<Package>,
    /// (class, world, package index)
    pub rules: Vec<(String, String, usize)>,
}

impl DispatchCfg {
    pub fn load(path: &str) -> Result<DispatchCfg, String> {
        let j = kagg_engine::json::parse(&std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?)?;
        let dir = std::path::Path::new(path).parent().map(|p| p.to_path_buf()).unwrap_or_default();
        let rel = |f: &str| dir.join(f).to_string_lossy().into_owned();
        let mut pkgs = vec![];
        for (name, p) in j.get("packages").obj() {
            let s = |k: &str| if p.get(k).is_null() { None } else { Some(rel(p.get(k).str())) };
            pkgs.push(Package {
                name: name.clone(),
                gt: s("gt").map(|f| crate::gt::GtCfg::load(&f).map(Arc::new)).transpose()?,
                policy: s("policy").map(|f| policy::Net::load(&f).map(Arc::new)).transpose()?,
                rshell: s("rshell").map(|f| crate::rshell::RConfig::load(&f).map(Arc::new)).transpose()?,
                preempt: s("preempt").map(|f| crate::preempt::PreCfg::load(&f).map(Arc::new)).transpose()?,
                endg: s("endg").map(|f| crate::endg::EndgCfg::load(&f).map(Arc::new)).transpose()?,
            });
        }
        let mut rules = vec![];
        for r in j.get("rules").arr() {
            let pk = r.get("pkg").str();
            let i = pkgs.iter().position(|p| p.name == pk).ok_or(format!("dispatch: no package {pk}"))?;
            let w = if r.get("world").is_null() { "*".to_string() } else { r.get("world").str().to_string() };
            rules.push((r.get("class").str().to_string(), w, i));
        }
        let n = |k: &str, d: f64| if j.get(k).is_null() { d } else { j.get(k).f64() };
        Ok(DispatchCfg { at: n("at", 144.0) as i64, lin_cash: n("lin_cash", 0.9), pkgs, rules })
    }
}

#[derive(Clone, Debug, Default)]
pub struct Dispatch {
    pub cfg: Arc<DispatchCfg>,
    /// (class, world, chosen package name) once decided
    pub chosen: Option<(String, String, String)>,
}

impl Dispatch {
    pub fn new(cfg: Arc<DispatchCfg>) -> Dispatch {
        Dispatch { cfg, chosen: None }
    }

    /// A2: the rival class at D6.
    pub fn class(group: Option<usize>, tr: &crate::gt::RivalTracker, lin_cash: f64) -> &'static str {
        match group {
            Some(0) => "DIFF",
            Some(1) => "PARTIAL",
            Some(2) => {
                if tr.fp_n > 0 && tr.fp_cash as f64 / tr.fp_n as f64 >= lin_cash {
                    "LINEAGE"
                } else {
                    "CLONE"
                }
            }
            _ => "NONE",
        }
    }

    /// Returns the package to install, once, at the first step >= `at`.
    pub fn decide(&mut self, v: &View, group: Option<usize>, tr: &crate::gt::RivalTracker) -> Option<Package> {
        if self.chosen.is_some() || v.step < self.cfg.at {
            return None;
        }
        let class = Self::class(group, tr, self.cfg.lin_cash);
        let sh = &v.obs.shops;
        let world = format!("{}|{}", sh.first().copied().unwrap_or("-"), sh.get(1).copied().unwrap_or("-"));
        let hit = self.cfg.rules.iter().find(|(c, w, _)| {
            (c == "*" || c == class || (c == "COPY" && (class == "LINEAGE" || class == "CLONE"))) && (w == "*" || *w == world)
        });
        let pkg = hit.map(|r| self.cfg.pkgs[r.2].clone());
        self.chosen = Some((class.to_string(), world, pkg.as_ref().map(|p| p.name.clone()).unwrap_or_else(|| "-".into())));
        pkg
    }
}
