//! `agent`: the Rust port of the v61.1 agent (docs/port.md).
//!
//! * [`act`] — Kaggle action token lists, kept verbatim.
//! * [`view`] — the per-step observation snapshot (engine `State` + helpers).
//! * [`router`] — route selection by the first two shops.
//! * [`chassis`] — the route-replay chassis and its reactive layers.
//! * [`base`] — a base agent loaded from `configs/bases/<name>/`, plus the `_SHOP` rescue.
//!
//! Layer chain status: chassis + `_SHOP` (layer 1 of 69). Later layers are appended to the
//! chain in Python call order (P2.7-P2.12).
pub mod act;
pub mod base;
pub mod chassis;
pub mod disguise;
pub mod layers;
pub mod market;
pub mod obs;
pub mod router;
pub mod sim;
pub mod terminal;
pub mod trie;
pub mod view;


/// The agent as the runtime sees it: one observation JSON in, one action out.
pub struct Agent {
    pub base: base::Base,
}

impl Agent {
    pub fn load(base_dir: &str) -> Result<Agent, String> {
        Ok(Agent { base: base::Base::load(base_dir)? })
    }
    /// Replace the base between turns (per-player state starts fresh on the new base).
    pub fn swap_base(&mut self, base_dir: &str) -> Result<(), String> {
        let mut b = base::Base::load(base_dir)?;
        // keep the lever-profile table and controller settings across the swap
        b.chain.profiles = std::mem::take(&mut self.base.chain.profiles);
        b.chain.schedule = self.base.chain.schedule.take();
        b.chain.clone_profile = self.base.chain.clone_profile;
        b.chain.clone_strict = self.base.chain.clone_strict;
        b.chain.afr_profile = self.base.chain.afr_profile;
        b.chain.afr_trigger = self.base.chain.afr_trigger;
        b.chain.group_ctl = self.base.chain.group_ctl.take();
        b.chain.dispatch = layers::dispatch::Dispatch::with_sets(std::mem::take(&mut self.base.chain.dispatch.sets), &b.chassis);
        b.cut = self.base.cut;
        self.base = b;
        Ok(())
    }
    /// Replace the lever-profile table (and its dispatch sets) between games; `profile` fixes the
    /// schedule like `--profile`.
    pub fn swap_profiles(&mut self, path: &str, profile: Option<usize>) -> Result<(), String> {
        self.base.set_profiles(path, profile)
    }
    pub fn act_json(&mut self, obs_text: &str) -> String {
        match obs::Obs::parse(obs_text) {
            Ok(obs) => self.base.act(obs).0.dump(),
            Err(_) => act::Action::pass().dump(),
        }
    }
}
