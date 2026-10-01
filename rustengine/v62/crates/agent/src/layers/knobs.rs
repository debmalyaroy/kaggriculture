//! Layer knobs and lever profiles (PLAN.md §4, §17.4-17.6; tasks P3.1/P3.2).
//!
//! A profile overrides market-side / timing knobs only (WHEN and HOW MUCH we sell), plus a few
//! labour-safe on/off switches — never tape planting, hiring, land or animals. The active profile
//! changes only at a day boundary (hour 1, step 24d+1) and holds for the day; every game starts
//! on profile 0, which is v61.1 exactly (`Knobs::default()`).
use kagg_engine::json::Json;

#[derive(Clone, Debug, PartialEq)]
pub struct Knobs {
    // RSA route sale advance (7494)
    pub rsa_on: bool,
    pub rsa_look: i64,
    pub rsa_min_frac: f64,
    // window lead-sells EV (15-20h) / DP (0-2h) / MP (10-13h), MPX model lead
    pub ev_on: bool,
    pub ev_h: i64,
    pub dp_on: bool,
    pub dp_h: i64,
    pub mp_on: bool,
    pub mp_h: i64,
    pub mpx_on: bool,
    // order-book reorders
    pub cxd_on: bool,
    pub v44y_on: bool,
    // ADV ready-stock advance
    pub adv_on: bool,
    pub adv_look: i64,
    // RACE clone horizons (5440; clone level reassigned to 9 at 5692)
    pub race_clone: i64,
    pub race_escalated: i64,
    pub race_mirror: i64,
    // V9 RACE reservation horizon clamp
    pub v9_race_default: i64,
    pub v9_race_max: i64,
    pub v9_race_margin: i64,
    // glut gates: lead-sell (chassis RACEPX) and reservation (RACEGATE)
    pub racepx_margin: i64,
    pub racegate_margin: i64,
    // labour-safe switches
    pub r85_feed_on: bool,
    pub e410_on: bool,
    pub r51_input_on: bool,
    pub v9_fert_on: bool,
    // AFR anti-front-run (ours; off = v61.1)
    pub afr_on: bool,
    pub afr_extra: i64,
    pub afr_jitter: i64,
    pub afr_hold: i64,
    pub afr_look_max: i64,
    pub afr_min_frac: f64,
    // CARROT2 "carrot pays" margin and ORDERPRI2 slot-swap margin (v61.1 -20 / 12; herd_safe ca25: -25..-15 / 8)
    pub ca_margin: f64,
    pub or2_slot_margin: f64,
    // terminal closure planner (v61.1: on, start 712, 64 sims, 1 pass, 4 proposals per actor)
    pub term_on: bool,
    pub term_start: i64,
    pub term_sims: i64,
    pub term_passes: i64,
    pub term_props: i64,
    // v92 rival-sales predictor (ca25/shepherds; off = v61.1). ext_window 0 = ca25, 4 = shepherds
    pub v92_on: bool,
    pub v92_h: i64,
    pub v92_k: i64,
    pub v92_every: i64,
    pub v92_top: i64,
    pub v92_ext_window: i64,
    // counter D rival model: 0 = copy (v61.1), 1 = v92 forecast, 2 = worst case of both
    pub cxd_model: i64,
    // v29 market pressure (off = v61.1)
    pub press_on: bool,
    pub press_from: i64,
    pub press_trigger: f64,
    pub press_h: i64,
    pub press_tranche: i64,
    pub press_max: i64,
    // end-game sale-timing search (off = v61.1)
    pub tsell_on: bool,
    pub tsell_from: i64,
    pub tsell_window: i64,
    pub tsell_model: i64,
    pub tsell_min: f64,
    // D6 dispatch rule set (layers/dispatch.rs; -1 = off = v61.1)
    pub dispatch_set: i64,
    // shepherds_ledger feed-risk pickup (off = v61.1)
    pub hfeed_on: bool,
    /// crate::disguise: zero-quantity orders that change our action stream, not the game
    pub disguise_stream: bool,
    /// crate::disguise: sell N wheat at step 1 so our visible farm never exactly equals a copy's (0 = off)
    pub disguise_money: i64,
    /// Chain stages bypassed in both phases (bit i = `CUTS[i]`; 0 = none = v61.1). Built from a list of stage
    /// names or indices (`"skip": ["ca", "fx", 37]`); used for Rust stand-ins of public agents of our family.
    pub skip: u128,
}

impl Default for Knobs {
    fn default() -> Self {
        Knobs {
            rsa_on: true,
            rsa_look: 5,
            rsa_min_frac: 0.5,
            ev_on: true,
            ev_h: 8,
            dp_on: true,
            dp_h: 8,
            mp_on: true,
            mp_h: 8,
            mpx_on: true,
            cxd_on: true,
            v44y_on: true,
            adv_on: true,
            adv_look: 3,
            race_clone: 9,
            race_escalated: 24,
            race_mirror: 24,
            v9_race_default: 44,
            v9_race_max: 48,
            v9_race_margin: 12,
            racepx_margin: 0,
            racegate_margin: 0,
            r85_feed_on: true,
            e410_on: true,
            r51_input_on: true,
            v9_fert_on: true,
            afr_on: false,
            afr_extra: 1,
            afr_jitter: 2,
            afr_hold: 48,
            afr_look_max: 24,
            afr_min_frac: 0.5,
            ca_margin: -20.0,
            or2_slot_margin: 12.0,
            term_on: true,
            term_start: 712,
            term_sims: 64,
            term_passes: 1,
            term_props: 4,
            v92_on: false,
            v92_h: 48,
            v92_k: 4,
            v92_every: 2,
            v92_top: 1,
            v92_ext_window: 0,
            cxd_model: 0,
            press_on: false,
            press_from: 456,
            press_trigger: 2.0,
            press_h: 24,
            press_tranche: 4,
            press_max: 18,
            tsell_on: false,
            tsell_from: 672,
            tsell_window: 24,
            tsell_model: 0,
            tsell_min: 1.0,
            dispatch_set: -1,
            hfeed_on: false,
            disguise_stream: false,
            disguise_money: 0,
            skip: 0,
        }
    }
}

impl Knobs {
    /// `self` with the keys present in `j` overridden. Unknown keys are an error (typos must
    /// not silently become profile 0).
    pub fn with(&self, j: &Json) -> Result<Knobs, String> {
        let mut k = self.clone();
        for (key, v) in j.obj() {
            let b = || v.bool();
            let i = || v.i64();
            match key.as_str() {
                "name" | "note" => {}
                "rsa_on" => k.rsa_on = b(),
                "rsa_look" => k.rsa_look = i(),
                "rsa_min_frac" => k.rsa_min_frac = v.f64(),
                "ev_on" => k.ev_on = b(),
                "ev_h" => k.ev_h = i(),
                "dp_on" => k.dp_on = b(),
                "dp_h" => k.dp_h = i(),
                "mp_on" => k.mp_on = b(),
                "mp_h" => k.mp_h = i(),
                "mpx_on" => k.mpx_on = b(),
                "cxd_on" => k.cxd_on = b(),
                "v44y_on" => k.v44y_on = b(),
                "adv_on" => k.adv_on = b(),
                "adv_look" => k.adv_look = i(),
                "race_clone" => k.race_clone = i(),
                "race_escalated" => k.race_escalated = i(),
                "race_mirror" => k.race_mirror = i(),
                "v9_race_default" => k.v9_race_default = i(),
                "v9_race_max" => k.v9_race_max = i(),
                "v9_race_margin" => k.v9_race_margin = i(),
                "racepx_margin" => k.racepx_margin = i(),
                "racegate_margin" => k.racegate_margin = i(),
                "r85_feed_on" => k.r85_feed_on = b(),
                "e410_on" => k.e410_on = b(),
                "r51_input_on" => k.r51_input_on = b(),
                "v9_fert_on" => k.v9_fert_on = b(),
                "afr_on" => k.afr_on = b(),
                "afr_extra" => k.afr_extra = i(),
                "afr_jitter" => k.afr_jitter = i(),
                "afr_hold" => k.afr_hold = i(),
                "afr_look_max" => k.afr_look_max = i(),
                "afr_min_frac" => k.afr_min_frac = v.f64(),
                "ca_margin" => k.ca_margin = v.f64(),
                "or2_slot_margin" => k.or2_slot_margin = v.f64(),
                "term_on" => k.term_on = b(),
                "term_start" => k.term_start = i(),
                "term_sims" => k.term_sims = i(),
                "term_passes" => k.term_passes = i(),
                "term_props" => k.term_props = i(),
                "v92_on" => k.v92_on = b(),
                "v92_h" => k.v92_h = i(),
                "v92_k" => k.v92_k = i(),
                "v92_every" => k.v92_every = i(),
                "v92_top" => k.v92_top = i(),
                "v92_ext_window" => k.v92_ext_window = i(),
                "cxd_model" => k.cxd_model = i(),
                "press_on" => k.press_on = b(),
                "press_from" => k.press_from = i(),
                "press_trigger" => k.press_trigger = v.f64(),
                "press_h" => k.press_h = i(),
                "press_tranche" => k.press_tranche = i(),
                "press_max" => k.press_max = i(),
                "tsell_on" => k.tsell_on = b(),
                "tsell_from" => k.tsell_from = i(),
                "tsell_window" => k.tsell_window = i(),
                "tsell_model" => k.tsell_model = i(),
                "tsell_min" => k.tsell_min = v.f64(),
                "dispatch_set" => k.dispatch_set = i(),
                "hfeed_on" => k.hfeed_on = b(),
                "disguise_stream" => k.disguise_stream = b(),
                "disguise_money" => k.disguise_money = i(),
                "skip" => {
                    k.skip = 0;
                    for x in v.arr() {
                        let idx = if x.is_str() {
                            super::cut_index(x.str()).ok_or(format!("skip: unknown stage {:?}", x.str()))?
                        } else {
                            x.i64() as usize
                        };
                        if idx == 0 || idx >= super::CUTS.len() {
                            return Err(format!("skip: stage index {idx} out of range"));
                        }
                        k.skip |= 1u128 << idx;
                    }
                }
                other => return Err(format!("unknown knob {other:?}")),
            }
        }
        Ok(k)
    }
}

impl Knobs {
    /// A random member of the "clone lineage" population: v61.1 with random market/timing knobs
    /// (labour switches stay default). `r` yields uniform u64s.
    pub fn random(r: &mut impl FnMut() -> u64) -> Knobs {
        let mut pick = |n: u64| (r() % n) as i64;
        let mut k = Knobs::default();
        k.rsa_on = pick(10) != 0;
        k.rsa_look = 2 + pick(8);
        k.rsa_min_frac = [0.0, 0.3, 0.5, 0.8][pick(4) as usize];
        k.ev_on = pick(5) != 0;
        k.ev_h = 4 + pick(13);
        k.dp_on = pick(5) != 0;
        k.dp_h = 4 + pick(13);
        k.mp_on = pick(5) != 0;
        k.mp_h = 4 + pick(13);
        k.mpx_on = pick(5) != 0;
        k.cxd_on = pick(5) != 0;
        k.v44y_on = pick(5) != 0;
        k.adv_on = pick(5) != 0;
        k.adv_look = 1 + pick(5);
        k.race_clone = [9, 24][pick(2) as usize];
        k.v9_race_default = 24 + pick(37);
        k.v9_race_max = k.v9_race_default + pick(25);
        k.racepx_margin = [-20, 0, 0, 20][pick(4) as usize];
        k.racegate_margin = [-20, 0, 0, 20][pick(4) as usize];
        k
    }
}

/// A profile table: `{"profiles": [{"name": ..., <knob overrides>}, ...]}`; entry 0 must be
/// the empty override (v61.1).
pub fn load_profiles(j: &Json) -> Result<Vec<(String, Knobs)>, String> {
    let base = Knobs::default();
    let mut out = vec![];
    for p in j.get("profiles").arr() {
        let name = p.get("name").str().to_string();
        out.push((name, base.with(p)?));
    }
    if out.first().is_none_or(|(_, k)| *k != base) {
        return Err("profile 0 must be v61.1 (no overrides)".into());
    }
    Ok(out)
}
