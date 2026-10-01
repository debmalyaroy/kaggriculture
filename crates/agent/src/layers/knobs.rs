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
    /// PPO options (rl5): V9 courier / carrot, E402, read only when nothing of theirs is in flight
    pub courier_on: bool,
    pub carrot_on: bool,
    pub e402_on: bool,
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
    // wb3 (wheat buys first in BRUNCH worlds): 0 = legacy (unconditional), 1 = only on the rival-wheat signal
    // (ma_mirror), 2 = signal + cash guard (the wheat buys must leave cash for the route's other buys)
    pub wb3_mode: i64,
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
    // ---- constants the v61.1 chain hard-coded (defaults = v61.1 exactly) ----
    /// R36 reservation starts at this step (v61.1: 192).
    pub r36_from: i64,
    /// R37 horizons: default, after a similarity streak, late (from `r37_late_from`); similarity bar.
    pub r37_base: i64,
    pub r37_streak: i64,
    pub r37_late: i64,
    pub r37_late_from: i64,
    pub r37_sim: f64,
    /// RACE clone horizon applies from this step (v61.1: 216).
    pub race_from: i64,
    /// Lead-sell windows: share of the next h turns' planned sells sold now (v61.1: 3/4), first step,
    /// and the hours of day (bitmask, bit h = hour h) of EV, DP and MP.
    pub lead_frac: f64,
    pub lead_from: i64,
    pub ev_hours: i64,
    pub dp_hours: i64,
    pub mp_hours: i64,
    /// Market-side chain stages switched off (bit = stage index in layers::CUTS; see MARKET_STAGES).
    pub off: u64,
    /// V219 late tomato investment: PIZZA / FARMERS_MARKET shops unlocked on day 18 (v61.1: 3). RCA 2026-09-27: the
    /// public field plants tomatoes with 2 (FARMERS|FARMERS, PIZZA|FARMERS, PIZZA|PIZZA): 52 of 63 tomato losses.
    /// Whole-game setting (read once at step 432): use --knob-over, not a per-day profile.
    /// +100 = ALSO qualify when the world itself (the first two shops) is PIZZA / FARMERS_MARKET (e.g. 103).
    pub v219_min_shops: i64,
    /// Project options for the PPO planner (read at each project's commit point; a started project always
    /// finishes): V233 six-sheep, V231 sheep->cow, HD2 herd choice, CS cow<->goose, Y yarn herd. Default on = v61.1.
    pub v233_on: bool,
    pub v231_on: bool,
    pub hd2_on: bool,
    pub cs_on: bool,
    pub y_on: bool,
}

/// Chain stages a profile may switch off mid-game: they change only WHEN / HOW MUCH we sell (or the
/// order of the market list), never what the farm does, so a day-boundary switch cannot desync the tape.
pub const MARKET_STAGES: [(&str, usize); 17] = [
    ("sales_first", 6), ("r36", 9), ("r37", 10), ("v9_race", 26), ("ctrtable", 29), ("overflow", 30), ("or2", 32),
    ("r127", 38), ("preguard", 39), ("e335", 42), ("t62a", 47), ("wb3", 50), ("fx", 51), ("bd", 54), ("sm", 56),
    ("mg", 60), ("ig", 61),
];

/// Economy stages (they move units or buy inputs): switchable for a WHOLE game only (`--chain-off`),
/// never by a day-boundary profile.
pub const ECON_STAGES: [(&str, usize); 19] = [
    ("v219", 4), ("v231", 8), ("v233", 12), ("r51_warehouse", 15), ("r85", 18), ("r95", 19), ("r97", 20), ("courier", 21),
    ("carrot", 22), ("opening", 25), ("ca", 31), ("ch", 33), ("sr", 34), ("hd2", 35), ("cs", 36), ("y", 41), ("wl", 45),
    ("pipe", 48), ("e402", 59),
];

/// Stage bits for a comma-separated list of stage names (`econ` = economy stages allowed too).
pub fn stage_bits(names: &str, econ: bool) -> Result<u64, String> {
    let mut b = 0u64;
    for n in names.split(',').map(|x| x.trim()).filter(|x| !x.is_empty()) {
        let hit = MARKET_STAGES.iter().find(|(k, _)| *k == n).or_else(|| if econ { ECON_STAGES.iter().find(|(k, _)| *k == n) } else { None });
        match hit {
            Some((_, i)) => b |= 1u64 << i,
            None => return Err(format!("stage {n:?} cannot be switched off here")),
        }
    }
    Ok(b)
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
            courier_on: true,
            carrot_on: true,
            e402_on: true,
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
            wb3_mode: 0,
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
            r36_from: 192,
            r37_base: 2,
            r37_streak: 3,
            r37_late: 4,
            r37_late_from: 288,
            r37_sim: 0.90,
            race_from: 216,
            lead_frac: 0.75,
            lead_from: 96,
            ev_hours: (15..=20).map(|h| 1i64 << h).sum(),
            dp_hours: 0b111,
            mp_hours: (10..=13).map(|h| 1i64 << h).sum(),
            off: 0,
            v219_min_shops: 3,
            v233_on: true,
            v231_on: true,
            hd2_on: true,
            cs_on: true,
            y_on: true,
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
                "courier_on" => k.courier_on = b(),
                "carrot_on" => k.carrot_on = b(),
                "e402_on" => k.e402_on = b(),
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
                "wb3_mode" => k.wb3_mode = i(),
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
                "r36_from" => k.r36_from = i(),
                "v219_min_shops" => k.v219_min_shops = i(),
                "v233_on" => k.v233_on = b(),
                "v231_on" => k.v231_on = b(),
                "hd2_on" => k.hd2_on = b(),
                "cs_on" => k.cs_on = b(),
                "y_on" => k.y_on = b(),
                "r37_base" => k.r37_base = i(),
                "r37_streak" => k.r37_streak = i(),
                "r37_late" => k.r37_late = i(),
                "r37_late_from" => k.r37_late_from = i(),
                "r37_sim" => k.r37_sim = v.f64(),
                "race_from" => k.race_from = i(),
                "lead_frac" => k.lead_frac = v.f64(),
                "lead_from" => k.lead_from = i(),
                "ev_hours" | "dp_hours" | "mp_hours" => {
                    let m: i64 = v.arr().iter().map(|h| 1i64 << h.i64().clamp(0, 23)).sum();
                    match key.as_str() {
                        "ev_hours" => k.ev_hours = m,
                        "dp_hours" => k.dp_hours = m,
                        _ => k.mp_hours = m,
                    }
                }
                "off" => {
                    let names: Vec<String> = v.arr().iter().map(|x| x.str().to_string()).collect();
                    k.off = stage_bits(&names.join(","), false)?;
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
