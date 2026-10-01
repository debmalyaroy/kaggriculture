//! Track-neutral core shared by the bandit seat, the trackp seat and the
//! offline harness. Nothing here knows about either track's strategy: it is the
//! shell vocabulary they all speak -- an action `Op`, a per-turn `Row`, and the
//! canonical JSON emitter for a Row.
//!
//! G0.1 (harness separation, 2026-09-18): `bandit.rs` and `mbandit.rs` used to
//! `use crate::policy::{row_json, Row}`, which coupled the bandit lane to the
//! trackp economy policy purely for these two shell types. They now depend on
//! this track-neutral CORE instead, so bandit and trackp share a substrate but
//! not each other. `policy.rs` re-uses the same definitions from here, so the
//! wire format stays byte-identical (the compiled-agent identity gate proves
//! it).

/// One unit- or market-action: space-separated tokens, e.g. `["PLANT","MELON"]`
/// or `["SELL","MELON","3"]`. The empty vec is a PASS placeholder.
pub type Op = Vec<String>;

/// One seat's full action for one turn, positionally aligned with the
/// observation's hands: the farmer op, one op per hand, and the market queue.
#[derive(Clone)]
pub struct Row {
    pub farmer: Op,
    pub hands: Vec<Op>,
    pub market: Vec<Op>,
}

/// One op as a JSON array of quoted string tokens.
fn json_op(o: &[String]) -> String {
    let parts: Vec<String> = o.iter().map(|t| format!("\"{t}\"")).collect();
    format!("[{}]", parts.join(", "))
}

/// A Row as the action JSON the bridges write, one per line:
/// `{"farmer": [...], "hands": [[...],...], "market": [[...],...]}`.
pub fn row_json(r: &Row) -> String {
    let hands: Vec<String> = r.hands.iter().map(|h| json_op(h)).collect();
    let market: Vec<String> = r.market.iter().map(|m| json_op(m)).collect();
    format!(
        "{{\"farmer\": {}, \"hands\": [{}], \"market\": [{}]}}",
        json_op(&r.farmer),
        hands.join(", "),
        market.join(", ")
    )
}
