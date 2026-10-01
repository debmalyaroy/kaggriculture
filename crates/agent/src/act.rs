//! Kaggle actions as the Python agent builds them: a unit action or a market order is a
//! list of tokens (`["PLANT", "WHEAT"]`, `["SELL", "MILK", 12]`, `["HIRE"]`, `[]`).
//!
//! The token list is kept verbatim, including its length (`["PLACE", "WHEAT"]` and
//! `["PLACE", "WHEAT", 3]` are different actions to the engine) and empty entries (an
//! empty market order keeps its slot in the lockstep market race).
use kagg_engine::json::{self, Json};
use std::sync::{Mutex, OnceLock};

/// Every token the route data and the layers use; `intern` hands out these statics so a
/// token compare is a pointer-sized string compare and cloning never allocates.
const KNOWN: &[&str] = &[
    "PASS", "NORTH", "SOUTH", "EAST", "WEST", "PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG",
    "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE", "DROP", "PICKUP", "PLACE",
    "SELL", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "BUY_LAND", "HIRE",
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER",
    "GOOSE", "COW", "SHEEP", "COOP", "PASTURE", "WEED", "LOCKED",
];

pub fn intern(s: &str) -> &'static str {
    // Hot path: the tile kinds / items / ops of every observation, as a compiled match.
    let hit: Option<&'static str> = match s {
        "PASS" => Some("PASS"), "NORTH" => Some("NORTH"), "SOUTH" => Some("SOUTH"), "EAST" => Some("EAST"),
        "WEST" => Some("WEST"), "PLANT" => Some("PLANT"), "WATER" => Some("WATER"), "HARVEST" => Some("HARVEST"),
        "WHEAT" => Some("WHEAT"), "CARROT" => Some("CARROT"), "TOMATO" => Some("TOMATO"),
        "STRAWBERRY" => Some("STRAWBERRY"), "MELON" => Some("MELON"), "EGG" => Some("EGG"), "MILK" => Some("MILK"),
        "WOOL" => Some("WOOL"), "FERTILIZER" => Some("FERTILIZER"), "GOOSE" => Some("GOOSE"), "COW" => Some("COW"),
        "SHEEP" => Some("SHEEP"), "COOP" => Some("COOP"), "PASTURE" => Some("PASTURE"), "WEED" => Some("WEED"),
        "LOCKED" => Some("LOCKED"), "SELL" => Some("SELL"),
        _ => None,
    };
    if let Some(k) = hit {
        return k;
    }
    if let Some(k) = KNOWN.iter().find(|k| **k == s) {
        return k;
    }
    static EXTRA: OnceLock<Mutex<Vec<&'static str>>> = OnceLock::new();
    let mut v = EXTRA.get_or_init(|| Mutex::new(Vec::new())).lock().unwrap();
    if let Some(k) = v.iter().find(|k| **k == s) {
        return k;
    }
    let k: &'static str = Box::leak(s.to_string().into_boxed_str());
    v.push(k);
    k
}

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    S(&'static str),
    I(i64),
    F(f64),
    Null,
}

impl Tok {
    fn from_json(j: &Json) -> Tok {
        match j {
            Json::Str(s) => Tok::S(intern(s)),
            Json::Num(n) if n.fract() == 0.0 && n.abs() < 9e15 => Tok::I(*n as i64),
            Json::Num(n) => Tok::F(*n),
            Json::Bool(b) => Tok::I(*b as i64),
            _ => Tok::Null,
        }
    }
    fn dump(&self, out: &mut String) {
        match self {
            Tok::S(s) => out.push_str(&json::quote(s)),
            Tok::I(i) => out.push_str(&i.to_string()),
            Tok::F(f) => out.push_str(&json::num(*f)),
            Tok::Null => out.push_str("null"),
        }
    }
    /// Python `_int(v, 0)`: int() of a number truncates, of a numeric string parses.
    pub fn int(&self) -> i64 {
        match self {
            Tok::I(i) => *i,
            Tok::F(f) => f.trunc() as i64,
            Tok::S(s) => s.trim().parse().unwrap_or(0),
            Tok::Null => 0,
        }
    }
}

/// One unit action or market order.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Cmd(pub Vec<Tok>);

impl Cmd {
    pub fn new(op: &str) -> Cmd {
        Cmd(vec![Tok::S(intern(op))])
    }
    pub fn pass() -> Cmd {
        Cmd::new("PASS")
    }
    pub fn order(op: &str, item: &str, n: i64) -> Cmd {
        Cmd(vec![Tok::S(intern(op)), Tok::S(intern(item)), Tok::I(n)])
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    /// Token i as a string ("" when absent or not a string).
    pub fn s(&self, i: usize) -> &'static str {
        match self.0.get(i) {
            Some(Tok::S(s)) => s,
            _ => "",
        }
    }
    /// The op token (`act[0]`); "" for an empty action.
    pub fn op(&self) -> &'static str {
        self.s(0)
    }
    /// `_int(act[i])`.
    pub fn n(&self, i: usize) -> i64 {
        self.0.get(i).map(Tok::int).unwrap_or(0)
    }
    pub fn set_n(&mut self, i: usize, v: i64) {
        if i < self.0.len() {
            self.0[i] = Tok::I(v);
        }
    }
    /// A SELL order with a quantity: `o and o[0] == "SELL" and len(o) >= 3`.
    pub fn is_sell3(&self) -> bool {
        self.op() == "SELL" && self.0.len() >= 3
    }
    fn from_json(j: &Json) -> Cmd {
        Cmd(j.arr().iter().map(Tok::from_json).collect())
    }
    fn dump(&self, out: &mut String) {
        out.push('[');
        for (i, t) in self.0.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            t.dump(out);
        }
        out.push(']');
    }
}

/// `{"farmer": [...], "hands": [[...], ...], "market": [[...], ...]}`.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Action {
    pub farmer: Cmd,
    pub hands: Vec<Cmd>,
    pub market: Vec<Cmd>,
}

impl Action {
    pub fn pass() -> Action {
        Action { farmer: Cmd::pass(), hands: vec![], market: vec![] }
    }
    pub fn from_json(j: &Json) -> Action {
        Action {
            farmer: Cmd::from_json(j.get("farmer")),
            hands: j.get("hands").arr().iter().map(Cmd::from_json).collect(),
            market: j.get("market").arr().iter().map(Cmd::from_json).collect(),
        }
    }
    /// `[farmer or ["PASS"]] + hands`.
    pub fn units(&self) -> Vec<Cmd> {
        let mut u = Vec::with_capacity(1 + self.hands.len());
        u.push(if self.farmer.is_empty() { Cmd::pass() } else { self.farmer.clone() });
        u.extend(self.hands.iter().cloned());
        u
    }
    /// Unit i without cloning (farmer falls back to PASS when empty).
    pub fn unit(&self, i: usize) -> Option<&Cmd> {
        static PASS: OnceLock<Cmd> = OnceLock::new();
        if i == 0 {
            Some(if self.farmer.is_empty() { PASS.get_or_init(Cmd::pass) } else { &self.farmer })
        } else {
            self.hands.get(i - 1)
        }
    }
    pub fn n_units(&self) -> usize {
        1 + self.hands.len()
    }
    pub fn set_units(&mut self, mut units: Vec<Cmd>) {
        let rest = units.split_off(1);
        self.farmer = units.pop().unwrap_or_default();
        self.hands = rest;
    }
    pub fn dump(&self) -> String {
        let mut out = String::with_capacity(256);
        out.push_str("{\"farmer\": ");
        self.farmer.dump(&mut out);
        out.push_str(", \"hands\": [");
        for (i, h) in self.hands.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            h.dump(&mut out);
        }
        out.push_str("], \"market\": [");
        for (i, m) in self.market.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            m.dump(&mut out);
        }
        out.push_str("]}");
        out
    }
}
