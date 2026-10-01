//! `_r37_market_price` (agent line 1763): the engine price curve as the agent's layers
//! compute it (hinge gain 8, floor 1, Python banker's rounding).
pub struct P {
    pub base: f64,
    pub i0: f64,
    pub t: f64,
    pub below: &'static str,
    pub below_target: f64,
    pub above: &'static str,
    pub above_target: f64,
}

pub fn params(item: &str) -> Option<P> {
    let p = |base, t, below, bt, above, at| P { base, i0: 10000.0, t, below, below_target: bt, above, above_target: at };
    Some(match item {
        "WHEAT" => p(25.0, 400.0, "sqrt", 0.8, "log", 0.2),
        "CARROT" => p(35.0, 450.0, "hinge", 1.0, "sqrt", 0.7),
        "TOMATO" => p(60.0, 200.0, "hinge", 0.4, "sqrt", 0.6),
        "STRAWBERRY" => p(120.0, 100.0, "sqrt", 0.7, "linear", 1.6),
        "MELON" => p(250.0, 300.0, "log", 0.2, "sq", 3.6),
        "EGG" => p(50.0, 332.0, "hinge", 0.4, "log", 0.2),
        "MILK" => p(160.0, 122.0, "sqrt", 0.6, "linear", 1.6),
        "WOOL" => p(200.0, 105.0, "log", 0.2, "sq", 3.2),
        "FERTILIZER" => p(100.0, 200.0, "linear", 0.4, "linear", 0.4),
        _ => return None,
    })
}

pub const HINGE_GAIN: f64 = 8.0;
pub const PRICE_FLOOR: i64 = 1;

pub fn shape(f: &str, x: f64, t: f64) -> f64 {
    let x = x.max(0.0);
    match f {
        "linear" => x,
        "sq" => x * x,
        "sqrt" => x.sqrt(),
        "log" => x.ln_1p_py(),
        "log10" => (1.0 + x).log10(),
        "hinge" => {
            if t <= 0.0 {
                return x;
            }
            let u = x / t;
            u + HINGE_GAIN * (u - 1.0).max(0.0).powi(2)
        }
        _ => x,
    }
}

/// Python `math.log(1.0 + x)` (NOT log1p: the addition rounds first).
trait LnPy {
    fn ln_1p_py(self) -> f64;
}
impl LnPy for f64 {
    fn ln_1p_py(self) -> f64 {
        (1.0 + self).ln()
    }
}

/// Python `round(x)` to an integer: ties to even.
pub fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

pub fn price(item: &str, inventory: f64) -> i64 {
    let Some(p) = params(item) else { return PRICE_FLOOR };
    let price = if inventory < p.i0 {
        let amp = p.below_target * p.base / shape(p.below, p.t, p.t);
        p.base + amp * shape(p.below, p.i0 - inventory, p.t)
    } else {
        let amp = p.above_target * p.base / shape(p.above, p.t, p.t);
        p.base - amp * shape(p.above, inventory - p.i0, p.t)
    };
    (round_half_even(price) as i64).max(PRICE_FLOOR)
}

const MEMO_ITEMS: [&str; 9] = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"];
const MEMO_INV: usize = 1 << 15;

pub fn item_index(item: &str) -> Option<usize> {
    MEMO_ITEMS.iter().position(|x| *x == item)
}

thread_local! {
    static MEMO: std::cell::RefCell<Vec<Vec<i64>>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// `price(item, inv)` for an integer inventory, memoised per thread (pure function). `idx` is
/// `item_index(item)`; out-of-table inventories are computed directly.
pub fn price_i(idx: Option<usize>, item: &str, inv: i64) -> i64 {
    let Some(k) = idx else { return price(item, inv as f64) };
    if inv < 0 || inv as usize >= MEMO_INV {
        return price(item, inv as f64);
    }
    MEMO.with(|m| {
        let mut m = m.borrow_mut();
        if m.is_empty() {
            *m = vec![Vec::new(); MEMO_ITEMS.len()];
        }
        let t = &mut m[k];
        if t.is_empty() {
            *t = vec![i64::MIN; MEMO_INV];
        }
        let slot = &mut t[inv as usize];
        if *slot == i64::MIN {
            *slot = price(item, inv as f64);
        }
        *slot
    })
}
