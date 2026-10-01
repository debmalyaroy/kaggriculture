//! Action parsing (engine semantics) and Python-canonical JSON for stream hashes.
use crate::consts::{index_of, ANIMALS, CROPS, PRODUCTS};
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderKind {
    Sell,
    BuyProduct,
    BuySeed,
    BuyAnimal,
    Hire,
    BuyLand,
    Other,
}

/// One market order. `item` indexes PRODUCTS (Sell/BuyProduct), CROPS (BuySeed) or
/// ANIMALS (BuyAnimal). `qty` follows the engine's `_parse_order`: Python `int(order[2])`,
/// and an order with qty <= 0 or an unparseable qty is dropped by the engine (qty = None).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Order {
    pub kind: OrderKind,
    pub item: Option<usize>,
    pub qty: Option<i64>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParsedAction {
    /// The raw market list (length = orders submitted, including ones the engine drops
    /// past `MAX_MARKET_ORDERS`); non-list entries are `Other`.
    pub orders: Vec<Order>,
    /// PLANT unit actions (farmer + hands) per crop, CROPS order.
    pub plants: [u16; 5],
}

/// Python `int(x)` for a JSON scalar: numbers truncate, bools are 0/1, strings parse as an
/// integer literal (surrounding whitespace allowed). Anything else -> None.
pub fn py_int(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(i)
            } else if let Some(u) = n.as_u64() {
                Some(u as i64)
            } else {
                n.as_f64().filter(|f| f.is_finite()).map(|f| f.trunc() as i64)
            }
        }
        Value::Bool(b) => Some(*b as i64),
        Value::String(s) => {
            let t = s.trim().replace('_', "");
            t.parse::<i64>().ok()
        }
        _ => None,
    }
}

fn parse_order(o: &Value) -> Order {
    let other = Order { kind: OrderKind::Other, item: None, qty: None };
    let a = match o.as_array() {
        Some(a) if !a.is_empty() => a,
        _ => return other,
    };
    let op = a[0].as_str().unwrap_or("");
    let kind = match op {
        "SELL" => OrderKind::Sell,
        "BUY_PRODUCT" => OrderKind::BuyProduct,
        "BUY_SEED" => OrderKind::BuySeed,
        "BUY_ANIMAL" => OrderKind::BuyAnimal,
        "HIRE" => return Order { kind: OrderKind::Hire, item: None, qty: None },
        "BUY_LAND" => return Order { kind: OrderKind::BuyLand, item: None, qty: None },
        _ => return other,
    };
    if a.len() < 3 {
        return Order { kind, item: None, qty: None };
    }
    let list: &[&str] = match kind {
        OrderKind::BuySeed => &CROPS,
        OrderKind::BuyAnimal => &ANIMALS,
        _ => &PRODUCTS,
    };
    let item = a[1].as_str().and_then(|s| index_of(list, s));
    let qty = py_int(&a[2]).filter(|&n| n > 0);
    Order { kind, item, qty }
}

fn plant_crop(u: &Value) -> Option<usize> {
    let a = u.as_array()?;
    if a.len() >= 2 && a[0].as_str() == Some("PLANT") {
        a[1].as_str().and_then(|s| index_of(&CROPS, s))
    } else {
        None
    }
}

/// Parse one seat's action (a JSON object; anything else is treated as `{}`).
pub fn parse_action(v: &Value) -> ParsedAction {
    let mut pa = ParsedAction::default();
    let obj = match v.as_object() {
        Some(o) => o,
        None => return pa,
    };
    if let Some(m) = obj.get("market").and_then(|m| m.as_array()) {
        pa.orders = m.iter().map(parse_order).collect();
    }
    if let Some(f) = obj.get("farmer") {
        if let Some(c) = plant_crop(f) {
            pa.plants[c] += 1;
        }
    }
    if let Some(h) = obj.get("hands").and_then(|h| h.as_array()) {
        for u in h {
            if let Some(c) = plant_crop(u) {
                pa.plants[c] += 1;
            }
        }
    }
    pa
}

/// Python truthiness of a JSON value.
pub fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Python `repr(float)`.
fn py_float(f: f64, out: &mut String) {
    if f == f.trunc() && f.abs() < 1e16 {
        // integral and in fixed range: Python prints "3.0"
        let s = format!("{}", f);
        out.push_str(&s);
        if !s.contains('.') {
            out.push_str(".0");
        }
        return;
    }
    let e = format!("{:e}", f); // shortest round-trip digits, e.g. "1.5e-5"
    let (mant, exp) = e.split_once('e').unwrap_or((&e, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    if (-4..16).contains(&exp) {
        out.push_str(&format!("{}", f));
    } else {
        out.push_str(mant);
        out.push('e');
        out.push(if exp < 0 { '-' } else { '+' });
        out.push_str(&format!("{:02}", exp.abs()));
    }
}

fn py_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            _ => {
                let mut buf = [0u16; 2];
                for u in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{:04x}", u));
                }
            }
        }
    }
    out.push('"');
}

/// `json.dumps(v, sort_keys=True, separators=(",", ":"))` (ensure_ascii on).
pub fn py_canonical(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.push_str(&i.to_string());
            } else if let Some(u) = n.as_u64() {
                out.push_str(&u.to_string());
            } else {
                py_float(n.as_f64().unwrap_or(0.0), out);
            }
        }
        Value::String(s) => py_str(s, out),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                py_canonical(x, out);
            }
            out.push(']');
        }
        Value::Object(o) => {
            // serde_json's default Map is a BTreeMap: keys already sorted (UTF-8 byte
            // order == code-point order == Python's str order).
            out.push('{');
            for (i, (k, x)) in o.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                py_str(k, out);
                out.push(':');
                py_canonical(x, out);
            }
            out.push('}');
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn canonical_matches_python() {
        let v = json!({"market": [["SELL", "WHEAT", 3]], "farmer": ["PASS"], "hands": []});
        let mut s = String::new();
        py_canonical(&v, &mut s);
        assert_eq!(s, r#"{"farmer":["PASS"],"hands":[],"market":[["SELL","WHEAT",3]]}"#);
        let mut s = String::new();
        py_canonical(&json!([1.0, 1.5e-5, 1e16, 0.1, "é"]), &mut s);
        assert_eq!(s, r#"[1.0,1.5e-05,1e+16,0.1,"é"]"#);
    }

    #[test]
    fn orders_follow_engine() {
        let pa = parse_action(&json!({"market": [["SELL", "MELON", "4"], ["SELL", "MELON", 0],
            ["HIRE"], ["BUY_SEED", "WHEAT", 2.7], "junk"], "farmer": ["PLANT", "MELON"],
            "hands": [["PLANT", "WHEAT"], ["PLANT", "NOPE"]]}));
        assert_eq!(pa.orders.len(), 5);
        assert_eq!(pa.orders[0].qty, Some(4));
        assert_eq!(pa.orders[1].qty, None);
        assert_eq!(pa.orders[2].kind, OrderKind::Hire);
        assert_eq!(pa.orders[3].qty, Some(2));
        assert_eq!(pa.orders[4].kind, OrderKind::Other);
        assert_eq!(pa.plants, [0, 1, 0, 0, 1]);
    }
}
