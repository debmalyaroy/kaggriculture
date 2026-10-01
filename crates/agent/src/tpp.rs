//! Top-player policy (TPP, operator 28 Sep): a per-turn policy cloned from the top players' own turns --
//! every unit action (farmer + each hand) and every market order -- instead of replaying a fixed route tape.
//!
//! This module is the ONE encoder shared by the dataset dumper (`bcdump`) and the agent at play time, so
//! train and serve see identical features:
//!   board  `C` channels x 10 x 10 (own farm), quantized to u8 (value in [0, 1] -> 0..255)
//!   glob   `G` floats (clock, money, shed, seeds, prices, market stock, shops, rival summary)
//!   units  up to `MAXU` rows of `UF` u8 (position, farmer flag, hand index, inventory)
//! Labels: per unit a class from `UNIT_VOCAB` + a quantity bucket; per market order a `MKT_VOCAB` id + bucket.
use crate::act::{Action, Cmd};
use crate::obs::{qget, FarmObs, Obs, Tile};

pub const B: usize = 10;
pub const C: usize = 29;
pub const G: usize = 97;
pub const MAXU: usize = 17;
pub const UF: usize = 18;
pub const MAXM: usize = 10;

pub const ITEMS: [&str; 12] = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP"];
pub const CROPS: [&str; 5] = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"];
pub const ANIMALS: [&str; 3] = ["GOOSE", "COW", "SHEEP"];
pub const SHOPS: [&str; 8] = ["BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP", "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE"];
pub const QUADS: [&str; 4] = ["NW", "NE", "SW", "SE"];

/// Unit classes: 15 plain ops, PLANT x5 crops, PICKUP x12 items, PLACE x12 items (44).
pub const PLAIN: [&str; 15] = ["PASS", "NORTH", "SOUTH", "EAST", "WEST", "WATER", "HARVEST", "FERTILIZE", "DIG", "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE", "DROP"];
pub const NUNIT: usize = 15 + 5 + 12 + 12;
/// Unit quantity buckets (PICKUP/PLACE count): 0 = no count token.
pub const UQ_REP: [i64; 8] = [0, 1, 2, 3, 5, 8, 15, 30];
/// Market ids: HIRE, BUY_LAND, SELL x12, BUY_SEED x5, BUY_PRODUCT x12, BUY_ANIMAL x3 (34).
pub const NMKT: usize = 2 + 12 + 5 + 12 + 3;
pub const MQ_REP: [i64; 11] = [1, 2, 3, 5, 7, 10, 15, 24, 45, 120, 1000];

fn idx(xs: &[&str], s: &str) -> Option<usize> {
    xs.iter().position(|x| *x == s)
}
fn q8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}
fn l1p(v: f64, scale: f64) -> f32 {
    ((v.max(0.0) + 1.0).ln() / scale) as f32
}

pub fn unit_class(c: &Cmd) -> (usize, usize) {
    let op = c.op();
    if let Some(i) = idx(&PLAIN, op) {
        return (i, 0);
    }
    let qb = |c: &Cmd| -> usize {
        if c.len() < 3 {
            return 0;
        }
        let n = c.n(2);
        UQ_REP.iter().enumerate().skip(1).min_by_key(|(_, r)| (**r - n).abs()).map(|(i, _)| i).unwrap_or(1)
    };
    match op {
        "PLANT" => idx(&CROPS, c.s(1)).map(|i| (15 + i, 0)).unwrap_or((0, 0)),
        "PICKUP" => idx(&ITEMS, c.s(1)).map(|i| (20 + i, qb(c))).unwrap_or((0, 0)),
        "PLACE" => idx(&ITEMS, c.s(1)).map(|i| (32 + i, qb(c))).unwrap_or((0, 0)),
        _ => (0, 0),
    }
}

pub fn unit_cmd(cls: usize, qb: usize) -> Cmd {
    if cls < 15 {
        return Cmd::new(PLAIN[cls]);
    }
    if cls < 20 {
        return Cmd(vec![crate::act::Tok::S(crate::act::intern("PLANT")), crate::act::Tok::S(crate::act::intern(CROPS[cls - 15]))]);
    }
    let (op, item) = if cls < 32 { ("PICKUP", ITEMS[cls - 20]) } else { ("PLACE", ITEMS[cls - 32]) };
    let mut v = vec![crate::act::Tok::S(crate::act::intern(op)), crate::act::Tok::S(crate::act::intern(item))];
    if qb > 0 {
        v.push(crate::act::Tok::I(UQ_REP[qb.min(7)]));
    }
    Cmd(v)
}

pub fn mkt_id(c: &Cmd) -> Option<usize> {
    match c.op() {
        "HIRE" => Some(0),
        "BUY_LAND" => Some(1),
        "SELL" => idx(&ITEMS, c.s(1)).map(|i| 2 + i),
        "BUY_SEED" => idx(&CROPS, c.s(1)).map(|i| 14 + i),
        "BUY_PRODUCT" => idx(&ITEMS, c.s(1)).map(|i| 19 + i),
        "BUY_ANIMAL" => idx(&ANIMALS, c.s(1)).map(|i| 31 + i),
        _ => None,
    }
}

pub fn mkt_bucket(n: i64) -> usize {
    let l = (n.max(1) as f64).ln();
    let d = |r: i64| ((r as f64).ln() - l).abs();
    (0..MQ_REP.len()).min_by(|&a, &b| d(MQ_REP[a]).partial_cmp(&d(MQ_REP[b])).unwrap()).unwrap_or(0)
}

pub fn mkt_cmd(id: usize, qb: usize) -> Cmd {
    let n = MQ_REP[qb.min(10)];
    match id {
        0 => Cmd::new("HIRE"),
        1 => Cmd::new("BUY_LAND"),
        i if i < 14 => Cmd::order("SELL", ITEMS[i - 2], n),
        i if i < 19 => Cmd::order("BUY_SEED", CROPS[i - 14], n),
        i if i < 31 => Cmd::order("BUY_PRODUCT", ITEMS[i - 19], n),
        i => Cmd::order("BUY_ANIMAL", ANIMALS[(i - 31).min(2)], n),
    }
}

fn tile_ch(t: &Tile, day: i64, out: &mut [f32]) {
    // 0 none 1 locked 2 weed 3 plant 4 coop 5 pasture | 6-10 crop | 11 age 12 watered 13 unwatered 14 yield
    // 15 fertilized | 16-18 animal | 19 fed 20 cared 21 fert avail 22 unfed 23 animal age 24 care bonus
    let k = match t.kind {
        "" => 0,
        "LOCKED" => 1,
        "WEED" => 2,
        "PLANT" => 3,
        "COOP" => 4,
        "PASTURE" => 5,
        _ => 0,
    };
    out[k] = 1.0;
    if k == 3 {
        if let Some(i) = idx(&CROPS, t.crop) {
            out[6 + i] = 1.0;
        }
        out[11] = ((day - t.planted_day) as f32 / 12.0).clamp(0.0, 1.0);
        out[12] = t.watered_today as i32 as f32;
        out[13] = (t.consecutive_unwatered as f32 / 2.0).min(1.0);
        out[14] = (t.yield_units as f32 / 10.0).min(1.0);
        out[15] = (t.fertilized_until_day >= day) as i32 as f32;
    }
    if let Some(i) = idx(&ANIMALS, t.animal) {
        out[16 + i] = 1.0;
        out[14] = (t.yield_units as f32 / 10.0).min(1.0);
        out[19] = t.fed_today as i32 as f32;
        out[20] = t.cared_today as i32 as f32;
        out[21] = t.fertilizer_available as i32 as f32;
        out[22] = (t.consecutive_unfed as f32 / 2.0).min(1.0);
        out[23] = ((day - t.placed_day) as f32 / 20.0).clamp(0.0, 1.0);
        out[24] = (t.pending_care_bonus as f32 / 5.0).min(1.0);
    }
}

pub struct Enc {
    pub board: Vec<u8>, // C * B * B, channel-major
    pub glob: [f32; G],
    pub units: Vec<[u8; UF]>,
}

fn farm_summary(f: &FarmObs, day: i64, g: &mut [f32]) {
    // 16 floats: plants per crop, ripe-ish (yield>0) plants, weeds, animals per kind, empty tiles, locked
    for t in &f.tiles {
        if let Some(i) = idx(&CROPS, t.crop) {
            g[i] += 1.0 / 30.0;
            if t.yield_units > 0 {
                g[5] += 1.0 / 30.0;
            }
            g[15] += ((day - t.planted_day) as f32 / 12.0).clamp(0.0, 1.0) / 30.0;
        }
        match t.kind {
            "WEED" => g[6] += 1.0 / 20.0,
            "" => g[7] += 1.0 / 100.0,
            "LOCKED" => g[8] += 1.0 / 100.0,
            "COOP" | "PASTURE" => g[12] += 1.0 / 10.0,
            _ => {}
        }
        if let Some(i) = idx(&ANIMALS, t.animal) {
            g[9 + i] += 1.0 / 10.0;
        }
    }
    g[13] = f.hands.len() as f32 / 16.0;
    g[14] = f.hires_today as f32 / 5.0;
}

pub fn encode(o: &Obs) -> Enc {
    let me = o.player as usize;
    let f = &o.farms[me];
    let r = &o.farms[1 - me];
    let day = o.day;
    let mut board = vec![0u8; C * B * B];
    let mut ch = [0f32; C];
    let mut occ = [0f32; B * B];
    if let Some((x, y)) = f.farmer {
        if (0..B as i64).contains(&x) && (0..B as i64).contains(&y) {
            occ[y as usize * B + x as usize] += 0.5;
        }
    }
    for &(x, y) in &f.hands {
        if (0..B as i64).contains(&x) && (0..B as i64).contains(&y) {
            occ[y as usize * B + x as usize] += 0.25;
        }
    }
    for y in 0..B {
        for x in 0..B {
            ch.iter_mut().for_each(|v| *v = 0.0);
            tile_ch(f.tile(x as i64, y as i64), day, &mut ch[..25]);
            ch[25] = (f.farmer == Some((x as i64, y as i64))) as i32 as f32;
            ch[26] = occ[y * B + x].min(1.0);
            ch[27] = kagg_engine::rules::is_shed_adjacent((x as i64, y as i64), B as i64) as i32 as f32;
            // rival's same tile: planted or animal (their layout is visible)
            let rt = r.tile(x as i64, y as i64);
            ch[28] = if rt.kind == "PLANT" { 0.5 } else if !rt.animal.is_empty() { 1.0 } else { 0.0 };
            for c in 0..C {
                board[c * B * B + y * B + x] = q8(ch[c]);
            }
        }
    }
    let mut g = [0f32; G];
    let step = o.step();
    g[0] = step as f32 / 720.0;
    g[1] = o.hour as f32 / 24.0;
    g[2] = day as f32 / 30.0;
    g[3] = l1p(f.money, 10.0);
    g[4] = l1p(r.money, 10.0);
    g[5] = (f.money / 1e4) as f32;
    g[6] = (r.money / 1e4) as f32;
    g[7] = ((f.money - r.money) / 1e4) as f32;
    for (i, q) in QUADS.iter().enumerate() {
        g[8 + i] = f.quadrants.contains(q) as i32 as f32;
        g[12 + i] = r.quadrants.contains(q) as i32 as f32;
    }
    for (i, s) in SHOPS.iter().enumerate() {
        g[16 + i] = o.shops.contains(s) as i32 as f32;
    }
    for (i, it) in ITEMS.iter().enumerate() {
        g[24 + i] = l1p(qget(&o.shed, it) as f64, 5.0);
        g[36 + i] = (qget(&o.prices, it) as f32 / 200.0).min(5.0);
        g[48 + i] = l1p(qget(&o.mkt_inventory, it) as f64, 6.0);
    }
    for (i, c) in CROPS.iter().enumerate() {
        g[60 + i] = l1p(qget(&o.seeds, c) as f64, 4.0);
    }
    farm_summary(f, day, &mut g[65..81]);
    farm_summary(r, day, &mut g[81..97]);
    let mut units = vec![];
    let mut push = |pos: (i64, i64), farmer: bool, hi: usize, inv: Option<&crate::obs::Qty>| {
        if units.len() >= MAXU {
            return;
        }
        let mut u = [0u8; UF];
        u[0] = q8(pos.0 as f32 / 9.0);
        u[1] = q8(pos.1 as f32 / 9.0);
        u[2] = farmer as u8 * 255;
        u[3] = q8(hi as f32 / 16.0);
        let mut tot = 0i64;
        if let Some(inv) = inv {
            for (i, it) in ITEMS.iter().enumerate() {
                let n = qget(inv, it);
                tot += n;
                u[4 + i] = q8(l1p(n as f64, 4.0));
            }
        }
        u[16] = q8(l1p(tot as f64, 4.0));
        u[17] = 255;
        units.push(u);
    };
    push(f.farmer.unwrap_or((0, 0)), true, 0, o.invs.first());
    for (i, &p) in f.hands.iter().enumerate() {
        push(p, false, i + 1, o.invs.get(i + 1));
    }
    Enc { board, glob: g, units }
}

/// Labels of `a` for the units of `o` (farmer first) and the market list.
pub fn labels(o: &Obs, a: &Action) -> (Vec<(u8, u8)>, Vec<(u8, u8)>) {
    let me = o.player as usize;
    let nh = o.farms[me].hands.len();
    let mut u = vec![];
    let (c, q) = unit_class(&a.farmer);
    u.push((c as u8, q as u8));
    for i in 0..nh.min(MAXU - 1) {
        let (c, q) = a.hands.get(i).map(unit_class).unwrap_or((0, 0));
        u.push((c as u8, q as u8));
    }
    let mut m = vec![];
    for c in a.market.iter().take(MAXM) {
        if let Some(id) = mkt_id(c) {
            let qb = if c.len() >= 3 { mkt_bucket(c.n(2)) } else { 0 };
            m.push((id as u8, qb as u8));
        }
    }
    (u, m)
}

// ---------------------------------------------------------------------------------------------------
// Inference: the same forward as python/tpp/train.py `Net`, weights from its net.json export.

fn relu(v: f32) -> f32 {
    if v > 0.0 { v } else { 0.0 }
}

fn argmax(v: &[f32]) -> usize {
    let mut k = 0;
    for i in 1..v.len() {
        if v[i] > v[k] {
            k = i;
        }
    }
    k
}

#[derive(Clone, Debug)]
pub struct Lin {
    pub w: Vec<f32>, // out x in
    pub b: Vec<f32>,
    pub nin: usize,
    pub nout: usize,
}
impl Lin {
    fn run(&self, x: &[f32], out: &mut Vec<f32>) {
        out.clear();
        for o in 0..self.nout {
            let row = &self.w[o * self.nin..(o + 1) * self.nin];
            let mut s = self.b[o];
            for (a, b) in row.iter().zip(x) {
                s += a * b;
            }
            out.push(s);
        }
    }
}

#[derive(Clone, Debug)]
pub struct Conv {
    pub w: Vec<f32>, // out x in x 3 x 3
    pub b: Vec<f32>,
    pub nin: usize,
    pub nout: usize,
    pub d: i64,
}

#[derive(Clone, Debug)]
pub struct Net {
    pub genc: Lin,
    pub convs: Vec<Conv>,
    pub u1: Lin,
    pub ucls: Lin,
    pub uq: Lin,
    pub m1: Lin,
    pub missue: Lin,
    pub mq: Lin,
    pub mhire: Lin,
    pub ch: usize,
    pub nb: Vec<(i64, i64)>,
}

pub struct Out {
    pub ucls: Vec<Vec<f32>>,
    pub uq: Vec<Vec<f32>>,
    pub issue: Vec<f32>,
    pub mq: Vec<f32>,
    pub hire: Vec<f32>,
}

impl Net {
    pub fn load(path: &str) -> Result<Net, String> {
        let j = kagg_engine::json::parse(&std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?)?;
        let arr = |k: &str| -> Result<(Vec<f32>, Vec<usize>), String> {
            let v = j.get(k);
            if v.is_null() {
                return Err(format!("{path}: missing {k}"));
            }
            let s: Vec<usize> = j.get(&format!("{k}.shape")).arr().iter().map(|x| x.i64() as usize).collect();
            Ok((v.arr().iter().map(|x| x.f64() as f32).collect(), s))
        };
        let lin = |k: &str| -> Result<Lin, String> {
            let (w, s) = arr(&format!("{k}.weight"))?;
            let (b, _) = arr(&format!("{k}.bias"))?;
            Ok(Lin { w, b, nout: s[0], nin: s[1] })
        };
        let dil: Vec<i64> = j.get("dil").arr().iter().map(|x| x.i64()).collect();
        let mut convs = vec![];
        for (i, d) in dil.iter().enumerate() {
            let (w, s) = arr(&format!("convs.{i}.weight"))?;
            let (b, _) = arr(&format!("convs.{i}.bias"))?;
            convs.push(Conv { w, b, nout: s[0], nin: s[1], d: *d });
        }
        let nb = j.get("nb").arr().iter().map(|p| (p.arr()[0].i64(), p.arr()[1].i64())).collect();
        Ok(Net {
            genc: lin("genc")?, convs, u1: lin("u1")?, ucls: lin("ucls")?, uq: lin("uq")?, m1: lin("m1")?,
            missue: lin("missue")?, mq: lin("mq")?, mhire: lin("mhire")?, ch: j.get("CH").i64() as usize, nb,
        })
    }

    /// Forward on the u8/f32 encoding exactly as the dataset stores it.
    pub fn forward(&self, board: &[u8], glob: &[f32], units: &[[u8; UF]]) -> Out {
        const P: usize = B * B;
        let bi = B as i64;
        let mut g = vec![];
        self.genc.run(glob, &mut g);
        g.iter_mut().for_each(|v| *v = relu(*v));
        let ge = g.len();
        let mut x = vec![0f32; (C + ge) * P];
        for (i, v) in board.iter().enumerate() {
            x[i] = *v as f32 / 255.0;
        }
        for (k, gv) in g.iter().enumerate() {
            x[(C + k) * P..(C + k + 1) * P].iter_mut().for_each(|v| *v = *gv);
        }
        let mut h1: Vec<f32> = vec![];
        let nl = self.convs.len();
        for (li, c) in self.convs.iter().enumerate() {
            // im2col: cols[pixel][in*9 + k] (zero where the tap falls off the board), then one dense dot
            // product per (pixel, out) -- contiguous rows the compiler vectorizes
            let k9 = c.nin * 9;
            let mut cols = vec![0f32; P * k9];
            for i in 0..c.nin {
                let xi = &x[i * P..(i + 1) * P];
                for ky in 0..3i64 {
                    let dy = (ky - 1) * c.d;
                    for kx in 0..3i64 {
                        let dx = (kx - 1) * c.d;
                        let k = i * 9 + (ky * 3 + kx) as usize;
                        for yy in 0..bi {
                            let sy = yy + dy;
                            if !(0..bi).contains(&sy) {
                                continue;
                            }
                            for xx in 0..bi {
                                let sx = xx + dx;
                                if (0..bi).contains(&sx) {
                                    cols[(yy * bi + xx) as usize * k9 + k] = xi[(sy * bi + sx) as usize];
                                }
                            }
                        }
                    }
                }
            }
            let mut y = vec![0f32; c.nout * P];
            for p in 0..P {
                let col = &cols[p * k9..(p + 1) * k9];
                for o in 0..c.nout {
                    let w = &c.w[o * k9..(o + 1) * k9];
                    let mut s = [0f32; 8];
                    let (wc, cc) = (w.chunks_exact(8), col.chunks_exact(8));
                    let (wr, cr) = (wc.remainder(), cc.remainder());
                    for (a, b) in wc.zip(cc) {
                        for j in 0..8 {
                            s[j] += a[j] * b[j];
                        }
                    }
                    let mut t = c.b[o] + s.iter().sum::<f32>();
                    for (a, b) in wr.iter().zip(cr) {
                        t += a * b;
                    }
                    y[o * P + p] = t;
                }
            }
            y.iter_mut().for_each(|v| *v = relu(*v));
            if li == 1 {
                h1 = y.clone();
            }
            if li == nl - 1 && !h1.is_empty() {
                for (a, b) in y.iter_mut().zip(&h1) {
                    *a += b;
                }
            }
            x = y;
        }
        let ch = self.ch;
        let mut pooled = vec![0f32; 2 * ch];
        for c in 0..ch {
            let s = &x[c * P..(c + 1) * P];
            pooled[c] = s.iter().sum::<f32>() / P as f32;
            pooled[ch + c] = s.iter().cloned().fold(f32::MIN, f32::max);
        }
        let mut out = Out { ucls: vec![], uq: vec![], issue: vec![], mq: vec![], hire: vec![] };
        let mut uin: Vec<f32> = Vec::with_capacity(self.u1.nin);
        let (mut hbuf, mut o1, mut o2) = (vec![], vec![], vec![]);
        for u in units {
            let ux = ((u[0] as f32 / 255.0) * 9.0 + 0.5).floor().clamp(0.0, 9.0) as i64;
            let uy = ((u[1] as f32 / 255.0) * 9.0 + 0.5).floor().clamp(0.0, 9.0) as i64;
            uin.clear();
            for &(dx, dy) in &self.nb {
                let (xx, yy) = (ux + dx, uy + dy);
                let on = (0..bi).contains(&xx) && (0..bi).contains(&yy);
                for c in 0..ch {
                    uin.push(if on { x[c * P + (yy as usize) * B + xx as usize] } else { 0.0 });
                }
            }
            uin.extend_from_slice(&pooled);
            uin.extend_from_slice(&g);
            uin.extend(u.iter().map(|v| *v as f32 / 255.0));
            self.u1.run(&uin, &mut hbuf);
            hbuf.iter_mut().for_each(|v| *v = relu(*v));
            self.ucls.run(&hbuf, &mut o1);
            self.uq.run(&hbuf, &mut o2);
            out.ucls.push(o1.clone());
            out.uq.push(o2.clone());
        }
        let mut min: Vec<f32> = pooled.clone();
        min.extend_from_slice(&g);
        min.extend_from_slice(glob);
        self.m1.run(&min, &mut hbuf);
        hbuf.iter_mut().for_each(|v| *v = relu(*v));
        self.missue.run(&hbuf, &mut out.issue);
        self.mq.run(&hbuf, &mut out.mq);
        self.mhire.run(&hbuf, &mut out.hire);
        out
    }

    /// The policy's action. Units: the most likely class the engine would not ignore (`view::is_noop`), with
    /// seeds reserved as units plant. Market: each order type is issued with its predicted probability
    /// (sampled from a hash of the turn, so a game is reproducible; KRL_TPP_ARGMAX=1 = the old 50 percent
    /// threshold), cash-checked against the price, HIRE x the predicted count, at most 10 orders.
    pub fn act(&self, v: &crate::view::View) -> Action {
        let o = &v.obs;
        let e = encode(o);
        let out = self.forward(&e.board, &e.glob, &e.units);
        let me = o.player as usize;
        let f = &o.farms[me];
        let step = o.step();
        let mut h: u64 = 0x9E37_79B9_7F4A_7C15 ^ (step as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9) ^ (f.money as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
        let mut rnd = move || {
            h ^= h << 13;
            h ^= h >> 7;
            h ^= h << 17;
            (h >> 11) as f64 / (1u64 << 53) as f64
        };
        let argmax_mode = std::env::var_os("KRL_TPP_ARGMAX").is_some();
        let empty: crate::obs::Qty = vec![];
        let mut seeds: Vec<(&'static str, i64)> = CROPS.iter().map(|c| (*c, qget(&o.seeds, c))).collect();
        let mut pick = |ui: usize, pos: (i64, i64), seeds: &mut Vec<(&'static str, i64)>| -> Cmd {
            let lg = &out.ucls[ui];
            let mut ord: Vec<usize> = (0..lg.len()).collect();
            ord.sort_by(|a, b| lg[*b].partial_cmp(&lg[*a]).unwrap_or(std::cmp::Ordering::Equal));
            let qb = argmax(&out.uq[ui]);
            let inv = o.invs.get(ui).unwrap_or(&empty);
            let tile = v.tile(pos);
            for &c in ord.iter().take(12) {
                let cmd = unit_cmd(c, if c >= 20 { qb } else { 0 });
                if (15..20).contains(&c) {
                    let k = c - 15;
                    if seeds[k].1 <= 0 || !tile.is_none() {
                        continue;
                    }
                    seeds[k].1 -= 1;
                    return cmd;
                }
                if c == 0 || !crate::view::is_noop(&cmd, tile, inv, v, pos) {
                    return cmd;
                }
            }
            Cmd::pass()
        };
        let farmer = pick(0, f.farmer.unwrap_or((0, 0)), &mut seeds);
        let hands: Vec<Cmd> = f.hands.iter().enumerate().map(|(i, &p)| if i + 1 < out.ucls.len() { pick(i + 1, p, &mut seeds) } else { Cmd::pass() }).collect();
        let market = Self::market_from(&out, o, if argmax_mode { None } else { Some(&mut rnd) });
        Action { farmer, hands, market }
    }

    /// Market orders only (the hybrid: the chassis keeps the labour). `rnd` None = issue when p > 0.5.
    pub fn market_orders(&self, v: &crate::view::View) -> Vec<Cmd> {
        let e = encode(&v.obs);
        let out = self.forward(&e.board, &e.glob, &e.units);
        Self::market_from(&out, &v.obs, None)
    }

    fn market_from(out: &Out, o: &Obs, mut rnd: Option<&mut dyn FnMut() -> f64>) -> Vec<Cmd> {
        let f = &o.farms[o.player as usize];
        let sig = |x: f32| 1.0 / (1.0 + (-x as f64).exp());
        let mut cash = f.money;
        let mut orders: Vec<(f64, Cmd)> = vec![];
        let nh = argmax(&out.hire);
        for _ in 0..nh {
            orders.push((sig(out.issue[0]), Cmd::new("HIRE")));
        }
        for id in 1..NMKT {
            let p = sig(out.issue[id]);
            let go = match rnd.as_mut() { Some(r) => r() < p, None => p > 0.5 };
            if !go {
                continue;
            }
            let qb = argmax(&out.mq[id * MQ_REP.len()..(id + 1) * MQ_REP.len()]);
            let mut cmd = mkt_cmd(id, qb);
            // buys: clamp the count to what the cash covers at the quoted price (seed / animal / product)
            if id >= 14 {
                let item = cmd.s(1);
                let unit = match cmd.op() {
                    "BUY_SEED" => crate::view::seed_price(item).unwrap_or(0) as f64,
                    "BUY_ANIMAL" => crate::view::animal_cost(item).unwrap_or(0) as f64,
                    _ => qget(&o.prices, item).max(1) as f64,
                };
                let n = if unit > 0.0 { ((cash / unit).floor() as i64).min(cmd.n(2)) } else { cmd.n(2) };
                if n <= 0 {
                    continue;
                }
                cmd.set_n(2, n);
                cash -= unit * n as f64;
            }
            orders.push((p, cmd));
        }
        orders.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        orders.into_iter().take(MAXM).map(|x| x.1).collect()
    }
}
