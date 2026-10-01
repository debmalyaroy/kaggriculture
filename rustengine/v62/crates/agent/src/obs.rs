//! Zero-allocation-per-token observation parser -> compact [`Obs`].
//!
//! The generic engine JSON reader allocates a String per key and per tile field (~1,000
//! allocations per observation, ~150 us); this walker compares keys as byte slices, interns
//! every string value to a `&'static str`, and writes straight into fixed structs.
//! Dict order is preserved wherever Python iterates it (shed, inventories, prices).
use crate::act::intern;

/// Insertion-ordered `{item: n}` (a Python dict) with interned keys.
pub type Qty = Vec<(&'static str, i64)>;
pub fn qget(m: &Qty, k: &str) -> i64 {
    m.iter().find(|(n, _)| *n == k).map(|(_, v)| *v).unwrap_or(0)
}
pub fn qhas(m: &Qty, k: &str) -> bool {
    m.iter().any(|(n, _)| *n == k)
}
pub fn qadd(m: &mut Qty, k: &'static str, n: i64) {
    if let Some(e) = m.iter_mut().find(|(nm, _)| *nm == k) {
        e.1 += n;
    } else {
        m.push((k, n));
    }
}
pub fn qset(m: &mut Qty, k: &'static str, n: i64) {
    if let Some(e) = m.iter_mut().find(|(nm, _)| *nm == k) {
        e.1 = n;
    } else {
        m.push((k, n));
    }
}

/// One tile. `kind` = "" for None (empty ground), "LOCKED", or the dict's `kind`
/// (WEED / PLANT / COOP / PASTURE). `animal` = "" when the dict has none.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Tile {
    pub kind: &'static str,
    pub crop: &'static str,
    pub animal: &'static str,
    pub planted_day: i64,
    pub watered_today: bool,
    pub consecutive_unwatered: i64,
    pub yield_units: i64,
    pub max_lifespan_step: i64,
    pub fertilized_until_day: i64,
    pub placed_day: i64,
    pub consecutive_unfed: i64,
    pub fed_today: bool,
    pub cared_today: bool,
    pub fertilizer_available: bool,
    pub pending_care_bonus: i64,
}

impl Tile {
    pub const LOCKED: Tile = Tile {
        kind: "LOCKED", crop: "", animal: "", planted_day: 0, watered_today: false, consecutive_unwatered: 0,
        yield_units: 0, max_lifespan_step: 0, fertilized_until_day: 0, placed_day: 0, consecutive_unfed: 0,
        fed_today: false, cared_today: false, fertilizer_available: false, pending_care_bonus: 0,
    };
    /// Python `tile is None`.
    pub fn is_none(&self) -> bool {
        self.kind.is_empty()
    }
    pub fn is_locked(&self) -> bool {
        self.kind == "LOCKED"
    }
    /// `isinstance(tile, dict)`.
    pub fn is_dict(&self) -> bool {
        !self.is_none() && !self.is_locked()
    }
    pub fn has_animal(&self) -> bool {
        !self.animal.is_empty()
    }
}

#[derive(Clone, Debug, Default)]
pub struct FarmObs {
    pub money: f64,
    pub farmer: Option<(i64, i64)>,
    pub hands: Vec<(i64, i64)>,
    pub hires_today: i64,
    pub quadrants: Vec<&'static str>,
    /// Row-major `tiles[y][x]`; `rows` x `cols`.
    pub tiles: Vec<Tile>,
    pub rows: usize,
    pub cols: usize,
}

impl FarmObs {
    pub fn tile(&self, x: i64, y: i64) -> &Tile {
        if x < 0 || y < 0 || y as usize >= self.rows || x as usize >= self.cols {
            return &Tile::LOCKED;
        }
        &self.tiles[y as usize * self.cols + x as usize]
    }
}

#[derive(Clone, Debug, Default)]
pub struct Obs {
    pub step: Option<i64>,
    pub day: i64,
    pub hour: i64,
    pub player: i64,
    pub farms: Vec<FarmObs>,
    pub mkt_inventory: Qty,
    pub prices: Qty,
    pub shops: Vec<&'static str>,
    pub has_private: bool,
    pub shed: Qty,
    pub seeds: Qty,
    pub invs: Vec<Qty>,
}

impl Obs {
    /// `_step_of`.
    pub fn step(&self) -> i64 {
        self.step.unwrap_or(self.day * 24 + self.hour)
    }
    pub fn parse(s: &str) -> Result<Obs, String> {
        let mut p = P { b: s.as_bytes(), i: 0 };
        let mut o = Obs::default();
        p.obj(|p, k| {
            match k {
                b"step" => o.step = p.num_opt()?.map(|x| x as i64),
                b"day" => o.day = p.num_or0()? as i64,
                b"hour" => o.hour = p.num_or0()? as i64,
                b"player" => o.player = p.num_or0()? as i64,
                b"farms" => {
                    p.arr(|p| {
                        let f = p.farm()?;
                        o.farms.push(f);
                        Ok(())
                    })?;
                }
                b"market" => {
                    p.obj(|p, k| {
                        match k {
                            b"inventory" => o.mkt_inventory = p.qty()?,
                            b"prices" => o.prices = p.qty()?,
                            _ => p.skip()?,
                        }
                        Ok(())
                    })?;
                }
                b"town" => {
                    p.obj(|p, k| {
                        if k == b"unlocked_shops" {
                            p.arr(|p| {
                                if let Some(s) = p.str_opt()? {
                                    o.shops.push(intern(s));
                                }
                                Ok(())
                            })?;
                        } else {
                            p.skip()?;
                        }
                        Ok(())
                    })?;
                }
                b"private" => {
                    if p.peek() == b'{' {
                        o.has_private = true;
                        p.obj(|p, k| {
                            match k {
                                b"shed" => o.shed = p.qty()?,
                                b"seeds" => o.seeds = p.qty()?,
                                b"inventories" => p.arr(|p| {
                                    let q = p.qty()?;
                                    o.invs.push(q);
                                    Ok(())
                                })?,
                                _ => p.skip()?,
                            }
                            Ok(())
                        })?;
                    } else {
                        p.skip()?;
                    }
                }
                _ => p.skip()?,
            }
            Ok(())
        })?;
        Ok(o)
    }
}

impl Obs {
    /// The seat's observation built straight from the engine state: the same `Obs` as
    /// `Obs::parse(&seat_obs_json(st, seat))` without writing and re-reading ~40 KB of JSON per turn
    /// (simulation harnesses only; a real match hands the agent JSON). `KAGG_OBS_CHECK` in the runner
    /// compares the two on every step.
    pub fn from_state(st: &kagg_engine::state::State, seat: usize) -> Obs {
        use kagg_engine::state::{Cell, Farm, OMap};
        let qty = |m: &OMap| -> Qty { m.0.iter().map(|(k, v)| (intern(k), *v)).collect() };
        let farm = |f: &Farm| -> FarmObs {
            let rows = f.tiles.len();
            let cols = f.tiles.first().map_or(0, |r| r.len());
            let mut tiles = Vec::with_capacity(rows * cols);
            for row in &f.tiles {
                for c in row {
                    tiles.push(match c {
                        Cell::Empty => Tile::default(),
                        Cell::Locked => Tile::LOCKED,
                        Cell::Weed => Tile { kind: intern("WEED"), ..Tile::default() },
                        Cell::Plant { crop, planted_day, watered_today, consecutive_unwatered, yield_units, max_lifespan_step, fertilized_until_day } => Tile {
                            kind: intern("PLANT"),
                            crop: intern(crop),
                            planted_day: *planted_day,
                            watered_today: *watered_today,
                            consecutive_unwatered: *consecutive_unwatered,
                            yield_units: *yield_units,
                            max_lifespan_step: *max_lifespan_step,
                            fertilized_until_day: *fertilized_until_day,
                            ..Tile::default()
                        },
                        Cell::Structure { kind, animal: None } => Tile { kind: intern(kind), ..Tile::default() },
                        Cell::Structure { kind, animal: Some(a) } => Tile {
                            kind: intern(kind),
                            animal: intern(a.animal),
                            placed_day: a.placed_day,
                            yield_units: a.yield_units,
                            consecutive_unfed: a.consecutive_unfed,
                            fed_today: a.fed_today,
                            cared_today: a.cared_today,
                            fertilizer_available: a.fertilizer_available,
                            pending_care_bonus: a.pending_care_bonus,
                            ..Tile::default()
                        },
                    });
                }
            }
            FarmObs {
                money: f.money,
                farmer: Some(f.farmer),
                hands: f.hands.clone(),
                hires_today: f.hires_today,
                quadrants: f.unlocked_quadrants.iter().map(|q| intern(q)).collect(),
                tiles,
                rows,
                cols,
            }
        };
        let p = &st.private[seat.min(1)];
        Obs {
            step: Some(st.step),
            day: st.day(),
            hour: st.step % kagg_engine::state::TURNS_PER_DAY,
            player: seat as i64,
            farms: vec![farm(&st.farms[0]), farm(&st.farms[1])],
            mkt_inventory: qty(&st.market.inventory),
            prices: qty(&st.market.prices),
            shops: st.town.unlocked_shops.iter().map(|s| intern(s)).collect(),
            has_private: true,
            shed: qty(&p.shed),
            seeds: qty(&p.seeds),
            invs: p.inventories.iter().map(qty).collect(),
        }
    }
}

/// Fast route-table loader: `{"<id>": [ {farmer, hands, market}, ... ], ...}` -> actions, without
/// building a JSON tree (the generic parser costs ~150 ms on the 4.8 MB v61.1 table).
pub fn parse_routes(s: &str) -> Result<Vec<(String, Vec<crate::act::Action>)>, String> {
    use crate::act::{Action, Cmd, Tok};
    fn cmd(p: &mut P) -> R<Cmd> {
        let mut toks = vec![];
        p.arr(|p| {
            let t = match p.peek() {
                b'"' => Tok::S(intern(std::str::from_utf8(p.raw_str()?).map_err(|e| e.to_string())?)),
                b'n' => {
                    p.lit(b"null")?;
                    Tok::Null
                }
                b't' => {
                    p.lit(b"true")?;
                    Tok::I(1)
                }
                b'f' => {
                    p.lit(b"false")?;
                    Tok::I(0)
                }
                _ => {
                    let x = p.num()?;
                    if x.fract() == 0.0 && x.abs() < 9e15 {
                        Tok::I(x as i64)
                    } else {
                        Tok::F(x)
                    }
                }
            };
            toks.push(t);
            Ok(())
        })?;
        Ok(Cmd(toks))
    }
    let mut p = P { b: s.as_bytes(), i: 0 };
    let mut out = vec![];
    p.obj(|p, k| {
        let id = std::str::from_utf8(k).map_err(|e| e.to_string())?.to_string();
        let mut tape = vec![];
        p.arr(|p| {
            let mut a = Action::default();
            p.obj(|p, k| {
                match k {
                    b"farmer" => a.farmer = cmd(p)?,
                    b"hands" => p.arr(|p| {
                        a.hands.push(cmd(p)?);
                        Ok(())
                    })?,
                    b"market" => p.arr(|p| {
                        a.market.push(cmd(p)?);
                        Ok(())
                    })?,
                    _ => p.skip()?,
                }
                Ok(())
            })?;
            tape.push(a);
            Ok(())
        })?;
        out.push((id, tape));
        Ok(())
    })?;
    Ok(out)
}

struct P<'a> {
    b: &'a [u8],
    i: usize,
}

type R<T> = Result<T, String>;

impl<'a> P<'a> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\n' | b'\r' | b'\t') {
            self.i += 1;
        }
    }
    fn peek(&mut self) -> u8 {
        self.ws();
        *self.b.get(self.i).unwrap_or(&0)
    }
    fn expect(&mut self, c: u8) -> R<()> {
        if self.peek() == c {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("expected '{}' at {}", c as char, self.i))
        }
    }
    fn lit(&mut self, w: &[u8]) -> R<()> {
        if self.b[self.i..].starts_with(w) {
            self.i += w.len();
            Ok(())
        } else {
            Err(format!("bad literal at {}", self.i))
        }
    }
    /// Raw string contents (escapes are not decoded: observation tokens never contain any).
    fn raw_str(&mut self) -> R<&'a [u8]> {
        self.expect(b'"')?;
        let s = self.i;
        loop {
            match self.b[self.i..].iter().position(|&c| c == b'"' || c == b'\\') {
                Some(k) if self.b[self.i + k] == b'\\' => self.i += k + 2,
                Some(k) => {
                    self.i += k;
                    break;
                }
                None => {
                    self.i = self.b.len();
                    break;
                }
            }
        }
        let e = self.i.min(self.b.len());
        self.i += 1;
        Ok(&self.b[s..e])
    }
    fn str_opt(&mut self) -> R<Option<&'a str>> {
        match self.peek() {
            b'"' => Ok(Some(std::str::from_utf8(self.raw_str()?).map_err(|e| e.to_string())?)),
            _ => {
                self.skip()?;
                Ok(None)
            }
        }
    }
    fn num(&mut self) -> R<f64> {
        self.ws();
        let s = self.i;
        let mut int: i64 = 0;
        let mut neg = false;
        let mut simple = true;
        if self.b.get(self.i) == Some(&b'-') {
            neg = true;
            self.i += 1;
        }
        while self.i < self.b.len() {
            let c = self.b[self.i];
            if c.is_ascii_digit() {
                int = int.wrapping_mul(10).wrapping_add((c - b'0') as i64);
            } else if matches!(c, b'.' | b'e' | b'E' | b'+' | b'-') {
                simple = false;
            } else {
                break;
            }
            self.i += 1;
        }
        if self.i == s {
            return Err(format!("expected number at {s}"));
        }
        if simple && self.i - s < 18 {
            return Ok(if neg { -(int as f64) } else { int as f64 });
        }
        let t = std::str::from_utf8(&self.b[s..self.i]).map_err(|e| e.to_string())?;
        t.parse::<f64>().map_err(|e| format!("{e} at {s}"))
    }
    /// A number, or None for null / non-numbers (skipped). Booleans read as 0/1.
    fn num_opt(&mut self) -> R<Option<f64>> {
        match self.peek() {
            b'-' | b'0'..=b'9' => Ok(Some(self.num()?)),
            b't' => {
                self.lit(b"true")?;
                Ok(Some(1.0))
            }
            b'f' => {
                self.lit(b"false")?;
                Ok(Some(0.0))
            }
            _ => {
                self.skip()?;
                Ok(None)
            }
        }
    }
    fn num_or0(&mut self) -> R<f64> {
        Ok(self.num_opt()?.unwrap_or(0.0))
    }
    fn boolean(&mut self) -> R<bool> {
        Ok(self.num_or0()? != 0.0)
    }
    fn skip(&mut self) -> R<()> {
        match self.peek() {
            b'{' => self.obj(|p, _| p.skip()),
            b'[' => self.arr(|p| p.skip()),
            b'"' => self.raw_str().map(|_| ()),
            b't' => self.lit(b"true"),
            b'f' => self.lit(b"false"),
            b'n' => self.lit(b"null"),
            _ => self.num().map(|_| ()),
        }
    }
    fn obj(&mut self, mut f: impl FnMut(&mut Self, &'a [u8]) -> R<()>) -> R<()> {
        if self.peek() != b'{' {
            return self.skip();
        }
        self.i += 1;
        if self.peek() == b'}' {
            self.i += 1;
            return Ok(());
        }
        loop {
            let k = self.raw_str()?;
            self.expect(b':')?;
            f(self, k)?;
            match self.peek() {
                b',' => self.i += 1,
                b'}' => {
                    self.i += 1;
                    return Ok(());
                }
                _ => return Err(format!("bad object at {}", self.i)),
            }
        }
    }
    fn arr(&mut self, mut f: impl FnMut(&mut Self) -> R<()>) -> R<()> {
        if self.peek() != b'[' {
            return self.skip();
        }
        self.i += 1;
        if self.peek() == b']' {
            self.i += 1;
            return Ok(());
        }
        loop {
            f(self)?;
            match self.peek() {
                b',' => self.i += 1,
                b']' => {
                    self.i += 1;
                    return Ok(());
                }
                _ => return Err(format!("bad array at {}", self.i)),
            }
        }
    }
    /// `{item: number}` in key order (Python `int(v)` truncation; null -> 0).
    fn qty(&mut self) -> R<Qty> {
        let mut q = Qty::with_capacity(12);
        self.obj(|p, k| {
            let v = p.num_or0()?;
            q.push((intern(std::str::from_utf8(k).map_err(|e| e.to_string())?), v.trunc() as i64));
            Ok(())
        })?;
        Ok(q)
    }
    fn xy(&mut self) -> R<Option<(i64, i64)>> {
        if self.peek() != b'[' {
            self.skip()?;
            return Ok(None);
        }
        let mut v = [0i64; 2];
        let mut n = 0;
        self.arr(|p| {
            let x = p.num_or0()? as i64;
            if n < 2 {
                v[n] = x;
            }
            n += 1;
            Ok(())
        })?;
        Ok(if n >= 2 { Some((v[0], v[1])) } else { None })
    }
    fn tile(&mut self) -> R<Tile> {
        match self.peek() {
            b'n' => {
                self.lit(b"null")?;
                Ok(Tile::default())
            }
            b'"' => {
                let s = self.raw_str()?;
                // Any string tile reads as LOCKED (only "LOCKED" exists).
                let _ = s;
                Ok(Tile::LOCKED)
            }
            b'{' => {
                let mut t = Tile::default();
                let mut kind: Option<&'static str> = None;
                self.obj(|p, k| {
                    match k {
                        b"kind" => kind = p.str_opt()?.map(intern),
                        b"crop" => t.crop = p.str_opt()?.map(intern).unwrap_or(""),
                        b"animal" => t.animal = p.str_opt()?.map(intern).unwrap_or(""),
                        b"planted_day" => t.planted_day = p.num_or0()? as i64,
                        b"watered_today" => t.watered_today = p.boolean()?,
                        b"consecutive_unwatered" => t.consecutive_unwatered = p.num_or0()? as i64,
                        b"yield_units" => t.yield_units = p.num_or0()? as i64,
                        b"max_lifespan_step" => t.max_lifespan_step = p.num_or0()? as i64,
                        b"fertilized_until_day" => t.fertilized_until_day = p.num_or0()? as i64,
                        b"placed_day" => t.placed_day = p.num_or0()? as i64,
                        b"consecutive_unfed" => t.consecutive_unfed = p.num_or0()? as i64,
                        b"fed_today" => t.fed_today = p.boolean()?,
                        b"cared_today" => t.cared_today = p.boolean()?,
                        b"fertilizer_available" => t.fertilizer_available = p.boolean()?,
                        b"pending_care_bonus" => t.pending_care_bonus = p.num_or0()? as i64,
                        _ => p.skip()?,
                    }
                    Ok(())
                })?;
                // A dict tile without a kind still is a dict: mark it with a non-empty kind.
                t.kind = kind.unwrap_or("?");
                Ok(t)
            }
            _ => {
                self.skip()?;
                Ok(Tile::LOCKED)
            }
        }
    }
    fn farm(&mut self) -> R<FarmObs> {
        let mut f = FarmObs::default();
        self.obj(|p, k| {
            match k {
                b"money" => f.money = p.num_or0()?,
                b"farmer" => f.farmer = p.xy()?,
                b"hands" => p.arr(|p| {
                    if let Some(h) = p.xy()? {
                        f.hands.push(h);
                    }
                    Ok(())
                })?,
                b"hires_today" => f.hires_today = p.num_or0()? as i64,
                b"unlocked_quadrants" => p.arr(|p| {
                    if let Some(s) = p.str_opt()? {
                        f.quadrants.push(intern(s));
                    }
                    Ok(())
                })?,
                b"tiles" => {
                    let (mut rows, mut cols) = (0usize, 0usize);
                    let tiles = &mut f.tiles;
                    tiles.reserve(100);
                    p.arr(|p| {
                        let mut c = 0;
                        p.arr(|p| {
                            tiles.push(p.tile()?);
                            c += 1;
                            Ok(())
                        })?;
                        if rows == 0 {
                            cols = c;
                        } else if c != cols {
                            return Err("ragged tiles".into());
                        }
                        rows += 1;
                        Ok(())
                    })?;
                    f.rows = rows;
                    f.cols = cols;
                }
                _ => p.skip()?,
            }
            Ok(())
        })?;
        Ok(f)
    }
}
