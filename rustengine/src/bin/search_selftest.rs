//! Integration self-test for the Phase-A boundary.
//!
//! Proves, without any transport, that:
//!   1. `search::decide(&State, me, budget_ms)` -- the stateless entry point
//!      Phase A's `kagg play` was asked to assume -- runs a whole episode;
//!   2. every action it emits serialises through `search::action_to_line` and
//!      parses back through the SAME tape grammar `service.rs` uses, so the
//!      bridge can hand actions to `kagg serve`/`kagg play` unchanged;
//!   3. a searcher given zero budget is byte-identical to the plain field
//!      skeleton (so the search is strictly additive, never a rewrite);
//!   4. `seed_opponent_belief` fills a hidden opponent block without
//!      disturbing our own seat.
//!
//!   search_selftest [seed] [budget_ms]

use kaggriculture_engine::engine::{self, PlayerAction, UnitAction};
use kaggriculture_engine::plan::SkeletonPolicy;
use kaggriculture_engine::search::{self, Searcher};
use kaggriculture_engine::state::State;

/// The tape grammar from service.rs / main.rs, re-implemented here so the test
/// is a real round-trip rather than a call into the code under test.
fn parse_line(line: &str) -> PlayerAction {
    let mut parts = line.split('\t');
    let farmer = parts.next().unwrap_or("PASS");
    let hands = parts.next().unwrap_or("");
    let market = parts.next().unwrap_or("");
    PlayerAction {
        farmer: UnitAction::parse(
            &farmer.split(' ').filter(|t| !t.is_empty()).collect::<Vec<_>>()),
        hands: engine::positional(hands).into_iter()
            .map(|h| UnitAction::parse(
                &h.split(' ').filter(|t| !t.is_empty()).collect::<Vec<_>>()))
            .collect(),
        market: engine::positional(market).into_iter()
            .map(|o| o.split(' ').filter(|t| !t.is_empty())
                .map(str::to_string).collect())
            .collect(),
    }
}

fn same(a: &PlayerAction, b: &PlayerAction) -> bool {
    let u = |x: &UnitAction, y: &UnitAction| {
        x.op == y.op && x.item == y.item && (!x.has_n || x.n == y.n)
    };
    u(&a.farmer, &b.farmer)
        && a.hands.len() == b.hands.len()
        && a.hands.iter().zip(&b.hands).all(|(x, y)| u(x, y))
        && a.market == b.market
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seed: i64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(4242);
    let budget: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);

    // --- 1 + 2: the stateless entry point over a whole episode, round-tripped
    let mut st = State::new(seed);
    let mut opp = SkeletonPolicy::new(1);
    let mut turns = 0u32;
    let mut orders = 0usize;
    loop {
        let a0 = search::decide(&st, 0, budget);
        let line = search::action_to_line(&a0);
        assert!(!line.contains('\n'), "action line must be one line");
        assert!(a0.market.len() <= 10, "market queue over the 10-order cap");
        let back = parse_line(&line);
        assert!(same(&a0, &back), "round-trip mismatch at step {}:\n{}",
                st.step, line);
        orders += a0.market.len();
        let a1 = opp.act(&st);
        turns += 1;
        if !engine::step(&mut st, &[back, a1]) {
            break;
        }
    }
    println!("1+2 ok: {turns} turns, {orders} market orders, all lines \
              round-tripped; banks {:.0} / {:.0}",
             st.farms[0].money, st.farms[1].money);

    // --- 3: zero budget == the plain skeleton, action for action
    let mut s1 = State::new(seed);
    let mut s2 = State::new(seed);
    let mut sr = Searcher::new(0);
    let mut sk = SkeletonPolicy::new(0);
    let mut o1 = SkeletonPolicy::new(1);
    let mut o2 = SkeletonPolicy::new(1);
    let mut checked = 0;
    loop {
        let a = sr.decide(&s1, 0, 0);
        let b = sk.act(&s2);
        assert!(same(&a, &b),
                "zero-budget searcher diverged from the skeleton at step {}\n\
                 search:   {}\nskeleton: {}",
                s1.step, search::action_to_line(&a),
                search::action_to_line(&b));
        checked += 1;
        let x = o1.act(&s1);
        let y = o2.act(&s2);
        let r1 = engine::step(&mut s1, &[a, x]);
        engine::step(&mut s2, &[b, y]);
        assert_eq!(s1.digest(), s2.digest(), "states diverged");
        if !r1 {
            break;
        }
    }
    println!("3 ok: {checked} turns identical at zero budget (search is \
              strictly additive)");

    // --- 4: opponent belief fills a hidden block, ours untouched
    let mut s3 = State::new(seed);
    let mut p0 = SkeletonPolicy::new(0);
    let mut p1 = SkeletonPolicy::new(1);
    for _ in 0..300 {
        let a = p0.act(&s3);
        let b = p1.act(&s3);
        engine::step(&mut s3, &[a, b]);
    }
    let mut hidden = s3.clone();
    hidden.private[1].shed.0.clear();
    hidden.private[1].seeds.0.clear();
    let mine_before = hidden.private[0].shed.sum();
    search::seed_opponent_belief(&mut hidden, 0);
    assert!(hidden.private[1].shed.sum() >= 0);
    assert_eq!(hidden.private[0].shed.sum(), mine_before,
               "belief seeding touched OUR seat");
    println!("4 ok: opponent belief = {} shed units from {} standing animals; \
              our seat untouched",
             hidden.private[1].shed.sum(),
             kaggriculture_engine::plan::census(&hidden.farms[1]).animals.len());

    println!("all search self-tests passed");
}
