//! Frozen vocabularies (engine 1.32.7). Every per-item column uses these orders.

/// Market products, alphabetical (the corpus column order).
pub const PRODUCTS: [&str; 9] = [
    "CARROT", "EGG", "FERTILIZER", "MELON", "MILK", "STRAWBERRY", "TOMATO", "WHEAT", "WOOL",
];
pub const P_CARROT: usize = 0;
pub const P_EGG: usize = 1;
pub const P_FERTILIZER: usize = 2;
pub const P_MELON: usize = 3;
pub const P_MILK: usize = 4;
pub const P_STRAWBERRY: usize = 5;
pub const P_TOMATO: usize = 6;
pub const P_WHEAT: usize = 7;
pub const P_WOOL: usize = 8;

/// Crops, alphabetical.
pub const CROPS: [&str; 5] = ["CARROT", "MELON", "STRAWBERRY", "TOMATO", "WHEAT"];
/// `CROPS[..].first_yield_day` from the engine, same order as `CROPS`.
pub const CROP_FIRST_YIELD_DAY: [i64; 5] = [2, 10, 10, 8, 2];

/// Animals, alphabetical.
pub const ANIMALS: [&str; 3] = ["COW", "GOOSE", "SHEEP"];

/// Town shops, alphabetical (bit i of the shop bitmask = SHOPS[i]).
pub const SHOPS: [&str; 8] = [
    "BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP", "PET_CAFE", "PIZZA_SHOP",
    "SMOOTHIE_SHOP", "YARN_STORE",
];

/// Products each shop instance pulls per shop tick (engine `SHOPS`), as product indices.
pub fn shop_items(shop: usize) -> &'static [usize] {
    match shop {
        0 => &[P_EGG, P_WHEAT],
        1 => &[P_EGG, P_WHEAT, P_STRAWBERRY],
        2 => &[P_WHEAT, P_CARROT, P_TOMATO, P_STRAWBERRY],
        3 => &[P_STRAWBERRY, P_MILK, P_WHEAT],
        4 => &[P_CARROT],
        5 => &[P_MILK, P_TOMATO, P_WHEAT],
        6 => &[P_STRAWBERRY, P_MILK],
        7 => &[P_WOOL],
        _ => &[],
    }
}

/// Engine cadences (default configuration).
pub const TURNS_PER_DAY: usize = 24;
pub const SHOP_SELL_INTERVAL: usize = 4;
pub const CENTER_SELL_INTERVAL: usize = 24;
pub const MAX_MARKET_ORDERS: usize = 10;

/// Hour windows for per-window sales (inclusive bounds).
pub const HOUR_WINDOWS: [(usize, usize, &str); 5] = [
    (0, 2, "h00_02"),
    (3, 9, "h03_09"),
    (10, 14, "h10_14"),
    (15, 20, "h15_20"),
    (21, 23, "h21_23"),
];

/// Trailing window (turns) for the "sell into strength" quote average.
pub const STRENGTH_WINDOW: usize = 12;

/// Stream-hash cuts (turns) reported per seat.
pub const HASH_CUTS: [usize; 4] = [24, 48, 100, 136];

#[inline]
pub fn index_of(list: &[&str], s: &str) -> Option<usize> {
    list.iter().position(|&x| x == s)
}

#[inline]
pub fn lower(s: &str) -> String {
    s.to_ascii_lowercase()
}
