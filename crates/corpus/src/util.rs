//! Small helpers: file fingerprints, dates without a date crate, peak RSS.
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// Cheap file fingerprint: `"<size>:<sha256 of the first 1 MiB, 16 hex>"`.
pub fn fingerprint(path: &Path) -> std::io::Result<String> {
    let size = std::fs::metadata(path)?.len();
    let mut f = File::open(path)?;
    let mut buf = vec![0u8; 1 << 20];
    let mut got = 0usize;
    while got < buf.len() {
        let k = f.read(&mut buf[got..])?;
        if k == 0 {
            break;
        }
        got += k;
    }
    let d = Sha256::digest(&buf[..got]);
    let hex: String = d.iter().take(8).map(|b| format!("{:02x}", b)).collect();
    Ok(format!("{size}:{hex}"))
}

/// (year, month, day) from days since 1970-01-01 (Howard Hinnant's civil_from_days).
fn civil(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// ISO-8601 UTC timestamp for a SystemTime.
pub fn iso(t: SystemTime) -> String {
    let secs = t.duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    let (y, m, d) = civil(secs.div_euclid(86_400));
    let s = secs.rem_euclid(86_400);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", s / 3600, (s / 60) % 60, s % 60)
}

pub fn mtime_iso(path: &Path) -> String {
    std::fs::metadata(path).and_then(|m| m.modified()).map(iso).unwrap_or_default()
}

/// First `YYYY-MM-DD` found in a string (e.g. a dataset directory name).
pub fn find_date(s: &str) -> Option<String> {
    let b = s.as_bytes();
    if b.len() < 10 {
        return None;
    }
    for i in 0..=b.len() - 10 {
        let w = &b[i..i + 10];
        let ok = w.iter().enumerate().all(|(k, c)| match k {
            4 | 7 => *c == b'-',
            _ => c.is_ascii_digit(),
        });
        if ok {
            return Some(String::from_utf8_lossy(w).into_owned());
        }
    }
    None
}

/// Deterministic negative id for a replay file whose stem is not a number.
pub fn hashed_id(stem: &str) -> i64 {
    let d = Sha256::digest(stem.as_bytes());
    let mut v = [0u8; 8];
    v.copy_from_slice(&d[..8]);
    -((u64::from_le_bytes(v) >> 2) as i64) - 1
}

/// Peak resident set size of this process in bytes (0 if unknown).
#[cfg(windows)]
pub fn peak_rss() -> u64 {
    #[repr(C)]
    struct Pmc {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> isize;
        fn K32GetProcessMemoryInfo(h: isize, p: *mut Pmc, cb: u32) -> i32;
    }
    let mut p = Pmc {
        cb: std::mem::size_of::<Pmc>() as u32,
        page_fault_count: 0,
        peak_working_set_size: 0,
        working_set_size: 0,
        quota_peak_paged_pool_usage: 0,
        quota_paged_pool_usage: 0,
        quota_peak_non_paged_pool_usage: 0,
        quota_non_paged_pool_usage: 0,
        pagefile_usage: 0,
        peak_pagefile_usage: 0,
    };
    // SAFETY: plain Win32 call writing into a correctly sized, owned struct.
    let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut p, p.cb) };
    if ok != 0 {
        p.peak_working_set_size as u64
    } else {
        0
    }
}

#[cfg(not(windows))]
pub fn peak_rss() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|k| k.parse::<u64>().ok()))
        })
        .map(|kb| kb * 1024)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates() {
        assert_eq!(civil(0), (1970, 1, 1));
        assert_eq!(civil(20_720), (2026, 9, 24));
        assert_eq!(find_date("kaggriculture-episodes-2026-09-22").as_deref(), Some("2026-09-22"));
        assert!(hashed_id("selfplay_a") < 0);
    }
}
