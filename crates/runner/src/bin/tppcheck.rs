//! TPP train/serve equivalence: run the Rust forward (crates/agent/src/tpp.rs) on bcdump records and write
//! every output as f32 so python/tpp/check.py can compare it with the PyTorch forward of the same records.
//!
//!     tppcheck --net NET.json --records part_00.bin --n 256 --out logits.bin
//! Per record: units MAXU x (ucls NUNIT ++ uq NUQ), issue NMKT, mq NMKT*NMQ, hire 11 (zeros for absent units).
use agent::tpp;
use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).map(|i| args[i + 1].clone());
    let net = tpp::Net::load(&get("--net").expect("--net")).expect("net");
    let data = std::fs::read(get("--records").expect("--records")).expect("records");
    let n: usize = get("--n").and_then(|s| s.parse().ok()).unwrap_or(256);
    let rec = 16 + tpp::C * tpp::B * tpp::B + tpp::G * 4 + tpp::MAXU * tpp::UF + tpp::MAXU * 2 + tpp::MAXM * 2;
    let mut out = std::io::BufWriter::new(std::fs::File::create(get("--out").expect("--out")).unwrap());
    let t0 = std::time::Instant::now();
    let mut done = 0;
    for r in data.chunks_exact(rec).take(n) {
        let nu = r[6] as usize;
        let mut o = 16;
        let board = &r[o..o + tpp::C * tpp::B * tpp::B];
        o += board.len();
        let glob: Vec<f32> = (0..tpp::G).map(|i| f32::from_le_bytes(r[o + 4 * i..o + 4 * i + 4].try_into().unwrap())).collect();
        o += tpp::G * 4;
        let units: Vec<[u8; tpp::UF]> = (0..nu).map(|i| r[o + i * tpp::UF..o + (i + 1) * tpp::UF].try_into().unwrap()).collect();
        let y = net.forward(board, &glob, &units);
        for i in 0..tpp::MAXU {
            let (a, b) = if i < nu { (y.ucls[i].clone(), y.uq[i].clone()) } else { (vec![0.0; tpp::NUNIT], vec![0.0; tpp::UQ_REP.len()]) };
            for v in a.iter().chain(b.iter()) {
                out.write_all(&v.to_le_bytes()).unwrap();
            }
        }
        for v in y.issue.iter().chain(y.mq.iter()).chain(y.hire.iter()) {
            out.write_all(&v.to_le_bytes()).unwrap();
        }
        done += 1;
    }
    eprintln!("[tppcheck] {done} records, {:.2} ms per forward", t0.elapsed().as_secs_f64() * 1e3 / done.max(1) as f64);
}
