//! Export equivalence: `policy-check WEIGHTS STATES EXPECTED K`
//! STATES = f32 [K, 30, 93], EXPECTED = f32 [K, 30, A + 1] (torch pi logits then v logit).
//! Prints the max abs difference; exit 1 if it exceeds 1e-4.
fn read_f32(p: &str) -> Vec<f32> {
    std::fs::read(p).expect(p).chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let net = policy::Net::load(&a[1]).expect("weights");
    let (xs, ys, k): (Vec<f32>, Vec<f32>, usize) = (read_f32(&a[2]), read_f32(&a[3]), a[4].parse().unwrap());
    let w = net.n_act + 1;
    assert_eq!(xs.len(), k * 30 * policy::NF);
    assert_eq!(ys.len(), k * 30 * w);
    let mut worst = 0f32;
    for i in 0..k {
        let mut h = [0f32; policy::H];
        for t in 0..30 {
            let x = &xs[(i * 30 + t) * policy::NF..(i * 30 + t + 1) * policy::NF];
            let o = net.step(&mut h, x);
            let y = &ys[(i * 30 + t) * w..(i * 30 + t + 1) * w];
            for j in 0..net.n_act {
                worst = worst.max((o.pi[j] - y[j]).abs());
            }
            worst = worst.max((o.v - y[net.n_act]).abs());
        }
    }
    println!("policy-check: {k} sequences x 30 days, {} actions, max abs diff {worst:e}", net.n_act);
    std::process::exit(if worst <= 1e-4 { 0 } else { 1 });
}
