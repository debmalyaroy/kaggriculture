//! Macro policy forward pass (arch gru64-v1), bit-compatible with `python/learn/model.py`.
//!
//! weights.bin = little-endian f32 tensors in model.EXPORT_ORDER:
//!   mu[93] sd[93] inp.W[64,93] inp.b[64] gru.W_ih[192,64] gru.W_hh[192,64] gru.b_ih[192]
//!   gru.b_hh[192] pi.W[A,64] pi.b[A] v.W[1,64] v.b[1] rnext.W[9,64] rnext.b[9] band.W[5,64] band.b[5]
//! The action count A is recovered from the file size. torch GRUCell gate order is (r, z, n):
//!   r = s(W_ir x + b_ir + W_hr h + b_hr); z = s(W_iz x + b_iz + W_hz h + b_hz)
//!   n = tanh(W_in x + b_in + r * (W_hn h + b_hn)); h' = (1 - z) * n + z * h
pub const NF: usize = 93; // = dayobs::N (FEAT_VERSION 2)
pub const H: usize = 64;
pub const N_BAND: usize = 5;
/// Optional trailer after the tensors: [day_1, .., day_n, n, MID_MAGIC] (f32). It lists the days (25-29 in
/// the ppo2 runs) that also get a second, hour-13 decision. Files without it (every older export) have none.
pub const MID_MAGIC: u32 = 0x4D49_4431; // "MID1"

#[derive(Clone, Debug)]
pub struct Net {
    pub n_act: usize,
    /// Days with a second decision at hour 13 (empty = one decision per day, hour 1).
    pub mid: Vec<usize>,
    mu: Vec<f32>,
    sd: Vec<f32>,
    inp_w: Vec<f32>,
    inp_b: Vec<f32>,
    w_ih: Vec<f32>,
    w_hh: Vec<f32>,
    b_ih: Vec<f32>,
    b_hh: Vec<f32>,
    pi_w: Vec<f32>,
    pi_b: Vec<f32>,
    v_w: Vec<f32>,
    v_b: Vec<f32>,
    band_w: Vec<f32>,
    band_b: Vec<f32>,
}

/// One day's outputs.
#[derive(Clone, Debug)]
pub struct Out {
    pub pi: Vec<f32>,
    pub v: f32,
    pub band: [f32; N_BAND],
}

fn mv(w: &[f32], b: &[f32], x: &[f32], out: &mut [f32]) {
    let n = x.len();
    for (i, o) in out.iter_mut().enumerate() {
        let row = &w[i * n..(i + 1) * n];
        let mut s = b[i];
        for j in 0..n {
            s += row[j] * x[j];
        }
        *o = s;
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl Net {
    pub fn from_bytes(b: &[u8]) -> Result<Net, String> {
        if b.len() % 4 != 0 {
            return Err("weights.bin size not a multiple of 4".into());
        }
        let mut f: Vec<f32> = b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let mut mid = vec![];
        if f.len() >= 2 && f.last().is_some_and(|x| x.to_bits() == MID_MAGIC) {
            let n = f[f.len() - 2] as usize;
            if f.len() < n + 2 {
                return Err("weights.bin: bad mid-day trailer".into());
            }
            mid = f[f.len() - 2 - n..f.len() - 2].iter().map(|x| *x as usize).collect();
            f.truncate(f.len() - 2 - n);
        }
        let fixed = 2 * NF + H * NF + H + 2 * 3 * H * H + 2 * 3 * H + (H + 1) + (9 * H + 9) + (N_BAND * H + N_BAND);
        if f.len() <= fixed || (f.len() - fixed) % (H + 1) != 0 {
            return Err(format!("weights.bin has {} floats; not a gru64-v1 export", f.len()));
        }
        let n_act = (f.len() - fixed) / (H + 1);
        let mut at = 0usize;
        let mut take = |n: usize| {
            let v = f[at..at + n].to_vec();
            at += n;
            v
        };
        let net = Net {
            n_act,
            mid,
            mu: take(NF),
            sd: take(NF),
            inp_w: take(H * NF),
            inp_b: take(H),
            w_ih: take(3 * H * H),
            w_hh: take(3 * H * H),
            b_ih: take(3 * H),
            b_hh: take(3 * H),
            pi_w: take(n_act * H),
            pi_b: take(n_act),
            v_w: take(H),
            v_b: take(1),
            band_w: {
                let _rnext_w = take(9 * H);
                let _rnext_b = take(9);
                take(N_BAND * H)
            },
            band_b: take(N_BAND),
        };
        Ok(net)
    }

    pub fn load(path: &str) -> Result<Net, String> {
        let b = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        Net::from_bytes(&b)
    }

    /// One day: update the hidden state `h` with observation `x` and return the heads.
    pub fn step(&self, h: &mut [f32; H], x: &[f32]) -> Out {
        let mut xn = [0f32; NF];
        for i in 0..NF {
            xn[i] = (x[i] - self.mu[i]) / self.sd[i];
        }
        let mut z = [0f32; H];
        mv(&self.inp_w, &self.inp_b, &xn, &mut z);
        for v in z.iter_mut() {
            *v = v.max(0.0);
        }
        let mut gi = [0f32; 3 * H];
        let mut gh = [0f32; 3 * H];
        mv(&self.w_ih, &self.b_ih, &z, &mut gi);
        mv(&self.w_hh, &self.b_hh, &h[..], &mut gh);
        let mut hn = [0f32; H];
        for i in 0..H {
            let r = sigmoid(gi[i] + gh[i]);
            let u = sigmoid(gi[H + i] + gh[H + i]);
            let n = (gi[2 * H + i] + r * gh[2 * H + i]).tanh();
            hn[i] = (1.0 - u) * n + u * h[i];
        }
        *h = hn;
        let mut pi = vec![0f32; self.n_act];
        mv(&self.pi_w, &self.pi_b, &h[..], &mut pi);
        let mut v = [0f32; 1];
        mv(&self.v_w, &self.v_b, &h[..], &mut v);
        let mut band = [0f32; N_BAND];
        mv(&self.band_w, &self.band_b, &h[..], &mut band);
        Out { pi, v: v[0], band }
    }
}
