//! debug "kv_write": how far the keys and values the last step of the sampler leaves in a block's cache are from the
//! ones the pass at t = 0 writes over them (video.rs, chunk). Every pass of a chunk writes the chunk's keys and values
//! at `at`; the pass at t = 0 (the clean chunk, as trained) overwrites the last step's (the noisy input of t = 833).
//! Its answer decides whether that pass, a third of a chunk, can go: cosine and relative L2 per block, of a sample of
//! the chunk's tokens, over every chunk of the run.

use serde_json::{json, Value};

/// Tokens of a chunk the comparison reads, evenly spread over it.
pub const SAMPLE: usize = 256;

/// float8 e4m3fn (torch.float8_e4m3fn): sign, 4 exponent bits of bias 7, 3 mantissa bits; no infinities, 0x7f NaN.
pub fn fp8_e4m3(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = ((b >> 3) & 0x0f) as i32;
    let man = (b & 0x07) as f32;
    if exp == 0x0f && b & 0x07 == 0x07 {
        return f32::NAN;
    }
    if exp == 0 { sign * man / 8.0 * 2f32.powi(-6) } else { sign * (1.0 + man / 8.0) * 2f32.powi(exp - 7) }
}

/// The chunk's clean latent noised to timestep `t` as the sampler's flow has it: (1 - t/1000) x0 + t/1000 noise.
pub fn noised(x0: &[f32], noise: &[f32], t: f64) -> Vec<f32> {
    let sigma = (t / 1000.0) as f32;
    x0.iter().zip(noise).map(|(x, n)| (1.0 - sigma) * x + sigma * n).collect()
}

/// The token indices of a chunk of `n` tokens the comparison reads.
pub fn sample(n: usize) -> Vec<usize> {
    let k = SAMPLE.min(n);
    (0..k).map(|i| i * n / k).collect()
}

/// One token's row of a cache in floats: `values` heads * dim FP8 bytes, `scales` one float32 per head.
pub fn dequantize(values: &[u8], scales: &[f32], dim: usize) -> Vec<f32> {
    values.iter().enumerate().map(|(i, &b)| fp8_e4m3(b) * scales[i / dim]).collect()
}

/// Sums of one block's comparison: a.b, a.a, b.b and |a - b|^2 of keys and of values.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Sums { k: [f64; 4], v: [f64; 4] }

impl Sums {
    fn add(acc: &mut [f64; 4], a: &[f32], b: &[f32]) {
        for (x, y) in a.iter().zip(b) {
            let (x, y) = (*x as f64, *y as f64);
            acc[0] += x * y;
            acc[1] += x * x;
            acc[2] += y * y;
            acc[3] += (x - y) * (x - y);
        }
    }
    /// a token of keys and of values, last step (`step`) against the pass at t = 0 (`clean`)
    pub fn add_token(&mut self, step_k: &[f32], clean_k: &[f32], step_v: &[f32], clean_v: &[f32]) {
        Self::add(&mut self.k, step_k, clean_k);
        Self::add(&mut self.v, step_v, clean_v);
    }
    fn stats(s: &[f64; 4]) -> Value {
        json!({"cosine": s[0] / (s[1].sqrt() * s[2].sqrt()).max(1e-30), "relative_l2": (s[3] / s[2].max(1e-30)).sqrt()})
    }
    pub fn json(&self) -> Value { json!({"k": Self::stats(&self.k), "v": Self::stats(&self.v)}) }
}

/// The comparison of a run: per block, summed over its chunks; the last step's, and of every probe timestep
/// (generate.kv_probe: a pass over the clean chunk noised to t).
#[derive(Clone, Debug, Default)]
pub struct Report { pub blocks: Vec<Sums>, pub probes: Vec<(f64, Vec<Sums>)>, pub chunks: usize }

fn blocks_json(blocks: &[Sums]) -> (Vec<Value>, Value) {
    let per: Vec<Value> = blocks.iter().map(Sums::json).collect();
    let mean = |key: &str, stat: &str| per.iter().map(|b| b[key][stat].as_f64().unwrap_or(f64::NAN)).sum::<f64>() / per.len().max(1) as f64;
    let m = json!({"k": {"cosine": mean("k", "cosine"), "relative_l2": mean("k", "relative_l2")},
                   "v": {"cosine": mean("v", "cosine"), "relative_l2": mean("v", "relative_l2")}});
    (per, m)
}

impl Report {
    pub fn new(blocks: usize) -> Report { Report { blocks: vec![Sums::default(); blocks], probes: Vec::new(), chunks: 0 } }
    /// The sums of probe timestep `t`, made on first use; None for a timestep that is no number.
    pub fn probe(&mut self, t: f64) -> Option<&mut Vec<Sums>> {
        if !t.is_finite() { return None; }
        let blocks = self.blocks.len();
        let at = match self.probes.iter().position(|(p, _)| *p == t) {
            Some(i) => i,
            None => { self.probes.push((t, vec![Sums::default(); blocks])); self.probes.len() - 1 }
        };
        Some(&mut self.probes[at].1)
    }
    pub fn json(&self) -> Value {
        let (per, mean) = blocks_json(&self.blocks);
        let probes: Vec<Value> = self.probes.iter().map(|(t, b)| {
            let (per, mean) = blocks_json(b);
            json!({"t": t, "blocks": per, "mean": mean})
        }).collect();
        json!({"chunks": self.chunks, "sample_tokens": SAMPLE, "blocks": per, "mean": mean, "probes": probes})
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// a probe pass sees the chunk's clean latent noised to t as the sampler would: (1 - t/1000) x0 + t/1000 noise
    #[test]
    fn the_probe_latent_is_the_flow_between_clean_and_noise() {
        assert_eq!(noised(&[1.0, 2.0], &[3.0, -2.0], 250.0), vec![1.5, 1.0]);
        assert_eq!(noised(&[1.0], &[9.0], 0.0), vec![1.0]);
    }

    #[test]
    fn probes_are_reported_per_timestep_beside_the_last_step() {
        let mut r = Report::new(1);
        r.probe(600.0).unwrap()[0].add_token(&[1.0], &[1.0], &[1.0], &[-1.0]);
        r.probe(100.0).unwrap()[0].add_token(&[1.0], &[1.0], &[1.0], &[1.0]);
        r.chunks = 1;
        let j = r.json();
        assert_eq!(j["probes"].as_array().unwrap().len(), 2);
        assert_eq!(j["probes"][0]["t"], 600.0);
        assert!((j["probes"][0]["mean"]["v"]["cosine"].as_f64().unwrap() + 1.0).abs() < 1e-12);
        assert!((j["probes"][1]["mean"]["v"]["cosine"].as_f64().unwrap() - 1.0).abs() < 1e-12);
        assert!(Report::new(1).probe(f64::NAN).is_none());
    }

    #[test]
    fn fp8_bytes_decode_as_torch_float8_e4m3fn() {
        // torch.tensor([...]).to(torch.float8_e4m3fn).view(torch.uint8)
        for (byte, value) in [(0x00u8, 0.0f32), (0x38, 1.0), (0xb8, -1.0), (0x40, 2.0), (0x7e, 448.0), (0x01, 2f32.powi(-9)),
                              (0x3c, 1.5), (0x34, 0.75)] {
            assert_eq!(fp8_e4m3(byte), value, "{byte:#x}");
        }
        assert!(fp8_e4m3(0x7f).is_nan());
    }

    #[test]
    fn the_sample_spreads_over_the_chunk_and_fits_a_small_one() {
        let s = sample(5320);
        assert_eq!((s.len(), s[0], *s.last().unwrap()), (SAMPLE, 0, 5320 - 5320 / SAMPLE - 1));
        assert!(s.windows(2).all(|w| w[1] > w[0]));
        assert_eq!(sample(10), (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn scales_are_per_head() {
        assert_eq!(dequantize(&[0x38, 0x40, 0x38, 0x40], &[0.5, 2.0], 2), vec![0.5, 1.0, 2.0, 4.0]);
    }

    #[test]
    fn equal_tokens_give_cosine_one_and_no_error_and_opposite_ones_minus_one() {
        let mut r = Report::new(2);
        r.blocks[0].add_token(&[1.0, 2.0], &[1.0, 2.0], &[3.0, 0.0], &[-3.0, 0.0]);
        r.blocks[1].add_token(&[1.0, 0.0], &[0.0, 1.0], &[2.0, 0.0], &[1.0, 0.0]);
        r.chunks = 1;
        let j = r.json();
        assert!((j["blocks"][0]["k"]["cosine"].as_f64().unwrap() - 1.0).abs() < 1e-12);
        assert_eq!(j["blocks"][0]["k"]["relative_l2"].as_f64().unwrap(), 0.0);
        assert!((j["blocks"][0]["v"]["cosine"].as_f64().unwrap() + 1.0).abs() < 1e-12);
        assert!((j["blocks"][1]["k"]["cosine"].as_f64().unwrap()).abs() < 1e-12);
        assert!((j["blocks"][1]["v"]["relative_l2"].as_f64().unwrap() - 1.0).abs() < 1e-12);
        assert!((j["mean"]["k"]["cosine"].as_f64().unwrap() - 0.5).abs() < 1e-12);
    }
}
