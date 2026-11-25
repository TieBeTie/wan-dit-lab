//! The descent in resolution (debug "descent", generate.descent): what makes a run at a lower resolution behave as one
//! at the native size, measured without depending on the resolution, and the knobs that move it without a new export.
//!
//! Knobs (generate.descent):
//! - attn_logit_scale lambda, global, per block or per block and head: the self-attention logits q.k/sqrt(d) times
//!   lambda. The host writes lambda * norm_q.weight (bfloat16) over that slice of a block's row on the GPU before the
//!   block runs, so the packages stay as exported (the RoPE table of export.rope.attn_logit_scale stays as it is);
//! - timestep_shift: the sampler's two timesteps and coefficients, computed here as export_live.causal_unipc gives
//!   them (two Euler steps over the shifted sigmas), checked at load against the ones the packages carry.
//!
//! Metrics (written into <run>/descent.json): for every pass of the sampler and every block, of the rotated queries and
//! keys of the probe package on a fixed sample of queries of the chunk, per head: the entropy of the attention and its
//! entropy relative to the uniform one (H - ln N: the same for a pattern that covers the same part of the frame at any
//! N), the mean distance of attention in fractions of the frame and in latent frames, each for a grid of multipliers of
//! the logits (what a lambda times that would give, for the fit); the RMS of the tokens after the block; the latent's
//! mean and deviation per channel. Against a reference run (generate.descent.reference): the reference's latent and
//! frames resized by area to this size, and the multiplier per block and head that matches its relative entropy.

use std::path::Path;

use serde_json::{json, Value};

/// Multipliers of the logits the entropy is measured at, for the fit of lambda (1.0 among them).
pub const MULTIPLIERS: [f64; 13] = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30];
/// Queries per pass and block the attention is measured at.
pub const QUERIES: usize = 256;

pub fn bf16_to_f32(bits: u16) -> f32 { f32::from_bits((bits as u32) << 16) }

/// Round to the nearest bfloat16, ties to even (as torch does).
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16
}

pub fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(2).map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect()
}

/// norm_q.weight (bfloat16 bytes of [heads * head_dim]) with head h's slice times lambda[h], as bfloat16 bytes.
pub fn scaled_norm(weight: &[u8], lambda: &[f64]) -> Vec<u8> {
    let values = bf16_bytes_to_f32(weight);
    let per = values.len() / lambda.len();
    assert_eq!(per * lambda.len(), values.len(), "norm_q.weight does not split into the heads");
    values.iter().enumerate().flat_map(|(i, w)| f32_to_bf16((*w as f64 * lambda[i / per]) as f32).to_le_bytes()).collect()
}

/// Timesteps and coefficients c_k of x_{k+1} = x_k - c_k flow_k of LongLive 2.0's UniPC over `steps` steps with
/// `shift`: its sigmas are linspace(0.999, 0, steps + 1) without the last, shifted, kept in float32 with a final 0;
/// a timestep is the shifted sigma times 1000 truncated; with two steps UniPC is two Euler steps (export_live checks it).
pub fn unipc(shift: f64, steps: usize) -> (Vec<f32>, Vec<f32>) {
    let base: Vec<f64> = (0..steps).map(|i| 0.999 - 0.999 * i as f64 / steps as f64).collect();
    let shifted: Vec<f64> = base.iter().map(|s| shift * s / (1.0 + (shift - 1.0) * s)).collect();
    let timesteps = shifted.iter().map(|s| (s * 1000.0).trunc() as f32).collect();
    let mut sigmas: Vec<f32> = shifted.iter().map(|&s| s as f32).collect();
    sigmas.push(0.0);
    let coefficients = sigmas.windows(2).map(|w| (w[0] as f64 - w[1] as f64) as f32).collect();
    (timesteps, coefficients)
}

/// (offset, bytes) of self_attn.norm_q.weight in a row of the pack, from its `<pack>.json` (bfloat16 only).
pub fn norm_q_slice(pack: &Value) -> Result<(usize, usize), String> {
    let entry = pack["layout"].as_array().and_then(|l| l.iter().find(|e| e["name"] == "self_attn.norm_q.weight"))
        .ok_or("the pack's layout has no self_attn.norm_q.weight")?;
    if entry["dtype"] != "bfloat16" { return Err(format!("self_attn.norm_q.weight is {}, not bfloat16", entry["dtype"])); }
    let at = |k: &str| entry[k].as_u64().map(|v| v as usize).ok_or(format!("self_attn.norm_q.weight has no {k}"));
    Ok((at("offset")?, at("bytes")?))
}

/// The sampler (timesteps, coefficients) of `shift`, or the packages' own one without it; refused unless the packages'
/// one (`timesteps`, `coefficients`) is unipc of their `packaged` shift, so that both come from the same formula.
pub fn sampler(timesteps: &[f32], coefficients: &[f32], packaged: f64, shift: Option<f64>) -> Result<(Vec<f32>, Vec<f32>), String> {
    let Some(shift) = shift else { return Ok((timesteps.to_vec(), coefficients.to_vec())) };
    let (t, c) = unipc(packaged, timesteps.len());
    let same = t == timesteps && c.iter().zip(coefficients).all(|(a, b)| (a - b).abs() < 1e-5);
    if !same {
        return Err(format!("the packages' sampler {timesteps:?} {coefficients:?} is not unipc of shift {packaged}: cannot replace it"));
    }
    Ok(unipc(shift, timesteps.len()))
}

/// The lambda of every block and head: one number, a number per block, or a list of numbers per block.
pub fn lambda_table(scale: &Value, blocks: usize, heads: usize) -> Result<Vec<Vec<f64>>, String> {
    let number = |v: &Value| v.as_f64().filter(|x| x.is_finite() && *x > 0.0 && *x <= 4.0)
        .ok_or_else(|| format!("generate.descent.attn_logit_scale: {v} is not a number in (0, 4]"));
    if scale.is_number() { return Ok(vec![vec![number(scale)?; heads]; blocks]); }
    let list = scale.as_array().ok_or("generate.descent.attn_logit_scale must be a number or a list")?;
    if list.len() != blocks { return Err(format!("generate.descent.attn_logit_scale has {} blocks, the model {blocks}", list.len())); }
    list.iter().map(|b| match b.as_array() {
        None => Ok(vec![number(b)?; heads]),
        Some(h) if h.len() == heads => h.iter().map(number).collect(),
        Some(h) => Err(format!("generate.descent.attn_logit_scale: a block has {} heads, the model {heads}", h.len())),
    }).collect()
}

/// lambda's numbers valid, before the model's sizes are known: lambda_table of its own shape.
pub fn check_scale(scale: &Value) -> Result<(), String> {
    let blocks = scale.as_array().map_or(1, Vec::len);
    let heads = scale.as_array().and_then(|l| l.iter().find_map(|b| b.as_array().map(Vec::len))).unwrap_or(1);
    lambda_table(scale, blocks, heads).map(|_| ())
}

/// `count` distinct token indices of `n` (all of them when count >= n), sorted, the same for the same seed.
pub fn sample_queries(n: usize, count: usize, seed: u64) -> Vec<usize> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut picked = rand::seq::index::sample(&mut rng, n, count.min(n)).into_vec();
    picked.sort_unstable();
    picked
}

/// Frames, height and width of the chunk in tokens.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grid { pub frames: usize, pub height: usize, pub width: usize }

impl Grid {
    pub fn tokens(&self) -> usize { self.frames * self.height * self.width }
    fn place(&self, t: usize) -> (f64, f64, f64) {
        let plane = self.height * self.width;
        ((t / plane) as f64, ((t % plane) / self.width) as f64, (t % self.width) as f64)
    }
}

/// Of one head at one multiplier of the logits, means over the queries.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Stat {
    /// -sum p ln p, nats
    pub entropy: f64,
    /// entropy - ln N
    pub relative: f64,
    /// sum p |position - query's|, the position in fractions of the frame's height and width
    pub spatial: f64,
    /// sum p |frame - query's frame|, latent frames
    pub temporal: f64,
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let tail: f32 = ca.remainder().iter().zip(cb.remainder()).map(|(x, y)| x * y).sum();
    for (x, y) in ca.zip(cb) { for i in 0..8 { acc[i] += x[i] * y[i]; } }
    acc.iter().sum::<f32>() + tail
}

/// softmax(m q.k / sqrt(d)) of every query of `queries` over all `grid.tokens()` keys, per head and multiplier m:
/// q and k are [tokens, heads, d] float32, the chunk's own keys (the first chunk sees only those).
pub fn attention(q: &[f32], k: &[f32], heads: usize, d: usize, grid: Grid, queries: &[usize], multipliers: &[f64])
    -> Vec<Vec<Stat>> {
    let n = grid.tokens();
    assert_eq!(q.len(), n * heads * d);
    assert_eq!(k.len(), n * heads * d);
    let scale = 1.0 / (d as f32).sqrt();
    let places: Vec<(f64, f64, f64)> = (0..n).map(|t| grid.place(t)).collect();
    let one_head = |h: usize| -> Vec<Stat> {
        let mut sums = vec![Stat::default(); multipliers.len()];
        let mut logits = vec![0f32; n];
        let (mut spatial, mut temporal) = (vec![0f64; n], vec![0f64; n]);
        for &i in queries {
            let qi = &q[(i * heads + h) * d..(i * heads + h + 1) * d];
            let (fq, yq, xq) = places[i];
            for j in 0..n {
                logits[j] = dot(qi, &k[(j * heads + h) * d..(j * heads + h + 1) * d]) * scale;
                let (f, y, x) = places[j];
                spatial[j] = (((y - yq) / grid.height as f64).powi(2) + ((x - xq) / grid.width as f64).powi(2)).sqrt();
                temporal[j] = (f - fq).abs();
            }
            let top = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
            for (m, sum) in multipliers.iter().zip(sums.iter_mut()) {
                let (mut z, mut zl, mut sp, mut tm) = (0f64, 0f64, 0f64, 0f64);
                for j in 0..n {
                    let shifted = m * (logits[j] as f64 - top);
                    let w = shifted.exp();
                    z += w; zl += w * shifted; sp += w * spatial[j]; tm += w * temporal[j];
                }
                let entropy = z.ln() - zl / z;
                sum.entropy += entropy;
                sum.relative += entropy - (n as f64).ln();
                sum.spatial += sp / z;
                sum.temporal += tm / z;
            }
        }
        let count = queries.len().max(1) as f64;
        sums.iter().map(|s| Stat { entropy: s.entropy / count, relative: s.relative / count,
                                    spatial: s.spatial / count, temporal: s.temporal / count }).collect()
    };
    let threads = std::thread::available_parallelism().map_or(4, |p| p.get()).min(heads.max(1));
    let mut out = vec![Vec::new(); heads];
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads).map(|w| {
            let one_head = &one_head;
            scope.spawn(move || (w..heads).step_by(threads).map(|h| (h, one_head(h))).collect::<Vec<_>>())
        }).collect();
        for worker in workers { for (h, stats) in worker.join().unwrap() { out[h] = stats; } }
    });
    out
}

/// Mean over the tokens of sqrt(mean of x^2) of [tokens, dim].
pub fn token_rms(x: &[f32], dim: usize) -> f64 {
    let rows = x.len() / dim;
    x.chunks_exact(dim).map(|r| (r.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / dim as f64).sqrt()).sum::<f64>() / rows.max(1) as f64
}

/// (mean, standard deviation) of every channel of [channels, rest].
pub fn channel_stats(x: &[f32], channels: usize) -> Vec<(f64, f64)> {
    x.chunks_exact(x.len() / channels).map(|c| {
        let mean = c.iter().map(|v| *v as f64).sum::<f64>() / c.len() as f64;
        let var = c.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / c.len() as f64;
        (mean, var.sqrt())
    }).collect()
}

/// Weights of an area resampling of `from` cells onto `to` cells of the same span: (source, weight) per target.
fn area_weights(from: usize, to: usize) -> Vec<Vec<(usize, f64)>> {
    let ratio = from as f64 / to as f64;
    (0..to).map(|i| {
        let (lo, hi) = (i as f64 * ratio, (i + 1) as f64 * ratio);
        (lo.floor() as usize..(hi.ceil() as usize).min(from)).filter_map(|s| {
            let w = (hi.min(s as f64 + 1.0) - lo.max(s as f64)) / ratio;
            (w > 1e-12).then_some((s, w))
        }).collect()
    }).collect()
}

/// Planes [planes, h, w] resized by area (each target cell the mean of the source over its span) to [planes, h2, w2].
pub fn area_resize(x: &[f32], planes: usize, h: usize, w: usize, h2: usize, w2: usize) -> Vec<f32> {
    assert_eq!(x.len(), planes * h * w);
    let (rows, cols) = (area_weights(h, h2), area_weights(w, w2));
    let mut out = Vec::with_capacity(planes * h2 * w2);
    let mut across = vec![0f64; h * w2];
    for p in x.chunks_exact(h * w) {
        for y in 0..h {
            for (c, weights) in cols.iter().enumerate() {
                across[y * w2 + c] = weights.iter().map(|&(s, wt)| p[y * w + s] as f64 * wt).sum();
            }
        }
        for weights in &rows {
            for c in 0..w2 { out.push(weights.iter().map(|&(s, wt)| across[s * w2 + c] * wt).sum::<f64>() as f32); }
        }
    }
    out
}

/// Noise [planes, h, w] of unit variance resized by area to [planes, h2, w2], each target value divided by the norm of
/// its weights, so that it is again of unit variance: the native size's noise field at another size
/// (generate.descent.noise "native_resized"). The same size gives the noise back as it is.
pub fn resize_noise(x: &[f32], planes: usize, h: usize, w: usize, h2: usize, w2: usize) -> Vec<f32> {
    if (h, w) == (h2, w2) { return x.to_vec(); }
    let (rows, cols) = (area_weights(h, h2), area_weights(w, w2));
    let norm = |weights: &[(usize, f64)]| weights.iter().map(|(_, wt)| wt * wt).sum::<f64>().sqrt();
    let (row_norm, col_norm): (Vec<f64>, Vec<f64>) = (rows.iter().map(|r| norm(r)).collect(), cols.iter().map(|c| norm(c)).collect());
    let resized = area_resize(x, planes, h, w, h2, w2);
    resized.chunks_exact(h2 * w2).flat_map(|p| p.iter().enumerate().map(|(i, v)| (*v as f64 / (row_norm[i / w2] * col_norm[i % w2])) as f32)
        .collect::<Vec<_>>()).collect()
}

/// Source frequency index of target bin `j` of `n2` bins taken from `n`: 0, 1, .., then the negative ones.
fn bin(j: usize, n2: usize, n: usize) -> usize {
    if j < n2.div_ceil(2) { j } else { n - (n2 - j) }
}

/// White noise [planes, h, w] resized to [planes, h2, w2] by cropping its spectrum: the 2D DFT of each plane, its
/// h2 x w2 lowest frequencies, the inverse DFT, the real part, each plane scaled to unit variance. The low frequencies
/// (what the composition grows from) are the native field's own, and the result is white again, since the DFT
/// coefficients of white noise are independent (generate.descent.noise "native_spectral"). The same size gives the
/// noise back as it is.
pub fn spectral_noise(x: &[f32], planes: usize, h: usize, w: usize, h2: usize, w2: usize) -> Vec<f32> {
    use std::f64::consts::PI;
    if (h, w) == (h2, w2) { return x.to_vec(); }
    let twiddle = |n: usize, sign: f64| -> Vec<(f64, f64)> { (0..n).map(|k| ((2.0 * PI * k as f64 / n as f64).cos(), sign * (2.0 * PI * k as f64 / n as f64).sin())).collect() };
    let (tw, th) = (twiddle(w, -1.0), twiddle(h, -1.0));
    let (iw, ih) = (twiddle(w2, 1.0), twiddle(h2, 1.0));
    let mut out = Vec::with_capacity(planes * h2 * w2);
    for p in x.chunks_exact(h * w) {
        // along the width: the kept columns' bins of every row
        let mut a = vec![(0f64, 0f64); h * w2];
        for r in 0..h {
            for l in 0..w2 {
                let k = bin(l, w2, w);
                let (mut re, mut im) = (0.0, 0.0);
                for c in 0..w { let (co, si) = tw[(k * c) % w]; let v = p[r * w + c] as f64; re += v * co; im += v * si; }
                a[r * w2 + l] = (re, im);
            }
        }
        // along the height: the kept rows' bins
        let mut spectrum = vec![(0f64, 0f64); h2 * w2];
        for j in 0..h2 {
            let k = bin(j, h2, h);
            for l in 0..w2 {
                let (mut re, mut im) = (0.0, 0.0);
                for r in 0..h { let (co, si) = th[(k * r) % h]; let (ar, ai) = a[r * w2 + l]; re += ar * co - ai * si; im += ar * si + ai * co; }
                spectrum[j * w2 + l] = (re, im);
            }
        }
        // the inverse at the new size, rows then columns, the real part
        let mut b = vec![(0f64, 0f64); h2 * w2];
        for y in 0..h2 {
            for l in 0..w2 {
                let (mut re, mut im) = (0.0, 0.0);
                for j in 0..h2 { let (co, si) = ih[(j * y) % h2]; let (sr, s_i) = spectrum[j * w2 + l]; re += sr * co - s_i * si; im += sr * si + s_i * co; }
                b[y * w2 + l] = (re, im);
            }
        }
        let mut plane: Vec<f64> = (0..h2 * w2).map(|i| {
            let (y, c) = (i / w2, i % w2);
            (0..w2).map(|l| { let (co, si) = iw[(l * c) % w2]; let (br, bi) = b[y * w2 + l]; br * co - bi * si }).sum::<f64>()
        }).collect();
        let mean = plane.iter().sum::<f64>() / plane.len() as f64;
        let std = (plane.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / plane.len() as f64).sqrt().max(1e-12);
        // one scale for the whole plane: unit variance, the native field's mean (its DC bin) scaled with the rest
        for v in &mut plane { *v /= std; }
        out.extend(plane.iter().map(|v| *v as f32));
    }
    out
}

/// Cosine and relative L2 of `x` to `reference`.
pub fn closeness(x: &[f32], reference: &[f32]) -> (f64, f64) {
    let (mut xy, mut xx, mut yy, mut dd) = (0f64, 0f64, 0f64, 0f64);
    for (a, b) in x.iter().zip(reference) {
        let (a, b) = (*a as f64, *b as f64);
        xy += a * b; xx += a * a; yy += b * b; dd += (a - b).powi(2);
    }
    (xy / (xx.sqrt() * yy.sqrt()).max(1e-30), (dd / yy.max(1e-30)).sqrt())
}

/// PSNR of uint8 pixels to a reference of the same size, dB.
pub fn psnr(x: &[f32], reference: &[f32]) -> f64 {
    let mse = x.iter().zip(reference).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>() / x.len().max(1) as f64;
    10.0 * (255.0f64.powi(2) / mse.max(1e-12)).log10()
}

/// The multiplier m of `multipliers` where the curve `values` (a value per multiplier) meets `target`, linear between
/// neighbours; the nearest end when the target is outside the curve.
pub fn fit(multipliers: &[f64], values: &[f64], target: f64) -> f64 {
    let mut best = (f64::INFINITY, 1.0);
    for i in 0..multipliers.len() - 1 {
        let (a, b) = (values[i], values[i + 1]);
        if (a - target) * (b - target) <= 0.0 && a != b {
            return multipliers[i] + (target - a) / (b - a) * (multipliers[i + 1] - multipliers[i]);
        }
    }
    for (m, v) in multipliers.iter().zip(values) {
        if (v - target).abs() < best.0 { best = ((v - target).abs(), *m); }
    }
    best.1
}

/// One block of one pass of the sampler.
#[derive(Clone, Debug)]
pub struct Record {
    pub pass: usize,
    pub block: usize,
    /// [head][multiplier]
    pub heads: Vec<Vec<Stat>>,
    /// RMS of the tokens after the block
    pub rms: f64,
}

/// What a run measured, gathered by the video and written by the offline run.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub records: Vec<Record>,
    /// [C, F, H, W] float32 of the first chunk
    pub latent: Vec<f32>,
    pub latent_shape: [usize; 4],
    pub grid: Option<Grid>,
}

fn stat_json(s: &Stat) -> Value {
    json!({"entropy": s.entropy, "relative": s.relative, "spatial": s.spatial, "temporal": s.temporal})
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let (s, n) = values.fold((0.0, 0usize), |(s, n), v| (s + v, n + 1));
    s / n.max(1) as f64
}

impl Report {
    /// The relative entropy at multiplier 1 of [block][head], the mean over the passes.
    pub fn relative(&self, blocks: usize, heads: usize) -> Vec<Vec<f64>> {
        let at = MULTIPLIERS.iter().position(|m| *m == 1.0).unwrap();
        self.curve(blocks, heads, |stats| stats[at].relative)
    }

    fn curve(&self, blocks: usize, heads: usize, pick: impl Fn(&[Stat]) -> f64) -> Vec<Vec<f64>> {
        (0..blocks).map(|b| (0..heads).map(|h| mean(self.records.iter().filter(|r| r.block == b && !r.heads.is_empty())
            .map(|r| pick(&r.heads[h])))).collect()).collect()
    }

    /// The JSON of descent.json: per pass and block the RMS and per head the stats at every multiplier, summaries,
    /// the latent's channels; with `reference` (its descent.json, its latent) the comparisons and the fit of lambda
    /// from the `lambda` this run had.
    pub fn json(&self, lambda: &[Vec<f64>], reference: Option<(&Value, &[f32], [usize; 4])>) -> Value {
        let blocks = lambda.len();
        let heads = lambda.first().map_or(0, Vec::len);
        let at = MULTIPLIERS.iter().position(|m| *m == 1.0).unwrap();
        let records: Vec<Value> = self.records.iter().map(|r| json!({
            "pass": r.pass, "block": r.block, "rms": r.rms,
            "heads": r.heads.iter().map(|h| h.iter().map(stat_json).collect::<Vec<_>>()).collect::<Vec<_>>()})).collect();
        let measured = self.records.iter().any(|r| !r.heads.is_empty());
        let per_block = |pick: &dyn Fn(&Stat) -> f64| -> Vec<f64> { (0..blocks).map(|b| mean(self.records.iter()
            .filter(|r| r.block == b && !r.heads.is_empty()).flat_map(|r| r.heads.iter().map(|h| pick(&h[at]))))).collect() };
        let rms: Vec<f64> = (0..blocks).map(|b| mean(self.records.iter().filter(|r| r.block == b).map(|r| r.rms))).collect();
        let channels = self.latent_shape[0];
        let stats = if self.latent.is_empty() { vec![] } else { channel_stats(&self.latent, channels) };
        let mut out = json!({
            "multipliers": MULTIPLIERS, "queries": QUERIES, "grid": self.grid.map(|g| [g.frames, g.height, g.width]),
            "attn_logit_scale": lambda, "records": records,
            "summary": {
                "relative_entropy": if measured { json!(mean(per_block(&|s| s.relative).into_iter())) } else { Value::Null },
                "entropy": if measured { json!(mean(per_block(&|s| s.entropy).into_iter())) } else { Value::Null },
                "spatial": if measured { json!(mean(per_block(&|s| s.spatial).into_iter())) } else { Value::Null },
                "temporal": if measured { json!(mean(per_block(&|s| s.temporal).into_iter())) } else { Value::Null },
                "relative_entropy_per_block": if measured { json!(per_block(&|s| s.relative)) } else { Value::Null },
                "spatial_per_block": if measured { json!(per_block(&|s| s.spatial)) } else { Value::Null },
                "rms_per_block": rms,
                "latent_mean": stats.iter().map(|s| s.0).collect::<Vec<_>>(),
                "latent_std": stats.iter().map(|s| s.1).collect::<Vec<_>>(),
            },
            "latent_shape": self.latent_shape,
        });
        let Some((reference, latent, shape)) = reference else { return out };
        let mut compared = serde_json::Map::new();
        if !latent.is_empty() && !self.latent.is_empty() && shape[0] == self.latent_shape[0] && shape[1] == self.latent_shape[1] {
            let [c, f, h, w] = self.latent_shape;
            let resized = area_resize(latent, shape[0] * shape[1], shape[2], shape[3], h, w);
            let (cosine, relative_l2) = closeness(&self.latent, &resized);
            let theirs = channel_stats(&resized, c);
            compared.insert("latent_cosine".into(), json!(cosine));
            compared.insert("latent_relative_l2".into(), json!(relative_l2));
            compared.insert("latent_mean_gap".into(), json!(mean(stats.iter().zip(&theirs).map(|(a, b)| (a.0 - b.0).abs()))));
            compared.insert("latent_std_ratio".into(), json!(mean(stats.iter().zip(&theirs).map(|(a, b)| a.1 / b.1.max(1e-12)))));
            let _ = f;
        }
        let theirs_rms = reference["summary"]["rms_per_block"].as_array().map(|a| a.iter().filter_map(Value::as_f64).collect::<Vec<_>>());
        if let Some(theirs) = theirs_rms.filter(|t| t.len() == blocks) {
            compared.insert("rms_log_ratio".into(), json!(mean(rms.iter().zip(&theirs).map(|(a, b)| (a / b).ln().abs()))));
        }
        // the reference's relative entropy of [block][head] at its multiplier 1, the mean over its passes
        let target: Option<Vec<Vec<f64>>> = (|| {
            let records = reference["records"].as_array()?;
            Some((0..blocks).map(|b| (0..heads).map(|h| mean(records.iter()
                .filter(|r| r["block"].as_u64() == Some(b as u64) && r["heads"].as_array().is_some_and(|a| !a.is_empty()))
                .filter_map(|r| r["heads"][h][at]["relative"].as_f64()))).collect()).collect())
        })();
        if let (true, Some(target)) = (measured, target.filter(|t| t.iter().flatten().all(|v| v.is_finite()))) {
            let mine = self.relative(blocks, heads);
            let curves: Vec<Vec<Vec<f64>>> = (0..blocks).map(|b| (0..heads).map(|h| (0..MULTIPLIERS.len()).map(|m| mean(self.records.iter()
                .filter(|r| r.block == b && !r.heads.is_empty()).map(|r| r.heads[h][m].relative))).collect()).collect()).collect();
            let gap = mean(mine.iter().flatten().zip(target.iter().flatten()).map(|(a, b)| (a - b).abs()));
            let signed = mean(mine.iter().flatten().zip(target.iter().flatten()).map(|(a, b)| a - b));
            let heads_fit: Vec<Vec<f64>> = (0..blocks).map(|b| (0..heads).map(|h| lambda[b][h] * fit(&MULTIPLIERS, &curves[b][h], target[b][h])).collect()).collect();
            let block_fit: Vec<f64> = (0..blocks).map(|b| {
                let curve: Vec<f64> = (0..MULTIPLIERS.len()).map(|m| mean((0..heads).map(|h| curves[b][h][m]))).collect();
                mean(lambda[b].iter().cloned()) * fit(&MULTIPLIERS, &curve, mean(target[b].iter().cloned()))
            }).collect();
            let all: Vec<f64> = (0..MULTIPLIERS.len()).map(|m| mean(curves.iter().flatten().map(|c| c[m]))).collect();
            let global = mean(lambda.iter().flatten().cloned()) * fit(&MULTIPLIERS, &all, mean(target.iter().flatten().cloned()));
            compared.insert("relative_entropy_gap".into(), json!(gap));
            compared.insert("relative_entropy_signed_gap".into(), json!(signed));
            compared.insert("fit".into(), json!({"global": global, "blocks": block_fit, "heads": heads_fit}));
            if let Some(theirs) = reference["summary"]["spatial"].as_f64() {
                let mine = mean(self.records.iter().filter(|r| !r.heads.is_empty()).flat_map(|r| r.heads.iter().map(|h| h[at].spatial)));
                compared.insert("spatial_ratio".into(), json!(mine / theirs));
            }
        }
        out["reference"] = Value::Object(compared);
        out
    }
}

/// The latent of a run folder: latent.f32 and its shape in descent.json.
pub fn read_reference(dir: &Path) -> Result<(Value, Vec<f32>, [usize; 4]), String> {
    let text = std::fs::read_to_string(dir.join("descent.json")).map_err(|e| format!("{}: {e}", dir.join("descent.json").display()))?;
    let value: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let shape: Vec<usize> = value["latent_shape"].as_array().ok_or("descent.json has no latent_shape")?.iter()
        .filter_map(|v| v.as_u64().map(|v| v as usize)).collect();
    let shape: [usize; 4] = shape.try_into().map_err(|_| "latent_shape is not four sizes")?;
    let bytes = std::fs::read(dir.join("latent.f32")).map_err(|e| e.to_string())?;
    let latent: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    if latent.len() != shape.iter().product::<usize>() { return Err("latent.f32 does not match latent_shape".into()); }
    Ok((value, latent, shape))
}

/// The frames of `video` as RGB [frames, h, w, 3] float, decoded by ffmpeg at their own size `size` (h, w).
pub fn read_frames(video: &Path, size: (usize, usize)) -> Result<Vec<Vec<u8>>, String> {
    let out = std::process::Command::new("ffmpeg").args(["-loglevel", "error", "-i"]).arg(video)
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]).output().map_err(|e| e.to_string())?;
    if !out.status.success() { return Err(format!("ffmpeg could not read {}", video.display())); }
    let frame = size.0 * size.1 * 3;
    Ok(out.stdout.chunks_exact(frame).map(<[u8]>::to_vec).collect())
}

/// PSNR of `frames` (h x w RGB) to `reference` frames (rh x rw RGB) resized by area to h x w, and of both resized to
/// the coarse 88 x 160 (the structure without the finest detail, one size for every resolution of the descent); the
/// means over the frames both have.
pub fn compare_frames(frames: &[Vec<u8>], (h, w): (usize, usize), reference: &[Vec<u8>], (rh, rw): (usize, usize)) -> Value {
    let planes = |f: &[u8], h: usize, w: usize| -> Vec<f32> {
        (0..3).flat_map(|c| (0..h * w).map(move |i| f[i * 3 + c] as f32)).collect()
    };
    let (mut full, mut coarse) = (Vec::new(), Vec::new());
    for (mine, theirs) in frames.iter().zip(reference) {
        let (a, b) = (planes(mine, h, w), planes(theirs, rh, rw));
        full.push(psnr(&a, &area_resize(&b, 3, rh, rw, h, w)));
        coarse.push(psnr(&area_resize(&a, 3, h, w, 88, 160), &area_resize(&b, 3, rh, rw, 88, 160)));
    }
    json!({"frames": full.len(), "psnr": mean(full.into_iter()), "psnr_coarse": mean(coarse.into_iter())})
}

pub const TABLE_HEADER: &str = "run\tsize\trope\tlambda\tshift\trelative_entropy\tspatial\ttemporal\tentropy_gap\tentropy_signed_gap\tspatial_ratio\trms_log_ratio\tlatent_cosine\tlatent_relative_l2\tpsnr\tpsnr_coarse\tfit_lambda";

/// A row of the table of a descent: `run` (run.json of the run) and `descent` (its descent.json).
pub fn table_row(name: &str, run: &Value, descent: &Value) -> String {
    let (e, g) = (&run["config"][0], &run["config"][1]);
    let rope = &e["rope"];
    let profile = if rope["interpolate"] != Value::Bool(true) { "native".to_string() } else {
        let kind = if rope["ntk_aware"] == Value::Bool(true) { "ntk" } else if rope["frequency_aware"] == Value::Bool(true) { "fa" }
                   else if !rope["by_parts"].is_null() && rope.get("by_parts").is_some() { "parts" } else { "uniform" };
        format!("{kind} {}", rope["stretch"].as_f64().unwrap_or(1.0))
    };
    let numbers = |v: &Value| -> Vec<f64> { match v {
        Value::Number(n) => n.as_f64().into_iter().collect(),
        Value::Array(a) => a.iter().flat_map(|b| match b { Value::Array(h) => h.iter().filter_map(Value::as_f64).collect(), _ => b.as_f64().into_iter().collect::<Vec<_>>() }).collect(),
        _ => vec![1.0],
    }};
    let lambda = mean(numbers(&g["descent"]["attn_logit_scale"]).into_iter()) * rope["attn_logit_scale"].as_f64().unwrap_or(1.0);
    // generate.sampling.schedule; the runs written before the shift moved there keep it in their descent
    let shift = g["sampling"]["schedule"]["timestep_shift"].as_f64().or(g["descent"]["timestep_shift"].as_f64())
        .or(e["sampling"]["timestep_shift"].as_f64()).unwrap_or(f64::NAN);
    let cell = |v: &Value, digits: usize| v.as_f64().map_or("-".to_string(), |x| format!("{x:.digits$}"));
    let (s, r) = (&descent["summary"], &descent["reference"]);
    [name.to_string(), e["video"]["size"].as_str().unwrap_or("-").to_string(), profile, format!("{lambda:.4}"), format!("{shift}"),
     cell(&s["relative_entropy"], 4), cell(&s["spatial"], 4), cell(&s["temporal"], 3), cell(&r["relative_entropy_gap"], 4),
     cell(&r["relative_entropy_signed_gap"], 4), cell(&r["spatial_ratio"], 4), cell(&r["rms_log_ratio"], 4), cell(&r["latent_cosine"], 4),
     cell(&r["latent_relative_l2"], 4), cell(&r["pixels"]["psnr"], 2), cell(&r["pixels"]["psnr_coarse"], 2), cell(&r["fit"]["global"], 4)].join("\t")
}

/// `wan descent table <run folder>...`: the table of the runs, a row each, named by the folder of their configuration's
/// output (the parent of the run folder).
pub fn main(argv: &[String]) -> i32 {
    if argv.first().map(String::as_str) == Some("search") { return crate::curve::main(&argv[1..]); }
    if argv.first().map(String::as_str) != Some("table") || argv.len() < 2 {
        eprintln!("wan descent table <run folder>...");
        return 2;
    }
    println!("{TABLE_HEADER}");
    for dir in &argv[1..] {
        let dir = Path::new(dir);
        let read = |f: &str| std::fs::read_to_string(dir.join(f)).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok());
        let (Some(run), Some(descent)) = (read("run.json"), read("descent.json")) else {
            eprintln!("{}: no run.json or descent.json", dir.display());
            return 1;
        };
        let name = dir.parent().and_then(|p| p.file_name()).map_or("-".into(), |n| n.to_string_lossy().to_string());
        println!("{}", table_row(&name, &run, &descent));
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bfloat16_rounds_to_nearest_even_and_back() {
        assert_eq!(bf16_to_f32(f32_to_bf16(1.0)), 1.0);
        assert_eq!(bf16_to_f32(f32_to_bf16(-2.5)), -2.5);
        // 1 + 2^-8 is halfway between 1 and 1 + 2^-7: to the even one, 1
        assert_eq!(bf16_to_f32(f32_to_bf16(1.0 + 1.0 / 256.0)), 1.0);
        assert_eq!(bf16_to_f32(f32_to_bf16(1.0 + 3.0 / 256.0)), 1.0 + 4.0 / 256.0);
    }

    #[test]
    fn a_head_of_norm_q_is_scaled_by_its_lambda() {
        let weight: Vec<u8> = [1.0f32, 2.0, 0.5, -1.0].iter().flat_map(|v| f32_to_bf16(*v).to_le_bytes()).collect();
        let scaled = bf16_bytes_to_f32(&scaled_norm(&weight, &[2.0, 0.5]));
        assert_eq!(scaled, vec![2.0, 4.0, 0.25, -0.5]);
    }

    /// the coefficients the packages carry (export_live.causal_unipc probes UniPC itself) for shift 5 and 3.63
    #[test]
    fn the_sampler_of_a_shift_is_the_one_export_probes() {
        let (t, c) = unipc(5.0, 2);
        assert_eq!(t, vec![999.0, 833.0]);
        assert!((c[0] as f64 - 0.16674446821107836).abs() < 1e-6 && (c[1] as f64 - 0.8330553770065308).abs() < 1e-7, "{c:?}");
        let (t, c) = unipc(3.63, 2);
        assert_eq!(t, vec![999.0, 783.0]);
        assert!((c[0] as f64 - 0.21604591680195817).abs() < 1e-6 && (c[1] as f64 - 0.783678412437439).abs() < 1e-7, "{c:?}");
    }

    #[test]
    fn lambda_is_one_number_a_number_per_block_or_per_head() {
        assert_eq!(lambda_table(&json!(0.9), 2, 3).unwrap(), vec![vec![0.9; 3]; 2]);
        assert_eq!(lambda_table(&json!([1.0, 0.8]), 2, 2).unwrap(), vec![vec![1.0, 1.0], vec![0.8, 0.8]]);
        assert_eq!(lambda_table(&json!([[1.0, 0.5], 0.8]), 2, 2).unwrap(), vec![vec![1.0, 0.5], vec![0.8, 0.8]]);
        for bad in [json!(0.0), json!(-1.0), json!(5.0), json!([1.0]), json!([[1.0], 1.0]), json!("x")] {
            assert!(lambda_table(&bad, 2, 2).is_err(), "{bad}");
        }
        assert!(check_scale(&json!([[1.0, 0.5], 0.8])).is_ok() && check_scale(&json!([[1.0, 0.5], [0.8]])).is_err());
        assert!(check_scale(&json!([1.0, 9.0])).is_err());
    }

    #[test]
    fn queries_are_distinct_sorted_and_fixed_by_the_seed() {
        let a = sample_queries(1000, 256, 3);
        assert_eq!(a.len(), 256);
        assert!(a.windows(2).all(|w| w[0] < w[1]) && *a.last().unwrap() < 1000);
        assert_eq!(a, sample_queries(1000, 256, 3));
        assert_ne!(a, sample_queries(1000, 256, 4));
        assert_eq!(sample_queries(5, 256, 0), vec![0, 1, 2, 3, 4]);
    }

    /// keys all alike: uniform attention, entropy ln N, relative 0, whatever the multiplier; a key far above the
    /// others: entropy near 0 and the distance that key's
    #[test]
    fn entropy_and_distance_of_uniform_and_peaked_attention() {
        let grid = Grid { frames: 2, height: 2, width: 4 };
        let (heads, d, n) = (2, 8, 16);
        let q = vec![1.0f32; n * heads * d];
        let k = vec![0.5f32; n * heads * d];
        let stats = attention(&q, &k, heads, d, grid, &[0, 5], &[0.5, 1.0]);
        for s in stats.iter().flatten() {
            assert!((s.entropy - (n as f64).ln()).abs() < 1e-9 && s.relative.abs() < 1e-9, "{s:?}");
        }
        // head 1: key 15 (frame 1, row 1, column 3) far above the rest
        let mut k = vec![0f32; n * heads * d];
        for x in &mut k[(15 * heads + 1) * d..(15 * heads + 2) * d] { *x = 100.0; }
        let stats = attention(&q, &k, heads, d, grid, &[0], &[1.0]);
        let peaked = stats[1][0];
        assert!(peaked.entropy < 1e-6, "{peaked:?}");
        let expected = ((1.0f64 / 2.0).powi(2) + (3.0f64 / 4.0).powi(2)).sqrt();
        assert!((peaked.spatial - expected).abs() < 1e-6 && (peaked.temporal - 1.0).abs() < 1e-6, "{peaked:?}");
        // a larger multiplier sharpens: lower entropy
        let mut k = vec![0f32; n * heads * d];
        for (j, row) in k.chunks_exact_mut(d).enumerate() { row[0] = j as f32 * 0.3; }
        let stats = attention(&q, &k, heads, d, grid, &[3], &[0.8, 1.2]);
        assert!(stats[0][1].entropy < stats[0][0].entropy);
    }

    #[test]
    fn rms_and_channel_statistics() {
        assert!((token_rms(&[3.0, 4.0, 0.0, 0.0], 2) - (12.5f64.sqrt() / 2.0)).abs() < 1e-12);
        let s = channel_stats(&[1.0, 3.0, 5.0, 5.0], 2);
        assert_eq!(s, vec![(2.0, 1.0), (5.0, 0.0)]);
    }

    #[test]
    fn area_resize_keeps_means_and_constants() {
        let x: Vec<f32> = (0..6 * 4).map(|v| v as f32).collect();
        let y = area_resize(&x, 1, 6, 4, 3, 2);
        assert_eq!(y.len(), 6);
        // the 2x2 means
        assert_eq!(y[0], (0.0 + 1.0 + 4.0 + 5.0) / 4.0);
        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        let z = area_resize(&x, 1, 6, 4, 4, 3); // fractional spans
        assert!((mean(&z) - mean(&x)).abs() < 1e-4);
        assert!(area_resize(&[7.0; 44 * 80], 1, 44, 80, 32, 58).iter().all(|v| (v - 7.0).abs() < 1e-4));
        assert_eq!(area_resize(&x, 1, 6, 4, 6, 4), x);
    }

    #[test]
    fn closeness_and_psnr() {
        let (c, r) = closeness(&[1.0, 0.0], &[1.0, 0.0]);
        assert!((c - 1.0).abs() < 1e-12 && r.abs() < 1e-12);
        let (c, r) = closeness(&[0.0, 1.0], &[1.0, 0.0]);
        assert!(c.abs() < 1e-12 && (r - 2f64.sqrt()).abs() < 1e-12);
        assert!((psnr(&[0.0, 255.0], &[255.0, 0.0]) - 0.0).abs() < 1e-9);
        assert!((psnr(&[10.0], &[11.0]) - 20.0 * 255f64.log10()).abs() < 1e-9);
    }

    #[test]
    fn the_fit_meets_the_target_between_neighbours_or_at_an_end() {
        let m = [0.8, 1.0, 1.2];
        let values = [-1.0, -2.0, -3.0];
        assert!((fit(&m, &values, -1.5) - 0.9).abs() < 1e-12);
        assert_eq!(fit(&m, &values, -2.0), 1.0);
        assert_eq!(fit(&m, &values, 0.0), 0.8);
        assert_eq!(fit(&m, &values, -9.0), 1.2);
    }

    /// a run against itself as the reference: gaps zero, the fit gives back its own lambda, the latent the same
    #[test]
    fn a_run_against_itself_fits_its_own_lambda() {
        let grid = Grid { frames: 1, height: 2, width: 2 };
        let (heads, d) = (2, 8);
        let q: Vec<f32> = (0..4 * heads * d).map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.3).collect();
        let k: Vec<f32> = (0..4 * heads * d).map(|i| ((i * 5 % 13) as f32 - 6.0) * 0.3).collect();
        let stats = attention(&q, &k, heads, d, grid, &[0, 1, 2, 3], &MULTIPLIERS);
        let report = Report {
            records: (0..2).map(|b| Record { pass: 0, block: b, heads: stats.clone(), rms: 1.0 + b as f64 }).collect(),
            latent: (0..2 * 4).map(|v| v as f32).collect(), latent_shape: [2, 1, 2, 2], grid: Some(grid),
        };
        let lambda = vec![vec![0.9, 1.1]; 2];
        let reference = report.json(&lambda, None);
        let again = report.json(&lambda, Some((&reference, &report.latent, report.latent_shape)));
        let r = &again["reference"];
        assert!(r["relative_entropy_gap"].as_f64().unwrap().abs() < 1e-12, "{r}");
        assert!((r["latent_cosine"].as_f64().unwrap() - 1.0).abs() < 1e-9);
        assert!(r["rms_log_ratio"].as_f64().unwrap().abs() < 1e-12);
        for (b, row) in r["fit"]["heads"].as_array().unwrap().iter().enumerate() {
            for (h, v) in row.as_array().unwrap().iter().enumerate() {
                assert!((v.as_f64().unwrap() - lambda[b][h]).abs() < 1e-9, "{r}");
            }
        }
        assert!((r["fit"]["global"].as_f64().unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn norm_q_of_self_attention_is_found_in_the_layout_of_the_pack() {
        let layout = json!({"layout": [
            {"name": "modulation", "dtype": "bfloat16", "shape": [1, 6, 4], "offset": 0, "bytes": 48},
            {"name": "self_attn.norm_q.weight", "dtype": "bfloat16", "shape": [4], "offset": 48, "bytes": 8},
            {"name": "cross_attn.norm_q.weight", "dtype": "bfloat16", "shape": [4], "offset": 56, "bytes": 8}]});
        assert_eq!(norm_q_slice(&layout).unwrap(), (48, 8));
        let mut f32_norm = layout.clone();
        f32_norm["layout"][1]["dtype"] = json!("float32");
        assert!(norm_q_slice(&f32_norm).is_err());
        assert!(norm_q_slice(&json!({"layout": []})).is_err());
    }

    /// the packages' sampler is checked to be the one of their shift before another shift replaces it
    #[test]
    fn a_shift_replaces_the_sampler_only_if_the_packages_carry_the_one_of_theirs() {
        let (t, c) = unipc(5.0, 2);
        assert_eq!(sampler(&t, &c, 5.0, None).unwrap(), (t.clone(), c.clone()));
        assert_eq!(sampler(&t, &c, 5.0, Some(3.63)).unwrap(), unipc(3.63, 2));
        assert!(sampler(&[999.0, 833.0], &[0.5, 0.5], 5.0, Some(3.63)).is_err());
        assert!(sampler(&t, &c, 3.0, Some(3.63)).is_err());
    }

    /// a row of the table of a descent: what the run was (size, RoPE, lambda, shift) and what it measured
    #[test]
    fn a_run_is_one_row_of_the_table() {
        let run = json!({"config": [{"video": {"size": "640x1152"}, "rope": {"interpolate": true, "stretch": 0.5, "ntk_aware": false},
                                     "sampling": {"timestep_shift": 5.0}},
                                    {"sampling": {"noise": {"mode": "fresh"}, "schedule": {"timestep_shift": 4.0}}, "descent": {"attn_logit_scale": [0.9, 1.1], "reference": "r"}}]});
        let descent = json!({"summary": {"relative_entropy": -3.5, "spatial": 0.26, "temporal": 1.9},
                             "reference": {"relative_entropy_gap": 0.2, "relative_entropy_signed_gap": -0.1, "latent_cosine": 0.8,
                                           "latent_relative_l2": 0.6, "rms_log_ratio": 0.05, "spatial_ratio": 1.02,
                                           "pixels": {"psnr": 20.0, "psnr_coarse": 25.0}, "fit": {"global": 0.95}}});
        let row = table_row("r640-s05", &run, &descent);
        let cells: Vec<&str> = row.split('\t').collect();
        assert_eq!(cells.len(), TABLE_HEADER.split('\t').count());
        assert_eq!(&cells[..6], &["r640-s05", "640x1152", "uniform 0.5", "1.0000", "4", "-3.5000"]);
        assert!(row.contains("25.00") && row.contains("0.9500"));
        let bare = table_row("ref", &json!({"config": [{"video": {"size": "704x1280"}, "rope": {"interpolate": false},
                                     "sampling": {"timestep_shift": 5.0}}, {}]}), &json!({"summary": {}}));
        assert!(bare.starts_with("ref\t704x1280\tnative\t1.0000\t5\t"), "{bare}");
    }

    /// the shift of a row is generate.sampling.schedule's; the runs written before it moved keep their descent's
    #[test]
    fn the_row_reads_the_shift_from_the_schedule_and_from_older_runs() {
        let descent = json!({"summary": {}});
        let export = json!({"video": {"size": "640x1152"}, "rope": {"interpolate": false}, "sampling": {"timestep_shift": 5.0}});
        let shift = |generate: Value| table_row("r", &json!({"config": [export.clone(), generate]}), &descent).split('\t').nth(4).unwrap().to_string();
        assert_eq!(shift(json!({"sampling": {"noise": {"mode": "fresh"}, "schedule": {"timestep_shift": 3.63}}})), "3.63");
        assert_eq!(shift(json!({"sampling": {"noise": {"mode": "fresh"}, "schedule": {"timestep_shift": null}}})), "5");
        assert_eq!(shift(json!({"descent": {"attn_logit_scale": 1.0, "timestep_shift": 4.0, "reference": null}})), "4");
        assert_eq!(shift(json!({})), "5");
    }

    /// the native size's noise resized: the same field at another size, every value still of unit variance
    #[test]
    fn native_noise_is_resized_by_area_keeping_unit_variance() {
        use rand::SeedableRng;
        use rand_distr::Distribution;
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let x: Vec<f32> = (0..4 * 44 * 80).map(|_| rand_distr::StandardNormal.sample(&mut rng)).collect();
        assert_eq!(resize_noise(&x, 4, 44, 80, 44, 80), x);
        let y = resize_noise(&x, 4, 44, 80, 32, 58);
        assert_eq!(y.len(), 4 * 32 * 58);
        let var = y.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / y.len() as f64;
        assert!((var - 1.0).abs() < 0.05, "{var}");
        // the same field: close to the plain area resize, up to the per-cell normalisation
        let (cosine, _) = closeness(&y, &area_resize(&x, 4, 44, 80, 32, 58));
        assert!(cosine > 0.98, "{cosine}");
    }

    /// white noise resized by area is correlated between neighbours (the model answers with colour blotches); the
    /// spectral crop keeps the native field's low frequencies and stays white
    #[test]
    fn the_spectral_crop_keeps_the_low_frequencies_and_stays_white() {
        use rand::SeedableRng;
        use rand_distr::Distribution;
        let mut rng = rand::rngs::StdRng::seed_from_u64(2);
        let (planes, h, w, h2, w2) = (6, 44, 80, 32, 58);
        let x: Vec<f32> = (0..planes * h * w).map(|_| rand_distr::StandardNormal.sample(&mut rng)).collect();
        assert_eq!(spectral_noise(&x, planes, h, w, h, w), x);
        let y = spectral_noise(&x, planes, h, w, h2, w2);
        assert_eq!(y.len(), planes * h2 * w2);
        let var = y.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / y.len() as f64;
        assert!((var - 1.0).abs() < 0.02, "variance {var}");
        let lag = |dy: usize, dx: usize| -> f64 {
            let mut s = 0.0;
            let mut n = 0.0;
            for p in y.chunks_exact(h2 * w2) {
                for r in 0..h2 - dy { for c in 0..w2 - dx { s += (p[r * w2 + c] * p[(r + dy) * w2 + c + dx]) as f64; n += 1.0; } }
            }
            s / n
        };
        assert!(lag(0, 1).abs() < 0.05 && lag(1, 0).abs() < 0.05, "neighbours correlated: {} {}", lag(0, 1), lag(1, 0));
        let area = area_resize(&y, planes, h2, w2, h2, w2);
        assert!(closeness(&area, &area_resize(&x, planes, h, w, h2, w2)).0 < 0.99, "a plain area resize, not white");
        // the same field: their coarse views (4 x 8) agree
        let coarse = |v: &[f32], h: usize, w: usize| area_resize(v, planes, h, w, 4, 8);
        let (cosine, _) = closeness(&coarse(&y, h2, w2), &coarse(&x, h, w));
        assert!(cosine > 0.9, "low frequencies lost: {cosine}");
        // area-resized noise is not white: what this replaces
        let a = resize_noise(&x, planes, h, w, h2, w2);
        let corr = a.chunks_exact(h2 * w2).map(|p| (0..h2).map(|r| (0..w2 - 1).map(|c| (p[r * w2 + c] * p[r * w2 + c + 1]) as f64).sum::<f64>()).sum::<f64>())
            .sum::<f64>() / (planes * h2 * (w2 - 1)) as f64;
        assert!(corr > 0.1, "{corr}");
    }

    #[test]
    fn frames_are_compared_after_the_reference_is_resized() {
        let frame = |h: usize, w: usize, v: u8| vec![v; h * w * 3];
        let same = compare_frames(&[frame(16, 32, 100)], (16, 32), &[frame(32, 64, 100)], (32, 64));
        assert!(same["psnr"].as_f64().unwrap() > 100.0 && same["psnr_coarse"].as_f64().unwrap() > 100.0);
        let off = compare_frames(&[frame(16, 32, 100)], (16, 32), &[frame(32, 64, 110)], (32, 64));
        assert!((off["psnr"].as_f64().unwrap() - 20.0 * (255.0f64 / 10.0).log10()).abs() < 1e-6);
    }
}
