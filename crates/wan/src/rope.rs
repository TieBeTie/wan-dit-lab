//! The height and width RoPE table that the block packages take as an input (export_live.py BlockStep, `hw`), computed
//! from export.rope as live_presets.rope_table computes it: a new M or profile needs no export. float32
//! [2, rows, 2 * pairs]: cos then sin, row = position along the side, the height pairs then the width pairs.

use crate::config::{Rope, RopeM, SweepPoint};

/// The table for a video of `size` (height, width in pixels), `head_dim` of attention, `rows` positions.
pub fn hw_table(rope: &Rope, size: (usize, usize), head_dim: usize, rows: usize) -> Result<Vec<f32>, String> {
    let (nh, nw) = rope.native_size.split_once('x').and_then(|(h, w)| Some((h.parse::<usize>().ok()?, w.parse::<usize>().ok()?)))
        .ok_or("rope.native_size must be HxW")?;
    let cell = 32; // a token is 2x2 latent cells of 16 pixels
    let trained = [(nh / cell) as f64, (nw / cell) as f64];
    let now = [(size.0 / cell) as f64, (size.1 / cell) as f64];
    let dim = 2 * (head_dim / 6); // of one side, as live_presets.rope_table splits the head
    let pairs = dim / 2;
    let theta: Vec<f64> = (0..pairs).map(|j| 10000f64.powf(-2.0 * j as f64 / dim as f64)).collect();
    // per side: the angle of pair j at position i is i * theta_j * factor_j (Position Interpolation: i * scale * theta_j)
    let side = |axis: usize| -> Vec<Vec<f64>> {
        let at = |scale: f64| -> Vec<Vec<f64>> { (0..rows).map(|i| theta.iter().map(|t| (i as f64 * scale) * t).collect()).collect() };
        if !rope.interpolate {
            return at(1.0);
        }
        let scale = match rope.m {
            Some(m) => [m.height, m.width][axis],
            None => (trained[axis] / now[axis]).powf(rope.stretch.unwrap_or(1.0)),
        };
        let factors: Vec<f64> = if rope.ntk_aware == Some(true) {
            let mu = rope.ntk_pi_mix.unwrap_or(0.0);
            (0..pairs).map(|j| scale.powf(mu + (1.0 - mu) * j as f64 / (pairs - 1) as f64)).collect()
        } else if rope.frequency_aware == Some(true) {
            if trained[axis] <= 2.0 * std::f64::consts::PI {
                return at(scale);
            }
            let target = pairs as f64 * (trained[axis] / (2.0 * std::f64::consts::PI)).ln() / 10000f64.ln();
            (0..pairs).map(|j| scale.powf((j as f64 / target).min(1.0))).collect()
        } else if let Some(p) = &rope.by_parts {
            theta.iter().map(|t| {
                let gamma = ((trained[axis] * t / (2.0 * std::f64::consts::PI) - p.alpha) / (p.beta - p.alpha)).clamp(0.0, 1.0);
                (1.0 - gamma) * scale + gamma
            }).collect()
        } else {
            return at(scale);
        };
        (0..rows).map(|i| theta.iter().zip(&factors).map(|(t, f)| i as f64 * t * f).collect()).collect()
    };
    let (h, w) = (side(0), side(1));
    let gain = rope.attn_logit_scale.unwrap_or(1.0).sqrt();
    let mut table = Vec::with_capacity(2 * rows * dim);
    for f in [f64::cos, f64::sin] {
        for i in 0..rows {
            for a in h[i].iter().chain(&w[i]) {
                table.push((f(*a) * gain) as f32);
            }
        }
    }
    Ok(table)
}

/// export.rope with M of height and width `m` in place of its stretch or m (a point of generate.rope_sweep).
pub fn with_m(rope: &Rope, m: RopeM) -> Rope {
    Rope { m: Some(m), stretch: None, ..rope.clone() }
}

/// export.rope at a point of generate.rope_sweep: its M per side, and its mix when it gives one.
pub fn with_point(rope: &Rope, p: SweepPoint) -> Rope {
    let mut r = with_m(rope, RopeM { height: p.height, width: p.width });
    if p.ntk_pi_mix.is_some() { r.ntk_pi_mix = p.ntk_pi_mix; }
    r
}

/// The folder of a sweep point: h1200-w1400, and -mix500 for a mix of 0.5 when the point gives one.
pub fn point_tag(p: SweepPoint) -> String {
    let base = tag(RopeM { height: p.height, width: p.width });
    match p.ntk_pi_mix { Some(mu) => format!("{base}-mix{:03}", (mu * 1000.0).round() as i64), None => base }
}

/// The folder of a sweep point: h1200-w1400 for M 1.2 of height and 1.4 of width.
pub fn tag(m: RopeM) -> String {
    format!("h{:04}-w{:04}", (m.height * 1000.0).round() as i64, (m.width * 1000.0).round() as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sweep_point_replaces_the_stretch_and_names_its_folder() {
        let base: Rope = serde_json::from_value(serde_json::json!({"interpolate": true, "native_size": "704x1280",
            "scene_offset_frames": 8, "stretch": 1.24, "by_parts": null, "ntk_aware": true})).unwrap();
        let m = RopeM { height: 1.2, width: 1.4 };
        let point = with_m(&base, m);
        assert_eq!((point.m, point.stretch, point.ntk_aware), (Some(m), None, Some(true)));
        assert_eq!(tag(m), "h1200-w1400");
        // M 1.2 per side is the stretch that gives 1.2 to the height: the height columns agree
        let s = 1.2f64.ln() / (22.0f64 / 19.0).ln();
        let by_stretch = hw_table(&Rope { stretch: Some(s), ..base.clone() }, (608, 1120), 128, 35).unwrap();
        let by_m = hw_table(&with_m(&base, RopeM { height: 1.2, width: (40.0f64 / 35.0).powf(s) }), (608, 1120), 128, 35).unwrap();
        assert!(by_stretch.iter().zip(&by_m).all(|(a, b)| (a - b).abs() <= 1e-6));
    }

    #[test]
    fn a_sweep_point_may_carry_the_ntk_pi_mix() {
        let base: Rope = serde_json::from_value(serde_json::json!({"interpolate": true, "native_size": "704x1280",
            "scene_offset_frames": 8, "stretch": 1.24, "by_parts": null, "ntk_aware": true, "ntk_pi_mix": 0.3})).unwrap();
        let point = SweepPoint { height: 1.2, width: 1.2, ntk_pi_mix: Some(0.5) };
        let rope = with_point(&base, point);
        assert_eq!((rope.m, rope.ntk_pi_mix), (Some(RopeM { height: 1.2, width: 1.2 }), Some(0.5)));
        assert_eq!(point_tag(point), "h1200-w1200-mix500");
        // without a mix of its own the point keeps the one of export.rope and the tag of M only
        let plain = SweepPoint { height: 1.2, width: 1.1, ntk_pi_mix: None };
        assert_eq!(with_point(&base, plain).ntk_pi_mix, Some(0.3));
        assert_eq!(point_tag(plain), "h1200-w1100");
    }

    /// tests/fixtures/rope_hw: the table of live_presets.rope_table for every profile (tests/test_rope_hw_fixture.py)
    #[test]
    fn the_table_is_the_one_of_python_for_every_profile() {
        let dir = crate::root().join("tests").join("fixtures").join("rope_hw");
        let cases: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("cases.json")).unwrap()).unwrap();
        for (name, case) in cases.as_object().unwrap() {
            let mut value = case["rope"].clone();
            value["native_size"] = "704x1280".into();
            value["scene_offset_frames"] = 8.into();
            let rope: Rope = serde_json::from_value(value).unwrap();
            let (h, w) = case["size"].as_str().unwrap().split_once('x').unwrap();
            let (h, w): (usize, usize) = (h.parse().unwrap(), w.parse().unwrap());
            let rows = h.max(w) / 32;
            let ours = hw_table(&rope, (h, w), 128, rows).unwrap();
            let bytes = std::fs::read(dir.join(format!("{name}.bin"))).unwrap();
            let theirs: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
            assert_eq!(ours.len(), theirs.len(), "{name}");
            let worst = ours.iter().zip(&theirs).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(worst <= 1e-6, "{name}: largest difference {worst}");
        }
    }
}
