//! The NTK curve M*(resolution): which NTK factor M of RoPE makes each resolution of the descent behave most like the
//! native 704x1280, found without gradients, every point an ordinary config run through the queue:
//!
//!     wan descent search --config configs/experiments/<...>/search.json
//!
//! Two methods over the same cached points: Bayesian optimization (a Gaussian process with noise over the log of
//! the resolution's scale and M, expected improvement) and, as the control, golden-section search on M per rung with a
//! warm start from the rung before. The loss of a point is the weighted mean over its scenes of the aligned LPIPS and
//! 1 - DINO CLS cosine to the native reference (scripts/reports/evaluate_videos.py).

use std::path::{Path, PathBuf};

use serde_json::{json, Value};

/// The stretch of export.rope that gives the NTK factor `m` at a size of `height` pixels (tokens of 32 pixels) against
/// the native 22 rows of tokens: M = (22 / h)^stretch.
pub fn stretch_for(m: f64, height: usize) -> f64 {
    m.ln() / (22.0 / (height / 32) as f64).ln()
}

/// Loss of a point: over `weights` (scene, weight), the weighted mean of lpips_aligned + 1 - dino_reference of its
/// evaluation rows (ids "<scene>__<tag>"); None if a scene has no row.
pub fn loss(rows: &[Value], weights: &[(String, f64)]) -> Option<f64> {
    let mut total = 0.0;
    for (scene, w) in weights {
        let row = rows.iter().find(|r| r["run"].as_str().is_some_and(|id| id.split("__").next() == Some(scene.as_str())))?;
        total += w * (row["lpips_aligned"].as_f64()? + 1.0 - row["dino_reference"].as_f64()?);
    }
    Some(total / weights.iter().map(|(_, w)| w).sum::<f64>())
}

/// A Gaussian process with a squared-exponential kernel of `length` per input dimension, `signal` variance and
/// `noise` variance, conditioned on the points `x` and values `y` (centred on their mean).
pub struct Gp { x: Vec<Vec<f64>>, alpha: Vec<f64>, chol: Vec<Vec<f64>>, mean: f64, length: Vec<f64>, signal: f64 }

impl Gp {
    pub fn fit(x: &[Vec<f64>], y: &[f64], length: &[f64], signal: f64, noise: f64) -> Gp {
        let n = x.len();
        let mean = y.iter().sum::<f64>() / n.max(1) as f64;
        let k = |a: &[f64], b: &[f64]| signal * (-0.5 * a.iter().zip(b).zip(length).map(|((p, q), l)| ((p - q) / l).powi(2)).sum::<f64>()).exp();
        let mut chol = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..=i {
                let mut v = k(&x[i], &x[j]) + if i == j { noise } else { 0.0 };
                for t in 0..j { v -= chol[i][t] * chol[j][t]; }
                chol[i][j] = if i == j { v.max(1e-12).sqrt() } else { v / chol[j][j] };
            }
        }
        let centred: Vec<f64> = y.iter().map(|v| v - mean).collect();
        let alpha = solve_t(&chol, &solve(&chol, &centred));
        Gp { x: x.to_vec(), alpha, chol, mean, length: length.to_vec(), signal }
    }

    fn kernel(&self, a: &[f64], b: &[f64]) -> f64 {
        self.signal * (-0.5 * a.iter().zip(b).zip(&self.length).map(|((p, q), l)| ((p - q) / l).powi(2)).sum::<f64>()).exp()
    }

    /// Posterior mean and variance at `p`.
    pub fn predict(&self, p: &[f64]) -> (f64, f64) {
        let ks: Vec<f64> = self.x.iter().map(|x| self.kernel(x, p)).collect();
        let mean = self.mean + ks.iter().zip(&self.alpha).map(|(a, b)| a * b).sum::<f64>();
        let v = solve(&self.chol, &ks);
        (mean, (self.signal - v.iter().map(|x| x * x).sum::<f64>()).max(0.0))
    }
}

/// Expected improvement below `best` of a normal of `mean` and standard deviation `sd` (minimization).
pub fn expected_improvement(mean: f64, sd: f64, best: f64) -> f64 {
    if sd < 1e-9 { return (best - mean).max(0.0); }
    let z = (best - mean) / sd;
    let pdf = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let cdf = 0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2));
    (best - mean) * cdf + sd * pdf
}

/// The error function (Abramowitz and Stegun 7.1.26, error below 1.5e-7).
fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-x * x).exp();
    if x < 0.0 { -y } else { y }
}

/// L z = b for the lower triangle L.
fn solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let mut z = vec![0.0; b.len()];
    for i in 0..b.len() { z[i] = (b[i] - (0..i).map(|t| l[i][t] * z[t]).sum::<f64>()) / l[i][i]; }
    z
}

/// Lᵀ x = z for the lower triangle L.
fn solve_t(l: &[Vec<f64>], z: &[f64]) -> Vec<f64> {
    let n = z.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() { x[i] = (z[i] - (i + 1..n).map(|t| l[t][i] * x[t]).sum::<f64>()) / l[i][i]; }
    x
}

/// What golden-section search on [a, b] needs next, replaying the values it already has (`known(m)`): the next m to
/// evaluate, or the best m found once the bracket is narrower than `tol`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Golden { Need(f64), Done(f64) }

pub fn golden(a: f64, b: f64, tol: f64, known: &dyn Fn(f64) -> Option<f64>) -> Golden {
    let g = (5f64.sqrt() - 1.0) / 2.0;
    let (mut a, mut b) = (a, b);
    // the inner points carried over (not recomputed: rounded to the cache's grid they would drift off the old ones)
    let (mut c, mut d) = (key(b - g * (b - a)), key(a + g * (b - a)));
    loop {
        let Some(fc) = known(c) else { return Golden::Need(c) };
        let Some(fd) = known(d) else { return Golden::Need(d) };
        if b - a < tol { return Golden::Done(if fc <= fd { c } else { d }); }
        if fc <= fd { b = d; d = c; c = key(b - g * (b - a)); } else { a = c; c = d; d = key(a + g * (b - a)); }
    }
}

/// The name of a point: its size and M in thousandths.
pub fn tag(size: &str, m: f64) -> String { format!("{size}-m{:04}", (key(m) * 1000.0).round() as i64) }

fn height(size: &str) -> Result<usize, String> {
    size.split('x').next().and_then(|v| v.parse().ok()).ok_or_else(|| format!("{size} is not HxW"))
}

/// The name of a point with its shift: the shift of the packages (5, or none given) leaves the name of M alone.
pub fn tag_shift(size: &str, m: f64, shift: Option<f64>) -> String {
    match shift.filter(|s| (s - 5.0).abs() > 1e-9) {
        None => tag(size, m),
        Some(s) => format!("{}-s{:03}", tag(size, m), (s * 100.0).round() as i64),
    }
}

/// A point with a shift: the config of `point_config` with generate.sampling.schedule.timestep_shift (the packages stay those of M).
pub fn point_config_shift(template: &Value, size: &str, m: f64, shift: Option<f64>, packages: &str, output: &str) -> Result<Value, String> {
    let mut c = point_config(template, size, m, packages, output)?;
    if let Some(s) = shift.filter(|s| (s - 5.0).abs() > 1e-9) {
        c[1]["sampling"]["schedule"]["timestep_shift"] = json!(s);
    }
    c[1]["output"]["directory"] = json!(format!("{output}/points/{}/videos", tag_shift(size, m, shift)));
    Ok(c)
}

/// The name of a point with its shift and, for the by-parts profile of RoPE, the suffix -parts.
pub fn tag_profile(size: &str, m: f64, shift: Option<f64>, parts: bool) -> String {
    format!("{}{}", tag_shift(size, m, shift), if parts { "-parts" } else { "" })
}

/// A point with the by-parts profile of RoPE (`by_parts`: alpha and beta) instead of NTK at the same M, its packages and
/// videos in folders of their own; without `by_parts` the NTK point of `point_config_shift`.
pub fn point_config_profile(template: &Value, size: &str, m: f64, shift: Option<f64>, by_parts: Option<&Value>, packages: &str,
                            output: &str) -> Result<Value, String> {
    let mut c = point_config_shift(template, size, m, shift, packages, output)?;
    if let Some(parts) = by_parts {
        c[0]["rope"]["by_parts"] = parts.clone();
        c[0]["rope"]["ntk_aware"] = json!(false);
        c[0]["output"] = json!(format!("{}-parts", c[0]["output"].as_str().unwrap_or("")));
        c[1]["output"]["directory"] = json!(format!("{output}/points/{}/videos", tag_profile(size, m, shift, true)));
    }
    Ok(c)
}

/// The config of a point: `template` (an [export, generate] config, the start point with the search's generate
/// settings) with only the size, the RoPE (interpolated, NTK, the stretch of M), the packages' folder and the videos'
/// folder changed.
pub fn point_config(template: &Value, size: &str, m: f64, packages: &str, output: &str) -> Result<Value, String> {
    let h = height(size)?;
    let mut c = template.clone();
    let e = c.get_mut(0).filter(|e| e["component"] == "export").ok_or("the template is not [export, generate]")?;
    let offset = e["rope"]["scene_offset_frames"].clone();
    e["video"]["size"] = json!(size);
    e["rope"] = json!({"interpolate": true, "native_size": "704x1280", "scene_offset_frames": offset,
                       "stretch": stretch_for(m, h), "by_parts": null, "ntk_aware": true});
    e["output"] = json!(format!("{packages}/{size}-ntk-curve-m{:04}", (key(m) * 1000.0).round() as i64));
    c[1]["output"]["directory"] = json!(format!("{output}/points/{}/videos", tag(size, m)));
    Ok(c)
}

pub type Eval<'a> = dyn FnMut(&str, f64) -> Result<f64, String> + 'a;

/// Golden-section search on M rung by rung, each bracket `half` around the best M of the rung before (`start` for the
/// first), clipped to `range`; at most `budget` evaluations in all. The best M of every rung searched.
pub fn golden_rungs(sizes: &[String], start: f64, half: f64, tol: f64, range: (f64, f64), budget: usize, eval: &mut Eval)
    -> Result<Vec<(String, f64)>, String> {
    let (mut best, mut center, mut used) = (vec![], start, 0);
    for size in sizes {
        let (a, b) = ((center - half).max(range.0), (center + half).min(range.1));
        let mut cache: Vec<(f64, f64)> = vec![];
        let found = loop {
            let known = |m: f64| cache.iter().find(|(k, _)| *k == key(m)).map(|(_, v)| *v);
            match golden(a, b, tol, &known) {
                Golden::Done(m) => break Some(m),
                Golden::Need(_) if used >= budget => break cache.iter().min_by(|x, y| x.1.total_cmp(&y.1)).map(|x| x.0),
                Golden::Need(m) => { let v = eval(size, m)?; used += 1; cache.push((key(m), v)); }
            }
        };
        let Some(m) = found else { break };
        best.push((size.clone(), m));
        center = m;
        if used >= budget { break; }
    }
    Ok(best)
}

/// Bayesian optimization over (log of the rung's scale, M): one point per rung on the line M = (22/h)^`initial`, then
/// the point of the grid (sizes x M from `range` by `step`) of the largest expected improvement under a Gaussian
/// process with `noise_sd` of noise on the loss, until `budget` evaluations; then per rung the M of the lowest
/// posterior mean.
#[allow(clippy::too_many_arguments)]
pub fn bayes(sizes: &[String], range: (f64, f64), step: f64, initial: f64, length: [f64; 2], noise_sd: f64, budget: usize,
             eval: &mut Eval) -> Result<Vec<(String, f64)>, String> {
    let scale = |size: &str| -> Result<f64, String> { Ok((22.0 / (height(size)? / 32) as f64).ln() / (22.0f64 / 18.0).ln()) };
    let x_of = |size: &str, m: f64| -> Result<Vec<f64>, String> { Ok(vec![scale(size)?, (m - 1.0) / 0.3]) };
    let ms: Vec<f64> = (0..).map(|i| key(range.0 + i as f64 * step)).take_while(|m| *m <= range.1 + 1e-9).collect();
    let mut obs: Vec<(String, f64, f64)> = vec![];
    for size in sizes {
        if obs.len() >= budget { break; }
        let h = height(size)? / 32;
        let m = key((22.0 / h as f64).powf(initial).clamp(range.0, range.1));
        obs.push((size.clone(), m, eval(size, m)?));
    }
    let fit = |obs: &[(String, f64, f64)]| -> Result<(Gp, f64, f64), String> {
        let y: Vec<f64> = obs.iter().map(|o| o.2).collect();
        let mu = y.iter().sum::<f64>() / y.len() as f64;
        let sd = (y.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / y.len() as f64).sqrt().max(1e-6);
        let x: Vec<Vec<f64>> = obs.iter().map(|o| x_of(&o.0, o.1)).collect::<Result<_, _>>()?;
        let ys: Vec<f64> = y.iter().map(|v| (v - mu) / sd).collect();
        Ok((Gp::fit(&x, &ys, &length, 1.0, (noise_sd / sd).powi(2)), mu, sd))
    };
    while obs.len() < budget {
        let (gp, mu, sd) = fit(&obs)?;
        let best = obs.iter().map(|o| (o.2 - mu) / sd).fold(f64::INFINITY, f64::min);
        let mut choice: Option<(f64, &String, f64)> = None;
        for size in sizes {
            for &m in &ms {
                if obs.iter().any(|o| &o.0 == size && o.1 == m) { continue; }
                let (mean, var) = gp.predict(&x_of(size, m)?);
                let ei = expected_improvement(mean, var.sqrt(), best);
                if choice.is_none_or(|c| ei > c.0) { choice = Some((ei, size, m)); }
            }
        }
        let Some((_, size, m)) = choice else { break };
        let size = size.clone();
        let v = eval(&size, m)?;
        obs.push((size, m, v));
    }
    let (gp, _, _) = fit(&obs)?;
    sizes.iter().map(|size| {
        let mut best = (f64::INFINITY, ms[0]);
        for &m in &ms {
            let (mean, _) = gp.predict(&x_of(size, m)?);
            if mean < best.0 { best = (mean, m); }
        }
        Ok((size.clone(), best.1))
    }).collect()
}

fn io<E: std::fmt::Display>(e: E) -> String { e.to_string() }

/// One evaluation of the search: the point's config, its run (a child wan.exe inside the search's queue ticket), its
/// metrics (evaluate_videos.py against the reference) and loss, kept in points/<tag>/loss.json (a point already there
/// is not run again) and appended to <method>.tsv.
fn evaluate_point(spec: &Value, size: &str, m: f64, shift: Option<f64>, exe: &Path, ticket: &str, method: &str, count: usize) -> Result<f64, String> {
    let root = crate::root();
    let text = |k: &str| spec[k].as_str().map(String::from).ok_or_else(|| format!("search.{k} is missing"));
    let output = PathBuf::from(text("output")?);
    let by_parts = spec.get("by_parts").filter(|v| !v.is_null());
    let t = tag_profile(size, m, shift, by_parts.is_some());
    let point = output.join("points").join(&t);
    let scenes: Vec<(String, f64)> = spec["scenes"].as_object().ok_or("search.scenes is missing")?.iter()
        .map(|(k, v)| (k.clone(), v.as_f64().unwrap_or(1.0))).collect();
    let cached = std::fs::read_to_string(point.join("loss.json")).ok().and_then(|s| serde_json::from_str::<Value>(&s).ok());
    let row = if let Some(row) = cached { row } else {
        let template: Value = serde_json::from_str(&std::fs::read_to_string(root.join(text("template")?)).map_err(io)?).map_err(io)?;
        let config = point_config_profile(&template, size, m, shift, by_parts, &text("packages")?, &output.display().to_string().replace('\\', "/"))?;
        let configs = root.join(text("configs")?);
        std::fs::create_dir_all(&configs).map_err(io)?;
        let path = configs.join(format!("{t}.json"));
        std::fs::write(&path, serde_json::to_string_pretty(&config).map_err(io)?).map_err(io)?;
        crate::config::Config::read(&path).map_err(|e| format!("{t}: {e}"))?.validate()?;
        std::fs::create_dir_all(output.join("logs")).map_err(io)?;
        let log = output.join("logs").join(format!("{t}.log"));
        let f = std::fs::File::create(&log).map_err(io)?;
        println!("[descent search] {method} #{count}: {t} (stretch {:.4})", stretch_for(m, height(size)?));
        let status = std::process::Command::new(exe).arg("--config").arg(&path).current_dir(&root)
            .env(crate::queue::HELD, ticket).env("PYTHONUTF8", "1").env("PYTHONUNBUFFERED", "1")
            .stdout(f.try_clone().map_err(io)?).stderr(f).status().map_err(io)?;
        if !status.success() { return Err(format!("{t}: the run failed ({status}), see {}", log.display())); }
        // the evaluation folder: a run per scene (hard links to the videos) against the reference run's
        let videos = newest_run(&point.join("videos")).ok_or_else(|| format!("{t}: no videos"))?;
        let eval = point.join("eval");
        evaluation_folder(&videos, &eval, &scenes.iter().map(|s| (format!("{}__{t}", s.0), s.0.clone())).collect::<Vec<_>>())?;
        let reference = output.join("reference");
        if !reference.join("experiment.json").exists() {
            evaluation_folder(&PathBuf::from(text("reference")?), &reference, &scenes.iter().map(|s| (s.0.clone(), s.0.clone())).collect::<Vec<_>>())?;
        }
        let elog = std::fs::File::create(point.join("evaluate.log")).map_err(io)?;
        let args: Vec<String> = spec["evaluate_args"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default();
        let status = std::process::Command::new(root.join(text("python")?)).arg("scripts/reports/evaluate_videos.py")
            .arg("--experiment").arg(&eval).arg("--reference").arg(&reference).args(&args).current_dir(&root)
            .env("PYTHONUTF8", "1").stdout(elog.try_clone().map_err(io)?).stderr(elog).status().map_err(io)?;
        if !status.success() { return Err(format!("{t}: the evaluation failed, see {}", point.join("evaluate.log").display())); }
        let rows: Vec<Value> = serde_json::from_str::<Value>(&std::fs::read_to_string(eval.join("evaluation.json")).map_err(io)?)
            .map_err(io)?.as_array().cloned().unwrap_or_default();
        let value = loss(&rows, &scenes).ok_or_else(|| format!("{t}: a scene has no metrics"))?;
        let metrics = std::fs::read_to_string(&log).ok().and_then(|l| l.lines().find_map(|x| x.strip_prefix("Metrics: ").map(String::from)))
            .and_then(|p| std::fs::read_to_string(p).ok()).and_then(|t| t.lines().last().and_then(|l| serde_json::from_str::<Value>(l).ok()));
        let pick = |f: &dyn Fn(&Value) -> Option<f64>| metrics.as_ref().and_then(f);
        let row = json!({"tag": t, "size": size, "m": key(m), "shift": shift.unwrap_or(5.0), "stretch": stretch_for(m, height(size)?), "loss": value,
            "fps": pick(&|m| m["fps"].as_f64()),
            "vram_peak": pick(&|m| m["memory"]["gpu_used_max_sampled_bytes"].as_f64()),
            "ram_peak": pick(&|m| m["memory"]["latest"]["process_peak_working_bytes"].as_f64()),
            "kv_vram": pick(&|m| m["memory"]["latest"]["kv_gpu_bytes"].as_f64()),
            "scenes": rows.iter().map(|r| json!({"run": r["run"], "lpips_aligned": r["lpips_aligned"], "dino_reference": r["dino_reference"],
                                                 "sharpness": r["sharpness"], "zoom": r["zoom"]})).collect::<Vec<_>>()});
        std::fs::write(point.join("loss.json"), serde_json::to_vec_pretty(&row).map_err(io)?).map_err(io)?;
        row
    };
    let value = row["loss"].as_f64().ok_or("loss.json without a loss")?;
    use std::io::Write;
    let tsv = output.join(format!("{method}.tsv"));
    let shift_cell = row["shift"].as_f64().unwrap_or(5.0);
    let new = !tsv.exists();
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&tsv).map_err(io)?;
    if new { writeln!(f, "n\ttag\tsize\tm\tshift\tstretch\tloss\tfps\tvram_peak_gib\tscenes").map_err(io)?; }
    writeln!(f, "{count}\t{}\t{size}\t{:.3}\t{shift_cell}\t{:.4}\t{value:.4}\t{}\t{}\t{}", row["tag"].as_str().unwrap_or(""), key(m), stretch_for(m, height(size)?),
             row["fps"].as_f64().map_or("-".into(), |v| format!("{v:.2}")),
             row["vram_peak"].as_f64().map_or("-".into(), |v| format!("{:.2}", v / (1u64 << 30) as f64)), row["scenes"]).map_err(io)?;
    println!("[descent search] {method} #{count}: {} loss {value:.4}", row["tag"].as_str().unwrap_or(""));
    Ok(value)
}

/// The points already evaluated in <output>/points: (size, M, shift, loss) from their loss.json (a point of before the
/// shift axis is at shift 5).
fn cached_points(output: &Path) -> Vec<(String, f64, f64, f64)> {
    let Ok(dirs) = std::fs::read_dir(output.join("points")) else { return vec![] };
    dirs.flatten().filter_map(|d| {
        let row: Value = serde_json::from_str(&std::fs::read_to_string(d.path().join("loss.json")).ok()?).ok()?;
        Some((row["size"].as_str()?.to_string(), row["m"].as_f64()?, row["shift"].as_f64().unwrap_or(5.0), row["loss"].as_f64()?))
    }).collect()
}

/// The newest run folder (one with run.json) of `dir`.
fn newest_run(dir: &Path) -> Option<PathBuf> {
    let mut runs: Vec<PathBuf> = std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path()).filter(|p| p.join("run.json").exists()).collect();
    runs.sort();
    runs.pop()
}

/// An evaluation folder of evaluate_videos.py: experiment.json and <id>/video.mp4 (hard links to `run`'s <chain>.mp4)
/// for every (id, chain) of `runs`, the prompt of each its chain's first.
fn evaluation_folder(run: &Path, folder: &Path, runs: &[(String, String)]) -> Result<(), String> {
    let chains: Value = serde_json::from_str(&std::fs::read_to_string(run.join("run.json")).map_err(io)?).map_err(io)?;
    let mut list = vec![];
    for (id, chain) in runs {
        let dir = folder.join(id);
        std::fs::create_dir_all(&dir).map_err(io)?;
        let (src, dst) = (run.join(format!("{chain}.mp4")), dir.join("video.mp4"));
        if !dst.exists() { std::fs::hard_link(&src, &dst).map_err(|e| format!("{}: {e}", src.display()))?; }
        list.push(json!({"id": id, "prompts": [chains["chains"][chain][0]["text"].clone()], "source": src.display().to_string()}));
    }
    std::fs::write(folder.join("experiment.json"), serde_json::to_string_pretty(&json!({"runs": list})).map_err(io)?).map_err(io)
}

/// `wan descent search --config <search.json>`: the search of search.json under one queue ticket; the best M per rung
/// into <output>/<method>-best.json.
pub fn main(argv: &[String]) -> i32 {
    let Some(config) = argv.iter().position(|a| a == "--config").and_then(|i| argv.get(i + 1)) else {
        eprintln!("wan descent search --config <search.json>");
        return 2;
    };
    match run(&crate::root().join(config), argv) {
        Ok(()) => 0,
        Err(e) => { eprintln!("[descent search] {e}"); 1 }
    }
}

fn run(path: &Path, argv: &[String]) -> Result<(), String> {
    let spec: Value = serde_json::from_str::<Value>(&std::fs::read_to_string(path).map_err(io)?).map_err(io)?["search"].clone();
    let sizes: Vec<String> = spec["sizes"].as_array().ok_or("search.sizes is missing")?.iter().filter_map(|v| v.as_str().map(String::from)).collect();
    let range = (spec["range"][0].as_f64().ok_or("search.range")?, spec["range"][1].as_f64().ok_or("search.range")?);
    let budget = spec["budget"].as_u64().ok_or("search.budget is missing")? as usize;
    let method = spec["method"].as_str().ok_or("search.method is missing")?.to_string();
    // the files of this search (<name>.tsv, <name>-best.json): a search may repeat a method on the same cache
    let name = spec["name"].as_str().map(String::from).unwrap_or_else(|| method.clone());
    let exe = std::env::current_exe().map_err(io)?;
    let ticket = crate::queue::enter(&[&["descent".to_string(), "search".to_string()], argv].concat(), Some(path)).map_err(io)?;
    let id = ticket.id.clone();
    let mut count = 0;
    let output = PathBuf::from(spec["output"].as_str().unwrap_or("."));
    // points every method must have (an equal cover of M on the rungs compared), evaluated first, then in the cache
    for point in spec["coverage"].as_array().cloned().unwrap_or_default() {
        let (Some(size), Some(m)) = (point[0].as_str(), point[1].as_f64()) else { return Err("search.coverage: [size, M] or [size, M, shift]".into()) };
        count += 1;
        if let Err(e) = evaluate_point(&spec, size, m, point[2].as_f64(), &exe, &id, &name, count) { ticket.finish(1); return Err(e); }
    }
    if method == "bayes_shift" {
        let b = &spec["bayes"];
        let shifts = (spec["shifts"][0].as_f64().ok_or("search.shifts")?, spec["shifts"][1].as_f64().ok_or("search.shifts")?);
        let prior = cached_points(&output);
        let mut eval3 = |size: &str, m: f64, sh: f64| -> Result<f64, String> { count += 1; evaluate_point(&spec, size, m, Some(sh), &exe, &id, &name, count) };
        let result = bayes_shift(&sizes, range, b["step"].as_f64().unwrap_or(0.005), shifts, b["shift_step"].as_f64().unwrap_or(0.05),
                                 [b["length"][0].as_f64().unwrap_or(0.5), b["length"][1].as_f64().unwrap_or(0.25), b["length"][2].as_f64().unwrap_or(0.3)],
                                 b["noise_sd"].as_f64().unwrap_or(0.02), budget, &prior, &mut eval3);
        ticket.finish(if result.is_ok() { 0 } else { 1 });
        let best = result?;
        let best_json = json!(best.iter().map(|(s, m, sh)| json!({"size": s, "m": m, "shift": sh, "stretch": stretch_for(*m, height(s).unwrap_or(704))})).collect::<Vec<_>>());
        std::fs::write(output.join(format!("{name}-best.json")), serde_json::to_vec_pretty(&best_json).map_err(io)?).map_err(io)?;
        println!("[descent search] {name}: best {best_json}");
        return Ok(());
    }
    let mut eval = |size: &str, m: f64| -> Result<f64, String> { count += 1; evaluate_point(&spec, size, m, None, &exe, &id, &name, count) };
    let result = match method.as_str() {
        "golden" => golden_rungs(&sizes, spec["golden"]["start"].as_f64().unwrap_or(1.0), spec["golden"]["half"].as_f64().unwrap_or(0.06),
                                 spec["golden"]["tol"].as_f64().unwrap_or(0.02), range, budget, &mut eval),
        "bayes" => {
            let b = &spec["bayes"];
            bayes(&sizes, range, b["step"].as_f64().unwrap_or(0.005), b["initial"].as_f64().unwrap_or(0.3),
                  [b["length"][0].as_f64().unwrap_or(0.5), b["length"][1].as_f64().unwrap_or(0.25)],
                  b["noise_sd"].as_f64().unwrap_or(0.02), budget, &mut eval)
        }
        other => Err(format!("search.method {other}: golden, bayes or bayes_shift")),
    };
    ticket.finish(if result.is_ok() { 0 } else { 1 });
    let best = result?;
    let best_json = json!(best.iter().map(|(s, m)| json!({"size": s, "m": m, "stretch": stretch_for(*m, height(s).unwrap_or(704))})).collect::<Vec<_>>());
    std::fs::write(output.join(format!("{name}-best.json")), serde_json::to_vec_pretty(&best_json).map_err(io)?).map_err(io)?;
    println!("[descent search] {name}: best {best_json}");
    Ok(())
}

pub type Eval3<'a> = dyn FnMut(&str, f64, f64) -> Result<f64, String> + 'a;

/// Bayesian optimization over (log of the rung's scale, M, shift): `prior` (size, M, shift, loss) are points already
/// known (free), one point per rung on the line M = (22/h)^`initial` at shift 5 if there are none; then `budget` new
/// points of the largest expected improvement per cost (a point whose packages are built, a known (size, M), costs a
/// third of one that needs them); then per rung the (M, shift) of the lowest posterior mean.
#[allow(clippy::too_many_arguments)]
pub fn bayes_shift(sizes: &[String], range: (f64, f64), step: f64, shifts: (f64, f64), shift_step: f64, length: [f64; 3],
                   noise_sd: f64, budget: usize, prior: &[(String, f64, f64, f64)], eval: &mut Eval3)
    -> Result<Vec<(String, f64, f64)>, String> {
    let scale = |size: &str| -> Result<f64, String> { Ok((22.0 / (height(size)? / 32) as f64).ln() / (22.0f64 / 18.0).ln()) };
    let x_of = |size: &str, m: f64, sh: f64| -> Result<Vec<f64>, String> { Ok(vec![scale(size)?, (m - 1.0) / 0.3, (sh - 5.0) / 2.5]) };
    let grid = |lo: f64, hi: f64, d: f64| -> Vec<f64> { (0..).map(|i| key(lo + i as f64 * d)).take_while(|v| *v <= hi + 1e-9).collect() };
    let (ms, shs) = (grid(range.0, range.1, step), grid(shifts.0, shifts.1, shift_step));
    let mut obs: Vec<(String, f64, f64, f64)> = prior.iter().filter(|p| sizes.contains(&p.0)).cloned().collect();
    let mut new = 0;
    if obs.is_empty() {
        for size in sizes {
            if new >= budget { break; }
            let m = key((22.0 / (height(size)? / 32) as f64).powf(0.3).clamp(range.0, range.1));
            obs.push((size.clone(), m, 5.0, eval(size, m, 5.0)?));
            new += 1;
        }
    }
    let fit = |obs: &[(String, f64, f64, f64)]| -> Result<(Gp, f64, f64), String> {
        let y: Vec<f64> = obs.iter().map(|o| o.3).collect();
        let mu = y.iter().sum::<f64>() / y.len() as f64;
        let sd = (y.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / y.len() as f64).sqrt().max(1e-6);
        let x: Vec<Vec<f64>> = obs.iter().map(|o| x_of(&o.0, o.1, o.2)).collect::<Result<_, _>>()?;
        let ys: Vec<f64> = y.iter().map(|v| (v - mu) / sd).collect();
        Ok((Gp::fit(&x, &ys, &length, 1.0, (noise_sd / sd).powi(2)), mu, sd))
    };
    while new < budget {
        let (gp, mu, sd) = fit(&obs)?;
        let best = obs.iter().map(|o| (o.3 - mu) / sd).fold(f64::INFINITY, f64::min);
        let mut choice: Option<(f64, String, f64, f64)> = None;
        for size in sizes {
            for &m in &ms {
                let built = obs.iter().any(|o| &o.0 == size && o.1 == m);
                for &sh in &shs {
                    if obs.iter().any(|o| &o.0 == size && o.1 == m && o.2 == sh) { continue; }
                    let (mean, var) = gp.predict(&x_of(size, m, sh)?);
                    let gain = expected_improvement(mean, var.sqrt(), best) / if built { 1.0 } else { 3.0 };
                    if choice.as_ref().is_none_or(|c| gain > c.0) { choice = Some((gain, size.clone(), m, sh)); }
                }
            }
        }
        let Some((_, size, m, sh)) = choice else { break };
        let v = eval(&size, m, sh)?;
        obs.push((size, m, sh, v));
        new += 1;
    }
    let (gp, _, _) = fit(&obs)?;
    sizes.iter().map(|size| {
        let mut best = (f64::INFINITY, ms[0], shs[0]);
        for &m in &ms {
            for &sh in &shs {
                let (mean, _) = gp.predict(&x_of(size, m, sh)?);
                if mean < best.0 { best = (mean, m, sh); }
            }
        }
        Ok((size.clone(), best.1, best.2))
    }).collect()
}

/// M rounded to the grid points are cached on (thousandths).
pub fn key(m: f64) -> f64 { (m * 1000.0).round() / 1000.0 }

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// a synthetic loss with its minimum on the line M = (22/h)^0.3 and a curvature in M
    fn synthetic(size: &str, m: f64) -> f64 {
        let h: usize = size.split('x').next().unwrap().parse().unwrap();
        (m - (22.0 / (h / 32) as f64).powf(0.3)).powi(2) * 10.0 + 0.2
    }
    const SIZES: [&str; 4] = ["672x1216", "640x1184", "608x1120", "576x1056"];

    /// the by-parts profile of RoPE (instead of NTK) at the same M: packages and a tag of its own
    #[test]
    fn a_point_can_take_the_by_parts_profile() {
        let template = json!([{"component": "export", "video": {"size": "704x1280"}, "rope": {"scene_offset_frames": 8}, "output": "x"},
                              {"component": "generate", "sampling": {"schedule": {"timestep_shift": null}}, "output": {"directory": "o"}}]);
        let parts = json!({"alpha": 0.25, "beta": 2.0});
        let c = point_config_profile(&template, "640x1184", 1.2, None, Some(&parts), "artifacts/live", "D:/out").unwrap();
        assert_eq!(c[0]["rope"]["by_parts"], parts);
        assert_eq!(c[0]["rope"]["ntk_aware"], false);
        assert!(((22.0f64 / 20.0).powf(c[0]["rope"]["stretch"].as_f64().unwrap()) - 1.2).abs() < 1e-9);
        assert_eq!(c[0]["output"], "artifacts/live/640x1184-ntk-curve-m1200-parts");
        assert_eq!(c[1]["output"]["directory"], "D:/out/points/640x1184-m1200-parts/videos");
        assert_eq!(tag_profile("640x1184", 1.2, None, true), "640x1184-m1200-parts");
        let ntk = point_config_profile(&template, "640x1184", 1.2, None, None, "artifacts/live", "D:/out").unwrap();
        assert_eq!(ntk, point_config_shift(&template, "640x1184", 1.2, None, "artifacts/live", "D:/out").unwrap());
    }

    /// shift 5 (the packages' own) keeps the tag and the config of M alone, so the points of M are reused
    #[test]
    fn a_shift_is_a_generate_setting_and_part_of_the_tag() {
        assert_eq!(tag_shift("640x1184", 1.029, None), "640x1184-m1029");
        assert_eq!(tag_shift("640x1184", 1.029, Some(5.0)), "640x1184-m1029");
        assert_eq!(tag_shift("640x1184", 1.029, Some(3.63)), "640x1184-m1029-s363");
        let template = json!([{"component": "export", "video": {"size": "704x1280"}, "rope": {"scene_offset_frames": 8}, "output": "x"},
                              {"component": "generate", "sampling": {"schedule": {"timestep_shift": null}}, "output": {"directory": "o"}}]);
        let plain = point_config_shift(&template, "640x1184", 1.029, None, "artifacts/live", "D:/out").unwrap();
        let shifted = point_config_shift(&template, "640x1184", 1.029, Some(3.63), "artifacts/live", "D:/out").unwrap();
        assert_eq!(plain[1]["sampling"]["schedule"]["timestep_shift"], Value::Null);
        assert_eq!(shifted[1]["sampling"]["schedule"]["timestep_shift"], json!(3.63));
        assert_eq!(plain[0], shifted[0], "the same packages");
        assert_eq!(shifted[1]["output"]["directory"], "D:/out/points/640x1184-m1029-s363/videos");
    }

    /// three axes: the minimum in M and in shift moves with the resolution; a new shift on packages already built
    /// costs less than a new M, and the search still finds both
    #[test]
    fn bayes_over_m_and_shift_finds_both_minima() {
        let sizes: Vec<String> = SIZES.iter().map(|s| s.to_string()).collect();
        let shift_star = |size: &str| { let h: f64 = size.split('x').next().unwrap().parse::<f64>().unwrap() / 32.0; 5.0 * (h / 22.0) };
        let f = |s: &str, m: f64, sh: f64| synthetic(s, m) + 0.05 * (sh - shift_star(s)).powi(2);
        let mut seen: Vec<(String, f64, f64)> = vec![];
        let best = bayes_shift(&sizes, (0.95, 1.30), 0.005, (3.0, 5.5), 0.05, [0.5, 0.25, 0.3], 0.02, 30, &[],
                               &mut |s, m, sh| { seen.push((s.to_string(), m, sh)); Ok(f(s, m, sh)) }).unwrap();
        assert!(seen.len() <= 30);
        for (size, m, sh) in &best {
            let h: usize = size.split('x').next().unwrap().parse().unwrap();
            assert!((m - (22.0 / (h / 32) as f64).powf(0.3)).abs() < 0.04, "{size} {m}");
            assert!((sh - shift_star(size)).abs() < 0.8, "{size} {sh}");
        }
    }

    #[test]
    fn golden_rungs_follow_the_minimum_down_with_a_warm_start() {
        let sizes: Vec<String> = SIZES.iter().map(|s| s.to_string()).collect();
        let mut calls = 0;
        let best = golden_rungs(&sizes, 1.0, 0.06, 0.02, (0.95, 1.30), 40, &mut |s, m| { calls += 1; Ok(synthetic(s, m)) }).unwrap();
        for (size, m) in &best {
            let h: usize = size.split('x').next().unwrap().parse().unwrap();
            assert!((m - (22.0 / (h / 32) as f64).powf(0.3)).abs() < 0.015, "{size} {m}");
        }
        assert!(calls <= 24, "{calls}");
    }

    #[test]
    fn bayes_finds_the_curve_within_its_budget() {
        let sizes: Vec<String> = SIZES.iter().map(|s| s.to_string()).collect();
        let mut seen = vec![];
        let best = bayes(&sizes, (0.95, 1.30), 0.005, 0.3, [0.5, 0.25], 0.02, 16, &mut |s, m| { seen.push((s.to_string(), m)); Ok(synthetic(s, m)) }).unwrap();
        assert!(seen.len() <= 16);
        let distinct: std::collections::BTreeSet<String> = seen.iter().map(|(s, m)| format!("{s}{m}")).collect();
        assert_eq!(distinct.len(), seen.len(), "a point evaluated twice");
        for (size, m) in &best {
            let h: usize = size.split('x').next().unwrap().parse().unwrap();
            assert!((m - (22.0 / (h / 32) as f64).powf(0.3)).abs() < 0.03, "{size} {m}");
        }
    }

    #[test]
    fn a_point_changes_only_the_size_the_rope_and_the_outputs() {
        let template = json!([{"component": "export", "video": {"size": "704x1280", "fps": 24},
                                "rope": {"interpolate": false, "native_size": "704x1280", "scene_offset_frames": 8},
                                "output": "artifacts/live/x", "kernels": {"attention": "sage_sm89"}},
                               {"component": "generate", "seed": 0, "output": {"directory": "o"}}]);
        let c = point_config(&template, "608x1120", 1.05, "artifacts/live", "D:/out").unwrap();
        assert_eq!(c[0]["video"]["size"], "608x1120");
        assert_eq!(c[0]["rope"]["ntk_aware"], true);
        assert!(((22.0f64 / 19.0).powf(c[0]["rope"]["stretch"].as_f64().unwrap()) - 1.05).abs() < 1e-9);
        assert_eq!(c[0]["output"], "artifacts/live/608x1120-ntk-curve-m1050");
        assert_eq!(c[1]["output"]["directory"], "D:/out/points/608x1120-m1050/videos");
        assert_eq!(c[0]["kernels"], template[0]["kernels"]);
        assert_eq!(tag("608x1120", 1.05), "608x1120-m1050");
    }

    #[test]
    fn the_stretch_gives_m_at_every_size() {
        assert!((stretch_for(1.10, 512) - 0.2992905106483072).abs() < 1e-12);
        assert_eq!(stretch_for(1.0, 576), 0.0);
        let s = stretch_for(1.05, 608);
        assert!(((22.0f64 / 19.0).powf(s) - 1.05).abs() < 1e-12);
    }

    #[test]
    fn the_loss_weights_the_scenes() {
        let rows = vec![json!({"run": "fox_runs__p", "lpips_aligned": 0.3, "dino_reference": 0.9}),
                        json!({"run": "teacup_pour__p", "lpips_aligned": 0.2, "dino_reference": 0.95})];
        let w = vec![("fox_runs".to_string(), 2.0), ("teacup_pour".to_string(), 1.0)];
        assert!((loss(&rows, &w).unwrap() - (2.0 * 0.4 + 0.25) / 3.0).abs() < 1e-12);
        assert!(loss(&rows[..1], &w).is_none());
    }

    /// with no noise the process goes through its points, far from them it returns to the mean with the full variance
    #[test]
    fn the_process_interpolates_and_is_unsure_far_away() {
        let x = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let y = [1.0, 2.0, 3.0];
        let gp = Gp::fit(&x, &y, &[0.5, 0.5], 1.0, 1e-9);
        for (p, v) in x.iter().zip(y) {
            let (m, var) = gp.predict(p);
            assert!((m - v).abs() < 1e-5 && var < 1e-5, "{m} {var}");
        }
        let (m, var) = gp.predict(&[20.0, 20.0]);
        assert!((m - 2.0).abs() < 1e-6 && (var - 1.0).abs() < 1e-6, "{m} {var}");
        // with noise it smooths: no longer through the points
        let noisy = Gp::fit(&x, &y, &[0.5, 0.5], 1.0, 0.5);
        assert!((noisy.predict(&x[0]).0 - 1.0).abs() > 0.05);
    }

    #[test]
    fn expected_improvement_prefers_low_means_and_uncertainty() {
        assert!(expected_improvement(0.0, 1.0, 1.0) > expected_improvement(0.5, 1.0, 1.0));
        assert!(expected_improvement(1.0, 1.0, 1.0) > expected_improvement(1.0, 0.1, 1.0));
        assert!(expected_improvement(2.0, 1e-9, 1.0).abs() < 1e-12);
        assert!((expected_improvement(0.0, 1e-12, 1.0) - 1.0).abs() < 1e-9);
    }

    /// on a parabola with its minimum at 1.07 the search asks for points one by one and ends near the minimum, within
    /// the evaluations golden section takes for the bracket
    #[test]
    fn golden_section_finds_the_minimum_of_a_parabola() {
        let f = |m: f64| (m - 1.07).powi(2);
        let mut cache: Vec<(f64, f64)> = vec![];
        let mut asked = 0;
        let result = loop {
            let known = |m: f64| cache.iter().find(|(k, _)| *k == key(m)).map(|(_, v)| *v);
            match golden(1.0, 1.12, 0.02, &known) {
                Golden::Need(m) => { cache.push((key(m), f(key(m)))); asked += 1; }
                Golden::Done(m) => break m,
            }
            assert!(asked < 20);
        };
        assert!((result - 1.07).abs() < 0.015, "{result}");
        assert!(asked <= 6, "{asked}");
    }
}
