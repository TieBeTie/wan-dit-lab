//! A grid search over settings of one experiment config, against reference videos, under one queue ticket:
//!
//!     wan search --config configs/experiments/<...>/search.json
//!
//! The search file names a base config (an export + generate pipeline), the axes (a path inside the export or
//! generate component and its values), the chains, the reference videos and the loss. Every point of the grid becomes
//! an ordinary config in `configs` (written before it runs, kept after), runs as a child `wan --config <point>` while
//! this process holds the GPU ticket (no other run slips in between), and is scored by
//! scripts/reports/evaluate_videos.py on the CPU while the next point already runs. Then a finer grid around the best
//! point, `refine.rounds` times. Points already scored are not run again (a stopped search resumes).
//! Results: <output>/search.tsv, a row per point, and <output>/best.json.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::video::Result;

/// One axis: the dotted path inside a component ("export.rope.stretch") and its values, in order.
#[derive(Clone, Debug, PartialEq)]
pub struct Axis {
    pub path: String,
    pub values: Vec<f64>,
}

/// The points of the grid of `axes`, the first axis slowest: every combination, each a value per axis.
pub fn grid(axes: &[Axis]) -> Vec<Vec<f64>> {
    axes.iter().fold(vec![vec![]], |points, axis| {
        points.iter().flat_map(|p| axis.values.iter().map(move |v| { let mut q = p.clone(); q.push(*v); q })).collect()
    })
}

/// Set `value` at "<component>.<key>.<key>..." of a config that is an array of components.
fn set(config: &mut Value, path: &str, value: Value) -> Result<()> {
    let mut parts = path.split('.');
    let name = parts.next().ok_or("empty axis path")?;
    let keys: Vec<&str> = parts.collect();
    let mut node = config.as_array_mut().ok_or("the base config is not an array of components")?
        .iter_mut().find(|c| c["component"] == name).ok_or_else(|| format!("no component {name} for the axis {path}"))?;
    let (last, inner) = keys.split_last().ok_or_else(|| format!("the axis {path} names no key"))?;
    for k in inner {
        if !node[*k].is_object() { node[*k] = json!({}); }
        node = &mut node[*k];
    }
    node[*last] = value;
    Ok(())
}

/// `base` (an array of components) with every axis value set at its path and the outputs of this point.
pub fn point_config(base: &Value, axes: &[Axis], point: &[f64], packages: &str, videos: &str, prompts: &Value) -> Result<Value> {
    if axes.len() != point.len() { return Err("a point needs a value per axis".into()); }
    let mut c = base.clone();
    for (axis, v) in axes.iter().zip(point) { set(&mut c, &axis.path, json!(v))?; }
    set(&mut c, "export.output", json!(packages))?;
    set(&mut c, "generate.output.directory", json!(videos))?;
    if prompts.is_object() && !prompts.as_object().unwrap().is_empty() { set(&mut c, "generate.prompts", prompts.clone())?; }
    Ok(c)
}

/// Axes of the next round: an axis of several values gets `n` values evenly between the neighbours of the best one
/// (the best itself at an end of the axis keeps that end); an axis of one value keeps it.
pub fn refine(axes: &[Axis], best: &[f64], n: usize) -> Vec<Axis> {
    axes.iter().zip(best).map(|(axis, b)| {
        if axis.values.len() < 2 || n < 2 { return axis.clone(); }
        let i = axis.values.iter().position(|v| (v - b).abs() < 1e-12).unwrap_or(0);
        let (lo, hi) = (axis.values[i.saturating_sub(1)], axis.values[(i + 1).min(axis.values.len() - 1)]);
        Axis { path: axis.path.clone(), values: (0..n).map(|k| lo + (hi - lo) * k as f64 / (n - 1) as f64).collect() }
    }).collect()
}

/// The loss of a point: `metric` of evaluation rows averaged over the chains that have it; None if none has it.
pub fn loss(rows: &[Value], metric: &str) -> Option<f64> {
    let values: Vec<f64> = rows.iter().filter_map(|r| r[metric].as_f64()).collect();
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

struct Spec {
    base: Value,
    configs: PathBuf,
    output: PathBuf,
    packages: String,
    prompts: Value,
    axes: Vec<Axis>,
    reference: PathBuf,
    metric: String,
    lower_is_better: bool,
    rounds: usize,
    values_per_axis: usize,
    python: PathBuf,
    evaluate_args: Vec<String>,
}

fn spec(path: &Path) -> Result<Spec> {
    let root = crate::root();
    let at = |p: &str| root.join(p);
    let s = serde_json::from_str::<Value>(&std::fs::read_to_string(path)?)?["search"].clone();
    let text = |k: &str| s[k].as_str().map(String::from).ok_or_else(|| format!("search.{k} is missing"));
    let base: Value = serde_json::from_str(&std::fs::read_to_string(at(&text("base")?))?)?;
    let axes = s["axes"].as_array().ok_or("search.axes is missing")?.iter().map(|a| Ok(Axis {
        path: a["path"].as_str().ok_or("an axis needs a path")?.to_string(),
        values: a["values"].as_array().ok_or("an axis needs values")?.iter().map(|v| v.as_f64().ok_or("a value is not a number")).collect::<std::result::Result<_, _>>()?,
    })).collect::<Result<Vec<_>>>()?;
    Ok(Spec {
        base, axes,
        configs: at(&text("configs")?),
        output: at(&text("output")?),
        packages: text("packages")?,
        prompts: s["prompts"].clone(),
        reference: at(&text("reference")?),
        metric: text("metric")?,
        lower_is_better: s["lower_is_better"].as_bool().unwrap_or(true),
        rounds: s["refine"]["rounds"].as_u64().unwrap_or(0) as usize,
        values_per_axis: s["refine"]["values_per_axis"].as_u64().unwrap_or(5) as usize,
        python: at(s["python"].as_str().unwrap_or(".venv/Scripts/python.exe")),
        evaluate_args: s["evaluate_args"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default(),
    })
}

/// The newest run folder of a point's videos (generate writes <directory>/<stamp>/).
fn newest_run(dir: &Path) -> Option<PathBuf> {
    let mut runs: Vec<PathBuf> = std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path())
        .filter(|p| p.join("run.json").exists()).collect();
    runs.sort();
    runs.pop()
}

/// The evaluation folder of a point: a run per chain (hard links to its videos) and experiment.json with the prompts.
fn evaluation_folder(videos: &Path, folder: &Path, tag: &str) -> Result<()> {
    let run = newest_run(videos).ok_or_else(|| format!("{tag}: no videos in {}", videos.display()))?;
    let chains: Value = serde_json::from_str(&std::fs::read_to_string(run.join("run.json"))?)?;
    let mut runs = vec![];
    for (chain, steps) in chains["chains"].as_object().ok_or("run.json without chains")? {
        let id = format!("{chain}__{tag}");
        let dir = folder.join(&id);
        std::fs::create_dir_all(&dir)?;
        let (src, dst) = (run.join(format!("{chain}.mp4")), dir.join("video.mp4"));
        if !dst.exists() { std::fs::hard_link(&src, &dst)?; }
        runs.push(json!({"id": id, "prompts": [steps[0]["text"].clone()], "source": src.display().to_string()}));
    }
    std::fs::write(folder.join("experiment.json"), serde_json::to_string_pretty(&json!({"runs": runs}))?)?;
    Ok(())
}

fn evaluation_rows(folder: &Path) -> Option<Vec<Value>> {
    serde_json::from_str::<Value>(&std::fs::read_to_string(folder.join("evaluation.json")).ok()?).ok()?.as_array().cloned()
}

fn number(v: f64) -> String { format!("{}", (v * 1e6).round() / 1e6) }

/// Run the search of `config`; the exit code.
fn run(config: &Path, argv: &[String]) -> Result<()> {
    let s = spec(config)?;
    let root = crate::root();
    let exe = std::env::current_exe()?;
    if argv.iter().any(|a| a == "--dry") { // the configs of the first round, checked, nothing run
        std::fs::create_dir_all(&s.configs)?;
        for (i, point) in grid(&s.axes).into_iter().enumerate() {
            let tag = format!("r0-p{:02}", i + 1);
            let c = point_config(&s.base, &s.axes, &point, &format!("{}-current", s.packages),
                                 &s.output.join("points").join(&tag).display().to_string(), &s.prompts)?;
            let path = s.configs.join(format!("{tag}.json"));
            std::fs::write(&path, serde_json::to_string_pretty(&c)?)?;
            crate::config::Config::read(&path)?.validate()?;
            println!("[search] {tag}: {:?} ok", point);
        }
        return Ok(());
    }
    for d in [&s.configs, &s.output.join("points"), &s.output.join("eval"), &s.output.join("logs")] { std::fs::create_dir_all(d)?; }
    let ticket = crate::queue::enter(argv, Some(config))?;
    let mut axes = s.axes.clone();
    let mut scored: Vec<(String, Vec<f64>, Option<f64>, Vec<Value>)> = vec![];
    let result = (|| -> Result<()> {
        for round in 0..=s.rounds {
            let mut pending: Vec<(String, Vec<f64>, std::process::Child)> = vec![];
            let mut done: Vec<(String, Vec<f64>)> = vec![];
            for (i, point) in grid(&axes).into_iter().enumerate() {
                if scored.iter().any(|(_, p, _, _)| p == &point) { continue; } // a point of an earlier round
                let tag = format!("r{round}-p{:02}", i + 1);
                let (videos, eval) = (s.output.join("points").join(&tag), s.output.join("eval").join(&tag));
                if evaluation_rows(&eval).is_some() { done.push((tag, point)); continue; } // resumed
                let packages = format!("{}-current", s.packages);
                let c = point_config(&s.base, &axes, &point, &packages, &videos.display().to_string(), &s.prompts)?;
                let path = s.configs.join(format!("{tag}.json"));
                std::fs::write(&path, serde_json::to_string_pretty(&c)?)?;
                let values: Vec<String> = axes.iter().zip(&point).map(|(a, v)| format!("{}={}", a.path, number(*v))).collect();
                println!("[search] {tag}: {}", values.join(" "));
                let log = std::fs::File::create(s.output.join("logs").join(format!("{tag}.log")))?;
                let status = std::process::Command::new(&exe).arg("--config").arg(&path).current_dir(&root)
                    .env(crate::queue::HELD, &ticket.id).env("PYTHONUTF8", "1").env("PYTHONUNBUFFERED", "1")
                    .stdout(log.try_clone()?).stderr(log).status()?;
                if !status.success() { println!("[search] {tag}: the run failed ({status})"); scored.push((tag, point, None, vec![])); continue; }
                evaluation_folder(&videos, &eval, &tag)?;
                while pending.len() >= 2 { let (t, p, mut child) = pending.remove(0); child.wait()?; done.push((t, p)); }
                let elog = std::fs::File::create(s.output.join("logs").join(format!("{tag}.eval.log")))?;
                let child = std::process::Command::new(&s.python).arg("scripts/reports/evaluate_videos.py")
                    .arg("--experiment").arg(&eval).arg("--reference").arg(&s.reference).args(&s.evaluate_args)
                    .current_dir(&root).env("PYTHONUTF8", "1").stdout(elog.try_clone()?).stderr(elog).spawn()?;
                pending.push((tag, point, child));
            }
            for (t, p, mut child) in pending { child.wait()?; done.push((t, p)); }
            for (tag, point) in done {
                let rows = evaluation_rows(&s.output.join("eval").join(&tag)).unwrap_or_default();
                scored.push((tag, point, loss(&rows, &s.metric), rows));
            }
            write_results(&s, &axes, &scored)?;
            let best = best(&scored, s.lower_is_better).ok_or("no point was scored")?;
            println!("[search] round {round}: best {} loss {}", best.0, best.2.map(number).unwrap_or_default());
            axes = refine(&axes, &best.1, s.values_per_axis);
        }
        Ok(())
    })();
    ticket.finish(if result.is_ok() { 0 } else { 1 });
    result
}

fn best(scored: &[(String, Vec<f64>, Option<f64>, Vec<Value>)], lower: bool) -> Option<&(String, Vec<f64>, Option<f64>, Vec<Value>)> {
    scored.iter().filter(|r| r.2.is_some_and(f64::is_finite)).min_by(|a, b| {
        let (x, y) = (a.2.unwrap(), b.2.unwrap());
        if lower { x.total_cmp(&y) } else { y.total_cmp(&x) }
    })
}

fn write_results(s: &Spec, axes: &[Axis], scored: &[(String, Vec<f64>, Option<f64>, Vec<Value>)]) -> Result<()> {
    let chains: Vec<String> = {
        let mut c: Vec<String> = scored.iter().flat_map(|r| r.3.iter().filter_map(|x| x["prompt"].as_str().map(String::from))).collect();
        c.sort(); c.dedup(); c
    };
    let mut out = format!("tag\t{}\t{}\t{}\n", axes.iter().map(|a| a.path.as_str()).collect::<Vec<_>>().join("\t"), s.metric,
                          chains.join("\t"));
    for (tag, point, l, rows) in scored {
        let per: Vec<String> = chains.iter().map(|c| rows.iter().find(|r| r["prompt"] == c.as_str())
            .and_then(|r| r[&s.metric].as_f64()).map(number).unwrap_or_default()).collect();
        out += &format!("{tag}\t{}\t{}\t{}\n", point.iter().map(|v| number(*v)).collect::<Vec<_>>().join("\t"),
                        l.map(number).unwrap_or_else(|| "failed".into()), per.join("\t"));
    }
    std::fs::write(s.output.join("search.tsv"), out)?;
    if let Some(b) = best(scored, s.lower_is_better) {
        let named: BTreeMap<&str, f64> = axes.iter().map(|a| a.path.as_str()).zip(b.1.iter().copied()).collect();
        std::fs::write(s.output.join("best.json"), serde_json::to_string_pretty(&json!({"tag": b.0, "point": named, s.metric.clone(): b.2}))?)?;
    }
    Ok(())
}

/// `wan search --config <search.json>`; the exit code.
pub fn main(argv: &[String]) -> i32 {
    let Some(config) = argv.iter().position(|a| a == "--config").and_then(|i| argv.get(i + 1)) else {
        eprintln!("wan search --config <search.json>");
        return 2;
    };
    match run(&crate::root().join(config), &[&["search".to_string()], argv].concat()) {
        Ok(()) => 0,
        Err(e) => { eprintln!("[search] {e}"); 1 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn axes() -> Vec<Axis> {
        vec![Axis { path: "export.rope.stretch".into(), values: vec![-1.0, 0.0] },
             Axis { path: "export.rope.attn_logit_scale".into(), values: vec![0.9, 1.0, 1.1] }]
    }
    fn base() -> Value {
        json!([{"component": "export", "output": "artifacts/live/x", "rope": {"stretch": 0.3, "ntk_aware": true},
                "sampling": {"timestep_shift": 5.0}},
               {"component": "generate", "prompts": {"source": "script", "file": "a.json", "chains": ["c"]},
                "output": {"directory": "out/x", "crf": 16}}])
    }

    #[test]
    fn the_grid_is_every_combination_with_the_first_axis_slowest() {
        let g = grid(&axes());
        assert_eq!(g.len(), 6);
        assert_eq!(g[0], vec![-1.0, 0.9]);
        assert_eq!(g[1], vec![-1.0, 1.0]);
        assert_eq!(g[5], vec![0.0, 1.1]);
    }

    #[test]
    fn a_point_config_sets_its_values_and_its_own_outputs_and_keeps_the_rest() {
        let prompts = json!({"source": "script", "file": "scenes.json", "chains": ["teacup_pour", "fox_cabin"]});
        let c = point_config(&base(), &axes(), &[-1.0, 0.93], "artifacts/live/grid-p01", "out/grid/p01", &prompts).unwrap();
        assert_eq!(c[0]["rope"]["stretch"], json!(-1.0));
        assert_eq!(c[0]["rope"]["attn_logit_scale"], json!(0.93));
        assert_eq!(c[0]["rope"]["ntk_aware"], json!(true));
        assert_eq!(c[0]["sampling"]["timestep_shift"], json!(5.0));
        assert_eq!(c[0]["output"], json!("artifacts/live/grid-p01"));
        assert_eq!(c[1]["output"]["directory"], json!("out/grid/p01"));
        assert_eq!(c[1]["output"]["crf"], json!(16));
        assert_eq!(c[1]["prompts"], prompts);
        assert_eq!(base()[0]["rope"]["stretch"], json!(0.3), "the base is not changed");
    }

    #[test]
    fn a_path_into_a_missing_component_is_an_error() {
        let bad = vec![Axis { path: "decode.rope.stretch".into(), values: vec![1.0] }];
        assert!(point_config(&base(), &bad, &[1.0], "p", "v", &json!({})).is_err());
    }

    #[test]
    fn refining_spreads_between_the_neighbours_of_the_best() {
        let a = vec![Axis { path: "s".into(), values: vec![-1.5, -1.0, -0.5, 0.0] },
                     Axis { path: "l".into(), values: vec![0.9, 1.0, 1.1] },
                     Axis { path: "t".into(), values: vec![5.0] }];
        let r = refine(&a, &[-1.0, 1.1, 5.0], 5);
        let close = |x: &[f64], y: &[f64]| x.len() == y.len() && x.iter().zip(y).all(|(a, b)| (a - b).abs() < 1e-9);
        assert!(close(&r[0].values, &[-1.5, -1.25, -1.0, -0.75, -0.5]), "{:?}", r[0].values);
        assert!(close(&r[1].values, &[1.0, 1.025, 1.05, 1.075, 1.1]), "{:?}", r[1].values);
        assert_eq!(r[2].values, vec![5.0]);
        assert_eq!(r[0].path, "s");
    }

    #[test]
    fn the_loss_averages_the_chains_that_have_the_metric() {
        let rows = vec![json!({"prompt": "a", "lpips_aligned": 0.2}), json!({"prompt": "b", "lpips_aligned": 0.4}),
                        json!({"prompt": "c"})];
        assert!((loss(&rows, "lpips_aligned").unwrap() - 0.3).abs() < 1e-12);
        assert_eq!(loss(&rows, "dino_reference"), None);
    }
}
