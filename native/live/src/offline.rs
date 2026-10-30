//! The offline mode (configs/offline.json, "mode": "offline"): chains of prompts of configs/prompts/demo_chains.json
//! made by the same video as the window, without it, into mp4 files, and beside them run.json: the configuration and
//! the flags of the run, the packages' manifest and the placement of the calibration, so that every video says what
//! made it.
//!
//! A chain begins with a new video and goes on with continuations ("same") and cuts ("scene"), each prompt held for
//! some chunks. A step is a whole "text", or a scene card ("card": the subject and the place) with an "action" and a
//! "light"; card and light are carried over from the step before, as the director builds a continuation. A chain can
//! also be a line of LongLive 1.0's interactive_example.jsonl ("ref": "<file>#<line>").

use std::path::Path;

use live_protocol::Cut;
use serde_json::Value;

#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    pub cut: Cut,
    pub text: String,
    pub chunks: usize,
}

/// The steps of `chain`, each prompt written out whole; `chunks` a prompt by default, `root` the repository.
pub fn chain_steps(chain: &Value, chunks: usize, root: &Path) -> Result<Vec<Step>, String> {
    let chunks = chain["chunks"].as_u64().map_or(chunks, |c| c as usize);
    if let Some(reference) = chain["ref"].as_str() {
        let (file, line) = reference.rsplit_once('#').ok_or(format!("{reference}: no #line"))?;
        let text = std::fs::read_to_string(root.join(file)).map_err(|e| format!("{file}: {e}"))?;
        let line: usize = line.parse().map_err(|_| format!("{reference}: the line is not a number"))?;
        let row: Value = serde_json::from_str(text.lines().nth(line - 1).ok_or(format!("{reference}: no such line"))?)
            .map_err(|e| format!("{reference}: {e}"))?;
        let prompts = row["prompts"].as_array().ok_or(format!("{reference}: no prompts"))?;
        return Ok(prompts.iter().enumerate()
            .map(|(i, p)| Step { cut: if i == 0 { Cut::New } else { Cut::Same }, text: p.as_str().unwrap_or("").into(), chunks })
            .collect());
    }
    let (mut card, mut light) = (None::<String>, None::<String>);
    let mut steps = Vec::new();
    for (i, s) in chain["steps"].as_array().ok_or("a chain has steps or a ref")?.iter().enumerate() {
        let cut = match s["cut"].as_str().unwrap_or(if i == 0 { "new" } else { "same" }) {
            "new" => Cut::New,
            "same" => Cut::Same,
            "scene" => Cut::Scene,
            other => return Err(format!("unknown cut {other}")),
        };
        let text = if let Some(t) = s["text"].as_str() {
            t.to_string()
        } else {
            if let Some(c) = s["card"].as_str() { card = Some(c.into()); }
            if let Some(l) = s["light"].as_str() { light = Some(l.into()); }
            let camera: Vec<&str> = [chain["camera"].as_str(), light.as_deref()].into_iter().flatten().collect();
            let action = s["action"].as_str().ok_or(format!("step {i}: no text and no action"))?;
            format!("{} {action} {}.", card.as_deref().ok_or(format!("step {i}: no card"))?, camera.join(", "))
        };
        steps.push(Step { cut, text, chunks: s["chunks"].as_u64().map_or(chunks, |c| c as usize) });
    }
    Ok(steps)
}

fn cut_name(cut: Cut) -> &'static str {
    match cut { Cut::New => "new", Cut::Same => "same", Cut::Scene => "scene" }
}

/// run.json of a run: `config` the text of its configuration file, `args` its command line, `packages` the manifest
/// of the packages, `calibration` the placement kept beside them, and each chain's steps with the second they begin
/// (a chunk is `frames_per_chunk` frames at `fps`).
pub fn snapshot(config: &str, args: &[String], packages: &str, calibration: Option<&str>,
                chains: &[(String, Vec<Step>)], fps: f64, frames_per_chunk: usize) -> Value {
    let parse = |t: &str| serde_json::from_str::<Value>(t).unwrap_or(Value::String(t.into()));
    let chains: serde_json::Map<String, Value> = chains.iter().map(|(id, steps)| {
        let mut chunks = 0;
        let steps: Vec<Value> = steps.iter().map(|s| {
            let begins = (chunks * frames_per_chunk) as f64 / fps;
            chunks += s.chunks;
            serde_json::json!({"cut": cut_name(s.cut), "text": s.text, "chunks": s.chunks, "begins_s": begins})
        }).collect();
        (id.clone(), Value::Array(steps))
    }).collect();
    serde_json::json!({"config": parse(config), "args": args, "packages": parse(packages),
                       "calibration": calibration.map(parse), "chains": chains})
}

/// The command line of ffmpeg that takes raw RGB frames of `height` x `width` on its input into an H.264 mp4.
pub fn ffmpeg_args(height: usize, width: usize, fps: f64, out: &Path) -> Vec<String> {
    let mut a: Vec<String> = ["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24"].map(String::from).into();
    a.extend(["-s".into(), format!("{width}x{height}"), "-r".into(), format!("{fps}"), "-i".into(), "-".into()]);
    a.extend(["-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p"].map(String::from));
    a.push(out.display().to_string());
    a
}

/// Today's date and time in UTC as yyyy-mm-dd_hhmm, the name of a run's folder.
fn stamp() -> String {
    let s = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;
    let (days, rest) = (s.div_euclid(86400), s.rem_euclid(86400));
    // days to a civil date (Howard Hinnant's algorithm)
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}_{:02}{:02}", rest / 3600, rest % 3600 / 60)
}

/// The chains `ids` of configs/prompts/demo_chains.json (all of them when empty) into
/// ~/Documents/Experiments/DIT/outputs/demo_chains/<time>/<id>.mp4, with run.json beside them.
pub fn run(args: &crate::Args, config_text: &str, argv: &[String],
           (mut video, mut t5, embeddings): (crate::Video, crate::T5, crate::DeviceBuffer)) -> crate::Result<()> {
    use std::io::Write;
    let root = crate::root();
    let pool: Value = serde_json::from_str(&std::fs::read_to_string(root.join("configs/prompts/demo_chains.json"))?)?;
    let chunks = pool["chunks"].as_u64().unwrap_or(8) as usize;
    let mut chains = Vec::new();
    for chain in pool["chains"].as_array().ok_or("demo_chains.json has no chains")? {
        let id = chain["id"].as_str().unwrap_or("").to_string();
        if args.chains.is_empty() || args.chains.contains(&id) {
            chains.push((id, chain_steps(chain, chunks, &root)?));
        }
    }
    for id in &args.chains {
        if !chains.iter().any(|(c, _)| c == id) {
            return Err(format!("no chain {id} in demo_chains.json").into());
        }
    }
    let home = std::env::var("USERPROFILE").map_err(|_| "no USERPROFILE")?;
    let out = Path::new(&home).join("Documents/Experiments/DIT/outputs/demo_chains").join(stamp());
    std::fs::create_dir_all(&out)?;
    let (h, w, fps) = (video.m.height, video.m.width, video.m.fps);
    let frames_per_chunk = video.m.chunk * 4;
    let write_run = |video_dir: &Path| -> crate::Result<()> {
        let packages = std::fs::read_to_string(args.packages.join("live.json"))?;
        let calibration = std::fs::read_to_string(args.packages.join("calibration.json")).ok();
        let run = snapshot(config_text, argv, &packages, calibration.as_deref(), &chains, fps, frames_per_chunk);
        std::fs::write(video_dir.join("run.json"), serde_json::to_string_pretty(&run)?)?;
        Ok(())
    };
    write_run(&out)?;
    for (id, steps) in &chains {
        let path = out.join(format!("{id}.mp4"));
        let mut ffmpeg = std::process::Command::new("ffmpeg").args(ffmpeg_args(h, w, fps, &path))
            .stdin(std::process::Stdio::piped()).spawn().map_err(|e| format!("ffmpeg: {e}"))?;
        let mut input = ffmpeg.stdin.take().unwrap();
        let start = std::time::Instant::now();
        for step in steps {
            crate::set_prompt(&mut video, &mut t5, &embeddings, &crate::Prompt { cut: step.cut, text: step.text.clone() })?;
            if step.cut == Cut::New {
                video.reset_decoder();
            }
            for _ in 0..step.chunks {
                let x0 = video.next_chunk(None)?;
                for frame in video.decode(&x0)? {
                    input.write_all(&frame)?;
                }
            }
        }
        drop(input);
        let status = ffmpeg.wait()?;
        if !status.success() {
            return Err(format!("ffmpeg failed on {id}: {status}").into());
        }
        println!("{}: {} chunks in {:.0} s", path.display(), steps.iter().map(|s| s.chunks).sum::<usize>(),
                 start.elapsed().as_secs_f64());
    }
    // again at the end: the calibration of the first chunks is in it now
    write_run(&out)?;
    println!("{}", out.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn step(cut: Cut, text: &str, chunks: usize) -> Step {
        Step { cut, text: text.into(), chunks }
    }

    #[test]
    fn a_step_keeps_the_card_and_the_light_of_the_one_before() {
        let chain = json!({"camera": "Medium close-up shot, static camera", "steps": [
            {"cut": "new", "card": "A red fox sits in the snow.", "action": "It turns its head.", "light": "day time"},
            {"action": "Snow begins to fall.", "chunks": 6},
            {"action": "It lies down.", "light": "sunset, warm light"},
            {"cut": "scene", "card": "The fox sleeps by a fireplace.", "action": "Flames flicker."}]});
        assert_eq!(chain_steps(&chain, 8, Path::new(".")).unwrap(), [
            step(Cut::New, "A red fox sits in the snow. It turns its head. Medium close-up shot, static camera, day time.", 8),
            step(Cut::Same, "A red fox sits in the snow. Snow begins to fall. Medium close-up shot, static camera, day time.", 6),
            step(Cut::Same, "A red fox sits in the snow. It lies down. Medium close-up shot, static camera, sunset, warm light.", 8),
            step(Cut::Scene, "The fox sleeps by a fireplace. Flames flicker. Medium close-up shot, static camera, sunset, warm light.", 8),
        ]);
    }

    #[test]
    fn a_whole_text_is_taken_as_it_is() {
        let chain = json!({"chunks": 4, "steps": [{"cut": "new", "text": "A teacup pours water."}, {"text": "Wine."}]});
        assert_eq!(chain_steps(&chain, 8, Path::new(".")).unwrap(),
                   [step(Cut::New, "A teacup pours water.", 4), step(Cut::Same, "Wine.", 4)]);
    }

    #[test]
    fn a_line_of_longlive_is_a_chain_begun_anew() {
        let dir = std::env::temp_dir().join(format!("live-chain-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("i.jsonl"), "{\"prompts\": [\"a\", \"b\"]}\n{\"prompts\": [\"c\", \"d\", \"e\"]}\n").unwrap();
        let steps = chain_steps(&json!({"ref": "i.jsonl#2"}), 8, &dir).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(steps, [step(Cut::New, "c", 8), step(Cut::Same, "d", 8), step(Cut::Same, "e", 8)]);
    }

    /// run.json beside the videos: the configuration file and the flags as they were, the packages' manifest, where
    /// the calibration put every resource, and the prompts of every chain with the second each one begins.
    #[test]
    fn a_run_keeps_what_made_it() {
        let steps = [step(Cut::New, "a fox", 8), step(Cut::Scene, "a beach", 6)];
        let run = snapshot(r#"{"mode": "offline", "seed": 0}"#, &["live".into(), "--chains".into(), "fox".into()],
                           r#"{"height": 512, "kernels": "sage"}"#, Some(r#"{"weights": "disk"}"#),
                           &[("fox".into(), steps.to_vec())], 24.0, 32);
        assert_eq!(run["config"]["mode"], "offline");
        assert_eq!(run["args"], json!(["live", "--chains", "fox"]));
        assert_eq!(run["packages"]["height"], 512);
        assert_eq!(run["calibration"]["weights"], "disk");
        assert_eq!(run["chains"]["fox"][1], json!({"cut": "scene", "text": "a beach", "chunks": 6,
                                                    "begins_s": 8.0 * 32.0 / 24.0}));
    }

    #[test]
    fn frames_go_to_ffmpeg_as_raw_rgb() {
        let a = ffmpeg_args(512, 928, 24.0, Path::new("o/fox.mp4"));
        let joined = a.join(" ");
        assert!(joined.contains("-f rawvideo -pix_fmt rgb24 -s 928x512 -r 24 -i -"), "{joined}");
        assert!(joined.ends_with("-pix_fmt yuv420p o/fox.mp4") || joined.ends_with("-pix_fmt yuv420p o\\fox.mp4"), "{joined}");
    }

    #[test]
    fn every_chain_of_the_repository_builds_and_begins_anew() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let pool: Value = serde_json::from_str(&std::fs::read_to_string(root.join("configs/prompts/demo_chains.json"))
            .unwrap()).unwrap();
        let chunks = pool["chunks"].as_u64().unwrap() as usize;
        for chain in pool["chains"].as_array().unwrap() {
            let steps = chain_steps(chain, chunks, &root).unwrap();
            assert_eq!(steps[0].cut, Cut::New, "{}", chain["id"]);
            assert!(steps[1..].iter().all(|s| s.cut != Cut::New && s.chunks >= 6), "{}", chain["id"]);
        }
    }
}
