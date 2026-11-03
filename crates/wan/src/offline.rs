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

use crate::prompts::Step;
#[cfg(test)]
use crate::prompts::chain_steps;

fn cut_name(cut: Cut) -> &'static str {
    match cut { Cut::New => "new", Cut::Same => "same", Cut::Scene => "scene" }
}

/// run.json of a run: `config` the text of its configuration file, `args` its command line, `packages` the manifest
/// of the packages, `calibration` the placement kept beside them, and each chain's steps with the second they begin
/// (a chunk is `frames_per_chunk` frames at `fps`).
pub fn snapshot(config: &str, args: &[String], packages: &str, calibration: Option<&str>,
                chains: &[(String, Vec<Step>)], fps: f64, frames_per_chunk: usize, decoder_trim: usize) -> Value {
    let parse = |t: &str| serde_json::from_str::<Value>(t).unwrap_or(Value::String(t.into()));
    let chains: serde_json::Map<String, Value> = chains.iter().map(|(id, steps)| {
        let mut frames = 0;
        let steps: Vec<Value> = steps.iter().enumerate().map(|(i, s)| {
            let begins = frames as f64 / fps;
            let made = s.chunks * frames_per_chunk;
            frames += made - if i == 0 || s.cut == Cut::New { decoder_trim.min(made) } else { 0 };
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
pub fn run<M: crate::telemetry::Probe>(args: &crate::Args, loaded: (crate::Video, crate::T5, crate::DeviceBuffer), metrics: M) -> crate::Result<()> {
    use std::io::Write;
    use crate::config::PromptSource;
    use crate::generation::Generator;
    let g = &args.pipeline.generate;
    let output = g.output.as_ref().ok_or("offline output is required")?;
    let mut sequences = crate::generation::sequences(&g.prompts)?;
    let home = std::path::PathBuf::from(std::env::var("USERPROFILE")?);
    let directory = if let Ok(relative) = output.directory.strip_prefix("outputs") {
        home.join("Documents/Experiments/DIT/outputs").join(relative)
    } else { crate::root().join(&output.directory) };
    let out = directory.join(format!("{}-{}", stamp(), std::process::id()));
    std::fs::create_dir_all(&out)?;
    let mut generator = Generator::new(loaded)?.observe(metrics);
    let (h,w,fps) = (generator.video.m.height, generator.video.m.width, generator.video.m.fps);
    let argv: Vec<String> = std::env::args().collect();
    let write_run = |sequences: &[crate::generation::Sequence]| -> crate::Result<()> {
        let packages = std::fs::read_to_string(args.packages().join("live.json"))?;
        let calibration = std::fs::read_to_string(args.packages().join("calibration.json")).ok();
        let snapshot = snapshot(&args.pipeline.value().to_string(), &argv, &packages, calibration.as_deref(),
                                sequences, fps, generator.video.m.chunk * 4, generator.video.m.decoder_trim);
        std::fs::write(out.join("run.json"), serde_json::to_vec_pretty(&snapshot)?)?;
        Ok(())
    };
    write_run(&sequences)?;
    let interactive = matches!(g.prompts, PromptSource::Interactive { .. });
    let inbox = crate::prompts::Inbox::default();
    if interactive {
        inbox.listen(live_protocol::PORT)?;
        let first = if let Some(p) = g.prompts.initial() { crate::Prompt { cut: Cut::New, text: p.into() } }
            else { loop { if let Some(p) = inbox.take() { break p; } std::thread::sleep(std::time::Duration::from_millis(20)); } };
        sequences.push(("interactive".into(), vec![Step { cut: first.cut, text: first.text, chunks: args.chunks }]));
    }
    let mut actual_sequences = Vec::new();
    for (id, steps) in &sequences {
        let mut actual_steps: Vec<Step> = Vec::new();
        let path = out.join(format!("{id}.mp4"));
        let mut a = ffmpeg_args(h,w,fps,&path);
        // Encoding is explicit in the offline config; validation currently supports this codec/pixel format.
        let at = a.iter().position(|s| s == "-crf").unwrap();
        a[at + 1] = output.crf.to_string();
        let mut child = std::process::Command::new("ffmpeg").args(a).stdin(std::process::Stdio::piped()).spawn()?;
        let mut input = child.stdin.take().unwrap();
        let mut emit = |frames: Vec<Vec<u8>>, _: &crate::Video, _: &str| -> crate::Result<()> {
            for frame in frames { input.write_all(&frame)?; }
            Ok(())
        };
        let result = if interactive {
            (|| -> crate::Result<()> {
                generator.request(crate::Prompt { cut: steps[0].cut, text: steps[0].text.clone() })?;
                emit(generator.poll(true)?, &generator.video, &steps[0].text)?;
                let mut revision = 0;
                for _ in 0..args.chunks {
                    if let Some(p) = inbox.take() { generator.request(p)?; }
                    emit(generator.poll(false)?, &generator.video, "")?;
                    if revision != generator.revision() {
                        let p = generator.active().unwrap();
                        actual_steps.push(Step { cut: p.cut, text: p.text.clone(), chunks: 0 });
                        revision = generator.revision();
                    }
                    emit(generator.advance()?, &generator.video, "")?;
                    actual_steps.last_mut().unwrap().chunks += 1;
                }
                emit(generator.flush()?, &generator.video, "")?;
                Ok(())
            })()
        } else { generator.script(steps, &mut emit) };
        drop(emit);
        drop(input);
        if let Err(e) = result { let _ = child.kill(); let _ = child.wait(); return Err(e); }
        if !child.wait()?.success() { return Err(format!("ffmpeg failed on {id}").into()); }
        actual_sequences.push((id.clone(), if interactive { actual_steps } else { steps.clone() }));
        println!("{}", path.display());
    }
    let packages = std::fs::read_to_string(args.packages().join("live.json"))?;
    let calibration = std::fs::read_to_string(args.packages().join("calibration.json")).ok();
    let snapshot = snapshot(&args.pipeline.value().to_string(), &argv, &packages, calibration.as_deref(),
                            &actual_sequences, fps, generator.video.m.chunk * 4, generator.video.m.decoder_trim);
    std::fs::write(out.join("run.json"), serde_json::to_vec_pretty(&snapshot)?)?;
    println!("{}", out.display());
    generator.finish_metrics()?;
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
                           &[("fox".into(), steps.to_vec())], 24.0, 32, 3);
        assert_eq!(run["config"]["mode"], "offline");
        assert_eq!(run["args"], json!(["live", "--chains", "fox"]));
        assert_eq!(run["packages"]["height"], 512);
        assert_eq!(run["calibration"]["weights"], "disk");
        assert_eq!(run["chains"]["fox"][1], json!({"cut": "scene", "text": "a beach", "chunks": 6,
                                                    "begins_s": (8.0 * 32.0 - 3.0) / 24.0}));
    }

    #[test]
    fn prompt_times_follow_encoded_frames_including_decoder_warmup() {
        let steps = [step(Cut::New, "a", 1), step(Cut::Same, "b", 1),
                     step(Cut::New, "c", 1), step(Cut::Same, "d", 1)];
        let run = snapshot("[]", &[], "{}", None, &[("test".into(), steps.to_vec())], 24.0, 32, 3);
        assert_eq!(run["chains"]["test"][1]["begins_s"], 29.0 / 24.0);
        assert_eq!(run["chains"]["test"][2]["begins_s"], 61.0 / 24.0);
        assert_eq!(run["chains"]["test"][3]["begins_s"], 90.0 / 24.0);
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
