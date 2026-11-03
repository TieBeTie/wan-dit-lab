//! wan.exe, the one entry point of wan-dit-lab (built into bin/ by build.ps1). Live: a video generated without end,
//! chunk by chunk, and a prompt over a local socket (or typed in the window) turns it, from the frames already shown;
//! offline: the same video into files. The model is the AOTInductor packages of scripts/conversion/export_live.py
//! (preset longlive2), everything around it is here.
//!
//!     wan --prompt "A lone oak tree on a grassy hill sways in the wind" [--size 512x928]
//!     wan --send "Snow starts to fall on the oak tree" [--scene | --new]
//!     wan check [--reference artifacts/live/480x864/reference.safetensors]   the chunks against the Python path (*)
//!     wan bench --chunks 12                                          time and memory, no window
//!     wan --config configs/offline.json [--chains fox_cabin,sea_turtle]  chains of demo_chains.json into mp4 files
//!     wan render --reference <directory>                             the videos of live_reference.py --render (*)
//!
//! (*) only in a build with `--features debug-tools`; the release build keeps the hot path alone.
//!
//! Two threads. The main one shows the frames at the rate they are made (24 fps at most), so that the video slows
//! down rather than stops; the other encodes a waiting prompt (the newest), generates a chunk, decodes it and queues
//! its frames. A prompt is plain text (the scene goes on), "/scene ..." (LongLive 2.0's cut to a new shot) or
//! "/new ..." (a new video). Esc or closing the window stops it.

mod cache;
mod budget;
mod kv_disk;
mod config;
mod disk;
mod encoder;
mod gpu;
mod manifest;
mod measure;
mod offline;
mod packages;
mod pipeline;
mod runtime;
mod generation;
mod telemetry;
mod prompts;
mod rows;
mod t5;
mod video;
mod window;

use std::path::{Path, PathBuf};

use live_protocol::Cut;

use crate::gpu::DeviceBuffer;
use crate::prompts::Prompt;
use crate::t5::T5;
use crate::video::{Result, Video};

/// Command-line controls only. Model and generation settings live once, in the parsed pipeline.
struct Args {
    pipeline: config::Config,
    command: String,
    send: Option<Prompt>,
    chunks: usize,
    reference: PathBuf,
    serial: bool,
    scale: f32,
    right: bool,
}
impl Args {
    fn packages(&self) -> PathBuf { root().join(&self.pipeline.export.output) }
}
fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}
#[cfg(test)]
fn defaults(path: &Path) -> Args {
    Args { pipeline: config::Config::read(path).expect("pipeline config"), command: "run".into(), send: None,
           chunks: 12, reference: root().join("artifacts/live/480x864/reference.safetensors"),
           serial: false, scale: 1.0, right: false }
}
fn parse_from(argv: &[String]) -> Result<Args> {
    let config_path = match argv.iter().position(|a| a == "--config") {
        Some(i) => root().join(argv.get(i + 1).ok_or("--config needs a value")?),
        None => root().join("configs/live.json"),
    };
    let config = config::Config::read(&config_path)?;
    let mut a = Args { pipeline: config, command: "run".into(), send: None, chunks: 12,
        reference: root().join("artifacts/live/480x864/reference.safetensors"), serial: false, scale: 1.0, right: false };
    let mut it = argv.iter();
    let mut cut = Cut::Same;
    while let Some(arg) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "check" | "bench" | "run" | "render" => a.command = arg.clone(),
            "--plan" => a.command = "plan".into(),
            "--export-only" => a.command = "export".into(),
            "--config" => { value()?; }
            "--packages" => a.pipeline.export.output = value()?.into(),
            "--size" => {
                let size = value()?.clone();
                a.pipeline.export.output = PathBuf::from("artifacts/live").join(&size);
                a.pipeline.export.video.size = size;
            }
            "--prompt" => a.pipeline.generate.prompts = config::PromptSource::Interactive { initial: Some(value()?.clone()) },
            "--chains" => match &mut a.pipeline.generate.prompts {
                config::PromptSource::Script { chains, .. } => *chains = value()?.split(',').map(str::trim).filter(|s| !s.is_empty()).map(String::from).collect(),
                _ => return Err("--chains needs prompts.source=script".into()),
            },
            "--send" => a.send = Some(Prompt { text: value()?.clone(), cut }),
            "--scene" => cut = Cut::Scene,
            "--new" => cut = Cut::New,
            "--seed" => a.pipeline.generate.seed = value()?.parse()?,
            "--chunks" => a.chunks = value()?.parse()?,
            "--reference" => a.reference = value()?.into(),
            "--scale" => a.scale = value()?.parse()?,
            "--right" => a.right = true,
            "--serial" => a.serial = true,
            "--rebuild" => a.pipeline.export.rebuild = value()?.clone(),
            "--calibration" => a.pipeline.generate.offloading.calibration = serde_json::from_value(serde_json::json!(value()?))?,
            "--tier" => {
                let v = value()?;
                let (r,t) = v.split_once('=').ok_or("--tier needs resource=tier")?;
                a.pipeline.generate.offloading.resources.insert(serde_json::from_value(serde_json::json!(r))?, serde_json::from_value(serde_json::json!(t))?);
            }
            "--no-offloading" => a.pipeline.generate.offloading.enabled = false,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    if let Some(p) = &mut a.send { p.cut = cut; }
    if a.chunks == 0 || !a.scale.is_finite() || a.scale <= 0.0 { return Err("chunks and scale must be positive".into()); }
    a.pipeline.validate()?;
    Ok(a)
}
// Kept as direct functions for the diagnostic callers; no configuration work in the chunk loop.
use runtime::load;
#[cfg(feature = "debug-tools")]
use runtime::set_prompt;

fn main() -> Result<()> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.iter().any(|a| a == "--help" || a == "-h") {
        println!("wan --config configs/live.json [--prompt TEXT] [--plan | --export-only]\nwan --config configs/offline.json [--chains IDS]\nwan bench --chunks N [--rebuild never]\nwan --send TEXT [--scene | --new]");
        return Ok(());
    }
    let args = parse_from(&argv)?;
    if let Some(p) = &args.send { prompts::send(p.cut, &p.text, live_protocol::PORT)?; return Ok(()); }
    if args.command == "plan" { println!("{}", serde_json::to_string_pretty(&args.pipeline.value())?); return Ok(()); }
    // Resolve prompt scripts before any expensive preparation or GPU allocation.
    let _ = generation::sequences(&args.pipeline.generate.prompts)?;
    #[cfg(not(feature = "fps"))]
    if args.command != "export" && args.pipeline.generate.debug.enabled() {
        return Err("this executable was built without diagnostics; use debug=false or rebuild with feature fps".into());
    }
    pipeline::prepare(&args.pipeline)?;
    if args.command == "export" { return Ok(()); }
    #[cfg(feature = "fps")]
    if args.pipeline.generate.debug.enabled() {
        let metrics = telemetry::Fps::spawn(&args.pipeline.generate.drain)?;
        return execute(args, metrics);
    }
    execute(args, telemetry::Off)
}

fn execute<M: telemetry::Probe>(args: Args, metrics: M) -> Result<()> {
    if args.command == "run" && args.pipeline.generate.mode == config::Mode::Offline {
        return offline::run(&args, load(&args)?, metrics);
    }
    match args.command.as_str() {
        #[cfg(feature = "debug-tools")]
        "check" => measure::check(&args.reference, load(&args)?),
        #[cfg(feature = "debug-tools")]
        "render" => measure::render(&args.reference, load(&args)?),
        #[cfg(not(feature = "debug-tools"))]
        "check" | "render" => Err("check and render need wan-debug.exe".into()),
        "bench" => measure::bench(load(&args)?, args.pipeline.generate.prompts.initial().ok_or("bench needs an initial prompt")?, args.chunks, args.serial, metrics),
        _ => window::run(args, load, metrics),
    }
}

#[cfg(test)]
mod kernel_tests {
    use super::*;
    #[test]
    fn mismatched_kernels_fail_before_loading_operators_or_the_gpu() {
        let mut args = defaults(&root().join("configs/live.json"));
        let dir = std::env::temp_dir().join(format!("wan-kernel-startup-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut manifest = crate::manifest::tests::fixture();
        manifest["kernels"]["cross_attention"] = serde_json::json!(["efficient"]);
        std::fs::write(dir.join("live.json"), manifest.to_string()).unwrap();
        args.pipeline.export.output = dir.clone();
        let error = load(&args).err().expect("mismatched packages loaded").to_string();
        assert!(error.contains("cross_attention"), "runtime touched before validation: {error}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
    #[test]
    fn overrides_change_the_effective_config_and_missing_values_are_errors() {
        let a = parse_from(&["--prompt".into(), "a fox".into(), "--seed".into(), "7".into()]).unwrap();
        assert_eq!(a.pipeline.generate.prompts.initial(), Some("a fox"));
        assert_eq!(a.pipeline.value()[1]["seed"], 7);
        assert!(parse_from(&["--config".into()]).is_err());
        assert!(parse_from(&["--seed".into()]).is_err());
        assert!(parse_from(&["--unknown".into()]).is_err());
    }
}
