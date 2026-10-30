//! Live: a video generated without end, chunk by chunk, and a prompt over a local socket (or typed in the window)
//! turns it, from the frames already shown. The Rust host of scripts/live/live_video.py (preset longlive2): the model
//! is the AOTInductor packages of scripts/conversion/export_live.py, everything around it is here.
//!
//!     live --prompt "A lone oak tree on a grassy hill sways in the wind" [--size 512x928]
//!     live --send "Snow starts to fall on the oak tree" [--scene | --new]
//!     live check [--reference artifacts/live/480x864/reference.safetensors]   the chunks against the Python path (*)
//!     live bench --chunks 12                                          time and memory, no window
//!     live --config configs/offline.json [--chains fox_cabin,sea_turtle]  chains of demo_chains.json into mp4 files
//!     live render --reference <directory>                             the videos of live_reference.py --render (*)
//!
//! (*) only in a build with `--features debug-tools`; the release build keeps the hot path alone.
//!
//! Two threads. The main one shows the frames at the rate they are made (24 fps at most), so that the video slows
//! down rather than stops; the other encodes a waiting prompt (the newest), generates a chunk, decodes it and queues
//! its frames. A prompt is plain text (the scene goes on), "/scene ..." (LongLive 2.0's cut to a new shot) or
//! "/new ..." (a new video). Esc or closing the window stops it.

mod cache;
mod config;
mod disk;
mod encoder;
mod gpu;
mod manifest;
mod measure;
mod offline;
mod packages;
mod prompts;
mod rows;
mod t5;
mod video;
mod window;

use std::path::{Path, PathBuf};
use std::time::Instant;

use live_protocol::Cut;

use crate::gpu::{DeviceBuffer, Gpu};
use crate::prompts::Prompt;
use crate::t5::T5;
use crate::video::{Result, Video};

struct Args {
    command: String,
    packages: PathBuf,
    prompt: Option<String>,
    send: Option<String>,
    cut: Cut,
    seed: u64,
    chunks: usize,
    reference: PathBuf,
    serial: bool,
    offloading: config::Offloading,
    recalibrate: bool,
    scale: f32,
    right: bool,
    /// the configuration file read, and its mode
    config: PathBuf,
    mode: config::Mode,
    /// the chains of the offline mode (all of demo_chains.json when empty)
    chains: Vec<String>,
}

fn root() -> PathBuf {
    // native/live -> wan-dit-lab
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

fn parse() -> Args {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    // --config before the other flags: they override the file
    let config = argv.iter().position(|a| a == "--config").map(|i| root().join(&argv[i + 1]))
        .unwrap_or_else(|| root().join("configs").join("live.json"));
    let mut a = defaults(&config);
    let mut it = argv.into_iter();
    flags(&mut a, &mut it);
    a
}

/// The arguments of the configuration file (configs/live.json by default), before the flags.
fn defaults(path: &Path) -> Args {
    let c = config::Config::read(path).unwrap_or_else(|e| panic!("{e}"));
    Args {
        config: path.to_path_buf(),
        mode: c.mode,
        chains: Vec::new(),
        command: "run".into(),
        packages: root().join("artifacts").join("live").join(&c.size),
        prompt: Some(c.prompt.clone()),
        send: None,
        cut: Cut::Same,
        seed: c.seed,
        chunks: 12,
        reference: root().join("artifacts").join("live").join("480x864").join("reference.safetensors"),
        scale: 1.0,
        serial: false,
        offloading: c.offloading.clone(),
        recalibrate: false,
        right: false,
    }
}

/// The flags of the command line over the arguments `a`.
fn flags(a: &mut Args, it: &mut impl Iterator<Item = String>) {
    while let Some(arg) = it.next() {
        let mut value = || it.next().unwrap_or_else(|| panic!("{arg} needs a value"));
        match arg.as_str() {
            "check" | "bench" | "run" | "render" => a.command = arg.clone(),
            "--config" => { value(); }
            "--chains" => a.chains = value().split(',').map(|c| c.trim().to_string()).filter(|c| !c.is_empty()).collect(),
            "--packages" => a.packages = value().into(),
            // the packages of one resolution, exported by export_live.py --height --width into artifacts/live/HxW
            "--size" => a.packages = root().join("artifacts").join("live").join(value()),
            "--prompt" => a.prompt = Some(value()),
            "--send" => a.send = Some(value()),
            "--scene" => a.cut = Cut::Scene,
            "--new" => a.cut = Cut::New,
            "--seed" => a.seed = value().parse().expect("--seed"),
            "--chunks" => a.chunks = value().parse().expect("--chunks"),
            "--reference" => a.reference = value().into(),
            "--scale" => a.scale = value().parse().expect("--scale"),
            "--right" => a.right = true,
            "--serial" => a.serial = true,
            // a resource's tier over configs/live.json: --tier kv_cache=vram, --tier block_weights=disk
            "--tier" => {
                let v = value();
                let (resource, tier) = v.split_once('=').unwrap_or_else(|| panic!("--tier resource=tier, not {v}"));
                let quoted = |s: &str| format!("\"{s}\"");
                let resource: config::Resource = serde_json::from_str(&quoted(resource)).expect("--tier: a resource");
                let tier: config::Tier = serde_json::from_str(&quoted(tier)).expect("--tier: vram, ram, disk or auto");
                a.offloading.resources.insert(resource, tier);
                a.offloading.validate().unwrap_or_else(|e| panic!("--tier {v}: {e}"));
            }
            "--no-offloading" => a.offloading.enabled = false,
            // calibrate again, not taking the placement kept in <packages>/calibration.json
            "--recalibrate" => a.recalibrate = true,
            "-h" | "--help" => {
                println!("{}", include_str!("main.rs").lines().take_while(|l| l.starts_with("//!"))
                    .map(|l| l.trim_start_matches("//!").trim_start_matches(' ')).collect::<Vec<_>>().join("\n"));
                std::process::exit(0);
            }
            other => panic!("unknown argument {other}"),
        }
    }
}

/// LibTorch, the operators live::* and the GPU, then the video and T5 over the packages.
fn load(args: &Args) -> Result<(Video, T5, DeviceBuffer)> {
    let operators: Vec<PathBuf> = ["live_w4a4", "live_sage"].iter()
        .map(|n| root().join("native").join("ops").join("build").join(format!("{n}.dll"))).collect();
    let runtime = aoti::Runtime::load(&aoti::Runtime::default_torch_lib(), &operators)?;
    let gpu = Gpu::new()?;
    gpu.bind()?;
    #[cfg(feature = "debug-tools")]
    let used = |what: &str| {
        let (free, total) = gpu.memory().unwrap_or((0, 0));
        let (allocated, _, _) = runtime.memory_peak(gpu::DEVICE);
        println!("GPU used after {what}: {:.0} MiB (torch allocated {:.0} MiB)", (total - free) as f64 / 1048576.0,
                 allocated as f64 / 1048576.0);
    };
    #[cfg(feature = "debug-tools")]
    used("the context and the operators");
    let video = Video::load(&args.packages, runtime.clone(), args.seed, &args.offloading, args.recalibrate)?;
    #[cfg(feature = "debug-tools")]
    used("the video's packages and buffers");
    let spec = video.m.t5.clone().ok_or("live.json has no T5: export it with export_live.py")?;
    let t5 = T5::load(&spec, video.m.text_len, video.m.text_dim, &runtime, &args.packages, &video.compute,
                      &args.offloading.t5)?;
    #[cfg(feature = "debug-tools")]
    used("T5");
    let embeddings = DeviceBuffer::new(video.m.text_len * video.m.text_dim * 4)?;
    Ok((video, t5, embeddings))
}

/// The prompt as the model takes it: a shot's prompt begins as LongLive 2.0 marks a cut.
pub fn model_prompt(video: &Video, p: &Prompt) -> String {
    if p.cut == Cut::Scene { format!("{}{}", video.m.scene_prefix, p.text) } else { p.text.clone() }
}

fn set_prompt(video: &mut Video, t5: &mut T5, embeddings: &DeviceBuffer, p: &Prompt) -> Result<f64> {
    let start = Instant::now();
    t5.encode(&model_prompt(video, p), embeddings, &video.compute)?;
    #[cfg(feature = "debug-tools")]
    let encoded = start.elapsed().as_secs_f64();
    let tensor = t5.tensor(embeddings);
    video.set_prompt(&tensor, p.cut)?;
    #[cfg(feature = "debug-tools")]
    {
        video.compute.synchronize()?;
        let all = start.elapsed().as_secs_f64();
        println!("a prompt: T5 {:.0} ms ({} from the disk), text embedding and cross-attention keys of 30 blocks {:.0} ms",
                 encoded * 1e3, t5.copy_waits()?, (all - encoded) * 1e3);
    }
    Ok(start.elapsed().as_secs_f64())
}

fn main() -> Result<()> {
    let args = parse();
    if let Some(text) = &args.send {
        prompts::send(args.cut, text, live_protocol::PORT)?;
        return Ok(());
    }
    if args.mode == config::Mode::Offline {
        let text = std::fs::read_to_string(&args.config)?;
        let argv: Vec<String> = std::env::args().collect();
        return offline::run(&args, &text, &argv, load(&args)?);
    }
    match args.command.as_str() {
        #[cfg(feature = "debug-tools")]
        "check" => measure::check(&args.reference, load(&args)?),
        #[cfg(feature = "debug-tools")]
        "render" => measure::render(&args.reference, load(&args)?),
        #[cfg(not(feature = "debug-tools"))]
        "check" | "render" => Err("check and render are built with --features debug-tools".into()),
        "bench" => measure::bench(load(&args)?, args.prompt.as_deref().unwrap_or(window::DEFAULT_PROMPT), args.chunks,
                                  args.serial),
        _ => window::run(args, load),
    }
}
