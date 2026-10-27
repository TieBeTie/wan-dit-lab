//! Live: a video generated without end, chunk by chunk, and a prompt over a local socket (or typed in the window)
//! turns it, from the frames already shown. The Rust host of scripts/live/live_video.py (preset longlive2): the model
//! is the AOTInductor packages of scripts/conversion/export_live.py, everything around it is here.
//!
//!     live --prompt "A lone oak tree on a grassy hill sways in the wind"
//!     live --send "Snow starts to fall on the oak tree" [--scene | --new]
//!     live check [--reference artifacts/live/reference.safetensors]   the chunks against the Python path
//!     live bench --chunks 12                                          time and memory, no window
//!     live render --reference <directory>                             the videos of live_reference.py --render
//!
//! Two threads. The main one shows the frames at the rate they are made (24 fps at most), so that the video slows
//! down rather than stops; the other encodes a waiting prompt (the newest), generates a chunk, decodes it and queues
//! its frames. A prompt is plain text (the scene goes on), "/scene ..." (LongLive 2.0's cut to a new shot) or
//! "/new ..." (a new video). Esc or closing the window stops it.

mod cache;
mod gpu;
mod manifest;
mod measure;
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
    scale: f32,
    right: bool,
}

fn root() -> PathBuf {
    // native/live -> wan-dit-lab
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

fn parse() -> Args {
    let mut a = Args {
        command: "run".into(),
        packages: root().join("artifacts").join("live"),
        prompt: None,
        send: None,
        cut: Cut::Same,
        seed: 0,
        chunks: 12,
        reference: root().join("artifacts").join("live").join("reference.safetensors"),
        scale: 1.0,
        right: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = || it.next().unwrap_or_else(|| panic!("{arg} needs a value"));
        match arg.as_str() {
            "check" | "bench" | "run" | "render" => a.command = arg.clone(),
            "--packages" => a.packages = value().into(),
            "--prompt" => a.prompt = Some(value()),
            "--send" => a.send = Some(value()),
            "--scene" => a.cut = Cut::Scene,
            "--new" => a.cut = Cut::New,
            "--seed" => a.seed = value().parse().expect("--seed"),
            "--chunks" => a.chunks = value().parse().expect("--chunks"),
            "--reference" => a.reference = value().into(),
            "--scale" => a.scale = value().parse().expect("--scale"),
            "--right" => a.right = true,
            "-h" | "--help" => {
                println!("{}", include_str!("main.rs").lines().take_while(|l| l.starts_with("//!"))
                    .map(|l| l.trim_start_matches("//!").trim_start_matches(' ')).collect::<Vec<_>>().join("\n"));
                std::process::exit(0);
            }
            other => panic!("unknown argument {other}"),
        }
    }
    a
}

/// LibTorch, the operators live::* and the GPU, then the video and T5 over the packages.
fn load(args: &Args) -> Result<(Video, T5, DeviceBuffer)> {
    let operators: Vec<PathBuf> = ["live_w4a4", "live_sage"].iter()
        .map(|n| root().join("native").join("ops").join("build").join(format!("{n}.dll"))).collect();
    let runtime = aoti::Runtime::load(&aoti::Runtime::default_torch_lib(), &operators)?;
    let gpu = Gpu::new()?;
    gpu.bind()?;
    let video = Video::load(&args.packages, runtime.clone(), args.seed)?;
    let spec = video.m.t5.clone().ok_or("live.json has no T5: export it with export_live.py")?;
    let t5 = T5::load(&spec, video.m.text_len, video.m.text_dim, &runtime, &args.packages, &video.compute)?;
    let embeddings = DeviceBuffer::new(video.m.text_len * video.m.text_dim * 4)?;
    Ok((video, t5, embeddings))
}

/// The prompt as the model takes it: a shot's prompt begins as LongLive 2.0 marks a cut.
fn model_prompt(video: &Video, p: &Prompt) -> String {
    if p.cut == Cut::Scene { format!("{}{}", video.m.scene_prefix, p.text) } else { p.text.clone() }
}

fn set_prompt(video: &mut Video, t5: &mut T5, embeddings: &DeviceBuffer, p: &Prompt) -> Result<f64> {
    let start = Instant::now();
    t5.encode(&model_prompt(video, p), embeddings, &video.compute)?;
    let tensor = t5.tensor(embeddings);
    video.set_prompt(&tensor, p.cut)?;
    Ok(start.elapsed().as_secs_f64())
}

fn main() -> Result<()> {
    let args = parse();
    if let Some(text) = &args.send {
        prompts::send(args.cut, text, live_protocol::PORT)?;
        return Ok(());
    }
    match args.command.as_str() {
        "check" => measure::check(&args.reference, load(&args)?),
        "render" => measure::render(&args.reference, load(&args)?),
        "bench" => measure::bench(load(&args)?, args.prompt.as_deref().unwrap_or(window::DEFAULT_PROMPT), args.chunks),
        _ => window::run(args, load),
    }
}
