//! Native runtime: loads validated packages after preparation has finished.
use std::path::PathBuf;
#[cfg(feature = "debug-tools")]
use std::time::Instant;
use live_protocol::Cut;
use crate::{root, Args};
use crate::gpu::{DeviceBuffer, Gpu};
use crate::prompts::Prompt;
use crate::t5::T5;
use crate::video::{Result, Video};
use crate::manifest;

/// LibTorch, the operators live::* and the GPU, then the video and T5 over the packages.
pub fn load(args: &Args) -> Result<(Video, T5, DeviceBuffer)> {
    crate::budget::configure(args.pipeline.generate.offloading.ram_margin_fraction);
    let manifest = manifest::Manifest::read_for(&args.packages(), &args.pipeline.export.kernels)?;
    let operators: Vec<PathBuf> = ["live_w4a4", "live_sage"].iter()
        .map(|n| root().join("ops").join("build").join(format!("{n}.dll"))).collect();
    let runtime = aoti::Runtime::load(&aoti::Runtime::default_torch_lib(), &operators)?;
    let gpu = Gpu::new()?;
    gpu.bind()?;
    #[cfg(feature = "debug-tools")]
    let used = |what: &str| {
        let (free, total) = gpu.memory().unwrap_or((0, 0));
        let (allocated, _, _) = runtime.memory_peak(crate::gpu::DEVICE);
        println!("GPU used after {what}: {:.0} MiB (torch allocated {:.0} MiB)", (total - free) as f64 / 1048576.0,
                 allocated as f64 / 1048576.0);
    };
    #[cfg(feature = "debug-tools")]
    used("the context and the operators");
    let video = Video::load(&args.packages(), manifest, runtime.clone(), args.pipeline.generate.seed, &args.pipeline.generate.offloading)?;
    #[cfg(feature = "debug-tools")]
    used("the video's packages and buffers");
    let spec = video.m.t5.clone().ok_or("live.json has no T5: export it with export_live.py")?;
    let mut t5_weights = args.pipeline.generate.offloading.t5.clone();
    if t5_weights.resident.as_deref() == Some("auto") {
        if video.m.generation_workspace_bytes == 0 {
            return Err("auto T5 residency requires measured generation workspace; re-export packages".into());
        }
        let (free, total) = gpu.memory()?;
        let reserve = args.pipeline.generate.offloading.memory_reserve(total as usize, spec.activation_bytes, spec.row_bytes)?;
        let budget = (free as usize).saturating_sub(reserve + video.m.generation_workspace_bytes);
        t5_weights.resident_gib = budget as f64 / (1u64 << 30) as f64;
        println!("auto T5: measured activations {:.0} MiB, resident budget {:.0} MiB",
                 spec.activation_bytes as f64 / 1048576.0, budget as f64 / 1048576.0);
    }
    let t5 = T5::load(&spec, video.m.text_len, video.m.text_dim, &runtime, &args.packages(), &video.compute,
                      &t5_weights)?;
    #[cfg(feature = "debug-tools")]
    used("T5");
    let embeddings = DeviceBuffer::new(video.m.text_len * video.m.text_dim * 4)?;
    Ok((video, t5, embeddings))
}

/// The prompt as the model takes it: a shot's prompt begins as LongLive 2.0 marks a cut.
pub fn model_prompt(video: &Video, p: &Prompt) -> String {
    if p.cut == Cut::Scene { format!("{}{}", video.m.scene_prefix, p.text) } else { p.text.clone() }
}

#[cfg(feature = "debug-tools")]
pub fn set_prompt(video: &mut Video, t5: &mut T5, embeddings: &DeviceBuffer, p: &Prompt) -> Result<f64> {
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
        println!("a prompt: T5 {:.0} ms, text embedding and cross-attention keys of 30 blocks {:.0} ms",
                 encoded * 1e3, (all - encoded) * 1e3);
    }
    Ok(start.elapsed().as_secs_f64())
}

