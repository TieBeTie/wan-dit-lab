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
    crate::stage::enter(crate::stage::Step::Runtime);
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
    crate::stage::enter(crate::stage::Step::Video);
    let mut video = Video::load(&args.packages(), manifest, runtime.clone(), args.pipeline.generate.seed, &args.pipeline.generate.offloading)?;
    if video.m.hw_rows > 0 {
        let table = crate::rope::hw_table(&args.pipeline.export.rope, (video.m.height, video.m.width), video.m.head_dim, video.m.hw_rows)?;
        video.set_hw_rope(&table)?;
    }
    let g = &args.pipeline.generate;
    video.set_descent(&args.packages(), g.descent.as_ref(), &g.sampling, args.pipeline.export.sampling.timestep_shift,
                      g.debug.includes(crate::config::DebugTool::Descent), g.seed)?;
    video.set_kv_write(g.debug.includes(crate::config::DebugTool::KvWrite), g.kv_probe.as_deref().unwrap_or(&[]))?;
    video.set_cache_write(g.cache_write);
    #[cfg(feature = "debug-tools")]
    used("the video's packages and buffers");
    crate::stage::enter(crate::stage::Step::T5);
    let spec = video.m.t5.clone().ok_or("mvp1.json has no T5: export it with export_live.py")?;
    let mut t5_weights = args.pipeline.generate.offloading.t5.clone();
    if t5_weights.resident.as_deref() == Some("auto") {
        if video.m.generation_workspace_bytes == 0 {
            return Err("auto T5 residency requires measured generation workspace; re-export packages".into());
        }
        let (free, total) = gpu.memory()?;
        let reserve = args.pipeline.generate.offloading.memory_reserve(total as usize, spec.activation_bytes, spec.row_bytes)?;
        let budget = t5_resident_budget(free as usize, reserve, video.m.generation_workspace_bytes, video.gpu_claim_bytes());
        t5_weights.resident_gib = budget as f64 / (1u64 << 30) as f64;
        println!("auto T5: measured activations {:.0} MiB, resident budget {:.0} MiB",
                 spec.activation_bytes as f64 / 1048576.0, budget as f64 / 1048576.0);
    }
    let mut t5 = T5::load(&spec, video.m.text_len, video.m.text_dim, &runtime, &args.packages(), &video.compute,
                      &t5_weights)?;
    let (_, total) = gpu.memory()?;
    let off = &args.pipeline.generate.offloading;
    let reserve = off.margin_bytes(total as usize) + off.activation_bytes(spec.activation_bytes)? + video.m.generation_workspace_bytes;
    let key = crate::config::Placement::key(off, &std::fs::read_to_string(args.packages().join("mvp1.json"))?);
    t5.configure_prefetch(off, reserve, &args.packages(), key, video.t5_slot_depth.clone(), &video.compute)?;
    #[cfg(feature = "debug-tools")]
    used("T5");
    let embeddings = DeviceBuffer::new(video.m.text_len * video.m.text_dim * 4)?;
    Ok((video, t5, embeddings))
}

/// GPU bytes T5's resident rows may take: what is free beyond the reserve, the model's workspace and what the
/// calibration may still put on the GPU for the model (T5 runs once per prompt, the model every chunk).
fn t5_resident_budget(free: usize, reserve: usize, workspace: usize, model_claim: usize) -> usize {
    free.saturating_sub(reserve + workspace + model_claim)
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


#[cfg(test)]
mod tests {
    #[test]
    fn t5_gets_only_what_the_dit_will_not_take() {
        const G: usize = 1 << 30;
        // 6 GiB free, 1.5 reserve, 1 workspace: without a DiT claim T5 took 3.5 GiB and pushed weights and KV out
        assert_eq!(super::t5_resident_budget(6 * G, 3 * G / 2, G, 0), 7 * G / 2);
        assert_eq!(super::t5_resident_budget(6 * G, 3 * G / 2, G, 3 * G), G / 2);
        assert_eq!(super::t5_resident_budget(6 * G, 3 * G / 2, G, 5 * G), 0);
    }
}
