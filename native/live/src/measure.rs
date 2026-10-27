//! `live check`: the chunks of the host against Live's Python path on the same noise and prompts
//! (scripts/conversion/live_reference.py); `live bench`: the time of a chunk and the memory of the process.

use std::path::Path;
use std::time::Instant;

use live_protocol::Cut;
use safetensors::SafeTensors;

use crate::gpu::DeviceBuffer;
use crate::prompts::Prompt;
use crate::t5::T5;
use crate::video::{Result, Video};

/// Private committed bytes and working set of this process.
pub fn process_memory() -> (usize, usize) {
    #[cfg(windows)]
    {
        #[repr(C)]
        #[derive(Default)]
        struct Counters {
            cb: u32,
            page_fault_count: u32,
            peak_working_set: usize,
            working_set: usize,
            quota_peak_paged: usize,
            quota_paged: usize,
            quota_peak_non_paged: usize,
            quota_non_paged: usize,
            pagefile_usage: usize,
            peak_pagefile_usage: usize,
            private_usage: usize,
        }
        #[link(name = "kernel32")]
        extern "system" {
            fn GetCurrentProcess() -> *mut std::ffi::c_void;
            fn K32GetProcessMemoryInfo(process: *mut std::ffi::c_void, counters: *mut Counters, cb: u32) -> i32;
        }
        let mut c = Counters { cb: std::mem::size_of::<Counters>() as u32, ..Default::default() };
        unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
        (c.private_usage, c.working_set)
    }
    #[cfg(not(windows))]
    {
        (0, 0)
    }
}

const GIB: f64 = (1u64 << 30) as f64;

pub fn memory_line(video: &Video) -> String {
    let (private, working) = process_memory();
    let (free, total) = crate::gpu::Gpu::new().and_then(|g| g.memory()).unwrap_or((0, 0));
    let (allocated, reserved) = video.runtime().memory(crate::gpu::DEVICE);
    format!("RAM private {:.2} GiB, working set {:.2} GiB, pinned rows {:.2} GiB | GPU used {:.2} GiB, torch allocated {:.2} reserved {:.2} GiB",
            private as f64 / GIB, working as f64 / GIB, video.pinned_bytes() as f64 / GIB,
            (total - free) as f64 / GIB, allocated as f64 / GIB, reserved as f64 / GIB)
}

fn relative(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut n) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += ((x - y) as f64).powi(2);
        n += (*y as f64).powi(2);
    }
    (d / n.max(1e-30)).sqrt()
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let mse = a.iter().zip(b).map(|(x, y)| ((*x as f64) - (*y as f64)).powi(2)).sum::<f64>() / a.len() as f64;
    if mse == 0.0 { f64::INFINITY } else { 10.0 * (255.0 * 255.0 / mse).log10() }
}

/// The host on the reference's script: T5 against FastTextEncoder, then chunk by chunk the clean latents and the
/// frames against the Python path's.
pub fn check(reference: &Path, (mut video, mut t5, embeddings): (Video, T5, DeviceBuffer)) -> Result<()> {
    let file = std::fs::File::open(reference)?;
    let map = unsafe { memmap2::Mmap::map(&file)? };
    let (_, metadata) = SafeTensors::read_metadata(&map)?;
    let r = SafeTensors::deserialize(&map)?;
    let meta = metadata.metadata().clone().ok_or("the reference has no metadata")?;
    let script: Vec<(String, Option<String>)> = serde_json::from_str(&meta["script"])?;
    let prompts: std::collections::HashMap<String, String> = serde_json::from_str(&meta["prompts"])?;
    println!("reference: {} (eager: {})", reference.display(), meta.get("eager").map(String::as_str).unwrap_or("?"));

    println!("| T5 prompt | relative L2 to FastTextEncoder | s |");
    println!("|---|---:|---:|");
    let mut host = vec![0f32; video.m.text_len * video.m.text_dim];
    for name in ["A", "B", "C"] {
        let start = Instant::now();
        t5.encode(&prompts[name], &embeddings, &video.compute)?;
        let seconds = start.elapsed().as_secs_f64();
        video.compute.copy_to_host(bytes_mut(&mut host), embeddings.ptr)?;
        video.compute.synchronize()?;
        let want = floats(r.tensor(&format!("embedding_{name}"))?.data());
        println!("| {name} | {:.2e} | {seconds:.2} |", relative(&host, &want));
    }

    // the decoder alone, on the reference's latents of chunk 0
    {
        let x0 = r.tensor("x0_0")?;
        let buffer = DeviceBuffer::new(x0.data().len())?;
        video.compute.copy_to_device(buffer.ptr, x0.data())?;
        let shape: Vec<i64> = x0.shape().iter().map(|&s| s as i64).collect();
        let frames = video.decode(&buffer.tensor(0, aoti::DType::F32, &shape))?;
        if let Ok(path) = std::env::var("LIVE_DUMP") {
            std::fs::write(path, frames.concat())?;
        }
        println!("decoder on the reference's x0 of chunk 0: PSNR {:.1} dB", psnr(&frames.concat(), r.tensor("frames_0")?.data()));
        video.reset_decoder();
    }
    println!("| chunk | prompt | x0 relative L2 | frames | PSNR, dB | DiT + decoder, s |");
    println!("|---:|---|---:|---:|---:|---:|");
    let mut current: Option<&str> = None;
    let mut worst = 0f64;
    for (i, (name, cut)) in script.iter().enumerate() {
        if current != Some(name.as_str()) {
            let cut = match cut.as_deref() { Some("shot") => Cut::Scene, Some("new") => Cut::New, _ => Cut::Same };
            // the reference's embeddings, so that the DiT is compared on its own
            let want = r.tensor(&format!("embedding_{name}"))?;
            video.compute.copy_to_device(embeddings.ptr, want.data())?;
            video.set_prompt(&t5.tensor(&embeddings), cut)?;
            current = Some(name);
        }
        let noise = floats(r.tensor(&format!("noise_{i}"))?.data());
        let start = Instant::now();
        let x0 = video.next_chunk(Some(&noise))?;
        let frames = video.decode(&x0)?;
        let seconds = start.elapsed().as_secs_f64();
        let mut got = vec![0f32; x0.numel()];
        video.compute.copy_to_host(bytes_mut(&mut got), x0.data as u64)?;
        video.compute.synchronize()?;
        let error = relative(&got, &floats(r.tensor(&format!("x0_{i}"))?.data()));
        worst = worst.max(error);
        let quality = match r.tensor(&format!("frames_{i}")) {
            Ok(want) => {
                let flat: Vec<u8> = frames.concat();
                assert_eq!(flat.len(), want.data().len(), "frames of chunk {i}");
                format!("{:.1}", psnr(&flat, want.data()))
            }
            Err(_) => "n/a".into(),
        };
        println!("| {i} | {name}{} | {error:.2e} | {} | {quality} | {seconds:.2} |",
                 cut.as_ref().map(|c| format!(", {c}")).unwrap_or_default(), frames.len());
    }
    println!("{}", memory_line(&video));
    println!("worst x0 relative L2: {worst:.2e}");
    Ok(())
}

/// Chunks of one prompt without a window: seconds of each, fps, memory.
pub fn bench((mut video, mut t5, embeddings): (Video, T5, DeviceBuffer), prompt: &str, chunks: usize) -> Result<()> {
    let p = Prompt { cut: Cut::Same, text: prompt.into() };
    let t = crate::set_prompt(&mut video, &mut t5, &embeddings, &p)?;
    println!("{}", memory_line(&video));
    println!("T5 and cross-attention: {t:.2} s");
    let mut times = Vec::new();
    for i in 0..chunks {
        let start = Instant::now();
        let x0 = video.next_chunk(None)?;
        let frames = video.decode(&x0)?;
        let seconds = start.elapsed().as_secs_f64();
        println!("chunk {i}: {seconds:.3} s, {} frames, {:.1} fps", frames.len(), frames.len() as f64 / seconds);
        times.push((seconds, frames.len()));
    }
    let steady: Vec<_> = times.iter().skip(4).collect(); // the ring is full from the fourth chunk on
    if !steady.is_empty() {
        let s: f64 = steady.iter().map(|t| t.0).sum::<f64>() / steady.len() as f64;
        println!("steady chunk: {s:.3} s, {:.2} fps", steady[0].1 as f64 / s);
    }
    println!("{}", memory_line(&video));
    Ok(())
}

fn bytes_mut<T>(values: &mut [T]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, std::mem::size_of_val(values)) }
}

/// For every video_<prompt>_<seed>.safetensors of `directory` (live_reference.py --render): the same video by the
/// host, a new video of its prompt on its noise, its frames into rust_<prompt>_<seed>.bin (uint8 [T, H, W, 3]).
pub fn render(directory: &Path, (mut video, mut t5, embeddings): (Video, T5, DeviceBuffer)) -> Result<()> {
    let mut names: Vec<_> = std::fs::read_dir(directory)?.flatten().map(|e| e.path())
        .filter(|p| p.file_name().unwrap().to_string_lossy().starts_with("video_") && p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    names.sort();
    for path in names {
        let file = std::fs::File::open(&path)?;
        let map = unsafe { memmap2::Mmap::map(&file)? };
        let (_, metadata) = SafeTensors::read_metadata(&map)?;
        let meta = metadata.metadata().clone().ok_or("no metadata")?;
        let r = SafeTensors::deserialize(&map)?;
        let chunks: usize = meta["chunks"].parse()?;
        crate::set_prompt(&mut video, &mut t5, &embeddings, &Prompt { cut: Cut::New, text: meta["prompt"].clone() })?;
        video.reset_decoder();
        let mut frames = Vec::new();
        let start = Instant::now();
        for i in 0..chunks {
            let noise = floats(r.tensor(&format!("noise_{i}"))?.data());
            let x0 = video.next_chunk(Some(&noise)).map_err(|e| format!("chunk {i}: {e}"))?;
            frames.extend(video.decode(&x0).map_err(|e| format!("decode {i}: {e}"))?);
        }
        let out = directory.join(path.file_name().unwrap().to_string_lossy().replace("video_", "rust_").replace(".safetensors", ".bin"));
        std::fs::write(&out, frames.concat())?;
        println!("{}: {} frames, {:.1} s", out.display(), frames.len(), start.elapsed().as_secs_f64());
    }
    Ok(())
}
