//! The video, chunk by chunk: causal.Stream of the preset longlive2 over the packages of export_live.py.
//!
//! A chunk is the sampler's two steps (each a pass of the model at timesteps 999 and 833, then x - c flow) and a
//! pass at t = 0 that writes the clean chunk into the caches, then the commit of the ring. The prompt changes
//! between chunks in one of three ways (live_protocol::Cut). Of causal.Stream only what Live runs is here: the
//! sampler UniPC with 2 steps (two linear steps, export_live.causal_unipc), positions in the video, cross-attention
//! keys and values once per prompt, and of the ways to change the prompt swap, shot and new.

use std::path::Path;

use aoti::{DType, Output, Runtime, Tensor};
use live_protocol::Cut;
use rand::SeedableRng;
use rand_distr::Distribution;

use crate::cache::{Buffers, Offloaded, Ring, Write};
use crate::gpu::{DeviceBuffer, Event, PinnedBuffer, Stream};

const SCALAR: usize = 256;
use crate::config::{choose, Offloading, Placement, Resource, Tier, Times};
use crate::manifest::Manifest;
use crate::packages::{self, Loaded};
use crate::rows::{spread, Rows};

pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, Error>;

struct Packages {
    text: Loaded,
    cross: Loaded,
    pre: Loaded,
    blocks: Vec<Loaded>,
    post: Loaded,
    decoder: Loaded,
}

/// What the calibration may decide: GPU bytes to leave free, the spare of copies, the resources it may offload.
#[derive(Clone, Copy)]
struct Calibrating {
    reserve: usize,
    /// what must stay free on the GPU whatever the margin: T5's slots and activations
    essential: usize,
    spare: f32,
    weights: bool,
    kv: bool,
}

/// The KV caches of the blocks: on the GPU, or offloaded to pinned RAM.
enum Kv {
    Resident(Vec<Buffers>),
    Offloaded(Offloaded),
}

/// The calibration of the offloading (Video::calibrate): armed at load, measuring during the first pass with a full
/// cache, applied at the start of the next chunk.
enum Calibration {
    Off,
    Armed(Calibrating),
    /// the blocks' times and the pass's of the first pass with a full cache, every resource offloaded
    Measuring(Vec<(Event, Event)>, (Event, Event), Calibrating),
    /// the trial of the KV cache, with the weights already placed: a pass with the cache offloaded (None), then one
    /// with it on the GPU (Some: the offloaded pass's milliseconds)
    Trial(Option<f32>),
    TrialMeasuring((Event, Event), Option<f32>),
    Done,
}

pub struct Video {
    pub m: Manifest,
    runtime: Runtime,
    p: Packages,
    rows: Rows,
    kv: Kv,
    pub ring: Ring,
    cross: Vec<Vec<Output>>,
    /// the time pairs of RoPE, [2, positions, pairs] float32, as export_live.py wrote them
    time_rope: Vec<f32>,
    time_rows: DeviceBuffer,
    /// t and c of the packages, one float32 each, SCALAR bytes apart: a package takes its inputs 16-byte aligned (an
    /// input off that is copied by AOTInductor first, with a warning on every call)
    scalars: DeviceBuffer,
    /// the pixels of the chunk being decoded, page-locked so that the copies are asynchronous
    pixels: PinnedBuffer,
    /// recorded after the last copy of the chunk being decoded; (offset, frames) of each latent frame's pixels
    decoded: Event,
    pending: Vec<(usize, usize)>,
    calibration: Calibration,
    /// where and under which key the calibration's placement is kept
    placement: (std::path::PathBuf, String),
    noise: DeviceBuffer,
    latent: DeviceBuffer,
    memory: Vec<Output>,
    zeros: Vec<DeviceBuffer>,
    trim: usize,
    /// latent frame of the next chunk in the video
    pub start: usize,
    rope_offset: usize,
    pin_next: bool,
    rng: rand::rngs::StdRng,
    pub compute: Stream,
}

fn f32_bytes(values: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) }
}

impl Video {
    /// `offloading` (configs/live.json): which resources are offloaded, and whether the calibration chooses which of
    /// them stay so (Video::calibrate) or they stay as configured.
    /// The placement a calibration of the same configuration and packages left in <directory>/calibration.json is
    /// taken as it is, unless `recalibrate`: every resource is made on its tier at once and nothing moves.
    pub fn load(directory: &Path, runtime: Runtime, seed: u64, offloading: &Offloading, recalibrate: bool)
        -> Result<Video> {
        let mut m = Manifest::read(directory)?;
        let key = Placement::key(offloading, &std::fs::read_to_string(directory.join("live.json"))?);
        let calibrates = Resource::ALL.iter().any(|&r| offloading.calibrates(r));
        let kept = if calibrates && !recalibrate { Placement::read(directory, &key) } else { None };
        let (weights, kv_tier) = match &kept {
            Some(p) => (p.weights, p.kv),
            None => (offloading.initial(Resource::BlockWeights), offloading.initial(Resource::KvCache)),
        };
        m.stream_blocks = match (&kept, weights) {
            (_, Tier::Vram) => 0,
            (Some(p), Tier::Ram) => p.rows_streamed,
            _ => m.blocks,
        };
        if let Some(p) = &kept {
            println!("offloading as calibrated before ({}): weights {:?} ({} streamed), KV cache {:?}",
                     directory.join("calibration.json").display(), p.weights, p.rows_streamed, p.kv);
        }
        // left free on the GPU when rows stay there: the margin, T5's two row slots and its activations
        let gib = |g: f64| (g * (1u64 << 30) as f64) as usize;
        let t5_slots = 2 * m.t5.as_ref().map_or(0, |t| t.row_bytes);
        let reserve = gib(offloading.vram_margin_gib) + t5_slots + gib(offloading.t5_activations_gib);
        let calibrate = (calibrates && kept.is_none()).then(|| Calibrating {
            reserve,
            essential: t5_slots + gib(offloading.t5_activations_gib),
            spare: offloading.copy_spare,
            weights: offloading.calibrates(Resource::BlockWeights),
            kv: offloading.calibrates(Resource::KvCache),
        });
        let compute = Stream::new()?;
        let package = |name: &str| packages::load(&runtime, directory, name, &compute);
        let p = Packages {
            text: { let mut t = package("text")?; t.park(&compute)?; t },
            cross: package("cross")?,
            pre: package("pre")?,
            blocks: (0..m.block_steps).map(|k| package(&format!("block_{k}"))).collect::<Result<_>>()?,
            post: package("post")?,
            decoder: package("decoder")?,
        };
        let rows = Rows::load(&m.pack, m.blocks, m.row_bytes, &spread(m.blocks, m.stream_blocks), weights == Tier::Disk,
                              &compute)?;
        let tokens = m.max_tokens + m.chunk_tokens;
        // an offloaded cache is made in RAM: on the GPU first, 704x1280 would not fit beside the slots
        let kv = if kv_tier == Tier::Ram {
            Kv::Offloaded(Offloaded::new(m.blocks, tokens, m.heads, m.head_dim, &compute)?)
        } else {
            Kv::Resident((0..m.blocks).map(|_| Buffers::new(tokens, m.heads, m.head_dim, &compute))
                .collect::<std::result::Result<_, _>>()?)
        };
        let bytes = std::fs::read(directory.join("time_rope.bin"))?;
        let time_rope: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        assert_eq!(time_rope.len(), 2 * m.positions * m.time_pairs);
        let latent_bytes = m.latent.iter().product::<usize>() * 4;
        let zeros = m.decoder_memory.iter()
            .map(|shape| DeviceBuffer::zeroed(shape.iter().product::<i64>() as usize * 2, &compute))
            .collect::<std::result::Result<_, _>>()?;
        Ok(Video {
            ring: Ring::new(m.max_tokens, m.sink_tokens, m.chunk, m.tokens_per_frame),
            time_rows: DeviceBuffer::new(2 * m.chunk * m.time_pairs * 4)?,
            scalars: DeviceBuffer::new(2 * SCALAR)?,
            pixels: PinnedBuffer::new(4 * m.chunk * m.height * m.width * 3)?,
            decoded: Event::new()?,
            pending: Vec::new(),
            calibration: match calibrate { Some(c) => Calibration::Armed(c), None => Calibration::Off },
            placement: (directory.to_path_buf(), key),
            noise: DeviceBuffer::new(latent_bytes)?,
            latent: DeviceBuffer::new(latent_bytes / m.chunk)?,
            trim: m.decoder_trim,
            m,
            runtime,
            p,
            rows,
            kv,
            cross: Vec::new(),
            time_rope,
            memory: Vec::new(),
            zeros,
            start: 0,
            rope_offset: 0,
            pin_next: false,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            compute,
        })
    }

    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    /// The calibration of the offloading: as little memory as there can be without losing speed. It takes the times
    /// of a pass with the cache full (the slowest blocks) and every block streamed. A streamed row is copied while
    /// the blocks before it compute: with k of the n blocks streamed and spread evenly, a copy has n / k blocks' time
    /// to hide behind, so k = n t_block / t_copy (a tenth to spare) is the most that can be streamed without waiting,
    /// and only the other blocks get rows on the GPU for good. Those are capped by the room the GPU has, `reserve`
    /// bytes left free.
    fn calibrate(&mut self, times: Vec<(Event, Event)>, whole: (Event, Event), c: Calibrating) -> Result<()> {
        self.compute.synchronize()?;
        let _ = whole; // the pass's own time is the trial's (decide_kv)
        let mut blocks: Vec<f32> = times.iter().map(|(a, b)| a.elapsed_ms(b)).collect::<std::result::Result<_, _>>()?;
        blocks.sort_by(f32::total_cmp);
        let block = blocks[blocks.len() / 2];
        // the times of copies taken alone: the cache's own copies would share the bus
        if let Kv::Offloaded(o) = &self.kv {
            o.synchronize()?;
        }
        let row = self.rows.time_copy()?;
        let n = self.m.blocks;
        // the read part of a block's cache, copied at the bandwidth a row was
        let kv_bytes = Offloaded::bytes_read(self.m.max_tokens, self.m.heads, self.m.head_dim);
        let kv_ms = row * kv_bytes as f32 / self.m.row_bytes as f32;
        let read = self.rows.time_read()?;
        let times = Times { block_ms: block, row_copy_ms: row, row_read_ms: read, kv_copy_ms: kv_ms };
        let (kv_tier, weights_tier, rows) = choose(n, &times, c.spare, c.kv, c.weights);
        if kv_tier == Tier::Vram {
            if let Kv::Offloaded(_) = self.kv {
                let Kv::Offloaded(o) = std::mem::replace(&mut self.kv, Kv::Resident(Vec::new())) else { unreachable!() };
                self.kv = Kv::Resident(o.into_resident(&self.compute)?);
            }
        }
        let mut note = "";
        let weights_where = match weights_tier {
            Tier::Disk => {
                self.rows.to_disk(&self.compute)?;
                "from the disk".to_string()
            }
            Tier::Ram | Tier::Vram => {
                self.rows.to_ram(&self.compute)?; // from the disk, if the weights started there
                let (free, _) = cudarc::driver::result::mem_get_info()?;
                let fit = free.saturating_sub(c.reserve) / self.m.row_bytes; // rows the GPU still has room for
                let streamed = self.rows.streamed();
                let keep = rows.min(streamed).max(streamed.saturating_sub(fit));
                if keep < streamed {
                    self.rows.keep_on_device(&spread(n, keep), &self.compute)?;
                    self.m.stream_blocks = keep;
                }
                if keep > rows {
                    note = ", fewer on the GPU than needed: it has no room";
                }
                format!("{keep} of {n} from RAM")
            }
            Tier::Auto => "as configured".to_string(),
        };
        // the copies hide by the times, but they can still slow the blocks down (they share the GPU's memory): with
        // the weights placed, a pass with the cache offloaded and one with it on the GPU (if there is room) are
        // compared, and it stays offloaded only if it costs no more than a percent (decide_kv)
        if let (Tier::Ram, Kv::Offloaded(_)) = (kv_tier, &self.kv) {
            let (free, _) = cudarc::driver::result::mem_get_info()?;
            let resident = self.m.blocks * Offloaded::bytes_read(self.m.max_tokens + self.m.chunk_tokens, self.m.heads,
                                                                 self.m.head_dim);
            // speed comes first: the trial runs if the cache fits beside what T5 needs, margin or not
            if free >= resident + c.essential {
                self.calibration = Calibration::Trial(None);
            }
        }
        let kv_where = match (kv_tier, &self.kv) {
            (Tier::Auto, _) => "as configured",
            (_, Kv::Offloaded(_)) if matches!(self.calibration, Calibration::Trial(_)) => "in RAM, on trial against the GPU",
            (_, Kv::Offloaded(_)) => "in RAM",
            (_, Kv::Resident(_)) => "on the GPU",
        };
        println!("offloading calibrated: block {block:.1} ms, row copy {row:.1} ms, row read from the disk {read:.1} ms, \
                  cache copy {kv_ms:.1} ms; weights {weights_where}, KV cache {kv_where}{note}");
        if !matches!(self.calibration, Calibration::Trial(_)) {
            self.calibration = Calibration::Done;
            self.keep_placement()?;
        }
        Ok(())
    }

    /// The calibration is done: where it put every resource, for the next run (calibration.json).
    fn keep_placement(&self) -> Result<()> {
        let weights = if self.rows.on_disk() {
            Tier::Disk
        } else if self.rows.streamed() > 0 {
            Tier::Ram
        } else {
            Tier::Vram
        };
        let kv = match self.kv { Kv::Offloaded(_) => Tier::Ram, Kv::Resident(_) => Tier::Vram };
        let (directory, key) = &self.placement;
        Placement { key: key.clone(), weights, rows_streamed: self.rows.streamed(), kv }.write(directory)?;
        Ok(())
    }

    /// The end of the trial: the KV cache stays offloaded unless the pass with it on the GPU was faster by more than
    /// a percent (as little memory as there can be at the most speed).
    fn decide_kv(&mut self, whole: (Event, Event), offloaded_ms: f32) -> Result<()> {
        self.compute.synchronize()?;
        let resident_ms = whole.0.elapsed_ms(&whole.1)?;
        let keep_offloaded = offloaded_ms <= 1.01 * resident_ms;
        if keep_offloaded {
            let Kv::Resident(caches) = std::mem::replace(&mut self.kv, Kv::Resident(Vec::new())) else { unreachable!() };
            self.kv = Kv::Offloaded(Offloaded::from_resident(caches, &self.compute)?);
        }
        println!("offloading calibrated: a pass {offloaded_ms:.0} ms with the KV cache in RAM, {resident_ms:.0} ms on the GPU; \
                  the cache {}", if keep_offloaded { "in RAM" } else { "on the GPU" });
        self.calibration = Calibration::Done;
        self.keep_placement()
    }

    /// debug-tools: the late copies of streamed rows since the last call (Rows::copy_waits).
    #[cfg(feature = "debug-tools")]
    pub fn copy_waits(&self) -> Result<(usize, usize, f32)> {
        Ok(self.rows.copy_waits(&self.compute)?)
    }

    /// debug-tools: the waits for copies of streamed rows the GPU has passed since the last call, without waiting.
    #[cfg(feature = "debug-tools")]
    pub fn passed_copy_waits(&self) -> Result<crate::rows::CopyWaits> {
        Ok(self.rows.passed_copy_waits()?)
    }

    /// GPU bytes the KV caches take: all of them when resident, two slots when offloaded (bench).
    pub fn kv_device_bytes(&self) -> usize {
        let bytes = |c: &Buffers| c.k.bytes + c.ks.bytes + c.v.bytes + c.vs.bytes;
        match &self.kv {
            Kv::Resident(caches) => caches.iter().map(bytes).sum(),
            Kv::Offloaded(o) => o.slot_bytes(),
        }
    }

    /// Pinned RAM of the offloaded resources: weight rows and KV caches.
    pub fn pinned_bytes(&self) -> usize {
        self.rows.pinned_bytes() + match &self.kv {
            Kv::Offloaded(o) => o.pinned_bytes(),
            Kv::Resident(_) => 0,
        }
    }

    fn scalar(&self, index: usize, value: f32) -> Result<Tensor> {
        self.compute.copy_to_device(self.scalars.at(index * SCALAR), &value.to_le_bytes())?;
        Ok(self.scalars.tensor(index * SCALAR, DType::F32, &[1]))
    }

    /// The prompt from its T5 embeddings [1, text_len, text_dim] float32 on the GPU, changed in the way of `cut`.
    pub fn set_prompt(&mut self, embeddings: &Tensor, cut: Cut) -> Result<()> {
        let s = self.compute.raw();
        // the text package is used only here: parked between prompts
        self.p.text.unpark(&self.compute)?;
        let text = self.p.text.run(&[*embeddings], s)?.remove(0);
        let mut cross = Vec::with_capacity(self.m.blocks);
        for b in 0..self.m.blocks {
            let row = self.rows.row(b, &self.compute)?;
            cross.push(self.p.cross.run(&[row, *text], s)?);
            self.rows.release(b, &self.compute)?;
        }
        self.cross = cross;
        drop(text);
        self.p.text.park(&self.compute)?;
        match cut {
            Cut::Same => {}
            Cut::Scene => {
                // LongLive 2.0's shot: positions move on, and the shot's first chunk is pinned in the ring
                self.rope_offset += self.m.shot_rope_offset;
                self.pin_next = true;
            }
            Cut::New => {
                self.ring.reset();
                self.start = 0;
                self.rope_offset = 0;
                self.pin_next = false;
            }
        }
        Ok(())
    }

    /// One pass of the model over the chunk `x` at timestep `t`: the tokens after the last block and the time
    /// embedding.
    /// `step` and `at`: the block package and where the chunk is written (the ring before this chunk's commit);
    /// `write`: the pass that writes the clean chunk, with the commit's copies (an offloaded cache does them here).
    fn pass(&mut self, x: Tensor, t: f32, step: usize, at: usize, write: Option<&Write>) -> Result<(Output, Output)> {
        let s = self.compute.raw();
        let t = self.scalar(0, t)?;
        let mut pre = self.p.pre.run(&[x, t], s)?.into_iter();
        let (mut tokens, e0, t_emb) = (pre.next().unwrap(), pre.next().unwrap(), pre.next().unwrap());
        let package = &self.p.blocks[step];
        if let Kv::Offloaded(o) = &mut self.kv {
            o.begin_pass(at)?;
        }
        let time = self.time_rows.tensor(0, DType::F32, &[2, self.m.chunk as i64, self.m.time_pairs as i64]);
        // the calibration times the blocks of the first pass that reads a full cache
        let full = step + 1 == self.m.block_steps;
        let mut times = match self.calibration {
            Calibration::Armed(..) if full => Some(Vec::with_capacity(self.m.blocks)),
            _ => None,
        };
        let whole = match self.calibration {
            Calibration::Armed(..) | Calibration::Trial(_) if full => {
                let (start, end) = (Event::timed()?, Event::timed()?);
                start.record(&self.compute)?;
                Some((start, end))
            }
            _ => None,
        };
        for b in 0..self.m.blocks {
            let row = self.rows.row(b, &self.compute)?;
            let [k, ks, v, vs] = match &self.kv {
                Kv::Resident(caches) => caches[b].tensors(),
                Kv::Offloaded(o) => o.tensors(b, &self.compute)?,
            };
            let cross = &self.cross[b];
            let span = match times {
                Some(_) => {
                    let (start, end) = (Event::timed()?, Event::timed()?);
                    start.record(&self.compute)?; // after the wait for the row: the block's own time
                    Some((start, end))
                }
                None => None,
            };
            let mut out = package.run(&[row, *tokens, *e0, *cross[0], *cross[1], time, k, ks, v, vs], s)?;
            if let (Some(times), Some((start, end))) = (&mut times, span) {
                end.record(&self.compute)?;
                times.push((start, end));
            }
            self.rows.release(b, &self.compute)?;
            if let Kv::Offloaded(o) = &mut self.kv {
                o.end_block(b, &self.compute, write)?;
            }
            tokens = out.remove(0);
        }
        if let Some((start, end)) = whole {
            end.record(&self.compute)?;
            self.calibration = match (times, &self.calibration) {
                (Some(times), Calibration::Armed(c)) => Calibration::Measuring(times, (start, end), *c),
                (_, Calibration::Trial(a)) => Calibration::TrialMeasuring((start, end), *a),
                _ => unreachable!(),
            };
        }
        Ok((tokens, t_emb))
    }

    /// The latents [C, chunk, H, W] float32 of the next chunk, from `noise` of that shape or fresh noise.
    pub fn next_chunk(&mut self, noise: Option<&[f32]>) -> Result<Output> {
        match std::mem::replace(&mut self.calibration, Calibration::Done) {
            Calibration::Measuring(times, whole, c) => self.calibrate(times, whole, c)?,
            Calibration::TrialMeasuring(whole, None) => {
                // the offloaded pass is measured: the next one with the cache on the GPU
                self.compute.synchronize()?;
                let offloaded = whole.0.elapsed_ms(&whole.1)?;
                let Kv::Offloaded(o) = std::mem::replace(&mut self.kv, Kv::Resident(Vec::new())) else { unreachable!() };
                self.kv = Kv::Resident(o.into_resident(&self.compute)?);
                self.calibration = Calibration::Trial(Some(offloaded));
            }
            Calibration::TrialMeasuring(whole, Some(offloaded)) => self.decide_kv(whole, offloaded)?,
            other => self.calibration = other,
        }
        let m = &self.m;
        let s0 = self.start + self.rope_offset;
        let (pairs, chunk) = (m.time_pairs, m.chunk);
        assert!(s0 + chunk <= m.positions, "the video is longer than the RoPE table");
        let mut rows = Vec::with_capacity(2 * chunk * pairs);
        for half in 0..2 {
            let from = (half * m.positions + s0) * pairs;
            rows.extend_from_slice(&self.time_rope[from..from + chunk * pairs]);
        }
        self.compute.copy_to_device(self.time_rows.ptr, f32_bytes(&rows))?;
        let count = m.latent.iter().product::<usize>();
        let fresh: Vec<f32>;
        let noise = match noise {
            Some(n) => n,
            None => {
                let normal = rand_distr::StandardNormal;
                fresh = (0..count).map(|_| normal.sample(&mut self.rng)).collect();
                &fresh
            }
        };
        assert_eq!(noise.len(), count);
        self.compute.copy_to_device(self.noise.ptr, f32_bytes(noise))?;
        let shape: Vec<i64> = self.m.latent.iter().map(|&v| v as i64).collect();
        let mut x = self.noise.tensor(0, DType::F32, &shape);
        let mut kept: Option<Output> = None;
        let (step, at) = (self.ring.step(), self.ring.saved);
        for k in 0..self.m.timesteps.len() {
            let (tokens, t_emb) = self.pass(x, self.m.timesteps[k], step, at, None)?;
            let c = self.scalar(1, self.m.sampler[k])?;
            let next = self.p.post.run(&[*tokens, *t_emb, x, c], self.compute.raw())?.remove(0);
            x = *next;
            kept = Some(next);
        }
        // the clean chunk written into the caches at t = 0, as trained (write_pass), and kept (the commit); the pass's
        // output is not needed
        let copies = self.ring.commit();
        let write = Write { at, n: self.m.chunk_tokens, copies: &copies };
        drop(self.pass(x, 0.0, step, at, Some(&write))?);
        if let Kv::Resident(caches) = &self.kv {
            for cache in caches {
                cache.copy(&copies, &self.compute)?;
            }
        }
        if self.pin_next {
            self.ring.pin_last();
            self.pin_next = false;
        }
        self.start += self.m.chunk;
        Ok(kept.unwrap())
    }

    /// The decoder from the start of a video again: no memory, the first frames trimmed.
    pub fn reset_decoder(&mut self) {
        self.memory.clear();
        self.trim = self.m.decoder_trim;
    }

    /// The frames of a chunk's latents as uint8 pixels [H, W, 3] each (TAEHVStream, one latent frame at a time).
    pub fn decode(&mut self, latents: &Tensor) -> Result<Vec<Vec<u8>>> {
        self.start_decode(latents)?;
        self.finish_decode()
    }

    /// Queues the decoding of a chunk and the copies of its pixels to the host; nothing waits for them. The next
    /// chunk can be queued behind it before finish_decode, so that the GPU does not wait for the host in between.
    pub fn start_decode(&mut self, latents: &Tensor) -> Result<()> {
        assert!(self.pending.is_empty(), "a chunk is still being decoded");
        let [channels, chunk, h, w] = self.m.latent;
        let plane = h * w * 4;
        let frame_bytes = self.m.height * self.m.width * 3;
        let mut offset = 0;
        for f in 0..chunk {
            // latents[:, f] contiguous: one row of h w floats per channel
            self.compute.copy_rows(self.latent.ptr, plane, latents.data as u64 + (f * plane) as u64, chunk * plane,
                                   plane, channels)?;
            let latent = self.latent.tensor(0, DType::F32, &[channels as i64, 1, h as i64, w as i64]);
            let mut inputs = vec![latent];
            if self.memory.is_empty() {
                for (buffer, shape) in self.zeros.iter().zip(&self.m.decoder_memory) {
                    inputs.push(buffer.tensor(0, DType::F16, shape));
                }
            } else {
                inputs.extend(self.memory.iter().map(|o| **o));
            }
            let mut out = self.p.decoder.run(&inputs, self.compute.raw())?;
            let pixels = out.remove(0);
            self.memory = out;
            let count = pixels.shape()[0] as usize;
            let bytes = count * frame_bytes;
            let host = &mut self.pixels.as_mut_slice()[offset..offset + bytes];
            self.compute.copy_to_host(host, pixels.data as u64)?;
            self.pending.push((offset, count));
            offset += bytes;
        }
        self.decoded.record(&self.compute)?;
        Ok(())
    }

    /// The frames of the chunk start_decode queued, once its copies are done.
    pub fn finish_decode(&mut self) -> Result<Vec<Vec<u8>>> {
        self.decoded.synchronize()?;
        let frame_bytes = self.m.height * self.m.width * 3;
        let mut frames = Vec::with_capacity(4 * self.m.chunk);
        for (offset, count) in std::mem::take(&mut self.pending) {
            for i in self.trim.min(count)..count {
                let at = offset + i * frame_bytes;
                frames.push(self.pixels.as_slice()[at..at + frame_bytes].to_vec());
            }
            self.trim = 0;
        }
        Ok(frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    /// The video of the packages of 480x864, the operators and LibTorch loaded once per test.
    fn video(offloading: &Offloading) -> Video {
        let operators: Vec<std::path::PathBuf> = ["live_w4a4", "live_sage"].iter()
            .map(|n| crate::root().join("native").join("ops").join("build").join(format!("{n}.dll"))).collect();
        let runtime = aoti::Runtime::load(&aoti::Runtime::default_torch_lib(), &operators).unwrap();
        // the primary context lives as long as the test process (a Gpu dropped would release it)
        static GPU: std::sync::OnceLock<crate::gpu::Gpu> = std::sync::OnceLock::new();
        GPU.get_or_init(|| crate::gpu::Gpu::new().unwrap()).bind().unwrap();
        let directory = crate::root().join("artifacts").join("live").join("480x864");
        let mut v = Video::load(&directory, runtime, 0, offloading, true).unwrap();
        // a prompt of zeros: what is compared is the cache, not the text
        let text = DeviceBuffer::zeroed(v.m.text_len * v.m.text_dim * 4, &v.compute).unwrap();
        let tensor = text.tensor(0, DType::F32, &[1, v.m.text_len as i64, v.m.text_dim as i64]);
        v.set_prompt(&tensor, Cut::Same).unwrap();
        v
    }

    /// The clean latents of `chunks` chunks on noise of seed 7.
    fn latents(v: &mut Video, chunks: usize) -> Vec<Vec<f32>> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let count = v.m.latent.iter().product::<usize>();
        (0..chunks).map(|_| {
            let noise: Vec<f32> = (0..count).map(|_| rand_distr::StandardNormal.sample(&mut rng)).collect();
            let x0 = v.next_chunk(Some(&noise)).unwrap();
            let mut host = vec![0f32; count];
            let bytes = unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr() as *mut u8, count * 4) };
            v.compute.copy_to_host(bytes, x0.data as u64).unwrap();
            v.compute.synchronize().unwrap();
            host
        }).collect()
    }

    fn relative(a: &[f32], b: &[f32]) -> f64 {
        let d: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
        let n: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
        (d / n).sqrt()
    }

    fn tiers(weights: Tier, kv: Tier) -> Offloading {
        let mut o = Config::default().offloading;
        o.resources.insert(Resource::BlockWeights, weights);
        o.resources.insert(Resource::KvCache, kv);
        o
    }

    /// The text package is used only when a prompt is set: between prompts its weights are off the GPU, and with
    /// them back from the disk it gives the text embedding it gave, bit for bit (the cross-attention keys after it
    /// differ in about one byte in a million between any two runs: the W4A4 GEMM is not deterministic).
    /// Needs the GPU and artifacts/live/480x864: cargo test -- --ignored
    #[test]
    #[ignore]
    fn the_text_package_is_off_the_gpu_between_prompts() {
        let mut v = video(&Config::default().offloading);
        assert_eq!(v.p.text.device_bytes(), 0, "the text package's weights stay on the GPU after the prompt");
        let embeddings = DeviceBuffer::zeroed(v.m.text_len * v.m.text_dim * 4, &v.compute).unwrap();
        let tensor = embeddings.tensor(0, DType::F32, &[1, v.m.text_len as i64, v.m.text_dim as i64]);
        let mut text = || {
            v.p.text.unpark(&v.compute).unwrap();
            let t = v.p.text.run(&[tensor], v.compute.raw()).unwrap().remove(0);
            let mut host = vec![0u8; t.bytes()];
            v.compute.copy_to_host(&mut host, t.data as u64).unwrap();
            v.p.text.park(&v.compute).unwrap();
            host
        };
        let (first, again) = (text(), text());
        assert!(!first.is_empty() && first == again, "the text embedding differs once the weights came back");
        v.set_prompt(&tensor, Cut::Same).unwrap();
        assert_eq!(v.p.text.device_bytes(), 0);
    }

    /// The KV cache in RAM gives chunks as close to those of the cache on the GPU as two runs of the cache on the GPU
    /// are to each other (the pipeline is not deterministic: Sage sums with atomics, and INT4 rounding amplifies the
    /// last bits, so two identical runs part by about half of the latents' norm), through the fill of the ring and
    /// its first evictions (6 chunks); and of the cache only two slots stay on the GPU. Needs the GPU and
    /// artifacts/live/480x864: cargo test -- --ignored
    #[test]
    #[ignore]
    fn the_kv_cache_in_ram_gives_the_chunks_of_the_cache_on_the_gpu() {
        let chunks = 6;
        let (resident, resident_bytes) = {
            let mut v = video(&tiers(Tier::Ram, Tier::Vram));
            (latents(&mut v, chunks), v.kv_device_bytes())
        };
        let again = latents(&mut video(&tiers(Tier::Ram, Tier::Vram)), chunks);
        let (offloaded, offloaded_bytes) = {
            let mut v = video(&tiers(Tier::Ram, Tier::Ram));
            (latents(&mut v, chunks), v.kv_device_bytes())
        };
        let (mut errors, mut noises) = (0.0, 0.0);
        for c in 0..chunks {
            let noise = relative(&again[c], &resident[c]);
            let error = relative(&offloaded[c], &resident[c]);
            println!("chunk {c}: offloaded {error:.2e}, two resident runs {noise:.2e}");
            errors += error;
            noises += noise;
        }
        assert!(errors <= 1.5 * noises, "offloaded {errors:.3} against two resident runs {noises:.3}");
        let m = Manifest::read(&crate::root().join("artifacts").join("live").join("480x864")).unwrap();
        let one = Offloaded::bytes_read(m.max_tokens + m.chunk_tokens, m.heads, m.head_dim);
        println!("KV cache on the GPU: {resident_bytes} resident, {offloaded_bytes} offloaded");
        assert_eq!(resident_bytes, m.blocks * one);
        assert_eq!(offloaded_bytes, 2 * one);
    }

    /// The transport of an offloaded cache alone, deterministic: over chunks that fill the ring, evict, pin a shot
    /// and start a new video, every block's cache in RAM after the pass that writes equals a resident cache that took
    /// the same chunks, byte for byte (the chunks written as known bytes in place of a block package). Needs the GPU.
    #[test]
    #[ignore]
    fn an_offloaded_cache_keeps_what_a_resident_one_keeps() {
        static GPU: std::sync::OnceLock<crate::gpu::Gpu> = std::sync::OnceLock::new();
        GPU.get_or_init(|| crate::gpu::Gpu::new().unwrap()).bind().unwrap();
        let (blocks, per, chunk, heads, head_dim) = (3, 5, 8, 2, 16);
        let (max_tokens, sink) = (24 * per, 8 * per);
        let n = chunk * per;
        let compute = Stream::new().unwrap();
        let new = || Buffers::new(max_tokens + n, heads, head_dim, &compute).unwrap();
        let resident: Vec<Buffers> = (0..blocks).map(|_| new()).collect();
        let mut offloaded = Offloaded::from_resident((0..blocks).map(|_| new()).collect(), &compute).unwrap();
        let mut ring = Ring::new(max_tokens, sink, chunk, per);
        let token = [heads * head_dim, heads * 4, heads * head_dim, heads * 4];
        // what a block package writes at `at`: bytes from the pass, block and buffer numbers
        let bytes = |pass: usize, b: usize, i: usize| -> Vec<u8> {
            (0..n * token[i]).map(|j| (pass * 31 + b * 7 + i * 3 + j) as u8).collect()
        };
        let write = |ptrs: [u64; 4], at: usize, pass: usize, b: usize| {
            for i in 0..4 {
                compute.copy_to_device(ptrs[i] + (at * token[i]) as u64, &bytes(pass, b, i)).unwrap();
            }
        };
        let script = ["c", "c", "c", "c", "c", "shot", "c", "c", "c", "new", "c", "c", "c", "c"];
        let (mut c, mut pin) = (0, false);
        for op in script {
            match op {
                "shot" => pin = true,
                "new" => ring.reset(),
                _ => {
                    let at = ring.saved;
                    let copies = ring.clone().commit();
                    for pass in 0..3 {
                        let last = pass == 2;
                        let w = Write { at, n, copies: &copies };
                        offloaded.begin_pass(at).unwrap();
                        for b in 0..blocks {
                            let t = offloaded.tensors(b, &compute).unwrap();
                            write([0, 1, 2, 3].map(|i| t[i].data as u64), at, c * 3 + pass, b);
                            offloaded.end_block(b, &compute, last.then_some(&w)).unwrap();
                            let r = &resident[b];
                            write([r.k.ptr, r.ks.ptr, r.v.ptr, r.vs.ptr], at, c * 3 + pass, b);
                        }
                    }
                    assert_eq!(ring.commit(), copies);
                    for r in &resident {
                        r.copy(&copies, &compute).unwrap();
                    }
                    if pin {
                        ring.pin_last();
                        pin = false;
                    }
                    c += 1;
                    // the kept part of every cache, [0, saved): in RAM and on the GPU alike
                    compute.synchronize().unwrap();
                    let back = offloaded.clone_host().unwrap();
                    for b in 0..blocks {
                        let r = &resident[b];
                        for (i, device) in [&r.k, &r.ks, &r.v, &r.vs].into_iter().enumerate() {
                            let len = ring.saved * token[i];
                            let mut want = vec![0u8; len];
                            compute.copy_to_host(&mut want, device.ptr).unwrap();
                            compute.synchronize().unwrap();
                            assert!(back[b][i][..len] == want[..], "chunk {c}, block {b}, buffer {i}");
                        }
                    }
                }
            }
        }
    }
}
