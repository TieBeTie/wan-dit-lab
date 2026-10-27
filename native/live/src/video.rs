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

use crate::cache::{Buffers, Ring};
use crate::gpu::{DeviceBuffer, Stream};
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

pub struct Video {
    pub m: Manifest,
    runtime: Runtime,
    p: Packages,
    rows: Rows,
    caches: Vec<Buffers>,
    pub ring: Ring,
    cross: Vec<Vec<Output>>,
    /// the time pairs of RoPE, [2, positions, pairs] float32, as export_live.py wrote them
    time_rope: Vec<f32>,
    time_rows: DeviceBuffer,
    /// t and c of the packages, two float32
    scalars: DeviceBuffer,
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
    pub fn load(directory: &Path, runtime: Runtime, seed: u64) -> Result<Video> {
        let m = Manifest::read(directory)?;
        let compute = Stream::new()?;
        let package = |name: &str| packages::load(&runtime, directory, name, &compute);
        let p = Packages {
            text: package("text")?,
            cross: package("cross")?,
            pre: package("pre")?,
            blocks: (0..m.block_steps).map(|k| package(&format!("block_{k}"))).collect::<Result<_>>()?,
            post: package("post")?,
            decoder: package("decoder")?,
        };
        let rows = Rows::load(&m.pack, m.blocks, m.row_bytes, &spread(m.blocks, m.stream_blocks), &compute)?;
        let tokens = m.max_tokens + m.chunk_tokens;
        let caches = (0..m.blocks).map(|_| Buffers::new(tokens, m.heads, m.head_dim, &compute))
            .collect::<std::result::Result<_, _>>()?;
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
            scalars: DeviceBuffer::new(8)?,
            noise: DeviceBuffer::new(latent_bytes)?,
            latent: DeviceBuffer::new(latent_bytes / m.chunk)?,
            trim: m.decoder_trim,
            m,
            runtime,
            p,
            rows,
            caches,
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

    pub fn pinned_bytes(&self) -> usize {
        self.rows.pinned_bytes()
    }

    fn scalar(&self, index: usize, value: f32) -> Result<Tensor> {
        self.compute.copy_to_device(self.scalars.at(index * 4), &value.to_le_bytes())?;
        Ok(self.scalars.tensor(index * 4, DType::F32, &[1]))
    }

    /// The prompt from its T5 embeddings [1, text_len, text_dim] float32 on the GPU, changed in the way of `cut`.
    pub fn set_prompt(&mut self, embeddings: &Tensor, cut: Cut) -> Result<()> {
        let s = self.compute.raw();
        let text = self.p.text.run(&[*embeddings], s)?.remove(0);
        let mut cross = Vec::with_capacity(self.m.blocks);
        for b in 0..self.m.blocks {
            let row = self.rows.row(b, &self.compute)?;
            cross.push(self.p.cross.run(&[row, *text], s)?);
            self.rows.release(b, &self.compute)?;
        }
        self.cross = cross;
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
    fn pass(&mut self, x: Tensor, t: f32) -> Result<(Output, Output)> {
        let s = self.compute.raw();
        let t = self.scalar(0, t)?;
        let mut pre = self.p.pre.run(&[x, t], s)?.into_iter();
        let (mut tokens, e0, t_emb) = (pre.next().unwrap(), pre.next().unwrap(), pre.next().unwrap());
        let step = &self.p.blocks[self.ring.step()];
        let time = self.time_rows.tensor(0, DType::F32, &[2, self.m.chunk as i64, self.m.time_pairs as i64]);
        for b in 0..self.m.blocks {
            let row = self.rows.row(b, &self.compute)?;
            let [k, ks, v, vs] = self.caches[b].tensors();
            let cross = &self.cross[b];
            let mut out = step.run(&[row, *tokens, *e0, *cross[0], *cross[1], time, k, ks, v, vs], s)?;
            self.rows.release(b, &self.compute)?;
            tokens = out.remove(0);
        }
        Ok((tokens, t_emb))
    }

    /// The latents [C, chunk, H, W] float32 of the next chunk, from `noise` of that shape or fresh noise.
    pub fn next_chunk(&mut self, noise: Option<&[f32]>) -> Result<Output> {
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
        for k in 0..self.m.timesteps.len() {
            let (tokens, t_emb) = self.pass(x, self.m.timesteps[k])?;
            let c = self.scalar(1, self.m.sampler[k])?;
            let next = self.p.post.run(&[*tokens, *t_emb, x, c], self.compute.raw())?.remove(0);
            x = *next;
            kept = Some(next);
        }
        // the clean chunk written into the caches at t = 0, as trained (write_pass); its output is not needed
        drop(self.pass(x, 0.0)?);
        let copies = self.ring.commit();
        for cache in &self.caches {
            cache.copy(&copies, &self.compute)?;
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
        let [channels, chunk, h, w] = self.m.latent;
        let plane = h * w * 4;
        let mut frames = Vec::with_capacity(4 * chunk);
        let frame_bytes = self.m.height * self.m.width * 3;
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
            let mut host = vec![0u8; count * frame_bytes];
            self.compute.copy_to_host(&mut host, pixels.data as u64)?;
            self.compute.synchronize()?;
            for i in self.trim.min(count)..count {
                frames.push(host[i * frame_bytes..(i + 1) * frame_bytes].to_vec());
            }
            self.trim = 0;
        }
        Ok(frames)
    }
}
