//! The KV caches of the 30 blocks and the one bookkeeping they share (causal.KVCache keeps a copy of it per block).
//!
//! A cache is [1, max_tokens + n, heads, head_dim] keys and values in FP8 with a scale per token and head. The first
//! sink tokens are the first chunk of the video, never evicted; the rest up to max_tokens is the ring. A chunk is
//! written at `at` (by the block package); while the ring fills `at` is where the saved tokens end and keeping the
//! chunk is moving the counter, once it is full the chunk is written into the last n tokens and kept by copying it
//! over the tokens of the oldest frames that are not pinned (LongLive 2.0 pins the first chunk of a shot until the
//! next shot). Of causal.KVCache only what Live runs is here: no Participative Compression, no positions in the
//! window, no forgetting, no Forcing-KV.

use crate::gpu::{DeviceBuffer, Event, PinnedBuffer, Stream};

use crate::video::Result;

pub fn check_host_capacity(required: usize, available: usize) -> crate::video::Result<()> {
    if required > available {
        return Err(format!("native buffers need {:.2} GiB pinned RAM, only {:.2} GiB available above the RAM reserve; reduce context/resolution or choose disk backing",
            required as f64 / (1u64 << 30) as f64, available as f64 / (1u64 << 30) as f64).into());
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format { Fp8, Bf16 }

impl Format {
    pub fn token_bytes(self, heads: usize, dim: usize) -> [usize; 4] {
        let value = heads * dim * if self == Self::Bf16 { 2 } else { 1 };
        // Keep the package's four-tensor ABI. Dense attention ignores the scales.
        [value, heads * 4, value, heads * 4]
    }

    pub fn bytes(self, tokens: usize, heads: usize, dim: usize) -> usize {
        tokens * self.token_bytes(heads, dim).iter().sum::<usize>()
    }

    fn dtype(self) -> aoti::DType {
        match self { Self::Fp8 => aoti::DType::F8E4M3, Self::Bf16 => aoti::DType::BF16 }
    }
}

/// Where a chunk goes and what keeping it copies.
#[derive(Clone, Debug)]
pub struct Ring {
    pub max_tokens: usize,
    pub sink: usize,
    pub n: usize,
    per_frame: usize,
    chunk: usize,
    /// tokens of finished frames in the buffer, and where the next chunk is written: right after them, the last n
    /// once they fill the ring
    pub saved: usize,
    /// latent frames committed so far
    frames: usize,
    frame_of: Vec<usize>,
    place: Vec<usize>,
    pinned: Vec<bool>,
    /// ring indices the last committed chunk went to
    pub last_slots: Option<Vec<usize>>,
}

/// Tokens [src, src + count) to [dst, dst + count) of every buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Copy {
    pub dst: usize,
    pub src: usize,
    pub count: usize,
}

impl Ring {
    pub fn new(max_tokens: usize, sink: usize, chunk: usize, per_frame: usize) -> Ring {
        let ring = max_tokens - sink;
        assert!(ring == 0 || ring >= chunk * per_frame, "the ring holds whole chunks");
        Ring {
            max_tokens,
            sink,
            n: chunk * per_frame,
            per_frame,
            chunk,
            saved: 0,
            frames: 0,
            frame_of: vec![0; ring],
            place: vec![0; ring],
            pinned: vec![false; ring],
            last_slots: None,
        }
    }

    /// The block package of the next chunk: its attention reads at + n tokens.
    pub fn step(&self) -> usize {
        self.saved / self.n
    }

    /// Keeps the chunk last written; returns the copies that do it (none while the ring fills).
    pub fn commit(&mut self) -> Vec<Copy> {
        let (n, at, per, sink) = (self.n, self.saved, self.per_frame, self.sink);
        let mut copies = Vec::new();
        let slots: Vec<(usize, usize)>; // (ring index, token of the chunk)
        if at + n <= self.max_tokens {
            self.saved += n;
            slots = (at.max(sink)..(at + n).max(sink)).map(|o| (o - sink, o - at)).collect();
        } else if self.max_tokens > sink {
            slots = self.evicted().into_iter().enumerate().map(|(i, slot)| (slot, i)).collect();
            for &(slot, i) in &slots {
                match copies.last_mut() {
                    Some(c @ Copy { .. }) if c.dst + c.count == sink + slot && c.src + c.count == at + i => c.count += 1,
                    _ => copies.push(Copy { dst: sink + slot, src: at + i, count: 1 }),
                }
            }
        } else {
            slots = Vec::new(); // the window is the chunk alone: nothing to keep
        }
        for &(slot, i) in &slots {
            self.frame_of[slot] = self.frames + i / per;
            self.place[slot] = i % per;
        }
        self.last_slots = Some(slots.iter().map(|&(slot, _)| slot).collect());
        self.frames += self.chunk;
        copies
    }

    /// The n ring indices of the oldest tokens that are not pinned, oldest first (topk of KVCache.evicted).
    fn evicted(&self) -> Vec<usize> {
        let order = |i: usize| if self.pinned[i] { usize::MAX } else { self.frame_of[i] * self.per_frame + self.place[i] };
        let mut indices: Vec<usize> = (0..self.frame_of.len()).collect();
        indices.sort_by_key(|&i| (order(i), i));
        indices.truncate(self.n);
        indices
    }

    /// The chunk just committed, the first of a new shot, stays in the ring until the next shot.
    pub fn pin_last(&mut self) {
        self.pinned.iter_mut().for_each(|p| *p = false);
        if let Some(slots) = &self.last_slots {
            assert!(slots.len() < self.pinned.len(), "the ring must hold more than the pinned chunk");
            for &slot in slots {
                self.pinned[slot] = true;
            }
        }
    }

    /// The static frame mask of block-sparse attention for the next chunk: for each block of `block` tokens of the
    /// buffer [0, max_tokens + n), 1 if the chunk's attention may run it. The chunk itself is always allowed; the
    /// sink, and a pinned chunk (the sink of its shot), only with `sink`; the other frames of the ring if they are
    /// among the `recent_frames` before the chunk. A block across frames is allowed if any of its tokens is.
    pub fn allowed_key_blocks(&self, sink: bool, recent_frames: usize, block: usize) -> Vec<i32> {
        let at = self.saved;
        let token = |t: usize| {
            if (at..at + self.n).contains(&t) {
                true
            } else if t < self.sink.min(at) {
                sink
            } else if t < at {
                let slot = t - self.sink;
                if self.pinned[slot] { sink } else { self.frame_of[slot] + recent_frames >= self.frames }
            } else {
                false
            }
        };
        (0..self.max_tokens + self.n).step_by(block)
            .map(|start| (start..(start + block).min(self.max_tokens + self.n)).any(token) as i32)
            .collect()
    }

    /// Empties the caches: a new video.
    pub fn reset(&mut self) {
        let fresh = Ring::new(self.max_tokens, self.sink, self.chunk, self.per_frame);
        *self = fresh;
    }
}

/// FP8 or BF16 keys and values, with scale tensors retained for the shared package ABI.
pub struct Buffers {
    pub k: DeviceBuffer,
    pub ks: DeviceBuffer,
    pub v: DeviceBuffer,
    pub vs: DeviceBuffer,
    pub shape: [i64; 4],
    pub format: Format,
}

impl Buffers {
    pub fn new(tokens: usize, heads: usize, head_dim: usize, format: Format, stream: &Stream) -> Result<Buffers> {
        let values = tokens * format.token_bytes(heads, head_dim)[0];
        let scales = tokens * heads * 4;
        Ok(Buffers {
            k: DeviceBuffer::zeroed(values, stream)?,
            ks: DeviceBuffer::zeroed(scales, stream)?,
            v: DeviceBuffer::zeroed(values, stream)?,
            vs: DeviceBuffer::zeroed(scales, stream)?,
            shape: [1, tokens as i64, heads as i64, head_dim as i64],
            format,
        })
    }

    /// The four tensors in the order of the block package: k, k scales, v, v scales.
    pub fn tensors(&self) -> [aoti::Tensor; 4] {
        let [b, t, h, d] = self.shape;
        [
            self.k.tensor(0, self.format.dtype(), &[b, t, h, d]),
            self.ks.tensor(0, aoti::DType::F32, &[b, t, h, 1]),
            self.v.tensor(0, self.format.dtype(), &[b, t, h, d]),
            self.vs.tensor(0, aoti::DType::F32, &[b, t, h, 1]),
        ]
    }

    /// The rows of `tokens` (FP8 keys, their scales, FP8 values, their scales), copied to the host after the work
    /// queued on `stream` (debug kv_write, kvwrite.rs).
    pub fn read_tokens(&self, tokens: &[usize], stream: &Stream) -> Result<(Vec<u8>, Vec<f32>, Vec<u8>, Vec<f32>)> {
        let [_, _, h, d] = self.shape;
        let (h, d) = (h as usize, d as usize);
        if self.format != Format::Fp8 {
            return Err("debug kv_write reads FP8 caches only".into());
        }
        let value = self.format.token_bytes(h, d)[0];
        let scale = h * 4;
        let (mut k, mut v) = (vec![0u8; tokens.len() * value], vec![0u8; tokens.len() * value]);
        let (mut ks, mut vs) = (vec![0u8; tokens.len() * scale], vec![0u8; tokens.len() * scale]);
        for (i, &t) in tokens.iter().enumerate() {
            stream.copy_to_host(&mut k[i * value..(i + 1) * value], self.k.at(t * value))?;
            stream.copy_to_host(&mut v[i * value..(i + 1) * value], self.v.at(t * value))?;
            stream.copy_to_host(&mut ks[i * scale..(i + 1) * scale], self.ks.at(t * scale))?;
            stream.copy_to_host(&mut vs[i * scale..(i + 1) * scale], self.vs.at(t * scale))?;
        }
        stream.synchronize()?;
        let floats = |b: Vec<u8>| b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<f32>>();
        Ok((k, floats(ks), v, floats(vs)))
    }

    /// Applies the copies of a commit on `stream`, after attention there has read the tokens.
    pub fn copy(&self, copies: &[Copy], stream: &Stream) -> Result<()> {
        let [_, _, h, d] = self.shape;
        let value = self.format.token_bytes(h as usize, d as usize)[0];
        let scale = h as usize * 4;
        for c in copies {
            for (buffer, bytes) in [(&self.k, value), (&self.v, value), (&self.ks, scale), (&self.vs, scale)] {
                stream.copy_on_device(buffer.at(c.dst * bytes), buffer.at(c.src * bytes), c.count * bytes)?;
            }
        }
        Ok(())
    }
}

/// The KV caches of the blocks kept in pinned RAM, with two sets of GPU buffers as slots (the KV cache offloaded,
/// configs/mvp1.json). Before a block, the part of its cache attention reads ([0, at)) is copied to a slot; after the
/// pass that writes the clean chunk, the commit's copies are done in the slot and only the tokens they wrote go back to
/// RAM (the other passes' writes of the chunk are overwritten by the next pass anyway). Copies to the GPU and back run
/// on streams of their own, a block ahead of compute.
pub struct Offloaded {
    host: Vec<[PinnedBuffer; 4]>,
    backing: Option<crate::kv_disk::Store>,
    blocks: usize,
    slots: [Buffers; 2],
    holds: [usize; 2],
    loaded: [Event; 2],
    freed: [Event; 2],
    written: Vec<Event>,
    up: Stream,
    down: Stream,
    read: usize,
    next: usize,
}

/// What the pass that writes the chunk leaves in a block's cache: the chunk at `at`, n tokens, and the commit's copies.
pub struct Write<'a> {
    pub at: usize,
    pub n: usize,
    pub copies: &'a [Copy],
}

impl Offloaded {
    /// `blocks` empty caches of `tokens` in pinned RAM, never on the GPU, with two slots of their size.
    pub fn new(blocks: usize, tokens: usize, heads: usize, head_dim: usize, format: Format, compute: &Stream) -> Result<Offloaded> {
        let sizes = format.token_bytes(heads, head_dim).map(|b| b * tokens);
        let host = (0..blocks).map(|_| {
            let buffers = sizes.map(|size| { let mut p = PinnedBuffer::new(size)?; p.as_mut_slice().fill(0); Ok::<_, cudarc::driver::DriverError>(p) });
            let [k, ks, v, vs] = buffers;
            Ok([k?, ks?, v?, vs?])
        }).collect::<Result<Vec<_>>>()?;
        Offloaded::with_host(host, tokens, heads, head_dim, format, compute)
    }

    pub fn new_disk(blocks: usize, tokens: usize, heads: usize, head_dim: usize, format: Format, compute: &Stream) -> Result<Self> {
        let mut cache = Self::new(2, tokens, heads, head_dim, format, compute)?;
        cache.backing = Some(crate::kv_disk::Store::new(blocks * format.bytes(tokens, heads, head_dim))?);
        cache.blocks = blocks;
        cache.written = (0..blocks).map(|_| Event::new()).collect::<std::result::Result<_, _>>()?;
        Ok(cache)
    }
    /// Measure actual KV uploads rather than extrapolating the weight-pack bandwidth.
    pub fn time_copy(&self) -> Result<f32> {
        self.synchronize()?;
        let bytes = self.token_bytes();
        let (start, end) = (Event::timed()?, Event::timed()?);
        let mut times = Vec::new();
        for _ in 0..3 {
            start.record(&self.up)?;
            for (i, device) in [&self.slots[0].k, &self.slots[0].ks, &self.slots[0].v, &self.slots[0].vs].iter().enumerate() {
                self.up.copy_to_device(device.ptr, &self.host[0][i].as_slice()[..self.read * bytes[i]])?;
            }
            end.record(&self.up)?; end.synchronize()?;
            times.push(start.elapsed_ms(&end)?);
        }
        times.sort_by(f32::total_cmp);
        Ok(times[1])
    }
    pub fn is_disk(&self) -> bool { self.backing.is_some() }
    pub fn disk_bytes(&self) -> (u64, u64) {
        self.backing.as_ref().map_or((0, 0), |s| (s.read_bytes, s.written))
    }
    /// Evict at a chunk boundary. Reuse two existing pinned blocks: no extra RAM peak.
    pub fn to_disk(&mut self) -> Result<()> {
        if self.is_disk() { return Ok(()); }
        self.synchronize()?;
        let row: usize = self.host[0].iter().map(|p| p.bytes).sum();
        let mut store = crate::kv_disk::Store::new(self.blocks * row)?;
        for (b, host) in self.host.iter().enumerate() {
            let mut offset = b * row;
            for p in host { store.write(offset, p.as_slice())?; offset += p.bytes; }
        }
        self.host.truncate(2);
        self.backing = Some(store);
        Ok(())
    }

    /// The caches of `resident` moved to pinned RAM (their GPU memory freed), with two slots of their size.
    pub fn from_resident(resident: Vec<Buffers>, compute: &Stream) -> Result<Offloaded> {
        compute.synchronize()?;
        let shape = resident[0].shape;
        let format = resident[0].format;
        let [_, tokens, heads, head_dim] = shape.map(|s| s as usize);
        let mut host = Vec::with_capacity(resident.len());
        for cache in resident {
            let mut pinned = Vec::with_capacity(4);
            for device in [&cache.k, &cache.ks, &cache.v, &cache.vs] {
                let mut buffer = PinnedBuffer::new(device.bytes)?;
                compute.copy_to_host(buffer.as_mut_slice(), device.ptr)?;
                pinned.push(buffer);
            }
            compute.synchronize()?; // before the device buffers go
            host.push(pinned.try_into().ok().unwrap());
        }
        Offloaded::with_host(host, tokens, heads, head_dim, format, compute)
    }

    fn with_host(host: Vec<[PinnedBuffer; 4]>, tokens: usize, heads: usize, head_dim: usize, format: Format, compute: &Stream)
        -> Result<Offloaded> {
        let blocks = host.len();
        let cache = Offloaded {
            host,
            backing: None,
            blocks,
            slots: [Buffers::new(tokens, heads, head_dim, format, compute)?, Buffers::new(tokens, heads, head_dim, format, compute)?],
            holds: [usize::MAX; 2],
            loaded: [Event::new()?, Event::new()?],
            freed: [Event::new()?, Event::new()?],
            written: (0..blocks).map(|_| Event::new()).collect::<std::result::Result<_, _>>()?,
            up: Stream::new()?,
            down: Stream::new()?,
            read: 0,
            next: 0,
        };
        compute.synchronize()?; // slot zeroing precedes copies on the independent transfer streams
        Ok(cache)
    }

    /// Bytes of one token in the four buffers, using the slots' value format.
    fn token_bytes(&self) -> [usize; 4] {
        let [_, _, h, d] = self.slots[0].shape.map(|s| s as usize);
        self.slots[0].format.token_bytes(h, d)
    }

    /// A pass that reads the tokens [0, read) of every cache begins: the first two blocks are loaded.
    pub fn begin_pass(&mut self, read: usize) -> Result<()> {
        self.read = read;
        self.next = 0;
        for slot in 0..2.min(self.blocks) {
            self.load(slot)?;
        }
        Ok(())
    }

    fn load(&mut self, slot: usize) -> Result<()> {
        let b = self.next;
        self.next += 1;
        // after the block that used the slot is done with it, and after this block's last write reached RAM
        self.up.wait(&self.freed[slot])?;
        self.up.wait(&self.written[b])?;
        let h = if self.backing.is_some() { slot } else { b };
        if self.backing.is_some() {
            self.up.synchronize()?;
            self.down.synchronize()?;
            let sizes = self.host[h].each_ref().map(|p| p.bytes);
            let row: usize = sizes.iter().sum();
            let token_bytes = self.token_bytes();
            let mut offset = b * row;
            for i in 0..4 {
                self.backing.as_mut().unwrap().read(offset, &mut self.host[h][i].as_mut_slice()[..self.read * token_bytes[i]])?;
                offset += sizes[i];
            }
        }
        let bytes = self.token_bytes();
        let device = [&self.slots[slot].k, &self.slots[slot].ks, &self.slots[slot].v, &self.slots[slot].vs];
        for i in 0..4 {
            let count = self.read * bytes[i];
            if count > 0 {
                self.up.copy_to_device(device[i].ptr, &self.host[h][i].as_slice()[..count])?;
            }
        }
        self.loaded[slot].record(&self.up)?;
        self.holds[slot] = b;
        Ok(())
    }

    /// The tensors of block `b`'s cache for work on `compute`, which waits for its load.
    pub fn tensors(&self, b: usize, compute: &Stream) -> Result<[aoti::Tensor; 4]> {
        let slot = self.holds.iter().position(|&h| h == b).expect("caches taken out of their order");
        compute.wait(&self.loaded[slot])?;
        Ok(self.slots[slot].tensors())
    }

    /// Block `b`'s work is queued on `compute`: in the pass that writes the chunk its tokens go back to RAM, then the
    /// slot loads the block two on.
    pub fn end_block(&mut self, b: usize, compute: &Stream, write: Option<&Write>) -> Result<()> {
        let slot = self.holds.iter().position(|&h| h == b).expect("caches taken out of their order");
        match write {
            Some(w) => {
                self.slots[slot].copy(w.copies, compute)?;
                let done = Event::new()?;
                done.record(compute)?;
                self.down.wait(&done)?;
                let ranges: Vec<(usize, usize)> = if w.copies.is_empty() {
                    vec![(w.at, w.n)]
                } else {
                    w.copies.iter().map(|c| (c.dst, c.count)).collect()
                };
                let bytes = self.token_bytes();
                let device = [self.slots[slot].k.ptr, self.slots[slot].ks.ptr, self.slots[slot].v.ptr, self.slots[slot].vs.ptr];
                let h = if self.backing.is_some() { slot } else { b };
                for &(start, count) in &ranges {
                    for i in 0..4 {
                        let (from, len) = (start * bytes[i], count * bytes[i]);
                        let host = &mut self.host[h][i].as_mut_slice()[from..from + len];
                        self.down.copy_to_host(host, device[i] + from as u64)?;
                    }
                }
                if self.backing.is_some() {
                    self.down.synchronize()?;
                    let sizes = self.host[h].each_ref().map(|p| p.bytes);
                    let row: usize = sizes.iter().sum();
                    for (start, count) in ranges {
                        let mut offset = b * row;
                        for i in 0..4 {
                            let from = start * bytes[i];
                            self.backing.as_mut().unwrap().write(offset + from, &self.host[h][i].as_slice()[from..from + count * bytes[i]])?;
                            offset += sizes[i];
                        }
                    }
                }
                self.written[b].record(&self.down)?;
                self.freed[slot].record(&self.down)?;
            }
            None => self.freed[slot].record(compute)?,
        }
        if self.next < self.blocks {
            self.load(slot)?;
        }
        Ok(())
    }

    /// The caches back on the GPU (the calibration found their copies could not hide), the pinned RAM freed.
    pub fn into_resident(self, compute: &Stream) -> Result<Vec<Buffers>> {
        if self.is_disk() { return Err("disk KV must stay streamed during calibration".into()); }
        compute.synchronize()?;
        self.up.synchronize()?;
        self.down.synchronize()?;
        let [_, tokens, heads, head_dim] = self.slots[0].shape.map(|s| s as usize);
        let mut caches = Vec::with_capacity(self.host.len());
        for host in &self.host {
            let cache = Buffers::new(tokens, heads, head_dim, self.slots[0].format, compute)?;
            for (device, pinned) in [&cache.k, &cache.ks, &cache.v, &cache.vs].into_iter().zip(host) {
                compute.copy_to_device(device.ptr, pinned.as_slice())?;
            }
            caches.push(cache);
        }
        compute.synchronize()?; // before the pinned buffers go
        Ok(caches)
    }

    pub fn pinned_bytes(&self) -> usize {
        self.host.iter().flat_map(|h| h.iter()).map(|p| p.bytes).sum()
    }

    /// GPU bytes of the two slots.
    pub fn slot_bytes(&self) -> usize {
        self.slots.iter().map(|s| s.k.bytes + s.ks.bytes + s.v.bytes + s.vs.bytes).sum()
    }

    /// Waits for every copy to and from the slots.
    pub fn synchronize(&self) -> Result<()> {
        self.up.synchronize()?;
        Ok(self.down.synchronize()?)
    }

    /// A copy of the caches in RAM, once every copy back is done (tests).
    #[cfg(test)]
    pub fn clone_host(&mut self) -> Result<Vec<[Vec<u8>; 4]>> {
        self.synchronize()?;
        if let Some(store) = &mut self.backing {
            let sizes = self.host[0].each_ref().map(|p| p.bytes);
            let row: usize = sizes.iter().sum();
            let mut result = Vec::new();
            for b in 0..self.blocks {
                let mut host = sizes.map(|n| vec![0; n]);
                let mut offset = b * row;
                for bytes in &mut host { store.read(offset, bytes)?; offset += bytes.len(); }
                result.push(host);
            }
            return Ok(result);
        }
        Ok(self.host.iter().map(|h| [0, 1, 2, 3].map(|i| h[i].as_slice().to_vec())).collect())
    }
}

impl Drop for Offloaded {
    fn drop(&mut self) {
        // Error paths must not free pinned buffers while a DMA still uses them.
        let _ = self.up.synchronize();
        let _ = self.down.synchronize();
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn oversized_pinned_cache_fails_before_allocation() {
        let required = Format::Bf16.bytes(32 * 22 * 40, 24, 128) * 30;
        assert_eq!(required, 10_543_104_000);
        let error = check_host_capacity(required, 8 << 30).unwrap_err().to_string();
        assert!(error.contains("9.82") && error.contains("RAM"), "{error}");
        assert!(check_host_capacity(1 << 30, 2 << 30).is_ok());
    }
    use super::*;

    #[test]
    fn dense_cache_accounts_for_two_bytes_per_value_in_every_transfer() {
        assert_eq!(Format::Bf16.token_bytes(24, 128), [6144, 96, 6144, 96]);
        assert_eq!(Format::Fp8.token_bytes(24, 128), [3072, 96, 3072, 96]);
        assert_eq!(Format::Bf16.bytes(100, 24, 128), 1_248_000);
    }

    #[derive(serde::Deserialize)]
    struct Commit {
        at: usize,
        saved: usize,
        slots: Option<Vec<usize>>,
    }

    #[derive(serde::Deserialize)]
    struct Fixture {
        tokens_per_frame: usize,
        chunk: usize,
        window: usize,
        sink: usize,
        script: Vec<String>,
        commits: Vec<Commit>,
    }

    /// The bookkeeping of causal.KVCache over chunks, shots and a new video (scripts/conversion/live_fixtures.py).
    #[test]
    fn commits_as_kv_cache_does() {
        let f: Fixture = serde_json::from_str(include_str!("../tests/fixtures/ring.json")).unwrap();
        let per = f.tokens_per_frame;
        let mut ring = Ring::new((f.window - f.chunk) * per, f.sink * per, f.chunk, per);
        let (mut commits, mut pin_next) = (f.commits.iter(), false);
        for op in &f.script {
            match op.as_str() {
                "shot" => pin_next = true,
                "new" => {
                    ring.reset();
                    pin_next = false;
                }
                _ => {
                    let want = commits.next().unwrap();
                    assert_eq!(ring.saved, want.at);
                    let at = ring.saved;
                    let copies = ring.commit();
                    if pin_next {
                        ring.pin_last();
                    }
                    pin_next = false;
                    assert_eq!(ring.saved, want.saved);
                    let slots = ring.last_slots.clone().unwrap();
                    assert_eq!(Some(&slots), want.slots.as_ref().or(Some(&vec![])));
                    // the copies put token i of the chunk at sink + slots[i]
                    let mut moved = vec![];
                    for c in &copies {
                        moved.extend((0..c.count).map(|j| (c.dst + j, c.src + j)));
                    }
                    if at + ring.n > ring.max_tokens {
                        let expected: Vec<_> = slots.iter().enumerate().map(|(i, s)| (ring.sink + s, at + i)).collect();
                        let mut sorted = moved.clone();
                        sorted.sort();
                        let mut expected_sorted = expected.clone();
                        expected_sorted.sort();
                        assert_eq!(sorted, expected_sorted);
                    } else {
                        assert!(copies.is_empty());
                    }
                }
            }
        }
        assert!(commits.next().is_none());
    }

    /// 2 tokens a frame, chunks of 2 frames, a window of 8: the sink is frames 0 and 1, the ring 4 frames.
    fn small() -> Ring { Ring::new(6 * 2, 2 * 2, 2, 2) }

    #[test]
    fn the_chunk_always_attends_to_itself() {
        let ring = small();
        assert_eq!(ring.allowed_key_blocks(true, 4, 2), [1, 1, 0, 0, 0, 0, 0, 0]);
        assert_eq!(ring.allowed_key_blocks(false, 0, 2), [1, 1, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn the_sink_is_allowed_by_its_switch() {
        let mut ring = small();
        ring.commit();
        assert_eq!(ring.allowed_key_blocks(true, 0, 2), [1, 1, 1, 1, 0, 0, 0, 0]);
        assert_eq!(ring.allowed_key_blocks(false, 4, 2), [0, 0, 1, 1, 0, 0, 0, 0]);
    }

    #[test]
    fn only_the_recent_frames_of_the_ring_are_allowed_wherever_the_ring_put_them() {
        let mut ring = small();
        for _ in 0..3 {
            ring.commit();
        } // the ring holds frames 2, 3 then 4, 5; the chunk, frames 6 and 7, goes to the last 4 tokens
        assert_eq!(ring.allowed_key_blocks(true, 2, 2), [1, 1, 0, 0, 1, 1, 1, 1]);
        assert_eq!(ring.allowed_key_blocks(true, 4, 2), [1, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(ring.allowed_key_blocks(true, 0, 2), [1, 1, 0, 0, 0, 0, 1, 1]);
        ring.commit(); // frames 6, 7 over the oldest, 2 and 3: now first in the ring
        assert_eq!(ring.allowed_key_blocks(true, 2, 2), [1, 1, 1, 1, 0, 0, 1, 1]);
    }

    #[test]
    fn a_pinned_chunk_is_a_sink() {
        let mut ring = small();
        ring.commit();
        ring.commit();
        ring.pin_last(); // frames 2, 3, the first chunk of a shot
        ring.commit();
        ring.commit(); // frames 6, 7 over 4, 5: the pinned are not evicted
        assert_eq!(ring.allowed_key_blocks(true, 0, 2), [1, 1, 1, 1, 0, 0, 1, 1]);
        assert_eq!(ring.allowed_key_blocks(false, 0, 2), [0, 0, 0, 0, 0, 0, 1, 1]);
    }

    #[test]
    fn a_block_across_frames_is_allowed_if_any_of_its_tokens_is() {
        let mut ring = small();
        for _ in 0..3 {
            ring.commit();
        }
        assert_eq!(ring.allowed_key_blocks(false, 2, 3), [0, 0, 1, 1, 1, 1]);
    }

    #[test]
    fn a_full_ring_copies_whole_frames_at_once() {
        let mut ring = Ring::new(24 * 405, 8 * 405, 8, 405);
        for _ in 0..3 {
            assert!(ring.commit().is_empty());
        }
        let copies = ring.commit();
        assert_eq!(copies.iter().map(|c| c.count).sum::<usize>(), 3240);
        assert!(copies.len() <= 8, "{} copies for a chunk of 8 frames", copies.len());
        assert_eq!(ring.step(), 3);
    }
}

#[cfg(test)]
mod disk_gpu_tests {
    use super::*;
    #[test]
    #[ignore = "requires CUDA; bounded transfer integration, no model packages"]
    fn disk_and_ram_match_after_commits_and_eviction() {
        let gpu = crate::gpu::Gpu::new().unwrap(); gpu.bind().unwrap();
        let compute = Stream::new().unwrap();
        for format in [Format::Fp8, Format::Bf16] {
            let mut ram = Offloaded::new(3, 8, 2, 4, format, &compute).unwrap();
            let mut disk = Offloaded::new_disk(3, 8, 2, 4, format, &compute).unwrap();
            for round in 0..4 {
                for cache in [&mut ram, &mut disk] {
                    let previous = cache.clone_host().unwrap();
                    cache.begin_pass(8).unwrap();
                    for b in 0..3 {
                        cache.tensors(b, &compute).unwrap();
                        let slot = cache.holds.iter().position(|&h| h == b).unwrap();
                        let sizes = cache.token_bytes();
                        for (i, device) in [&cache.slots[slot].k, &cache.slots[slot].ks, &cache.slots[slot].v, &cache.slots[slot].vs].iter().enumerate() {
                            let mut loaded = vec![0; device.bytes];
                            compute.copy_to_host(&mut loaded, device.ptr).unwrap(); compute.synchronize().unwrap();
                            assert_eq!(loaded, previous[b][i]);
                        }
                        for (i, device) in [&cache.slots[slot].k, &cache.slots[slot].ks, &cache.slots[slot].v, &cache.slots[slot].vs].iter().enumerate() {
                            let bytes = vec![(round * 30 + b * 4 + i) as u8; 2 * sizes[i]];
                            compute.copy_to_device(device.ptr + (round * 2 * sizes[i]) as u64, &bytes).unwrap();
                            compute.synchronize().unwrap();
                        }
                        cache.end_block(b, &compute, Some(&Write { at: round * 2, n: 2, copies: &[] })).unwrap();
                    }
                }
                assert_eq!(ram.clone_host().unwrap(), disk.clone_host().unwrap());
            }
            assert_eq!(disk.pinned_bytes(), 2 * format.bytes(8, 2, 4));
            assert_eq!(disk.disk_bytes().1, (3 * format.bytes(8, 2, 4)) as u64);
            // Ring compaction persists copied destinations, including across slot reuse.
            for cache in [&mut ram, &mut disk] {
                cache.begin_pass(8).unwrap();
                for b in 0..3 {
                    cache.tensors(b, &compute).unwrap();
                    cache.end_block(b, &compute, Some(&Write { at: 6, n: 2,
                        copies: &[Copy { src: 6, dst: 2, count: 2 }] })).unwrap();
                }
            }
            assert_eq!(ram.clone_host().unwrap(), disk.clone_host().unwrap());
            ram.to_disk().unwrap();
            assert_eq!(ram.clone_host().unwrap(), disk.clone_host().unwrap());
        }
    }
}
