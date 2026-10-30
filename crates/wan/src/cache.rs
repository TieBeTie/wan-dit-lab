//! The KV caches of the 30 blocks and the one bookkeeping they share (causal.KVCache keeps a copy of it per block).
//!
//! A cache is [1, max_tokens + n, heads, head_dim] keys and values in FP8 with a scale per token and head. The first
//! sink tokens are the first chunk of the video, never evicted; the rest up to max_tokens is the ring. A chunk is
//! written at `at` (by the block package); while the ring fills `at` is where the saved tokens end and keeping the
//! chunk is moving the counter, once it is full the chunk is written into the last n tokens and kept by copying it
//! over the tokens of the oldest frames that are not pinned (LongLive 2.0 pins the first chunk of a shot until the
//! next shot). Of causal.KVCache only what Live runs is here: no Participative Compression, no positions in the
//! window, no forgetting, no Forcing-KV.

use crate::gpu::{DeviceBuffer, Event, PinnedBuffer, Result, Stream};

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

    /// Empties the caches: a new video.
    pub fn reset(&mut self) {
        let fresh = Ring::new(self.max_tokens, self.sink, self.chunk, self.per_frame);
        *self = fresh;
    }
}

/// The FP8 keys and values of one block with their scales, [1, tokens, heads, head_dim] and [1, tokens, heads, 1].
pub struct Buffers {
    pub k: DeviceBuffer,
    pub ks: DeviceBuffer,
    pub v: DeviceBuffer,
    pub vs: DeviceBuffer,
    pub shape: [i64; 4],
}

impl Buffers {
    pub fn new(tokens: usize, heads: usize, head_dim: usize, stream: &Stream) -> Result<Buffers> {
        let values = tokens * heads * head_dim;
        let scales = tokens * heads * 4;
        Ok(Buffers {
            k: DeviceBuffer::zeroed(values, stream)?,
            ks: DeviceBuffer::zeroed(scales, stream)?,
            v: DeviceBuffer::zeroed(values, stream)?,
            vs: DeviceBuffer::zeroed(scales, stream)?,
            shape: [1, tokens as i64, heads as i64, head_dim as i64],
        })
    }

    /// The four tensors in the order of the block package: k, k scales, v, v scales.
    pub fn tensors(&self) -> [aoti::Tensor; 4] {
        let [b, t, h, d] = self.shape;
        [
            self.k.tensor(0, aoti::DType::F8E4M3, &[b, t, h, d]),
            self.ks.tensor(0, aoti::DType::F32, &[b, t, h, 1]),
            self.v.tensor(0, aoti::DType::F8E4M3, &[b, t, h, d]),
            self.vs.tensor(0, aoti::DType::F32, &[b, t, h, 1]),
        ]
    }

    /// Applies the copies of a commit on `stream`, after attention there has read the tokens.
    pub fn copy(&self, copies: &[Copy], stream: &Stream) -> Result<()> {
        let [_, _, h, d] = self.shape;
        let value = (h * d) as usize; // bytes of a token in FP8
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
/// configs/live.json). Before a block, the part of its cache attention reads ([0, at)) is copied to a slot; after the
/// pass that writes the clean chunk, the commit's copies are done in the slot and only the tokens they wrote go back to
/// RAM (the other passes' writes of the chunk are overwritten by the next pass anyway). Copies to the GPU and back run
/// on streams of their own, a block ahead of compute.
pub struct Offloaded {
    host: Vec<[PinnedBuffer; 4]>,
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
    pub fn new(blocks: usize, tokens: usize, heads: usize, head_dim: usize, compute: &Stream) -> Result<Offloaded> {
        let sizes = [tokens * heads * head_dim, tokens * heads * 4, tokens * heads * head_dim, tokens * heads * 4];
        let host = (0..blocks).map(|_| {
            let buffers = sizes.map(PinnedBuffer::new);
            let [k, ks, v, vs] = buffers;
            Ok([k?, ks?, v?, vs?])
        }).collect::<Result<Vec<_>>>()?;
        Offloaded::with_host(host, tokens, heads, head_dim, compute)
    }

    /// The caches of `resident` moved to pinned RAM (their GPU memory freed), with two slots of their size.
    pub fn from_resident(resident: Vec<Buffers>, compute: &Stream) -> Result<Offloaded> {
        compute.synchronize()?;
        let shape = resident[0].shape;
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
        Offloaded::with_host(host, tokens, heads, head_dim, compute)
    }

    fn with_host(host: Vec<[PinnedBuffer; 4]>, tokens: usize, heads: usize, head_dim: usize, compute: &Stream)
        -> Result<Offloaded> {
        let blocks = host.len();
        Ok(Offloaded {
            host,
            slots: [Buffers::new(tokens, heads, head_dim, compute)?, Buffers::new(tokens, heads, head_dim, compute)?],
            holds: [usize::MAX; 2],
            loaded: [Event::new()?, Event::new()?],
            freed: [Event::new()?, Event::new()?],
            written: (0..blocks).map(|_| Event::new()).collect::<Result<_>>()?,
            up: Stream::new()?,
            down: Stream::new()?,
            read: 0,
            next: 0,
        })
    }

    /// Bytes of one token in the four buffers: a key or value in FP8, a scale in float32.
    fn token_bytes(&self) -> [usize; 4] {
        let [_, _, h, d] = self.slots[0].shape.map(|s| s as usize);
        [h * d, h * 4, h * d, h * 4]
    }

    /// A pass that reads the tokens [0, read) of every cache begins: the first two blocks are loaded.
    pub fn begin_pass(&mut self, read: usize) -> Result<()> {
        self.read = read;
        self.next = 0;
        for slot in 0..2.min(self.host.len()) {
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
        let bytes = self.token_bytes();
        let device = [&self.slots[slot].k, &self.slots[slot].ks, &self.slots[slot].v, &self.slots[slot].vs];
        for i in 0..4 {
            let count = self.read * bytes[i];
            if count > 0 {
                self.up.copy_to_device(device[i].ptr, &self.host[b][i].as_slice()[..count])?;
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
                for (start, count) in ranges {
                    for i in 0..4 {
                        let (from, len) = (start * bytes[i], count * bytes[i]);
                        let host = &mut self.host[b][i].as_mut_slice()[from..from + len];
                        self.down.copy_to_host(host, device[i] + from as u64)?;
                    }
                }
                self.written[b].record(&self.down)?;
                self.freed[slot].record(&self.down)?;
            }
            None => self.freed[slot].record(compute)?,
        }
        if self.next < self.host.len() {
            self.load(slot)?;
        }
        Ok(())
    }

    /// Bytes of a block's cache that a pass with the ring full reads: `max_tokens` tokens of keys, values, scales.
    pub fn bytes_read(max_tokens: usize, heads: usize, head_dim: usize) -> usize {
        max_tokens * 2 * heads * (head_dim + 4)
    }

    /// The caches back on the GPU (the calibration found their copies could not hide), the pinned RAM freed.
    pub fn into_resident(self, compute: &Stream) -> Result<Vec<Buffers>> {
        compute.synchronize()?;
        self.up.synchronize()?;
        self.down.synchronize()?;
        let [_, tokens, heads, head_dim] = self.slots[0].shape.map(|s| s as usize);
        let mut caches = Vec::with_capacity(self.host.len());
        for host in &self.host {
            let cache = Buffers::new(tokens, heads, head_dim, compute)?;
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
        self.down.synchronize()
    }

    /// A copy of the caches in RAM, once every copy back is done (tests).
    #[cfg(test)]
    pub fn clone_host(&self) -> Result<Vec<[Vec<u8>; 4]>> {
        self.synchronize()?;
        Ok(self.host.iter().map(|h| [0, 1, 2, 3].map(|i| h[i].as_slice().to_vec())).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
