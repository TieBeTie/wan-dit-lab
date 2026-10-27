//! The KV caches of the 30 blocks and the one bookkeeping they share (causal.KVCache keeps a copy of it per block).
//!
//! A cache is [1, max_tokens + n, heads, head_dim] keys and values in FP8 with a scale per token and head. The first
//! sink tokens are the first chunk of the video, never evicted; the rest up to max_tokens is the ring. A chunk is
//! written at `at` (by the block package); while the ring fills `at` is where the saved tokens end and keeping the
//! chunk is moving the counter, once it is full the chunk is written into the last n tokens and kept by copying it
//! over the tokens of the oldest frames that are not pinned (LongLive 2.0 pins the first chunk of a shot until the
//! next shot). Of causal.KVCache only what Live runs is here: no Participative Compression, no positions in the
//! window, no forgetting, no Forcing-KV.

use crate::gpu::{DeviceBuffer, Result, Stream};

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
