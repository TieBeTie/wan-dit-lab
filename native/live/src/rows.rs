//! The blocks' weights, one row of bytes per block (the W4A4 pack, block_w4a4_v2): some rows on the GPU for good,
//! the others in pinned RAM and copied ahead of use into two GPU slots, as StreamedRows (offloading/offload.py)
//! does. The rows are used in a fixed cycle (block 0 to 29 in every pass), so a slot freed by block j is refilled
//! with the streamed row two places on in that cycle, the next pass's first rows included.

use std::fs::File;
use std::path::Path;

use crate::gpu::{DeviceBuffer, Event, PinnedBuffer, Result, Stream};

enum Place {
    Device(DeviceBuffer),
    /// index among the streamed rows
    Streamed(usize),
}

pub struct Rows {
    places: Vec<Place>,
    host: Vec<PinnedBuffer>,
    /// block of each streamed row, in the order of use
    streamed: Vec<usize>,
    slots: [DeviceBuffer; 2],
    copied: [Event; 2],
    freed: [Event; 2],
    copy: Stream,
    pub row_bytes: usize,
    /// the streamed row each slot holds, or is being filled with
    holds: [usize; 2],
}

/// The blocks live_video.py streams: `count` of `blocks`, spread over the model.
pub fn spread(blocks: usize, count: usize) -> Vec<usize> {
    (0..count).map(|j| ((j as f64 + 0.5) * blocks as f64 / count as f64) as usize).collect()
}

impl Rows {
    /// The rows of `pack` (blocks x row_bytes), `streamed` of them in pinned RAM, the rest on the GPU.
    pub fn load(pack: &Path, blocks: usize, row_bytes: usize, streamed: &[usize], stream: &Stream) -> std::io::Result<Rows> {
        let file = File::open(pack)?;
        let map = unsafe { memmap2::Mmap::map(&file)? };
        assert_eq!(map.len(), blocks * row_bytes, "the W4A4 pack has another size than its manifest says");
        let io = |e: cudarc::driver::DriverError| std::io::Error::other(e.to_string());
        let mut places = Vec::with_capacity(blocks);
        let mut host = Vec::new();
        for b in 0..blocks {
            let row = &map[b * row_bytes..(b + 1) * row_bytes];
            if let Some(j) = streamed.iter().position(|&s| s == b) {
                let mut pinned = PinnedBuffer::new(row_bytes).map_err(io)?;
                pinned.as_mut_slice().copy_from_slice(row);
                assert_eq!(j, host.len(), "streamed rows in the order of their blocks");
                host.push(pinned);
                places.push(Place::Streamed(j));
            } else {
                let device = DeviceBuffer::new(row_bytes).map_err(io)?;
                stream.copy_to_device(device.ptr, row).map_err(io)?;
                stream.synchronize().map_err(io)?; // the mapping is read before the next row
                places.push(Place::Device(device));
            }
        }
        let copy = Stream::new().map_err(io)?;
        let slots = [DeviceBuffer::new(row_bytes).map_err(io)?, DeviceBuffer::new(row_bytes).map_err(io)?];
        let mut rows = Rows {
            places,
            host,
            streamed: streamed.to_vec(),
            slots,
            copied: [Event::new().map_err(io)?, Event::new().map_err(io)?],
            freed: [Event::new().map_err(io)?, Event::new().map_err(io)?],
            copy,
            row_bytes,
            holds: [usize::MAX; 2],
        };
        for j in 0..rows.streamed.len().min(2) {
            rows.fill(j % 2, j).map_err(io)?;
        }
        Ok(rows)
    }

    fn fill(&mut self, slot: usize, j: usize) -> Result<()> {
        // after the block that used the slot last is done with it
        self.copy.wait(&self.freed[slot])?;
        self.copy.copy_to_device(self.slots[slot].ptr, self.host[j].as_slice())?;
        self.copied[slot].record(&self.copy)?;
        self.holds[slot] = j;
        Ok(())
    }

    /// The row of `block` as a uint8 tensor for work on `compute`, which waits for its copy if it is streamed.
    pub fn row(&self, block: usize, compute: &Stream) -> Result<aoti::Tensor> {
        let shape = [self.row_bytes as i64];
        match &self.places[block] {
            Place::Device(buffer) => Ok(buffer.tensor(0, aoti::DType::U8, &shape)),
            Place::Streamed(j) => {
                let slot = j % 2;
                assert_eq!(self.holds[slot], *j, "rows taken out of their cycle");
                compute.wait(&self.copied[slot])?;
                Ok(self.slots[slot].tensor(0, aoti::DType::U8, &shape))
            }
        }
    }

    /// `block`'s work is queued on `compute`: its slot gets the streamed row two on in the cycle.
    pub fn release(&mut self, block: usize, compute: &Stream) -> Result<()> {
        if let Place::Streamed(j) = self.places[block] {
            let slot = j % 2;
            self.freed[slot].record(compute)?;
            self.fill(slot, (j + 2) % self.streamed.len())?;
        }
        Ok(())
    }

    pub fn pinned_bytes(&self) -> usize {
        self.host.iter().map(|h| h.bytes).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::spread;

    #[test]
    fn streamed_blocks_as_live_video_spreads_them() {
        // live_video.py: [int((j + 0.5) * n / k) for j in range(k)], n = 30, k = 24
        let want = [0, 1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21, 23, 24, 25, 26, 28, 29];
        assert_eq!(spread(30, 24), want);
    }
}
