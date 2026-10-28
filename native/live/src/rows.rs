//! The blocks' weights, one row of bytes per block (the W4A4 pack, block_w4a4_v2): some rows on the GPU for good,
//! the others in pinned RAM and copied ahead of use into two GPU slots, as StreamedRows (offloading/offload.py)
//! does. The rows are used in a fixed cycle (block 0 to 29 in every pass), so a slot freed by block j is refilled
//! with the streamed row two places on in that cycle, the next pass's first rows included.

use std::fs::File;
use std::path::Path;

use crate::disk::{open_unbuffered, read_row, span, SECTOR};
use crate::gpu::{DeviceBuffer, Event, PinnedBuffer, Result, Stream};

/// Where the streamed rows come from: pinned RAM (one buffer per row), or the pack file on the disk, read into two
/// pinned staging buffers.
enum Source {
    Pinned(Vec<PinnedBuffer>),
    Disk { file: File, staging: [PinnedBuffer; 2] },
}

enum Place {
    Device(DeviceBuffer),
    /// index among the streamed rows
    Streamed(usize),
}

pub struct Rows {
    places: Vec<Place>,
    source: Source,
    pack: std::path::PathBuf,
    /// block of each streamed row, in the order of use
    streamed: Vec<usize>,
    slots: [DeviceBuffer; 2],
    copied: [Event; 2],
    freed: [Event; 2],
    copy: Stream,
    pub row_bytes: usize,
    /// the streamed row each slot holds, or is being filled with
    holds: [usize; 2],
    /// debug-tools: timed events around every wait of the compute stream for a row's copy
    #[cfg(feature = "debug-tools")]
    waits: std::cell::RefCell<Vec<(Event, Event)>>,
}

/// The blocks live_video.py streams: `count` of `blocks`, spread over the model.
pub fn spread(blocks: usize, count: usize) -> Vec<usize> {
    (0..count).map(|j| ((j as f64 + 0.5) * blocks as f64 / count as f64) as usize).collect()
}

impl Rows {
    /// The rows of `pack` (blocks x row_bytes): the `streamed` ones in pinned RAM, or read from the pack file when
    /// `disk`; the rest on the GPU.
    pub fn load(pack: &Path, blocks: usize, row_bytes: usize, streamed: &[usize], disk: bool, stream: &Stream)
        -> std::io::Result<Rows> {
        let file = File::open(pack)?;
        let map = unsafe { memmap2::Mmap::map(&file)? };
        assert_eq!(map.len(), blocks * row_bytes, "the W4A4 pack has another size than its manifest says");
        let io = |e: cudarc::driver::DriverError| std::io::Error::other(e.to_string());
        let mut places = Vec::with_capacity(blocks);
        let mut host = Vec::new();
        for b in 0..blocks {
            let row = &map[b * row_bytes..(b + 1) * row_bytes];
            if let Some(j) = streamed.iter().position(|&s| s == b) {
                if !disk {
                    let mut pinned = PinnedBuffer::new(row_bytes).map_err(io)?;
                    pinned.as_mut_slice().copy_from_slice(row);
                    assert_eq!(j, host.len(), "streamed rows in the order of their blocks");
                    host.push(pinned);
                }
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
        let source = if disk {
            let staging = [PinnedBuffer::new(span(row_bytes)).map_err(io)?, PinnedBuffer::new(span(row_bytes)).map_err(io)?];
            Source::Disk { file: open_unbuffered(pack)?, staging }
        } else {
            Source::Pinned(host)
        };
        let mut rows = Rows {
            places,
            source,
            pack: pack.to_path_buf(),
            streamed: streamed.to_vec(),
            slots,
            copied: [Event::new().map_err(io)?, Event::new().map_err(io)?],
            freed: [Event::new().map_err(io)?, Event::new().map_err(io)?],
            copy,
            row_bytes,
            holds: [usize::MAX; 2],
            #[cfg(feature = "debug-tools")]
            waits: Default::default(),
        };
        for j in 0..rows.streamed.len().min(2) {
            rows.fill(j % 2, j).map_err(io)?;
        }
        Ok(rows)
    }

    fn fill(&mut self, slot: usize, j: usize) -> Result<()> {
        // after the block that used the slot last is done with it
        self.copy.wait(&self.freed[slot])?;
        match &mut self.source {
            Source::Pinned(host) => self.copy.copy_to_device(self.slots[slot].ptr, host[j].as_slice())?,
            Source::Disk { file, staging } => {
                // the staging buffer's last row is on the GPU, then this row is read into it
                self.copied[slot].synchronize()?;
                let at = read_row(file, self.streamed[j], self.row_bytes, staging[slot].as_mut_slice())
                    .map_err(|_| cudarc::driver::DriverError(cudarc::driver::sys::cudaError_enum::CUDA_ERROR_FILE_NOT_FOUND))?;
                self.copy.copy_to_device(self.slots[slot].ptr, &staging[slot].as_slice()[at..at + self.row_bytes])?;
            }
        }
        self.copied[slot].record(&self.copy)?;
        self.holds[slot] = j;
        Ok(())
    }

    /// The slot that holds streamed row j. Not j % 2: with an odd number of streamed rows the cycle wraps onto the
    /// other parity (row k - 1 frees the slot that row 1 of the next pass is copied into).
    fn slot_of(&self, j: usize) -> usize {
        self.holds.iter().position(|&h| h == j).expect("rows taken out of their cycle")
    }

    /// The row of `block` as a uint8 tensor for work on `compute`, which waits for its copy if it is streamed.
    pub fn row(&self, block: usize, compute: &Stream) -> Result<aoti::Tensor> {
        let shape = [self.row_bytes as i64];
        match &self.places[block] {
            Place::Device(buffer) => Ok(buffer.tensor(0, aoti::DType::U8, &shape)),
            Place::Streamed(j) => {
                let slot = self.slot_of(*j);
                #[cfg(feature = "debug-tools")]
                let span = (Event::timed()?, Event::timed()?);
                #[cfg(feature = "debug-tools")]
                span.0.record(compute)?;
                compute.wait(&self.copied[slot])?;
                #[cfg(feature = "debug-tools")]
                {
                    span.1.record(compute)?;
                    self.waits.borrow_mut().push(span);
                }
                Ok(self.slots[slot].tensor(0, aoti::DType::U8, &shape))
            }
        }
    }

    /// `block`'s work is queued on `compute`: its slot gets the streamed row two on in the cycle.
    pub fn release(&mut self, block: usize, compute: &Stream) -> Result<()> {
        if let Place::Streamed(j) = self.places[block] {
            let slot = self.slot_of(j);
            self.freed[slot].record(compute)?;
            self.fill(slot, (j + 2) % self.streamed.len())?;
        }
        Ok(())
    }

    /// Streams only the blocks of `streamed` from now on (a subset of those streamed now, as `spread` picks them):
    /// the others get a device buffer of their own and give their pinned memory back. Waits for the GPU first.
    pub fn keep_on_device(&mut self, streamed: &[usize], compute: &Stream) -> Result<()> {
        compute.synchronize()?;
        self.copy.synchronize()?;
        let Source::Pinned(pinned) = std::mem::replace(&mut self.source, Source::Pinned(Vec::new())) else {
            panic!("rows on the disk tier are not placed on the GPU");
        };
        let mut host: Vec<Option<PinnedBuffer>> = pinned.into_iter().map(Some).collect();
        let mut kept = Vec::new();
        for b in 0..self.places.len() {
            if let Place::Streamed(j) = self.places[b] {
                let pinned = host[j].take().unwrap();
                if streamed.contains(&b) {
                    self.places[b] = Place::Streamed(kept.len());
                    kept.push(pinned);
                } else {
                    let device = DeviceBuffer::new(self.row_bytes)?;
                    self.copy.copy_to_device(device.ptr, pinned.as_slice())?;
                    self.copy.synchronize()?; // before the pinned buffer goes
                    self.places[b] = Place::Device(device);
                }
            }
        }
        self.source = Source::Pinned(kept);
        self.streamed = (0..self.places.len()).filter(|b| matches!(self.places[*b], Place::Streamed(_))).collect();
        self.holds = [usize::MAX; 2];
        for j in 0..self.streamed.len().min(2) {
            self.fill(j % 2, j)?;
        }
        Ok(())
    }

    /// Milliseconds of one row's copy from pinned RAM to the GPU, alone on the copy stream (the median of three).
    pub fn time_copy(&self) -> Result<f32> {
        self.copy.synchronize()?;
        let target = DeviceBuffer::new(self.row_bytes)?;
        let (start, end) = (Event::timed()?, Event::timed()?);
        let mut times = Vec::new();
        for _ in 0..3 {
            start.record(&self.copy)?;
            // a pinned buffer the rows already have: one made for the measure copied at a third of the speed
            let from = match &self.source {
                Source::Pinned(host) => host[0].as_slice(),
                Source::Disk { staging, .. } => &staging[0].as_slice()[..self.row_bytes],
            };
            self.copy.copy_to_device(target.ptr, from)?;
            end.record(&self.copy)?;
            end.synchronize()?;
            times.push(start.elapsed_ms(&end)?);
        }
        times.sort_by(f32::total_cmp);
        Ok(times[1])
    }

    /// debug-tools: of the waits for copies since the last call, how many were late (the compute stream stood more
    /// than 0.05 ms), how many there were, and the milliseconds it stood in all. Waits for the GPU.
    #[cfg(feature = "debug-tools")]
    pub fn copy_waits(&self, compute: &Stream) -> Result<(usize, usize, f32)> {
        compute.synchronize()?;
        let waits = std::mem::take(&mut *self.waits.borrow_mut());
        let mut late = 0;
        let mut total = 0.0;
        for (start, end) in &waits {
            let ms = start.elapsed_ms(end)?;
            total += ms;
            late += (ms > 0.05) as usize;
        }
        Ok((late, waits.len(), total))
    }

    pub fn on_disk(&self) -> bool {
        matches!(self.source, Source::Disk { .. })
    }

    pub fn streamed(&self) -> usize {
        self.streamed.len()
    }

    pub fn pinned_bytes(&self) -> usize {
        match &self.source {
            Source::Pinned(host) => host.iter().map(|h| h.bytes).sum(),
            Source::Disk { staging, .. } => staging.iter().map(|h| h.bytes).sum(),
        }
    }

    /// Milliseconds of reading one row from the pack file without the page cache (the median of three).
    pub fn time_read(&self) -> std::io::Result<f32> {
        let file = open_unbuffered(&self.pack)?;
        let mut buffer = vec![0u8; span(self.row_bytes) + SECTOR];
        let start = buffer.as_ptr() as usize % SECTOR;
        let aligned = &mut buffer[(SECTOR - start) % SECTOR..];
        let mut times = Vec::new();
        for i in 0..3 {
            let t = std::time::Instant::now();
            read_row(&file, i, self.row_bytes, aligned)?;
            times.push(t.elapsed().as_secs_f32() * 1000.0);
        }
        times.sort_by(f32::total_cmp);
        Ok(times[1])
    }

    /// The streamed rows in pinned RAM from now on, read from the pack (they were on the disk tier).
    pub fn to_ram(&mut self, compute: &Stream) -> std::io::Result<()> {
        let io = |e: cudarc::driver::DriverError| std::io::Error::other(e.to_string());
        if let Source::Pinned(_) = self.source {
            return Ok(());
        }
        compute.synchronize().map_err(io)?;
        self.copy.synchronize().map_err(io)?;
        let file = File::open(&self.pack)?;
        let map = unsafe { memmap2::Mmap::map(&file)? };
        let mut host = Vec::with_capacity(self.streamed.len());
        for &b in &self.streamed {
            let mut pinned = PinnedBuffer::new(self.row_bytes).map_err(io)?;
            pinned.as_mut_slice().copy_from_slice(&map[b * self.row_bytes..(b + 1) * self.row_bytes]);
            host.push(pinned);
        }
        self.source = Source::Pinned(host);
        self.holds = [usize::MAX; 2];
        for j in 0..self.streamed.len().min(2) {
            self.fill(j % 2, j).map_err(io)?;
        }
        Ok(())
    }

    /// Every streamed row from the pack file from now on, through two pinned staging buffers; their pinned RAM goes.
    pub fn to_disk(&mut self, compute: &Stream) -> std::io::Result<()> {
        let io = |e: cudarc::driver::DriverError| std::io::Error::other(e.to_string());
        compute.synchronize().map_err(io)?;
        self.copy.synchronize().map_err(io)?;
        let staging = [PinnedBuffer::new(span(self.row_bytes)).map_err(io)?, PinnedBuffer::new(span(self.row_bytes)).map_err(io)?];
        self.source = Source::Disk { file: open_unbuffered(&self.pack)?, staging };
        self.holds = [usize::MAX; 2];
        for j in 0..self.streamed.len().min(2) {
            self.fill(j % 2, j).map_err(io)?;
        }
        Ok(())
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
