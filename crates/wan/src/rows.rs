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
    /// two GPU slots for the streamed rows; none while parked
    slots: Vec<DeviceBuffer>,
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
        let io = |e: cudarc::driver::DriverError| std::io::Error::other(e.to_string());
        assert_eq!(std::fs::metadata(pack)?.len() as usize, blocks * row_bytes, "the pack has another size than its manifest says");
        // From the disk, the rows kept on the GPU go through the staging buffers too: a mapping of the pack would put
        // its pages into this process's working set (8.6 GiB for T5) and count them against the RAM reserve.
        let mut disk_source = if disk {
            let staging = [PinnedBuffer::new(span(row_bytes)).map_err(io)?, PinnedBuffer::new(span(row_bytes)).map_err(io)?];
            Some((open_unbuffered(pack)?, staging))
        } else {
            None
        };
        let map = if disk { None } else { Some(unsafe { memmap2::Mmap::map(&File::open(pack)?)? }) };
        let mut places = Vec::with_capacity(blocks);
        let mut host = Vec::new();
        for b in 0..blocks {
            if let Some(j) = streamed.iter().position(|&s| s == b) {
                if let Some(map) = &map {
                    let mut pinned = PinnedBuffer::new(row_bytes).map_err(io)?;
                    pinned.as_mut_slice().copy_from_slice(&map[b * row_bytes..(b + 1) * row_bytes]);
                    assert_eq!(j, host.len(), "streamed rows in the order of their blocks");
                    host.push(pinned);
                }
                places.push(Place::Streamed(j));
            } else {
                let device = DeviceBuffer::new(row_bytes).map_err(io)?;
                match (&map, &mut disk_source) {
                    (Some(map), _) => stream.copy_to_device(device.ptr, &map[b * row_bytes..(b + 1) * row_bytes]).map_err(io)?,
                    (None, Some((file, staging))) => {
                        let at = read_row(file, b, row_bytes, staging[0].as_mut_slice())?;
                        stream.copy_to_device(device.ptr, &staging[0].as_slice()[at..at + row_bytes]).map_err(io)?;
                    }
                    (None, None) => unreachable!(),
                }
                stream.synchronize().map_err(io)?; // the source is read before the next row
                places.push(Place::Device(device));
            }
        }
        drop(map);
        let copy = Stream::new().map_err(io)?;
        let slots = vec![DeviceBuffer::new(row_bytes).map_err(io)?, DeviceBuffer::new(row_bytes).map_err(io)?];
        let source = match disk_source {
            Some((file, staging)) => Source::Disk { file, staging },
            None => Source::Pinned(host),
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
    pub fn row<const TIMED: bool>(&self, block: usize, compute: &Stream) -> Result<aoti::Tensor> {
        let shape = [self.row_bytes as i64];
        match &self.places[block] {
            Place::Device(buffer) => Ok(buffer.tensor(0, aoti::DType::U8, &shape)),
            Place::Streamed(j) => {
                let slot = self.slot_of(*j);
                #[cfg(feature = "debug-tools")]
                let span = if TIMED { Some((Event::timed()?, Event::timed()?)) } else { None };
                #[cfg(feature = "debug-tools")]
                if let Some((start, _)) = &span { start.record(compute)?; }
                compute.wait(&self.copied[slot])?;
                #[cfg(feature = "debug-tools")]
                if let Some((start, end)) = span {
                    end.record(compute)?;
                    self.waits.borrow_mut().push((start, end));
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
        if let Source::Disk { file, staging } = &mut self.source {
            // through the staging buffer, as Rows::load: a mapping would count the pack's pages against the reserve
            let missing = |_| cudarc::driver::DriverError(cudarc::driver::sys::cudaError_enum::CUDA_ERROR_FILE_NOT_FOUND);
            let mut kept = Vec::new();
            for b in 0..self.places.len() {
                if matches!(self.places[b], Place::Streamed(_)) {
                    if streamed.contains(&b) {
                        self.places[b] = Place::Streamed(kept.len()); kept.push(b);
                    } else {
                        let device = DeviceBuffer::new(self.row_bytes)?;
                        let at = read_row(file, b, self.row_bytes, staging[0].as_mut_slice()).map_err(missing)?;
                        self.copy.copy_to_device(device.ptr, &staging[0].as_slice()[at..at + self.row_bytes])?;
                        self.copy.synchronize()?;
                        self.places[b] = Place::Device(device);
                    }
                }
            }
            self.streamed = kept;
            self.holds = [usize::MAX; 2];
            for j in 0..self.streamed.len().min(2) { self.fill(j % 2, j)?; }
            return Ok(());
        }
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
        let times = waits.iter().map(|(start, end)| start.elapsed_ms(end)).collect::<Result<Vec<_>>>()?;
        let w = CopyWaits::of(&times);
        Ok((w.late, w.waits, w.ms))
    }

    /// The slots go back to the GPU's free memory until unpark (rows used now and then, as T5's between prompts).
    pub fn park(&mut self, compute: &Stream) -> Result<()> {
        compute.synchronize()?;
        self.copy.synchronize()?;
        self.slots.clear();
        self.holds = [usize::MAX; 2];
        Ok(())
    }

    /// Slots again, the first streamed rows copied into them.
    pub fn unpark(&mut self) -> Result<()> {
        if self.slots.is_empty() && !self.streamed.is_empty() {
            self.slots = vec![DeviceBuffer::new(self.row_bytes)?, DeviceBuffer::new(self.row_bytes)?];
            for j in 0..self.streamed.len().min(2) {
                self.fill(j % 2, j)?;
            }
        }
        Ok(())
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
        if matches!(self.source, Source::Disk { .. }) { return Ok(()); }
        let io = |e: cudarc::driver::DriverError| std::io::Error::other(e.to_string());
        compute.synchronize().map_err(io)?;
        self.copy.synchronize().map_err(io)?;
        // Immutable rows can be discarded before allocating staging: no transition RAM peak.
        self.source = Source::Pinned(Vec::new());
        let staging = [PinnedBuffer::new(span(self.row_bytes)).map_err(io)?, PinnedBuffer::new(span(self.row_bytes)).map_err(io)?];
        self.source = Source::Disk { file: open_unbuffered(&self.pack)?, staging };
        self.holds = [usize::MAX; 2];
        for j in 0..self.streamed.len().min(2) {
            self.fill(j % 2, j).map_err(io)?;
        }
        Ok(())
    }
}

/// debug-tools: waits of the compute stream for rows' copies: how many stood more than 0.05 ms, how many there were,
/// and the milliseconds it stood in all.
#[cfg(feature = "debug-tools")]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct CopyWaits {
    pub late: usize,
    pub waits: usize,
    pub ms: f32,
}

#[cfg(feature = "debug-tools")]
impl CopyWaits {
    pub fn of(times: &[f32]) -> CopyWaits {
        CopyWaits { late: times.iter().filter(|&&ms| ms > 0.05).count(), waits: times.len(), ms: times.iter().sum() }
    }
}

#[cfg(feature = "debug-tools")]
impl std::fmt::Display for CopyWaits {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "late copies {} of {}, the GPU stood {:.1} ms for them", self.late, self.waits, self.ms)
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

    #[cfg(feature = "debug-tools")]
    #[test]
    fn a_wait_over_a_twentieth_of_a_millisecond_is_late() {
        let w = super::CopyWaits::of(&[0.0, 0.05, 0.06, 0.2]);
        assert_eq!((w.late, w.waits), (2, 4));
        assert!((w.ms - 0.31).abs() < 1e-6);
    }

    #[cfg(feature = "debug-tools")]
    #[test]
    fn copy_waits_read_as_a_log_line() {
        let w = super::CopyWaits { late: 3, waits: 60, ms: 4.25 };
        assert_eq!(w.to_string(), "late copies 3 of 60, the GPU stood 4.2 ms for them");
    }
}

#[cfg(test)]
mod placement_gpu_tests {
    use super::*;
    #[test]
    #[ignore = "requires CUDA; small weight placement integration"]
    fn disk_promotion_does_not_materialize_ram_weights() {
        let gpu = crate::gpu::Gpu::new().unwrap(); gpu.bind().unwrap();
        let compute = Stream::new().unwrap();
        let path = std::env::temp_dir().join(format!("wan-row-test-{}.bin", std::process::id()));
        let data: Vec<u8> = (0..3).flat_map(|b| vec![b as u8 + 1; 4096]).collect();
        std::fs::write(&path, &data).unwrap();
        let mut rows = Rows::load(&path, 3, 4096, &[0, 1, 2], true, &compute).unwrap();
        let before = rows.pinned_bytes();
        rows.keep_on_device(&[1], &compute).unwrap();
        assert!(rows.on_disk());
        assert_eq!(rows.pinned_bytes(), before);
        for pass in 0..3 {
            for b in 0..3 {
                rows.row::<false>(b, &compute).unwrap();
                let ptr = match &rows.places[b] { Place::Device(d) => d.ptr, Place::Streamed(j) => rows.slots[rows.slot_of(*j)].ptr };
                let mut bytes = vec![0; 4096];
                compute.copy_to_host(&mut bytes, ptr).unwrap(); compute.synchronize().unwrap();
                assert!(bytes.iter().all(|&v| v == b as u8 + 1));
                rows.release(b, &compute).unwrap();
            }
            if pass == 0 { rows.to_ram(&compute).unwrap(); }
            if pass == 1 { rows.to_disk(&compute).unwrap(); }
        }
        compute.synchronize().unwrap(); rows.copy.synchronize().unwrap();
        drop(rows);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    #[ignore = "requires CUDA; small weight placement integration"]
    fn disk_rows_kept_on_the_gpu_are_read_without_mapping_the_pack() {
        // T5's resident rows went through a mapping of the 8.6 GiB pack: its pages counted against the RAM reserve
        // and the staging buffers after them were refused. Rows of 5000 bytes: the file ends inside a sector.
        let gpu = crate::gpu::Gpu::new().unwrap(); gpu.bind().unwrap();
        let compute = Stream::new().unwrap();
        let path = std::env::temp_dir().join(format!("wan-row-resident-{}.bin", std::process::id()));
        let data: Vec<u8> = (0..3 * 5000).map(|i| (i % 251) as u8).collect();
        std::fs::write(&path, &data).unwrap();
        let rows = Rows::load(&path, 3, 5000, &[1], true, &compute).unwrap();
        // a mapping still open would keep the file from being replaced; reads without one do not
        std::fs::write(&path, b"replaced").unwrap();
        for b in [0, 2] {
            let Place::Device(d) = &rows.places[b] else { panic!("row {b} is not on the GPU") };
            let mut bytes = vec![0; 5000];
            compute.copy_to_host(&mut bytes, d.ptr).unwrap(); compute.synchronize().unwrap();
            assert_eq!(bytes, &data[b * 5000..(b + 1) * 5000]);
        }
        drop(rows);
        std::fs::remove_file(path).unwrap();
    }
}
