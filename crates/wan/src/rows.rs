//! The blocks' weights, one row of bytes per block (the W4A4 pack, block_w4a4_v2): some rows on the GPU for good,
//! the others in pinned RAM or staged from disk into independently sized RAM/VRAM windows.
//! Rows follow a cyclic order. Calibration chooses window depths; a freed device slot is refilled one
//! window ahead in that order, including the next pass's first rows.

use std::fs::File;
use std::path::Path;

use crate::disk::{open_unbuffered, read_row, span, SECTOR};
use crate::gpu::{DeviceBuffer, Event, PinnedBuffer, Result, Stream};

/// Where the streamed rows come from: pinned RAM (one buffer per row), or the pack file on the disk, read into
/// pinned staging buffers.
enum Source {
    Pinned(Vec<PinnedBuffer>),
    Disk { file: File, staging: Vec<PinnedBuffer>, owners: Vec<Option<usize>> },
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
    /// GPU slots for the streamed rows; none while parked
    slots: Vec<DeviceBuffer>,
    copied: Vec<Event>,
    freed: Vec<Event>,
    copy: Stream,
    staging_count: usize,
    device_count: usize,
    pub row_bytes: usize,
    /// the streamed row each slot holds, or is being filled with
    holds: Vec<usize>,
    profile_cpu: std::cell::Cell<bool>,
    cpu_stats: std::cell::RefCell<ProducerStats>,
    /// debug-tools: timed events around every wait of the compute stream for a row's copy
    #[cfg(feature = "fps")]
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
        Self::load_with_staging(pack, blocks, row_bytes, streamed, disk, stream, 2)
    }

    fn load_with_staging(pack: &Path, blocks: usize, row_bytes: usize, streamed: &[usize], disk: bool, stream: &Stream, staging_count: usize)
        -> std::io::Result<Rows> {
        Self::load_with_slots(pack, blocks, row_bytes, streamed, disk, stream, staging_count, 2)
    }

    pub fn load_with_slots(pack: &Path, blocks: usize, row_bytes: usize, streamed: &[usize], disk: bool, stream: &Stream, staging_count: usize, device_count: usize)
        -> std::io::Result<Rows> {
        assert!(staging_count > 0 && device_count > 0);
        let io = |e: cudarc::driver::DriverError| std::io::Error::other(e.to_string());
        assert_eq!(std::fs::metadata(pack)?.len() as usize, blocks * row_bytes, "the pack has another size than its manifest says");
        // From the disk, the rows kept on the GPU go through the staging buffers too: a mapping of the pack would put
        // its pages into this process's working set (8.6 GiB for T5) and count them against the RAM reserve.
        let mut disk_source = if disk {
            let staging = (0..staging_count).map(|_| PinnedBuffer::new(span(row_bytes)).map_err(io)).collect::<std::io::Result<Vec<_>>>()?;
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
        let slots = (0..device_count).map(|_| DeviceBuffer::new(row_bytes).map_err(io)).collect::<std::io::Result<Vec<_>>>()?;
        let source = match disk_source {
            Some((file, staging)) => Source::Disk { file, staging, owners: vec![None; staging_count] },
            None => Source::Pinned(host),
        };
        let mut rows = Rows {
            places,
            source,
            pack: pack.to_path_buf(),
            streamed: streamed.to_vec(),
            slots,
            copied: (0..device_count).map(|_| Event::new().map_err(io)).collect::<std::io::Result<Vec<_>>>()?,
            freed: (0..device_count).map(|_| Event::new().map_err(io)).collect::<std::io::Result<Vec<_>>>()?,
            copy,
            staging_count,
            device_count,
            row_bytes,
            holds: vec![usize::MAX; device_count],
            profile_cpu: std::cell::Cell::new(false),
            cpu_stats: Default::default(),
            #[cfg(feature = "fps")]
            waits: Default::default(),
        };
        for j in 0..rows.streamed.len().min(rows.device_count) {
            rows.fill(j, j).map_err(io)?;
        }
        Ok(rows)
    }

    fn fill(&mut self, slot: usize, j: usize) -> Result<()> {
        // after the block that used the slot last is done with it
        self.copy.wait(&self.freed[slot])?;
        match &mut self.source {
            Source::Pinned(host) => self.copy.copy_to_device(self.slots[slot].ptr, host[j].as_slice())?,
            Source::Disk { file, staging, owners } => {
                // the staging buffer's last row is on the GPU, then this row is read into it
                let stage = j % staging.len();
                let began = self.profile_cpu.get().then(std::time::Instant::now);
                if let Some(owner) = owners[stage] { self.copied[owner].synchronize()?; }
                let read_began = self.profile_cpu.get().then(std::time::Instant::now);
                let at = read_row(file, self.streamed[j], self.row_bytes, staging[stage].as_mut_slice())
                    .map_err(|_| cudarc::driver::DriverError(cudarc::driver::sys::cudaError_enum::CUDA_ERROR_FILE_NOT_FOUND))?;
                if let (Some(began), Some(read_began)) = (began, read_began) {
                    let mut stats = self.cpu_stats.borrow_mut();
                    stats.refills += 1;
                    stats.staging_wait_ms += (read_began - began).as_secs_f64() * 1000.0;
                    stats.disk_read_ms += read_began.elapsed().as_secs_f64() * 1000.0;
                }
                self.copy.copy_to_device(self.slots[slot].ptr, &staging[stage].as_slice()[at..at + self.row_bytes])?;
                owners[stage] = Some(slot);
            }
        }
        self.copied[slot].record(&self.copy)?;
        self.holds[slot] = j;
        Ok(())
    }

    /// The slot that holds streamed row j. Not j % depth: with an unaligned number of rows the cycle wraps onto the
    /// other parity (row k - 1 frees the slot that row 1 of the next pass is copied into).
    fn slot_of(&self, j: usize) -> usize {
        self.holds.iter().position(|&h| h == j).expect("rows taken out of their cycle")
    }

    /// The row of `block` as a uint8 tensor for work on `compute`, which waits for its copy if it is streamed.
    pub fn row<const TIMED: bool>(&self, block: usize, compute: &Stream) -> Result<aoti::Tensor> {
        if TIMED { self.profile_cpu.set(true); }
        let shape = [self.row_bytes as i64];
        match &self.places[block] {
            Place::Device(buffer) => Ok(buffer.tensor(0, aoti::DType::U8, &shape)),
            Place::Streamed(j) => {
                let slot = self.slot_of(*j);
                #[cfg(feature = "fps")]
                let span = if TIMED { Some((Event::timed()?, Event::timed()?)) } else { None };
                #[cfg(feature = "fps")]
                if let Some((start, _)) = &span { start.record(compute)?; }
                compute.wait(&self.copied[slot])?;
                #[cfg(feature = "fps")]
                if let Some((start, end)) = span {
                    end.record(compute)?;
                    self.waits.borrow_mut().push((start, end));
                }
                Ok(self.slots[slot].tensor(0, aoti::DType::U8, &shape))
            }
        }
    }

    /// `block`'s work is queued on `compute`: refill its slot one window ahead in the cycle.
    pub fn release(&mut self, block: usize, compute: &Stream) -> Result<()> {
        if let Place::Streamed(j) = self.places[block] {
            let slot = self.slot_of(j);
            self.freed[slot].record(compute)?;
            self.fill(slot, (j + self.device_count.min(self.streamed.len())) % self.streamed.len())?;
        }
        Ok(())
    }

    /// Streams only the blocks of `streamed` from now on (a subset of those streamed now, as `spread` picks them):
    /// the others get a device buffer of their own and give their pinned memory back. Waits for the GPU first.
    pub fn keep_on_device(&mut self, streamed: &[usize], compute: &Stream) -> Result<()> {
        compute.synchronize()?;
        self.copy.synchronize()?;
        if let Source::Disk { file, staging, .. } = &mut self.source {
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
            self.holds.fill(usize::MAX);
            for j in 0..self.streamed.len().min(self.device_count) { self.fill(j, j)?; }
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
        self.holds.fill(usize::MAX);
        for j in 0..self.streamed.len().min(self.device_count) {
            self.fill(j, j)?;
        }
        Ok(())
    }

    /// Milliseconds of one row's copy from pinned RAM to the GPU, alone on the copy stream (the median of three).
    pub fn time_copy(&self) -> Result<f32> {
        // every row resident on the GPU (block_weights vram): no row is ever copied, nothing to time
        let empty = match &self.source { Source::Pinned(host) => host.is_empty(), Source::Disk { staging, .. } => staging.is_empty() };
        if empty { return Ok(0.0); }
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

    /// Of the waits for copies since the last call, how many were late (the compute stream stood more
    /// than 0.05 ms), how many there were, and the milliseconds it stood in all. Waits for the GPU.
    #[cfg(feature = "fps")]
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
        self.holds.fill(usize::MAX);
        Ok(())
    }

    /// Slots again, the first streamed rows copied into them.
    pub fn unpark(&mut self) -> Result<()> {
        if self.slots.is_empty() && !self.streamed.is_empty() {
            self.slots = (0..self.device_count).map(|_| DeviceBuffer::new(self.row_bytes)).collect::<Result<Vec<_>>>()?;
            for j in 0..self.streamed.len().min(self.device_count) {
                self.fill(j, j)?;
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

    pub fn slot_counts(&self) -> (usize, usize) { (self.staging_count, self.device_count) }

    pub fn profile_cpu(&self, enabled: bool) { self.profile_cpu.set(enabled); }
    pub fn producer_stats(&self) -> ProducerStats { *self.cpu_stats.borrow() }

    /// Rebuild only transient windows. Immutable resident rows and KV placement are unchanged.
    pub fn reconfigure(&mut self, host: usize, device: usize, compute: &Stream) -> Result<()> {
        assert!(host > 0 && device > 0);
        compute.synchronize()?; self.copy.synchronize()?;
        self.slots.clear();
        if let Source::Disk { staging, owners, .. } = &mut self.source {
            staging.clear(); owners.clear();
            for _ in 0..host { staging.push(PinnedBuffer::new(span(self.row_bytes))?); owners.push(None); }
        }
        self.staging_count = host; self.device_count = device;
        self.copied = (0..device).map(|_| Event::new()).collect::<Result<Vec<_>>>()?;
        self.freed = (0..device).map(|_| Event::new()).collect::<Result<Vec<_>>>()?;
        self.holds = vec![usize::MAX; device];
        self.unpark()
    }

    /// Enumerate all feasible depths; no fixed upper bound such as two or five slots.
    pub fn calibrate_schedule(&mut self, read_ms: f64, copy_ms: f64, compute_ms: f64,
        spare: f64, frames_per_pass: f64, fps_tolerance: f64, fixed_seconds_per_pass: f64, gpu_reserve: usize, compute: &Stream) -> crate::video::Result<serde_json::Value> {
        compute.synchronize()?; self.copy.synchronize()?;
        let n = self.streamed.len();
        if n == 0 { return Ok(serde_json::json!({"reason":"all weight blocks resident"})); }
        let host_budget = crate::budget::available()?.saturating_add(if self.on_disk() { self.pinned_bytes() } else { 0 });
        let (free, _) = cudarc::driver::result::mem_get_info()?;
        let device_budget = (free as usize).saturating_sub(gpu_reserve).saturating_add(self.slots.iter().map(|b| b.bytes).sum::<usize>());
        let host_size = if self.on_disk() { span(self.row_bytes) } else { 0 };
        let mut candidates = Vec::new();
        for device in 1..=n.min(device_budget / self.row_bytes) {
            // More host slots than outstanding device transfers cannot help this producer.
            let max_host = if host_size == 0 { 1 } else { device.min(host_budget / host_size) };
            for host in 1..=max_host {
                if let Some(seconds) = crate::prefetch::simulate(n, host, device, read_ms * spare, copy_ms * spare, compute_ms, self.on_disk()) {
                    candidates.push(crate::prefetch::Candidate { host, device, host_bytes: host * host_size,
                        device_bytes: device * self.row_bytes, seconds: seconds + fixed_seconds_per_pass });
                }
            }
        }
        let chosen = crate::prefetch::choose(&candidates, host_budget, device_budget, frames_per_pass, fps_tolerance)
            .ok_or("no prefetch schedule fits remaining RAM/VRAM budgets")?;
        let old = self.slot_counts();
        if old != (chosen.host, chosen.device) { self.reconfigure(chosen.host, chosen.device, compute)?; }
        Ok(serde_json::json!({"source":"native producer/copy/compute cost model; FlexGen-inspired adaptation",
            "costs_ms":{"read":read_ms,"copy":copy_ms,"compute":compute_ms},"copy_spare":spare,
            "host_budget":host_budget,"device_budget":device_budget,"frames_per_pass":frames_per_pass,
            "fps_tolerance":fps_tolerance,"fixed_seconds_per_pass":fixed_seconds_per_pass,
            "fps_scope":"predicted steady throughput; observed fixed pipeline cost plus simulated block chain",
            "observed_producer":self.producer_stats(),
            "previous":{"host":old.0,"device":old.1},"chosen":chosen,"candidates":candidates}))
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
        self.holds.fill(usize::MAX);
        for j in 0..self.streamed.len().min(self.device_count) {
            self.fill(j, j).map_err(io)?;
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
        let staging = (0..self.staging_count).map(|_| PinnedBuffer::new(span(self.row_bytes)).map_err(io)).collect::<std::io::Result<Vec<_>>>()?;
        self.source = Source::Disk { file: open_unbuffered(&self.pack)?, staging, owners: vec![None; self.staging_count] };
        self.holds.fill(usize::MAX);
        for j in 0..self.streamed.len().min(self.device_count) {
            self.fill(j, j).map_err(io)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct ProducerStats {
    pub refills: usize,
    pub staging_wait_ms: f64,
    pub disk_read_ms: f64,
}

/// debug-tools: waits of the compute stream for rows' copies: how many stood more than 0.05 ms, how many there were,
/// and the milliseconds it stood in all.
#[cfg(feature = "fps")]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct CopyWaits {
    pub late: usize,
    pub waits: usize,
    pub ms: f32,
}

#[cfg(feature = "fps")]
impl CopyWaits {
    pub fn of(times: &[f32]) -> CopyWaits {
        CopyWaits { late: times.iter().filter(|&&ms| ms > 0.05).count(), waits: times.len(), ms: times.iter().sum() }
    }
}

#[cfg(feature = "fps")]
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

    #[cfg(feature = "fps")]
    #[test]
    fn a_wait_over_a_twentieth_of_a_millisecond_is_late() {
        let w = super::CopyWaits::of(&[0.0, 0.05, 0.06, 0.2]);
        assert_eq!((w.late, w.waits), (2, 4));
        assert!((w.ms - 0.31).abs() < 1e-6);
    }

    #[cfg(feature = "fps")]
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
    #[ignore = "requires CUDA; variable prefetch depths preserve cyclic weight ordering"]
    fn variable_device_depth_preserves_order() {
        let gpu = crate::gpu::Gpu::new().unwrap(); gpu.bind().unwrap();
        let compute = Stream::new().unwrap();
        let path = std::env::temp_dir().join(format!("wan-depth-{}.bin", std::process::id()));
        std::fs::write(&path, (0..5).flat_map(|b| vec![b as u8 + 1; 4096]).collect::<Vec<_>>()).unwrap();
        let mut rows = Rows::load_with_slots(&path, 5, 4096, &[0, 1, 2, 3, 4], true, &compute, 2, 2).unwrap();
        for depth in 1..=5 {
            rows.reconfigure(1, depth, &compute).unwrap();
            assert_eq!(rows.slot_counts(), (1, depth));
            rows.park(&compute).unwrap(); rows.unpark().unwrap();
            let snapshots = DeviceBuffer::new(15 * 4096).unwrap();
            for i in 0..15 {
                let b = i % 5;
                rows.row::<false>(b, &compute).unwrap();
                let Place::Streamed(j) = rows.places[b] else { panic!("expected streamed row") };
                compute.copy_on_device(snapshots.at(i * 4096), rows.slots[rows.slot_of(j)].ptr, 4096).unwrap();
                rows.release(b, &compute).unwrap();
            }
            compute.synchronize().unwrap(); rows.copy.synchronize().unwrap();
            let mut bytes = vec![0; 15 * 4096];
            compute.copy_to_host(&mut bytes, snapshots.ptr).unwrap(); compute.synchronize().unwrap();
            for i in 0..15 { assert!(bytes[i*4096..(i+1)*4096].iter().all(|&v| v == (i%5) as u8 + 1)); }
        }
        drop(rows); std::fs::remove_file(path).unwrap();
    }
    #[test]
    #[ignore = "requires CUDA; checks one shared disk staging buffer across repeated transfers"]
    fn one_staging_buffer_preserves_rows_across_wraparound() {
        let gpu = crate::gpu::Gpu::new().unwrap(); gpu.bind().unwrap();
        let compute = Stream::new().unwrap();
        let path = std::env::temp_dir().join(format!("wan-one-staging-{}.bin", std::process::id()));
        let data: Vec<u8> = (0..5).flat_map(|b| vec![b as u8 + 1; 4096]).collect();
        std::fs::write(&path, &data).unwrap();
        let mut rows = Rows::load_with_staging(&path, 5, 4096, &[0, 1, 2, 3, 4], true, &compute, 1).unwrap();
        assert_eq!(rows.pinned_bytes(), crate::disk::span(4096));
        let snapshots = DeviceBuffer::new(20 * 4096).unwrap();
        for pass in 0..4 {
            for b in 0..5 {
                rows.row::<false>(b, &compute).unwrap();
                let Place::Streamed(j) = rows.places[b] else { panic!("expected streamed row") };
                compute.copy_on_device(snapshots.at((pass * 5 + b) * 4096), rows.slots[rows.slot_of(j)].ptr, 4096).unwrap();
                rows.release(b, &compute).unwrap();
            }
        }
        compute.synchronize().unwrap(); rows.copy.synchronize().unwrap();
        let mut bytes = vec![0; 20 * 4096];
        compute.copy_to_host(&mut bytes, snapshots.ptr).unwrap(); compute.synchronize().unwrap();
        for i in 0..20 { assert!(bytes[i * 4096..(i + 1) * 4096].iter().all(|&v| v == (i % 5) as u8 + 1)); }
        drop(rows); std::fs::remove_file(path).unwrap();
    }
    #[cfg(feature = "fps")]
    #[test]
    #[ignore = "requires CUDA; tiny transfer telemetry integration"]
    fn transfer_metrics_are_opt_in_and_drained_once() {
        let gpu = crate::gpu::Gpu::new().unwrap(); gpu.bind().unwrap();
        let compute = Stream::new().unwrap();
        let path = std::env::temp_dir().join(format!("wan-metrics-test-{}.bin", std::process::id()));
        std::fs::write(&path, vec![1u8; 3 * 4096]).unwrap();
        let mut rows = Rows::load(&path, 3, 4096, &[0, 1, 2], true, &compute).unwrap();
        for b in 0..3 { rows.row::<false>(b, &compute).unwrap(); rows.release(b, &compute).unwrap(); }
        assert_eq!(rows.copy_waits(&compute).unwrap(), (0, 0, 0.0));
        for b in 0..3 { rows.row::<true>(b, &compute).unwrap(); rows.release(b, &compute).unwrap(); }
        let (late, waits, ms) = rows.copy_waits(&compute).unwrap();
        assert_eq!(waits, 3);
        assert!(late <= waits && ms.is_finite() && ms >= 0.0);
        assert_eq!(rows.copy_waits(&compute).unwrap(), (0, 0, 0.0));
        compute.synchronize().unwrap(); rows.copy.synchronize().unwrap();
        drop(rows);
        std::fs::remove_file(path).unwrap();
    }

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

