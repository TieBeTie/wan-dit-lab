//! Shared optional diagnostics for Live, offline and bench.

pub trait Probe: Send + 'static {
    fn ready(&mut self, frames: usize);
    /// Frames handed over with no compute of their own (the flush of a chunk whose compute the advance before it
    /// timed): no fps sample.
    fn drained(&mut self, _frames: usize) {}
    /// Whether the chunk just computed (the interval of the next `ready`) attended to the full KV window.
    fn window(&mut self, _full: bool) {}
    fn reset(&mut self) {}
    fn transfers(&self) -> bool { false }
    fn chunk(&mut self, _: &crate::video::Video) -> crate::Result<()> { Ok(()) }
    fn prompt(&mut self, _: f64) {}
    fn finish(&mut self) -> std::io::Result<()> { Ok(()) }
}

pub struct Off;
impl Probe for Off {
    #[inline(always)]
    fn ready(&mut self, _: usize) {}
}

#[cfg(feature = "fps")]
pub use enabled::Metrics;

#[cfg(feature = "fps")]
mod enabled {
    use std::io::{self, Write};
    use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
    use std::time::{Duration, Instant};
    use rtrb::{Consumer, Producer, RingBuffer};

    pub struct Metrics {
        producer: Option<Producer<Sample>>,
        worker: Option<std::thread::JoinHandle<io::Result<()>>>,
        dropped: Arc<AtomicU64>,
        last: Instant,
        warmup: bool,
        full: bool,
        selected: crate::config::DebugSelection,
    }
    impl Metrics {
        pub fn spawn(drain: &crate::config::Drain, selected: &crate::config::DebugSelection) -> crate::Result<Self> {
            let every = drain.interval()?;
            let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"))
                .ok_or("cannot locate metrics output directory")?;
            let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos();
            let path = std::path::PathBuf::from(home).join("Documents/Experiments/DIT/outputs/metrics")
                .join(format!("{stamp}-{}.jsonl", std::process::id()));
            let (mut fps, queue) = Self::channel(256);
            fps.selected = selected.clone();
            let dropped = fps.dropped.clone();
            eprintln!("Metrics: {}", path.display());
            fps.worker = Some(std::thread::Builder::new().name("metrics".into()).spawn(move || {
                std::fs::create_dir_all(path.parent().unwrap())?;
                let writer = io::BufWriter::new(std::fs::File::create(path)?);
                collect(queue, dropped, every, writer)
            })?);
            Ok(fps)
        }
        fn channel(capacity: usize) -> (Self, Consumer<Sample>) {
            let (producer, consumer) = RingBuffer::new(capacity);
            (Self { producer: Some(producer), worker: None, dropped: Arc::new(AtomicU64::new(0)),
                    last: Instant::now(), warmup: true, full: false, selected: crate::config::DebugSelection::Tool(crate::config::DebugTool::Fps) }, consumer)
        }
        #[inline]
        fn submit(&mut self, sample: Sample) {
            if self.producer.as_mut().unwrap().push(sample).is_err() {
                self.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    impl super::Probe for Metrics {
        #[inline]
        fn ready(&mut self, frames: usize) {
            if frames == 0 || !self.selected.includes(crate::config::DebugTool::Fps) { return; }
            let now = Instant::now();
            let sample = Sample { frames, elapsed: now - self.last, warmup: self.warmup, full: self.full, detail: None };
            self.last = now;
            self.warmup = false;
            self.submit(sample);
        }
        fn window(&mut self, full: bool) { self.full = full; }
        fn transfers(&self) -> bool { self.selected.includes(crate::config::DebugTool::Transfers) }
        fn chunk(&mut self, video: &crate::video::Video) -> crate::Result<()> {
            if self.selected.includes(crate::config::DebugTool::Memory) {
                self.submit(Sample::detail(Detail::Memory(crate::measure::memory_sample(video)?)));
            }
            if self.transfers() {
                let (late, waits, ms) = video.copy_waits()?;
                self.submit(Sample::detail(Detail::Transfers { late, waits, ms }));
            }
            Ok(())
        }
        fn prompt(&mut self, seconds: f64) {
            if self.selected.includes(crate::config::DebugTool::Prompts) {
                self.submit(Sample::detail(Detail::Prompt { seconds }));
            }
        }
        fn reset(&mut self) { self.last = Instant::now(); self.warmup = true; }
        fn finish(&mut self) -> io::Result<()> {
            self.producer.take();
            if let Some(worker) = self.worker.take() {
                worker.thread().unpark();
                worker.join().map_err(|_| io::Error::other("metrics collector panicked"))??;
            }
            Ok(())
        }
    }
    impl Drop for Metrics {
        fn drop(&mut self) {
            if let Err(e) = super::Probe::finish(self) { eprintln!("metrics collector failed: {e}"); }
        }
    }

    fn collect(mut queue: Consumer<Sample>, dropped: Arc<AtomicU64>, every: Option<Duration>,
               mut writer: impl Write) -> io::Result<()> {
        let mut totals = Totals::default();
        let mut last_write = Instant::now();
        let mut dirty = false;
        loop {
            // Check closing before draining, so every record published before close is included.
            let closed = queue.is_abandoned();
            while let Ok(sample) = queue.pop() { totals.add(sample); dirty = true; }
            if closed || (dirty && every.is_some_and(|period| last_write.elapsed() >= period)) {
                let mut value = totals.value(dropped.load(Ordering::Relaxed));
                value["final"] = closed.into();
                serde_json::to_writer(&mut writer, &value)?;
                writer.write_all(b"\n")?;
                writer.flush()?;
                dirty = false;
                last_write = Instant::now();
            }
            if closed { return Ok(()); }
            // No wake-up syscall in the producer, and no busy polling in the collector.
            std::thread::park_timeout(every.unwrap_or(Duration::from_millis(100)).min(Duration::from_millis(100)));
        }
    }

    #[derive(Clone, Copy)]
    struct Sample { frames: usize, elapsed: Duration, warmup: bool, full: bool, detail: Option<Detail> }
    impl Sample {
        fn detail(detail: Detail) -> Self { Self { frames: 0, elapsed: Duration::ZERO, warmup: false, full: false, detail: Some(detail) } }
    }
    #[derive(Clone, Copy)]
    enum Detail { Memory(crate::measure::MemorySample), Transfers { late: usize, waits: usize, ms: f32 }, Prompt { seconds: f64 } }

    #[derive(Default)]
    struct Totals {
        frames: u64, elapsed: Duration, chunks: u64, warmup_chunks: u64,
        // the chunks computed with the full KV window: the speed the run settles at
        steady_frames: u64, steady_elapsed: Duration, steady_chunks: u64,
        // the last chunk (full KV window by the end of a run) and the slowest one, as (frames, elapsed)
        full: Vec<Duration>, chunk_frames: usize,
        memory: Option<crate::measure::MemorySample>, available_min: Option<usize>, gpu_used_max: usize,
        transfer_samples: usize, late: usize, waits: usize, wait_ms: f64,
        prompts: usize, prompt_seconds: f64, prompt_max: f64,
    }
    impl Totals {
        fn add(&mut self, sample: Sample) {
            if let Some(detail) = sample.detail {
                match detail {
                    Detail::Memory(m) => {
                        self.available_min = Some(self.available_min.map_or(m.system_available_bytes, |v| v.min(m.system_available_bytes)));
                        self.gpu_used_max = self.gpu_used_max.max(m.gpu_used_bytes);
                        self.memory = Some(m);
                    }
                    Detail::Transfers { late, waits, ms } => {
                        self.transfer_samples += 1; self.late += late; self.waits += waits; self.wait_ms += ms as f64;
                    }
                    Detail::Prompt { seconds } => {
                        self.prompts += 1; self.prompt_seconds += seconds; self.prompt_max = self.prompt_max.max(seconds);
                    }
                }
                return;
            }
            if sample.warmup { self.warmup_chunks += 1; return; }
            self.frames += sample.frames as u64;
            self.elapsed += sample.elapsed;
            self.chunks += 1;
            if sample.full {
                self.steady_frames += sample.frames as u64; self.steady_elapsed += sample.elapsed; self.steady_chunks += 1;
            }
            let seconds = sample.elapsed.as_secs_f64();
            // a chunk is the largest number of frames seen: a smaller sample is the decoder's tail of a video
            if sample.frames > self.chunk_frames { self.chunk_frames = sample.frames; self.full.clear(); }
            if seconds > 0.0 && sample.frames == self.chunk_frames { self.full.push(sample.elapsed); }
        }
        /// The last chunk that was computed: a sample far shorter than the mean (under a quarter) is the flush of a
        /// chunk decoded already, its frames emitted right after the chunk before.
        fn last_chunk(&self) -> Option<Duration> {
            let mean = self.full.iter().sum::<Duration>().as_secs_f64() / self.full.len().max(1) as f64;
            self.full.iter().rev().find(|e| e.as_secs_f64() >= 0.25 * mean).copied()
        }
        fn value(&self, dropped: u64) -> serde_json::Value {
            let seconds = self.elapsed.as_secs_f64();
            serde_json::json!({"frames": self.frames, "seconds": seconds, "chunks": self.chunks,
                "fps": (seconds > 0.0).then(|| self.frames as f64 / seconds),
                "seconds_per_frame": (self.frames > 0).then(|| seconds / self.frames as f64),
                "steady_fps": (self.steady_elapsed > Duration::ZERO).then(|| self.steady_frames as f64 / self.steady_elapsed.as_secs_f64()),
                "steady_chunks": self.steady_chunks,
                "memory": self.memory.map(|latest| serde_json::json!({"latest": latest,
                    "system_available_min_sampled_bytes": self.available_min, "gpu_used_max_sampled_bytes": self.gpu_used_max})),
                "weight_transfers": (self.transfer_samples > 0).then(|| serde_json::json!({"samples": self.transfer_samples, "waits": self.waits, "late": self.late, "wait_ms": self.wait_ms})),
                "prompt_encoding": (self.prompts > 0).then(|| serde_json::json!({"count": self.prompts, "seconds": self.prompt_seconds, "max_seconds": self.prompt_max})),
                "last_chunk_fps": self.last_chunk().map(|e| self.chunk_frames as f64 / e.as_secs_f64()),
                "last_chunk_seconds": self.last_chunk().map(|e| e.as_secs_f64()),
                "slowest_chunk_fps": self.full.iter().max().map(|e| self.chunk_frames as f64 / e.as_secs_f64()),
                "warmup_chunks": self.warmup_chunks, "dropped": dropped, "complete": dropped == 0})
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn detailed_metrics_are_drained_without_counting_as_frames() {
            let (mut metrics, queue) = Metrics::channel(8);
            metrics.submit(Sample::detail(Detail::Memory(crate::measure::MemorySample {
                system_available_bytes: 100, gpu_used_bytes: 200, ..Default::default()
            })));
            metrics.submit(Sample::detail(Detail::Memory(crate::measure::MemorySample {
                system_available_bytes: 80, gpu_used_bytes: 150, ..Default::default()
            })));
            for seconds in [1.0, 2.0] { metrics.submit(Sample::detail(Detail::Prompt { seconds })); }
            metrics.submit(Sample::detail(Detail::Transfers { late: 1, waits: 3, ms: 0.5 }));
            metrics.submit(Sample::detail(Detail::Transfers { late: 2, waits: 4, ms: 1.0 }));
            metrics.producer.take();
            let mut bytes = Vec::new();
            collect(queue, metrics.dropped.clone(), None, &mut bytes).unwrap();
            let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(v["frames"], 0);
            assert_eq!(v["chunks"], 0);
            assert!(v["fps"].is_null());
            assert_eq!(v["memory"]["system_available_min_sampled_bytes"], 80);
            assert_eq!(v["memory"]["gpu_used_max_sampled_bytes"], 200);
            assert_eq!(v["memory"]["latest"]["gpu_used_bytes"], 150);
            assert_eq!(v["weight_transfers"], serde_json::json!({"samples": 2, "waits": 7, "late": 3, "wait_ms": 1.5}));
            assert_eq!(v["prompt_encoding"], serde_json::json!({"count": 2, "seconds": 3.0, "max_seconds": 2.0}));
            assert_eq!(v["final"], true);
        }

        #[test]
        fn steady_fps_counts_only_chunks_computed_with_the_full_window() {
            // the mean keeps every chunk after the warmup; steady_fps only the chunks whose KV window was full
            let mut totals = Totals::default();
            totals.add(Sample { frames: 29, elapsed: Duration::from_secs(20), warmup: true, full: false, detail: None });
            totals.add(Sample { frames: 32, elapsed: Duration::from_secs(1), warmup: false, full: false, detail: None });
            totals.add(Sample { frames: 32, elapsed: Duration::from_secs(4), warmup: false, full: true, detail: None });
            totals.add(Sample { frames: 32, elapsed: Duration::from_secs(4), warmup: false, full: true, detail: None });
            let v = totals.value(0);
            assert_eq!(v["fps"], 96.0 / 9.0);
            assert_eq!(v["steady_fps"], 8.0);
            assert_eq!(v["steady_chunks"], 2);
            assert_eq!(Totals::default().value(0)["steady_fps"], serde_json::Value::Null);
        }

        #[test]
        fn the_full_window_of_a_chunk_goes_with_the_sample_of_its_interval() {
            // advance k times chunk k's compute and hands over chunk k - 1's frames: the flag is chunk k's
            use crate::telemetry::Probe;
            let (mut metrics, mut queue) = Metrics::channel(8);
            metrics.window(true);
            metrics.ready(32);
            assert!(queue.pop().unwrap().full);
            metrics.window(false);
            metrics.ready(32);
            assert!(!queue.pop().unwrap().full);
        }

        #[test]
        fn frames_drained_without_compute_make_no_fps_sample() {
            // flush hands over the frames of a chunk whose compute was timed by the advance before it: a sample here
            // would count those frames a second time against a near zero interval (160 frames over 4 intervals)
            use crate::telemetry::Probe;
            let (mut metrics, mut queue) = Metrics::channel(8);
            metrics.ready(29);
            assert!(queue.pop().unwrap().warmup);
            metrics.drained(32);
            assert!(queue.pop().is_err());
            metrics.ready(32);
            assert_eq!(queue.pop().unwrap().frames, 32);
        }

        #[test]
        fn only_selected_callbacks_publish_metrics() {
            use crate::telemetry::Probe;
            let (mut metrics, mut queue) = Metrics::channel(8);
            metrics.prompt(1.0);
            assert!(queue.pop().is_err());
            assert!(!metrics.transfers());
            metrics.selected = crate::config::DebugSelection::Tool(crate::config::DebugTool::Prompts);
            metrics.ready(32);
            assert!(queue.pop().is_err());
            metrics.prompt(2.0);
            assert!(matches!(queue.pop().unwrap().detail, Some(Detail::Prompt { seconds: 2.0 })));
            metrics.selected = crate::config::DebugSelection::Switch(true);
            assert!(metrics.transfers());
        }

        #[test]
        fn fps_uses_actual_frames_and_total_time_and_excludes_warmup() {
            let mut totals = Totals::default();
            totals.add(Sample { frames: 29, elapsed: Duration::from_secs(20), warmup: true, full: false, detail: None });
            totals.add(Sample { frames: 32, elapsed: Duration::from_secs(2), warmup: false, full: false, detail: None });
            totals.add(Sample { frames: 16, elapsed: Duration::from_secs(4), warmup: false, full: false, detail: None });
            let v = totals.value(3);
            assert_eq!(v["frames"], 48);
            assert_eq!(v["seconds"], 6.0);
            assert_eq!(v["fps"], 8.0);
            assert_eq!(v["seconds_per_frame"], 0.125);
            assert_eq!(v["warmup_chunks"], 1);
            assert_eq!(v["dropped"], 3);
            assert_eq!(v["complete"], false);
            assert_eq!(Totals::default().value(0)["fps"], serde_json::Value::Null);
        }

        #[test]
        fn the_last_chunk_has_its_own_fps_and_the_slowest_chunk_is_kept() {
            // the last chunk runs with the full KV window, the average mixes in the chunks that filled it
            let mut totals = Totals::default();
            totals.add(Sample { frames: 32, elapsed: Duration::from_secs(20), warmup: true, full: false, detail: None });
            totals.add(Sample { frames: 32, elapsed: Duration::from_secs(2), warmup: false, full: false, detail: None });
            totals.add(Sample { frames: 32, elapsed: Duration::from_secs(8), warmup: false, full: false, detail: None });
            totals.add(Sample { frames: 32, elapsed: Duration::from_secs(4), warmup: false, full: false, detail: None });
            // the decoder's tail at the end of a video: fewer frames, no chunk of its own
            totals.add(Sample { frames: 3, elapsed: Duration::from_millis(10), warmup: false, full: false, detail: None });
            // the flush of the last decoded chunk right after the one before: all its frames, no time of its own
            totals.add(Sample { frames: 32, elapsed: Duration::from_millis(60), warmup: false, full: false, detail: None });
            let v = totals.value(0);
            assert_eq!(v["last_chunk_fps"], 8.0);
            assert_eq!(v["last_chunk_seconds"], 4.0);
            assert_eq!(v["slowest_chunk_fps"], 4.0);
            assert_eq!(Totals::default().value(0)["last_chunk_fps"], serde_json::Value::Null);
        }

        #[test]
        fn end_drains_remaining_records_once_without_keeping_a_history() {
            let (mut fps, queue) = Metrics::channel(2);
            fps.submit(Sample { frames: 29, elapsed: Duration::from_secs(10), warmup: true, full: false, detail: None });
            fps.submit(Sample { frames: 32, elapsed: Duration::from_secs(2), warmup: false, full: false, detail: None });
            fps.producer.take();
            let mut bytes = Vec::new();
            collect(queue, fps.dropped.clone(), None, &mut bytes).unwrap();
            let text = String::from_utf8(bytes).unwrap();
            let lines: Vec<_> = text.lines().collect();
            assert_eq!(lines.len(), 1);
            let v: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
            assert_eq!(v["frames"], 32);
            assert_eq!(v["fps"], 16.0);
            assert_eq!(v["final"], true);
            assert_eq!(v["complete"], true);
        }

        #[test]
        fn a_blocked_disk_does_not_block_the_producer_and_overflow_is_reported() {
            use std::sync::{Barrier, Mutex, mpsc};
            // Only the test writer blocks. The generating thread must keep going while it is blocked.
            struct Gate {
                entered: Option<mpsc::Sender<()>>,
                release: Arc<Barrier>,
                bytes: Arc<Mutex<Vec<u8>>>,
            }
            impl Write for Gate {
                fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                    if let Some(entered) = self.entered.take() {
                        entered.send(()).unwrap();
                        self.release.wait();
                    }
                    self.bytes.lock().unwrap().extend_from_slice(bytes);
                    Ok(bytes.len())
                }
                fn flush(&mut self) -> io::Result<()> { Ok(()) }
            }
            let (mut fps, queue) = Metrics::channel(2);
            let sample = Sample { frames: 32, elapsed: Duration::from_secs(2), warmup: false, full: false, detail: None };
            fps.submit(sample);
            let (entered, waiting) = mpsc::channel();
            let release = Arc::new(Barrier::new(2));
            let bytes = Arc::new(Mutex::new(Vec::new()));
            let writer = Gate { entered: Some(entered), release: release.clone(), bytes: bytes.clone() };
            let dropped = fps.dropped.clone();
            let worker = std::thread::spawn(move || collect(queue, dropped, Some(Duration::from_nanos(1)), writer));
            waiting.recv_timeout(Duration::from_secs(3)).expect("collector did not write on its interval");
            // The disk is still blocked here. A blocking producer would deadlock this test.
            for _ in 0..10 { fps.submit(sample); }
            assert_eq!(fps.dropped.load(Ordering::Relaxed), 8);
            fps.producer.take();
            release.wait();
            worker.thread().unpark();
            worker.join().unwrap().unwrap();
            let text = String::from_utf8(bytes.lock().unwrap().clone()).unwrap();
            let last: serde_json::Value = serde_json::from_str(text.lines().last().unwrap()).unwrap();
            assert_eq!(last["frames"], 96);
            assert_eq!(last["fps"], 16.0);
            assert_eq!(last["dropped"], 8);
            assert_eq!(last["complete"], false);
            assert_eq!(last["final"], true);
        }
    }
}
