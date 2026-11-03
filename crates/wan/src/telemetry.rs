//! Optional FPS collection. No-op and FPS loops are separate compiler specializations.

pub trait Probe: Send + 'static {
    fn ready(&mut self, frames: usize);
    fn reset(&mut self) {}
    fn finish(&mut self) -> std::io::Result<()> { Ok(()) }
}

pub struct Off;
impl Probe for Off {
    #[inline(always)]
    fn ready(&mut self, _: usize) {}
}

#[cfg(feature = "fps")]
pub use enabled::Fps;

#[cfg(feature = "fps")]
mod enabled {
    use std::io::{self, Write};
    use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
    use std::time::{Duration, Instant};
    use rtrb::{Consumer, Producer, RingBuffer};

    pub struct Fps {
        producer: Option<Producer<Sample>>,
        worker: Option<std::thread::JoinHandle<io::Result<()>>>,
        dropped: Arc<AtomicU64>,
        last: Instant,
        warmup: bool,
    }
    impl Fps {
        pub fn spawn(drain: &crate::config::Drain) -> crate::Result<Self> {
            let every = drain.interval()?;
            let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"))
                .ok_or("cannot locate metrics output directory")?;
            let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos();
            let path = std::path::PathBuf::from(home).join("Documents/Experiments/DIT/outputs/metrics")
                .join(format!("{stamp}-{}.jsonl", std::process::id()));
            let (mut fps, queue) = Self::channel(256);
            let dropped = fps.dropped.clone();
            eprintln!("FPS metrics: {}", path.display());
            fps.worker = Some(std::thread::Builder::new().name("fps".into()).spawn(move || {
                std::fs::create_dir_all(path.parent().unwrap())?;
                let writer = io::BufWriter::new(std::fs::File::create(path)?);
                collect(queue, dropped, every, writer)
            })?);
            Ok(fps)
        }
        fn channel(capacity: usize) -> (Self, Consumer<Sample>) {
            let (producer, consumer) = RingBuffer::new(capacity);
            (Self { producer: Some(producer), worker: None, dropped: Arc::new(AtomicU64::new(0)),
                    last: Instant::now(), warmup: true }, consumer)
        }
        #[inline]
        fn submit(&mut self, sample: Sample) {
            if self.producer.as_mut().unwrap().push(sample).is_err() {
                self.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    impl super::Probe for Fps {
        #[inline]
        fn ready(&mut self, frames: usize) {
            if frames == 0 { return; }
            let now = Instant::now();
            let sample = Sample { frames, elapsed: now - self.last, warmup: self.warmup };
            self.last = now;
            self.warmup = false;
            self.submit(sample);
        }
        fn reset(&mut self) { self.last = Instant::now(); self.warmup = true; }
        fn finish(&mut self) -> io::Result<()> {
            self.producer.take();
            if let Some(worker) = self.worker.take() {
                worker.thread().unpark();
                worker.join().map_err(|_| io::Error::other("FPS collector panicked"))??;
            }
            Ok(())
        }
    }
    impl Drop for Fps {
        fn drop(&mut self) {
            if let Err(e) = super::Probe::finish(self) { eprintln!("FPS metrics failed: {e}"); }
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
    struct Sample { frames: usize, elapsed: Duration, warmup: bool }

    #[derive(Default)]
    struct Totals { frames: u64, elapsed: Duration, chunks: u64, warmup_chunks: u64 }
    impl Totals {
        fn add(&mut self, sample: Sample) {
            if sample.warmup { self.warmup_chunks += 1; return; }
            self.frames += sample.frames as u64;
            self.elapsed += sample.elapsed;
            self.chunks += 1;
        }
        fn value(&self, dropped: u64) -> serde_json::Value {
            let seconds = self.elapsed.as_secs_f64();
            serde_json::json!({"frames": self.frames, "seconds": seconds, "chunks": self.chunks,
                "fps": (seconds > 0.0).then(|| self.frames as f64 / seconds),
                "seconds_per_frame": (self.frames > 0).then(|| seconds / self.frames as f64),
                "warmup_chunks": self.warmup_chunks, "dropped": dropped, "complete": dropped == 0})
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn fps_uses_actual_frames_and_total_time_and_excludes_warmup() {
            let mut totals = Totals::default();
            totals.add(Sample { frames: 29, elapsed: Duration::from_secs(20), warmup: true });
            totals.add(Sample { frames: 32, elapsed: Duration::from_secs(2), warmup: false });
            totals.add(Sample { frames: 16, elapsed: Duration::from_secs(4), warmup: false });
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
        fn end_drains_remaining_records_once_without_keeping_a_history() {
            let (mut fps, queue) = Fps::channel(2);
            fps.submit(Sample { frames: 29, elapsed: Duration::from_secs(10), warmup: true });
            fps.submit(Sample { frames: 32, elapsed: Duration::from_secs(2), warmup: false });
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
            let (mut fps, queue) = Fps::channel(2);
            let sample = Sample { frames: 32, elapsed: Duration::from_secs(2), warmup: false };
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
