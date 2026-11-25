//! One chunk/decode path for the window and MP4. No config parsing or export checks inside it.
use live_protocol::Cut;
use crate::config::PromptSource;
use crate::encoder::Encoder;
use crate::gpu::{DeviceBuffer, Event};
use crate::prompts::{Prompt, Step, chain_steps};
use crate::t5::T5;
use crate::video::{Result, Video};

pub type Sequence = (String, Vec<Step>);

pub fn sequences(source: &PromptSource) -> Result<Vec<Sequence>> {
    let PromptSource::Script { file, chains } = source else { return Ok(Vec::new()) };
    let pool: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(crate::root().join(file))?)?;
    let chunks = pool["chunks"].as_u64().filter(|&n| n > 0).ok_or("script needs positive chunks")? as usize;
    let mut result = Vec::new();
    for chain in pool["chains"].as_array().ok_or("script needs chains")? {
        let id = chain["id"].as_str().filter(|s| !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'))
            .ok_or("chain id must contain letters, digits or underscores")?;
        if result.iter().any(|(other, _)| other == id) { return Err(format!("duplicate chain {id}").into()); }
        if chains.is_empty() || chains.iter().any(|c| c == id) {
            let steps = chain_steps(chain, chunks, &crate::root())?;
            if steps.is_empty() || steps[0].cut != Cut::New || steps.iter().any(|s| s.chunks == 0 || s.text.trim().is_empty()) {
                return Err(format!("chain {id} must begin with new and have nonempty prompts and positive durations").into());
            }
            result.push((id.into(), steps));
        }
    }
    if result.is_empty() { return Err("no chains selected".into()); }
    for id in chains {
        if !result.iter().any(|(c, _)| c == id) { return Err(format!("no chain {id} in {}", file.display()).into()); }
    }
    Ok(result)
}

pub struct Generator<M: crate::telemetry::Probe = crate::telemetry::Off> {
    // Rust drops fields in declaration order: finish T5 while the video's streams still exist.
    encoder: Encoder,
    pub video: Video,
    active: Option<Prompt>,
    revision: u64,
    pending: bool,
    metrics: M,
}
impl Generator {
    pub fn new((video, t5, embeddings): (Video, T5, DeviceBuffer)) -> Result<Self> {
        Ok(Self { video, encoder: Encoder::spawn(t5, embeddings)?, active: None, revision: 0, pending: false, metrics: crate::telemetry::Off })
    }
}
impl<M: crate::telemetry::Probe> Generator<M> {
    pub fn observe<N: crate::telemetry::Probe>(self, metrics: N) -> Generator<N> {
        Generator { encoder: self.encoder, video: self.video, active: self.active,
                    revision: self.revision, pending: self.pending, metrics }
    }
    pub fn finish_metrics(&mut self) -> Result<()> { Ok(self.metrics.finish()?) }
    pub fn active(&self) -> Option<&Prompt> { self.active.as_ref() }
    pub fn revision(&self) -> u64 { self.revision }
    pub fn request(&mut self, p: Prompt) -> Result<()> {
        let text = crate::runtime::model_prompt(&self.video, &p);
        self.encoder.encode(p, text, read(&self.video)?)?;
        Ok(())
    }
    /// Accept an encoded prompt at a chunk boundary; a new video drains the previous decode first.
    pub fn poll(&mut self, wait: bool) -> Result<Vec<Vec<u8>>> {
        let Some(p) = self.encoder.ready(wait)? else { return Ok(Vec::new()) };
        self.metrics.prompt(self.encoder.encoding_seconds());
        let drained = if p.cut == Cut::New { self.flush()? } else { Vec::new() };
        if p.cut == Cut::New { self.video.reset_decoder(); self.metrics.reset(); }
        self.video.set_prompt(&self.encoder.tensor(), p.cut)?;
        self.active = Some(p);
        self.revision += 1;
        if let Some((next, text)) = self.encoder.take_waiting() {
            self.encoder.encode(next, text, read(&self.video)?)?;
        }
        Ok(drained)
    }
    /// Queue compute behind the previous decode before waiting for its pixels.
    pub fn advance(&mut self) -> Result<Vec<Vec<u8>>> {
        self.advance_timed::<false>()
    }
    pub fn advance_timed<const TIMED: bool>(&mut self) -> Result<Vec<Vec<u8>>> {
        if self.active.is_none() { return Err("generation needs a first prompt".into()); }
        let x0 = if self.metrics.transfers() { self.video.next_chunk_timed::<true>(None)? }
                 else { self.video.next_chunk_timed::<TIMED>(None)? };
        let frames = if self.pending { self.video.finish_decode()? } else { Vec::new() };
        self.video.start_decode(&x0)?;
        self.pending = true;
        self.metrics.chunk(&self.video)?;
        self.metrics.window(self.video.window_full());
        self.metrics.ready(frames.len());
        Ok(frames)
    }
    pub fn flush(&mut self) -> Result<Vec<Vec<u8>>> {
        if !self.pending { return Ok(Vec::new()); }
        let frames = self.video.finish_decode()?;
        self.pending = false;
        self.metrics.drained(frames.len());
        Ok(frames)
    }
    /// Exactly the scripted number of chunks. The final queued decode is emitted once.
    pub fn script(&mut self, steps: &[Step], mut emit: impl FnMut(Vec<Vec<u8>>, &Video, &str) -> Result<()>) -> Result<()> {
        for step in steps {
            self.request(Prompt { cut: step.cut, text: step.text.clone() })?;
            let frames = self.poll(true)?;
            if !frames.is_empty() { emit(frames, &self.video, &step.text)?; }
            for _ in 0..step.chunks {
                let frames = self.advance()?;
                if !frames.is_empty() { emit(frames, &self.video, &step.text)?; }
            }
        }
        let frames = self.flush()?;
        if !frames.is_empty() { emit(frames, &self.video, &self.active.as_ref().unwrap().text)?; }
        Ok(())
    }
}
fn read(video: &Video) -> Result<Event> {
    let e = Event::new()?;
    e.record(&video.compute)?;
    Ok(e)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scripts_are_resolved_before_starting_the_runtime() {
        let source = PromptSource::Script { file: "configs/prompts/demo_chains.json".into(), chains: vec!["fox_cabin".into()] };
        let plan = sequences(&source).unwrap();
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].0, "fox_cabin");
        assert_eq!(plan[0].1[0].cut, Cut::New);
        assert_eq!(plan[0].1.iter().map(|s| s.chunks).sum::<usize>(), 32);
        let missing = PromptSource::Script { file: "configs/prompts/demo_chains.json".into(), chains: vec!["missing".into()] };
        assert!(sequences(&missing).is_err());
        assert!(sequences(&PromptSource::Interactive { initial: None }).unwrap().is_empty());
    }

    #[test]
    #[ignore = "requires exported packages and CUDA"]
    fn scripted_output_drains_every_chunk_once_and_resets_decoder_for_new_videos() {
        use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
        struct Count(Arc<AtomicUsize>);
        impl crate::telemetry::Probe for Count {
            fn ready(&mut self, frames: usize) { self.0.fetch_add(frames, Ordering::Relaxed); }
        }
        let args = crate::defaults(&crate::root().join("configs/mvp1.json"));
        let observed = Arc::new(AtomicUsize::new(0));
        let mut generator = Generator::new(crate::runtime::load(&args).unwrap()).unwrap().observe(Count(observed.clone()));
        assert!(generator.advance().is_err(), "a hidden initial prompt was used");
        let step = |cut| Step { cut, text: "A lone oak tree on a grassy hill sways in the wind".into(), chunks: 1 };
        let (mut frames, mut batches) = (0, 0);
        generator.script(&[step(Cut::New), step(Cut::Same)], |chunk, video, _| {
            assert!(chunk.iter().all(|f| f.len() == video.m.height * video.m.width * 3));
            frames += chunk.len(); batches += 1; Ok(())
        }).unwrap();
        assert_eq!((frames, batches), (61, 2));
        assert_eq!(observed.load(Ordering::Relaxed), 61, "telemetry missed or duplicated completed frames");
        #[cfg(feature = "debug-tools")]
        assert_eq!(generator.video.copy_waits().unwrap(), (0, 0, 0.0), "normal generation still creates per-row GPU timing events");
        assert!(generator.flush().unwrap().is_empty(), "a decoded chunk was emitted twice");
        let mut next_frames = 0;
        generator.script(&[step(Cut::New)], |chunk, _, _| { next_frames += chunk.len(); Ok(()) }).unwrap();
        assert_eq!(next_frames, 29, "new video inherited decoder state");
        assert_eq!(observed.load(Ordering::Relaxed), 90);
    }
}
