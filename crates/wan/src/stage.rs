//! Where the start of Live is: the steps before the first frame, for the loading screen of the window. One state for
//! the process, moved on by whoever begins a step (the generating thread, the preparation's child output) and read by
//! the window every frame. Nothing here touches the GPU.

use std::sync::Mutex;
use std::time::Instant;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Step {
    /// export and checks of the AOTInductor packages (scripts/conversion/prepare_live.py): compiles when they changed
    Packages,
    /// LibTorch, the operators live::* and the GPU context
    Runtime,
    /// the video's packages, weights and buffers
    Video,
    /// the text encoder
    T5,
    /// the first prompt encoded and the first chunk generated (the offloading calibrates during it)
    FirstChunk,
}

pub const STEPS: [Step; 5] = [Step::Packages, Step::Runtime, Step::Video, Step::T5, Step::FirstChunk];

impl Step {
    pub fn name(self) -> &'static str {
        match self {
            Step::Packages => "packages",
            Step::Runtime => "runtime",
            Step::Video => "video model",
            Step::T5 => "text encoder",
            Step::FirstChunk => "first chunk",
        }
    }
    pub fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Debug)]
pub struct Progress {
    pub step: Step,
    /// when this step began
    pub since: Instant,
    /// seconds each finished step took, in order
    pub took: Vec<f32>,
    /// the last line said about the step (a package compiled, a calibration), empty if none
    pub detail: String,
}

impl Progress {
    /// From 0 to 1: finished steps count whole, the current one creeps towards its end and never reaches it.
    pub fn fraction(&self) -> f32 {
        let partial = 1.0 - (-self.since.elapsed().as_secs_f32() / 20.0).exp();
        (self.step.index() as f32 + 0.9 * partial) / STEPS.len() as f32
    }
}

static NOW: Mutex<Option<Progress>> = Mutex::new(None);

/// The step begins; the one before it is finished.
pub fn enter(step: Step) {
    let mut now = NOW.lock().unwrap();
    let mut took = now.as_ref().map(|p| p.took.clone()).unwrap_or_default();
    if let Some(p) = now.as_ref() {
        took.push(p.since.elapsed().as_secs_f32());
    }
    *now = Some(Progress { step, since: Instant::now(), took, detail: String::new() });
}

/// A line about the current step, shown under its name.
pub fn detail(text: &str) {
    if let Some(p) = NOW.lock().unwrap().as_mut() {
        p.detail = text.trim().chars().take(120).collect();
    }
}

/// Where the start is now; before any step, at the beginning of the first.
pub fn now() -> Progress {
    NOW.lock().unwrap().clone().unwrap_or_else(|| Progress { step: Step::Packages, since: Instant::now(), took: Vec::new(),
                                                              detail: String::new() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steps_accumulate_and_the_fraction_stays_below_the_next_step() {
        enter(Step::Packages);
        detail("compiling packages (one compiler worker)");
        assert_eq!(now().detail, "compiling packages (one compiler worker)");
        enter(Step::Runtime);
        let p = now();
        assert_eq!(p.step, Step::Runtime);
        assert_eq!(p.took.len(), 1);
        assert!(p.detail.is_empty());
        let f = p.fraction();
        assert!(f >= 1.0 / 5.0 && f < 2.0 / 5.0, "{f}");
    }
}
