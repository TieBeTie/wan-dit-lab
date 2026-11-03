//! The window (eframe): the video fills it with its proportions kept, a status line over its bottom, a prompt box
//! below it (Enter sends what is typed or dictated with Win+H, as a prompt over the socket would). The frames come
//! from the generating thread and are shown at the rate they are made, so that the video goes on smoothly.

use std::sync::mpsc::{sync_channel, Receiver, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use eframe::egui;
use live_protocol::Cut;

use crate::gpu::DeviceBuffer;
use crate::prompts::{Inbox, Prompt};
use crate::t5::T5;
use crate::video::{Result, Video};
use crate::Args;

const TITLE: &str = "live video (Esc to stop)";
const BOX: f32 = 44.0; // pixels of the prompt box below the video

#[derive(Default)]
struct Status {
    prompt: String,
    /// seconds of the last chunk, generated and decoded
    chunk_s: Option<f64>,
    frames_per_chunk: usize,
    error: Option<String>,
    closed: bool,
}

type Load = fn(&Args) -> Result<(Video, T5, DeviceBuffer)>;

/// Both scripted playback and interactive prompts use the common generator.
fn generate<M: crate::telemetry::Probe>(args: Args, load: Load, inbox: Inbox, frames: std::sync::mpsc::SyncSender<Vec<u8>>,
            status: Arc<Mutex<Status>>, repaint: egui::Context, metrics: M) -> Result<()> {
    let script = crate::generation::sequences(&args.pipeline.generate.prompts)?;
    let mut generator = crate::generation::Generator::new(load(&args)?)?.observe(metrics);
    let mut started = Instant::now();
    let mut emit = |chunk: Vec<Vec<u8>>, _: &Video, prompt: &str| -> Result<()> {
        if chunk.is_empty() { return Ok(()); }
        let seconds = started.elapsed().as_secs_f64();
        started = Instant::now();
        {
            let mut state = status.lock().unwrap();
            state.prompt = prompt.into(); state.chunk_s = Some(seconds);
            state.frames_per_chunk = chunk.len();
        }
        send(&frames, chunk)?;
        repaint.request_repaint();
        Ok(())
    };
    if !script.is_empty() {
        for (_, steps) in script { generator.script(&steps, &mut emit)?; }
        return generator.finish_metrics();
    }
    let first = if let Some(text) = args.pipeline.generate.prompts.initial() { Prompt { cut: Cut::Same, text: text.into() } }
            else { loop {
                if status.lock().unwrap().closed { return generator.finish_metrics(); }
                if let Some(p) = inbox.take() { break p; }
                std::thread::sleep(Duration::from_millis(20));
            } };
    generator.request(first)?;
    generator.poll(true)?;
    loop {
        if let Some(p) = inbox.take() { generator.request(p)?; }
        let drained = generator.poll(false)?;
        emit(drained, &generator.video, &generator.active().unwrap().text)?;
        let chunk = generator.advance()?;
        if emit(chunk, &generator.video, &generator.active().unwrap().text).is_err() { return generator.finish_metrics(); }
    }
}

fn send(frames: &std::sync::mpsc::SyncSender<Vec<u8>>, chunk: Vec<Vec<u8>>) -> Result<()> {
    for frame in chunk {
        frames.send(frame).map_err(|_| "the window is closed")?;
    }
    Ok(())
}

struct App {
    frames: Receiver<Vec<u8>>,
    status: Arc<Mutex<Status>>,
    inbox: Inbox,
    texture: Option<egui::TextureHandle>,
    size: [usize; 2],
    max_fps: f64,
    entry: String,
    pending: Option<String>,
    due: Instant,
    shown: bool,
    interactive: bool,
    show_fps: bool,
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        let (prompt, made, error) = {
            let s = self.status.lock().unwrap();
            let made = s.chunk_s.map(|c| s.frames_per_chunk as f64 / c).unwrap_or(self.max_fps);
            (s.prompt.clone(), made, s.error.clone())
        };
        // shown at the rate frames are made; the next is due 1/fps after this one was due, and after a stall the
        // schedule starts again instead of bursting
        let fps = self.max_fps.min(0.95 * made);
        let now = Instant::now();
        if now >= self.due {
            match self.frames.try_recv() {
                Ok(pixels) => {
                    let image = egui::ColorImage::from_rgb(self.size, &pixels);
                    match &mut self.texture {
                        Some(t) => t.set(image, egui::TextureOptions::LINEAR),
                        None => self.texture = Some(ctx.load_texture("video", image, egui::TextureOptions::LINEAR)),
                    }
                    self.shown = true;
                    let period = Duration::from_secs_f64(1.0 / fps);
                    self.due = (self.due + period).max(now - period / 2);
                }
                Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => {}
            }
        }
        if let Some(p) = &self.pending {
            if !prompt.is_empty() && *p == prompt {
                self.pending = None;
            }
        }
        egui::Panel::bottom("prompt").exact_size(BOX).show(ui, |ui| {
            let edit = ui.add_enabled(self.interactive, egui::TextEdit::singleline(&mut self.entry).desired_width(f32::INFINITY)
                .font(egui::FontId::proportional(17.0)));
            if edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                let text = self.entry.trim().to_owned();
                if self.inbox.put(&text) {
                    self.pending = live_protocol::parse(&text).map(|(_, t)| t);
                    self.entry.clear();
                }
                edit.request_focus();
            }
            if !edit.has_focus() && self.pending.is_none() && !self.shown {
                edit.request_focus();
            }
        });
        egui::CentralPanel::default().frame(egui::Frame::NONE.fill(egui::Color32::BLACK)).show(ui, |ui| {
            let area = ui.available_rect_before_wrap();
            let [w, h] = [self.size[0] as f32, self.size[1] as f32];
            let k = (area.width() / w).min(area.height() / h);
            let rect = egui::Rect::from_center_size(area.center(), egui::vec2(w * k, h * k));
            if let Some(t) = &self.texture {
                ui.painter().image(t.id(), rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                                   egui::Color32::WHITE);
            }
            let mut text = match (&error, self.shown) {
                (Some(e), _) => format!("error: {e}"),
                (None, false) => "warming up: loading the packages, T5".into(),
                (None, true) if self.show_fps => format!("{}   |   generation {made:.1} FPS", short(&prompt, 100)),
                (None, true) => short(&prompt, 100),
            };
            if let Some(p) = &self.pending {
                text += &format!("   |   next, T5 encoding: {}", short(p, 60));
            }
            let at = egui::pos2(rect.left() + 10.0, rect.bottom() - 26.0);
            let galley = ui.painter().layout_no_wrap(text, egui::FontId::proportional(13.0), egui::Color32::WHITE);
            ui.painter().rect_filled(egui::Rect::from_min_size(at, galley.size()).expand(4.0), 0.0, egui::Color32::BLACK);
            ui.painter().galley(at, galley, egui::Color32::WHITE);
        });
        ctx.request_repaint_after(self.due.saturating_duration_since(Instant::now()).max(Duration::from_millis(2)));
    }
}

fn short(text: &str, n: usize) -> String {
    text.chars().take(n).collect()
}

pub fn run<M: crate::telemetry::Probe>(args: Args, load: Load, metrics: M) -> Result<()> {
    let manifest = crate::manifest::Manifest::read_for(&args.packages(), &args.pipeline.export.kernels)?;
    let size = [manifest.width, manifest.height];
    let (w, h) = (manifest.width as f32 * args.scale, manifest.height as f32 * args.scale + BOX);
    let inbox = Inbox::default();
    let interactive = args.pipeline.generate.prompts.interactive();
    let show_fps = args.pipeline.generate.debug.enabled();
    if interactive { inbox.listen(live_protocol::PORT)?; }
    let (tx, rx) = sync_channel(4 * 4 * manifest.chunk); // four chunks ahead at most
    let status = Arc::new(Mutex::new(Status::default()));
    let mut viewport = egui::ViewportBuilder::default().with_title(TITLE).with_inner_size([w, h]);
    if args.right {
        viewport = viewport.with_position([960.0, 0.0]);
    }
    let options = eframe::NativeOptions { viewport, ..Default::default() };
    let (generating, max_fps) = (status.clone(), manifest.fps);
    let generator_inbox = inbox.clone();
    let shutdown = status.clone();
    let mut worker = None;
    let generating_thread = &mut worker;
    let result = eframe::run_native(TITLE, options, Box::new(move |cc| {
        let repaint = cc.egui_ctx.clone();
        *generating_thread = Some(std::thread::Builder::new().name("generate".into()).spawn(move || {
            if let Err(e) = generate(args, load, generator_inbox, tx, generating.clone(), repaint.clone(), metrics) {
                eprintln!("error: {e}");
                generating.lock().unwrap().error = Some(e.to_string());
                repaint.request_repaint();
            }
        })?);
        Ok(Box::new(App { frames: rx, status, inbox, texture: None, size, max_fps, entry: String::new(),
                          pending: None, due: Instant::now(), shown: false, interactive, show_fps }))
    }));
    shutdown.lock().unwrap().closed = true;
    if let Some(worker) = worker { worker.join().map_err(|_| "generating thread panicked")?; }
    result.map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A prompt changed while the video runs: T5 reads and computes beside the video, which goes on with the prompt it
    /// has, so that no chunk after the change takes more than a third longer than the chunks before it.
    /// Needs the GPU and artifacts/live/512x928: cargo test -- --ignored --nocapture
    #[test]
    #[ignore]
    fn the_video_goes_on_while_a_prompt_is_encoded() {
        let mut args = crate::defaults(&crate::root().join("configs").join("live.json"));
        args.pipeline.export.output = "artifacts/live/512x928".into();
        let inbox = Inbox::default();
        let (tx, rx) = sync_channel(1 << 16); // the test takes every frame at once: nothing waits for it
        let status = Arc::new(Mutex::new(Status::default()));
        let generating = {
            let inbox = inbox.clone();
            std::thread::spawn(move || generate(args, crate::load, inbox, tx, status, egui::Context::default(), crate::telemetry::Off))
        };
        // a chunk's frames come together; a chunk begins where a frame comes 0.3 s or more after the one before
        let mut chunks: Vec<Instant> = Vec::new();
        let mut last = Instant::now();
        let (before, after) = (8, 6);
        let mut changed = None;
        while chunks.len() < before + after {
            rx.recv().expect("the video stopped");
            let now = Instant::now();
            if chunks.is_empty() || now - last > Duration::from_millis(300) {
                chunks.push(now);
                if chunks.len() == before {
                    inbox.put(&live_protocol::line(Cut::Same, "A red fox walks through a snowy forest at dusk"));
                    changed = Some(chunks.len());
                }
            }
            last = now;
        }
        drop(rx);
        let _ = generating.join();
        let gaps: Vec<f64> = chunks.windows(2).map(|w| (w[1] - w[0]).as_secs_f64()).collect();
        let changed = changed.unwrap() - 1; // the gap from the chunk the prompt came in on
        let mut steady = gaps[3..changed].to_vec(); // the cache full and calibrated from the fourth chunk on
        steady.sort_by(f64::total_cmp);
        let typical = steady[steady.len() / 2];
        let worst = gaps[changed..].iter().cloned().fold(0.0, f64::max);
        println!("chunks every {typical:.2} s before the prompt, at most {worst:.2} s after it: {gaps:.2?}");
        assert!(worst < 1.33 * typical, "the video stood {:.2} s for the prompt", worst - typical);
    }
}
