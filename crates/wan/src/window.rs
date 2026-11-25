//! The window (eframe): a bar above, the video filling the middle with its proportions kept, a status line over its
//! bottom, a prompt box below (Enter sends what is typed or dictated with Win+H, as a prompt over the socket would).
//! The window opens at once: the packages are prepared and loaded in the generating thread behind a loading screen
//! (look.rs, stage.rs). The frames come from the generating thread and are shown at the rate they are made, so that
//! the video goes on smoothly; between frames the window sleeps until the next is due or the generator wakes it.

use std::sync::mpsc::{sync_channel, Receiver};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use eframe::egui;
use live_protocol::Cut;

use crate::gpu::DeviceBuffer;
use crate::prompts::{Inbox, Prompt};
use crate::t5::T5;
use crate::video::{Result, Video};
use crate::Args;

/// Escape does not close the window (it closed it by accident); the cross of the bar does, and stop_live.ps1.
const TITLE: &str = "live video";
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
    crate::stage::enter(crate::stage::Step::Packages);
    crate::pipeline::prepare(&args.pipeline)?;
    let mut generator = crate::generation::Generator::new(load(&args)?)?.observe(metrics);
    crate::stage::enter(crate::stage::Step::FirstChunk);
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
            else { crate::stage::detail("waiting for a prompt"); loop {
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
    /// the frame due next, taken from the channel ahead: with one waiting the window wakes when it is due, without one
    /// it sleeps until the generator wakes it
    next: Option<Vec<u8>>,
    status: Arc<Mutex<Status>>,
    inbox: Inbox,
    texture: Option<egui::TextureHandle>,
    /// the texture's pixels, refilled in place: egui has let go of them by the next frame, so a frame allocates nothing
    image: Arc<egui::ColorImage>,
    max_fps: f64,
    entry: String,
    pending: Option<String>,
    due: Instant,
    shown: bool,
    interactive: bool,
    show_fps: bool,
    loading: crate::look::Loading,
    painted: Instant,
}

impl App {
    /// RGB from the decoder into the texture, without a new buffer.
    fn show(&mut self, ctx: &egui::Context, pixels: &[u8]) {
        refill(&mut self.image, pixels);
        match &mut self.texture {
            Some(t) => t.set(self.image.clone(), egui::TextureOptions::LINEAR),
            None => self.texture = Some(ctx.load_texture("video", self.image.clone(), egui::TextureOptions::LINEAR)),
        }
        self.shown = true;
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let (prompt, made, chunk_s, error) = {
            let s = self.status.lock().unwrap();
            let made = s.chunk_s.map(|c| s.frames_per_chunk as f64 / c).unwrap_or(self.max_fps);
            (s.prompt.clone(), made, s.chunk_s, s.error.clone())
        };
        // shown at the rate frames are made; the next is due 1/fps after this one was due, and after a stall the
        // schedule starts again instead of bursting
        let fps = self.max_fps.min(0.95 * made);
        if self.next.is_none() { self.next = self.frames.try_recv().ok(); }
        let now = Instant::now();
        if now >= self.due {
            if let Some(pixels) = self.next.take() {
                self.show(&ctx, &pixels);
                let period = Duration::from_secs_f64(1.0 / fps);
                self.due = (self.due + period).max(now - period / 2);
                self.next = self.frames.try_recv().ok();
            }
        }
        let dt = now.duration_since(self.painted).as_secs_f32().min(0.1);
        self.painted = now;
        if let Some(p) = &self.pending {
            if !prompt.is_empty() && *p == prompt {
                self.pending = None;
            }
        }
        let progress = (!self.shown).then(crate::stage::now);
        let bar = match (&progress, chunk_s) {
            (Some(p), _) => format!("starting · {} · {:.0} s", p.step.name(), p.since.elapsed().as_secs_f32()),
            (None, Some(c)) => format!("{made:.1} FPS · chunk {c:.2} s"),
            (None, None) => String::new(),
        };
        egui::Panel::top("bar").exact_size(crate::look::BAR).frame(egui::Frame::NONE).show(ui, |ui| crate::look::bar(ui, &bar));
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
            if let Some(p) = &progress {
                self.loading.paint(ui, area, p, error.as_deref(), dt);
                return;
            }
            let [w, h] = [self.image.size[0] as f32, self.image.size[1] as f32];
            let k = (area.width() / w).min(area.height() / h);
            let rect = egui::Rect::from_center_size(area.center(), egui::vec2(w * k, h * k));
            if let Some(t) = &self.texture {
                ui.painter().image(t.id(), rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                                   egui::Color32::WHITE);
            }
            let mut text = match &error {
                Some(e) => format!("error: {e}"),
                None if self.show_fps => format!("{}   |   generation {made:.1} FPS", short(&prompt, 100)),
                None => short(&prompt, 100),
            };
            if let Some(p) = &self.pending {
                text += &format!("   |   next, T5 encoding: {}", short(p, 60));
            }
            let at = egui::pos2(rect.left() + 12.0, rect.bottom() - 28.0);
            let galley = ui.painter().layout_no_wrap(text, egui::FontId::proportional(13.0), egui::Color32::WHITE);
            let back = egui::Rect::from_min_size(at, galley.size()).expand(5.0);
            ui.painter().rect_filled(back, 4.0, egui::Color32::from_black_alpha(170));
            ui.painter().rect_filled(egui::Rect::from_min_size(back.min, egui::vec2(3.0, back.height())), 0.0,
                                     if error.is_some() { egui::Color32::RED } else { crate::look::ORANGE });
            ui.painter().galley(at, galley, egui::Color32::WHITE);
        });
        crate::look::edges(&ctx);
        ctx.request_repaint_after(wake(progress.is_some(), self.next.is_some(), self.due, Instant::now()));
    }
}

/// The decoder's RGB into the texture's pixels in place; a new buffer only if egui still holds the last one.
fn refill(image: &mut Arc<egui::ColorImage>, rgb: &[u8]) {
    let image = Arc::make_mut(image);
    for (to, p) in image.pixels.iter_mut().zip(rgb.chunks_exact(3)) {
        *to = egui::Color32::from_rgb(p[0], p[1], p[2]);
    }
}

/// When the window paints next. The loading screen moves at 60 Hz; the video wakes the window when its next frame is
/// due; with no frame waiting the window sleeps, since the generator wakes it as frames come (request_repaint in
/// generate). Before, it woke every 2 ms whenever a frame was late: 500 paints a second for nothing.
fn wake(loading: bool, frame_waiting: bool, due: Instant, now: Instant) -> Duration {
    if loading { Duration::from_millis(16) }
    else if frame_waiting { due.saturating_duration_since(now) }
    else { Duration::from_millis(250) }
}

fn short(text: &str, n: usize) -> String {
    text.chars().take(n).collect()
}

pub fn run<M: crate::telemetry::Probe>(args: Args, load: Load, metrics: M) -> Result<()> {
    // the size from the configuration, not from the packages: the window opens before they are prepared
    let video = &args.pipeline.export.video;
    let (height, width) = video.size.split_once('x').and_then(|(h, w)| Some((h.parse::<usize>().ok()?, w.parse::<usize>().ok()?)))
        .ok_or("video.size must be HxW")?;
    let (max_fps, chunk) = (video.fps, args.pipeline.export.context.chunk_frames);
    let (w, h) = (width as f32 * args.scale, height as f32 * args.scale + BOX + crate::look::BAR);
    let inbox = Inbox::default();
    let interactive = args.pipeline.generate.prompts.interactive();
    let show_fps = args.pipeline.generate.debug.includes(crate::config::DebugTool::Fps);
    if interactive { inbox.listen(live_protocol::PORT)?; }
    let (tx, rx) = sync_channel(4 * 4 * chunk); // four chunks ahead at most
    let status = Arc::new(Mutex::new(Status::default()));
    let mut viewport = egui::ViewportBuilder::default().with_title(TITLE).with_inner_size([w, h])
        .with_icon(Arc::new(crate::look::icon())).with_decorations(false); // the bar of look.rs is the frame
    if args.right {
        viewport = viewport.with_position([960.0, 0.0]);
    }
    let options = eframe::NativeOptions { viewport, ..Default::default() };
    let generating = status.clone();
    let generator_inbox = inbox.clone();
    let shutdown = status.clone();
    let mut worker = None;
    let generating_thread = &mut worker;
    let result = eframe::run_native(TITLE, options, Box::new(move |cc| {
        let repaint = cc.egui_ctx.clone();
        crate::look::style(&cc.egui_ctx);
        *generating_thread = Some(std::thread::Builder::new().name("generate".into()).spawn(move || {
            if let Err(e) = generate(args, load, generator_inbox, tx, generating.clone(), repaint.clone(), metrics) {
                eprintln!("error: {e}");
                generating.lock().unwrap().error = Some(e.to_string());
                repaint.request_repaint();
            }
        })?);
        Ok(Box::new(App { frames: rx, next: None, status, inbox, texture: None,
                          image: Arc::new(egui::ColorImage::filled([width, height], egui::Color32::BLACK)), max_fps,
                          entry: String::new(), pending: None, due: Instant::now(), shown: false, interactive, show_fps,
                          loading: Default::default(), painted: Instant::now() }))
    }));
    shutdown.lock().unwrap().closed = true;
    if let Some(worker) = worker { worker.join().map_err(|_| "generating thread panicked")?; }
    let failed = shutdown.lock().unwrap().error.take();
    outcome(result.map_err(|e| e.to_string()), failed)
}

/// The end of a windowed run: the generating thread's error is the run's error, not only a line in the window.
fn outcome(window: std::result::Result<(), String>, generation: Option<String>) -> Result<()> {
    if let Some(e) = generation { return Err(e.into()); }
    window?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Metric: a run whose generation failed (here, package preparation) ends with an error, so the queue records
    /// `failed` with code 1; on 2026-11-23 it recorded `done`, code 0, and the failure stayed in the closed window.
    #[test]
    fn a_failed_generation_fails_the_run() {
        let failed = Some("package preparation failed; generation was not started".to_string());
        assert!(outcome(Ok(()), failed.clone()).unwrap_err().to_string().contains("package preparation failed"));
        assert!(outcome(Err("window".into()), failed).is_err());
        assert!(outcome(Ok(()), None).is_ok());
        assert!(outcome(Err("window".into()), None).is_err());
    }

    /// Metric: with no frame waiting the window paints at most 4 times a second (it painted 500 before), and it
    /// paints a waiting frame exactly when it is due.
    #[test]
    fn the_window_sleeps_while_no_frame_waits() {
        let now = Instant::now();
        let due = now + Duration::from_millis(40);
        assert!(wake(false, false, due, now) >= Duration::from_millis(250));
        assert!(wake(false, false, now - Duration::from_secs(1), now) >= Duration::from_millis(250));
        assert_eq!(wake(false, true, due, now), Duration::from_millis(40));
        assert_eq!(wake(false, true, now - Duration::from_millis(5), now), Duration::ZERO);
        assert_eq!(wake(true, false, due, now), Duration::from_millis(16));
    }

    /// Metric: a frame refilled into the texture's pixels in place costs less than a new image built per frame, as
    /// the window did before (ColorImage::from_rgb), and gives the same pixels.
    #[test]
    fn a_frame_is_refilled_faster_than_it_was_built() {
        let size = [928, 512];
        let rgb: Vec<u8> = (0..size[0] * size[1] * 3).map(|i| (i * 7 % 251) as u8).collect();
        let mut image = Arc::new(egui::ColorImage::filled(size, egui::Color32::BLACK));
        let best = |f: &mut dyn FnMut()| (0..5).map(|_| { let t = Instant::now(); for _ in 0..20 { f(); } t.elapsed() }).min().unwrap();
        let built = best(&mut || { std::hint::black_box(Arc::new(egui::ColorImage::from_rgb(size, &rgb))); });
        let refilled = best(&mut || {
            refill(&mut image, &rgb);
            std::hint::black_box(image.clone()); // egui takes a clone and lets it go before the next frame
        });
        assert_eq!(image.pixels, egui::ColorImage::from_rgb(size, &rgb).pixels);
        println!("20 frames {}x{}: built {built:?}, refilled {refilled:?}", size[0], size[1]);
        assert!(refilled < built, "refilled {refilled:?}, built {built:?}");
    }

    /// A prompt changed while the video runs: T5 reads and computes beside the video, which goes on with the prompt it
    /// has, so that no chunk after the change takes more than a third longer than the chunks before it.
    /// Needs the GPU and artifacts/live/512x928: cargo test -- --ignored --nocapture
    #[test]
    #[ignore]
    fn the_video_goes_on_while_a_prompt_is_encoded() {
        let mut args = crate::defaults(&crate::root().join("configs").join("mvp1.json"));
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
