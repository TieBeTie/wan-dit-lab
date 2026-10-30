//! The window (eframe): the video fills it with its proportions kept, a status line over its bottom, a prompt box
//! below it (Enter sends what is typed or dictated with Win+H, as a prompt over the socket would). The frames come
//! from the generating thread and are shown at the rate they are made, so that the video goes on smoothly.

use std::sync::mpsc::{sync_channel, Receiver, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use eframe::egui;
use live_protocol::Cut;

use crate::encoder::Encoder;
use crate::gpu::{DeviceBuffer, Event};
use crate::prompts::{Inbox, Prompt};
use crate::t5::T5;
use crate::video::{Result, Video};
use crate::Args;

pub const DEFAULT_PROMPT: &str = "A lone oak tree on a grassy hill sways in the wind";
const TITLE: &str = "live video (Esc to stop)";
const BOX: f32 = 44.0; // pixels of the prompt box below the video

#[derive(Default)]
struct Status {
    prompt: String,
    /// seconds of the last chunk, generated and decoded
    chunk_s: Option<f64>,
    frames_per_chunk: usize,
    error: Option<String>,
}

type Load = fn(&Args) -> Result<(Video, T5, DeviceBuffer)>;

/// The prompt whose embeddings T5 has made: the video takes it between two chunks (the text embedding and the
/// cross-attention keys of every block); then the newest prompt that came meanwhile goes to T5.
fn take_prompt(video: &mut Video, encoder: &mut Encoder, wait: bool) -> Result<Option<Prompt>> {
    let Some(p) = encoder.ready(wait)? else { return Ok(None) };
    video.set_prompt(&encoder.tensor(), p.cut)?;
    if let Some((next, text)) = encoder.take_waiting() {
        encoder.encode(next, text, read(video)?)?;
    }
    Ok(Some(p))
}

/// The event after which the video has read the embeddings T5 last made.
fn read(video: &Video) -> Result<Event> {
    let e = Event::new()?;
    e.record(&video.compute)?;
    Ok(e)
}

/// Generates chunk after chunk until the window is gone. A prompt that comes goes to T5 on its own thread, and the
/// video goes on with the prompt it has until T5 is done.
fn generate(args: Args, load: Load, inbox: Inbox, frames: std::sync::mpsc::SyncSender<Vec<u8>>,
            status: Arc<Mutex<Status>>, repaint: egui::Context) -> Result<()> {
    let (mut video, t5, embeddings) = load(&args)?;
    let mut encoder = Encoder::spawn(t5, embeddings)?;
    let first = Prompt { cut: Cut::Same, text: args.prompt.clone().unwrap_or_else(|| DEFAULT_PROMPT.into()) };
    let started = Instant::now();
    encoder.encode(first.clone(), crate::model_prompt(&video, &first), read(&video)?)?;
    take_prompt(&mut video, &mut encoder, true)?;
    println!("{}x{}, chunk {}, {} blocks ({} streamed), prompt encoded in {:.1} s", video.m.height, video.m.width,
             video.m.chunk, video.m.blocks, video.m.stream_blocks, started.elapsed().as_secs_f64());
    #[cfg(feature = "debug-tools")]
    println!("{}", crate::measure::memory_line(&video));
    status.lock().unwrap().prompt = first.text;
    // the next chunk is queued behind the decoding of this one, before the host waits for its pixels: the GPU goes
    // from one to the other without waiting for the host
    let mut start = Instant::now();
    let x0 = video.next_chunk(None)?;
    video.start_decode(&x0)?;
    let mut asked = Instant::now();
    loop {
        if let Some(p) = inbox.take() {
            asked = Instant::now();
            let text = crate::model_prompt(&video, &p);
            encoder.encode(p, text, read(&video)?)?;
        }
        // between two chunks: the one queued is decoded, the next is made with the new prompt
        if let Some(p) = take_prompt(&mut video, &mut encoder, false)? {
            let kind = match p.cut { Cut::New => " (new video)", Cut::Scene => " (new shot)", Cut::Same => "" };
            println!("prompt encoded in {:.1} s{kind}", asked.elapsed().as_secs_f64());
            status.lock().unwrap().prompt = p.text;
        }
        let x0 = video.next_chunk(None)?;
        let chunk = video.finish_decode()?;
        video.start_decode(&x0)?;
        let seconds = start.elapsed().as_secs_f64();
        start = Instant::now();
        {
            let mut s = status.lock().unwrap();
            s.chunk_s = Some(seconds);
            s.frames_per_chunk = 4 * video.m.chunk;
        }
        // the waits of the chunks the GPU has passed; the one queued keeps running
        #[cfg(feature = "debug-tools")]
        println!("chunk {}: {seconds:.2} s, {} | {}", video.start / video.m.chunk - 2, video.passed_copy_waits()?,
                 crate::measure::memory_line(&video));
        if send(&frames, chunk).is_err() {
            return Ok(()); // the window is closed
        }
        repaint.request_repaint();
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
            let edit = ui.add(egui::TextEdit::singleline(&mut self.entry).desired_width(f32::INFINITY)
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
                (None, true) => format!("{}   |   {fps:.1} fps, chunk {:.2} s", short(&prompt, 100),
                                        self.status.lock().unwrap().chunk_s.unwrap_or(0.0)),
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

pub fn run(args: Args, load: Load) -> Result<()> {
    let manifest = crate::manifest::Manifest::read(&args.packages)?;
    let size = [manifest.width, manifest.height];
    let (w, h) = (manifest.width as f32 * args.scale, manifest.height as f32 * args.scale + BOX);
    let inbox = Inbox::default();
    inbox.listen(live_protocol::PORT)?;
    let (tx, rx) = sync_channel(4 * 4 * manifest.chunk); // four chunks ahead at most
    let status = Arc::new(Mutex::new(Status::default()));
    let mut viewport = egui::ViewportBuilder::default().with_title(TITLE).with_inner_size([w, h]);
    if args.right {
        viewport = viewport.with_position([960.0, 0.0]);
    }
    let options = eframe::NativeOptions { viewport, ..Default::default() };
    let (generating, max_fps) = (status.clone(), manifest.fps);
    let generator_inbox = inbox.clone();
    eframe::run_native(TITLE, options, Box::new(move |cc| {
        let repaint = cc.egui_ctx.clone();
        std::thread::Builder::new().name("generate".into()).spawn(move || {
            if let Err(e) = generate(args, load, generator_inbox, tx, generating.clone(), repaint.clone()) {
                eprintln!("error: {e}");
                generating.lock().unwrap().error = Some(e.to_string());
                repaint.request_repaint();
            }
        })?;
        Ok(Box::new(App { frames: rx, status, inbox, texture: None, size, max_fps, entry: String::new(),
                          pending: None, due: Instant::now(), shown: false }))
    })).map_err(|e| e.to_string())?;
    std::process::exit(0); // the generating thread is in the middle of a chunk: no need to wait for it
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
        args.packages = crate::root().join("artifacts").join("live").join("512x928");
        let inbox = Inbox::default();
        let (tx, rx) = sync_channel(1 << 16); // the test takes every frame at once: nothing waits for it
        let status = Arc::new(Mutex::new(Status::default()));
        let generating = {
            let inbox = inbox.clone();
            std::thread::spawn(move || generate(args, crate::load, inbox, tx, status, egui::Context::default()))
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
