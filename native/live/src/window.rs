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

/// Generates chunk after chunk until the window is gone, a waiting prompt first.
fn generate(args: Args, load: Load, inbox: Inbox, frames: std::sync::mpsc::SyncSender<Vec<u8>>,
            status: Arc<Mutex<Status>>, repaint: egui::Context) -> Result<()> {
    let (mut video, mut t5, embeddings) = load(&args)?;
    let first = Prompt { cut: Cut::Same, text: args.prompt.clone().unwrap_or_else(|| DEFAULT_PROMPT.into()) };
    let t = crate::set_prompt(&mut video, &mut t5, &embeddings, &first)?;
    println!("{}x{}, chunk {}, {} blocks ({} streamed), prompt encoded in {t:.1} s", video.m.height, video.m.width,
             video.m.chunk, video.m.blocks, video.m.stream_blocks);
    println!("{}", crate::measure::memory_line(&video));
    status.lock().unwrap().prompt = first.text;
    loop {
        if let Some(p) = inbox.take() {
            let t = crate::set_prompt(&mut video, &mut t5, &embeddings, &p)?;
            let kind = match p.cut { Cut::New => " (new video)", Cut::Scene => " (new shot)", Cut::Same => "" };
            println!("prompt encoded in {t:.1} s{kind}");
            status.lock().unwrap().prompt = p.text;
        }
        let start = Instant::now();
        let x0 = video.next_chunk(None)?;
        let chunk = video.decode(&x0)?;
        let seconds = start.elapsed().as_secs_f64();
        {
            let mut s = status.lock().unwrap();
            s.chunk_s = Some(seconds);
            s.frames_per_chunk = 4 * video.m.chunk;
        }
        println!("chunk {}: {seconds:.2} s", video.start / video.m.chunk - 1);
        for frame in chunk {
            if frames.send(frame).is_err() {
                return Ok(()); // the window is closed
            }
        }
        repaint.request_repaint();
    }
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
