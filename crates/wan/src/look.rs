//! The look of the window, in the tutor's manner and the clean test-chamber style of Portal 2: dark glass, one blue
//! and one orange accent, white signs with black type. While Live starts, a loading screen: a Lorenz attractor drawn
//! by the CPU over a sky of white twinkling stars, crowded towards the middle by a Gaussian and calming as the start goes
//! on (a latent being denoised), and a sign with the steps. Everything here is a few thousand vertices and no texture: the
//! start of Live holds nearly all the video memory, and the loading screen must not take from it.

use std::collections::VecDeque;

use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, Stroke, StrokeKind, Vec2};
use rand::Rng;
use rand_distr::StandardNormal;

use crate::stage::{Progress, STEPS};

pub const BLUE: Color32 = Color32::from_rgb(0x2e, 0xa7, 0xff);
pub const ORANGE: Color32 = Color32::from_rgb(0xff, 0x8c, 0x1a);
pub const GLASS: Color32 = Color32::from_rgb(0x0b, 0x0e, 0x12);
const DEEP: Color32 = Color32::from_rgb(0x05, 0x07, 0x0a);
const SIGN: Color32 = Color32::from_rgb(0xec, 0xee, 0xf0);
const INK: Color32 = Color32::from_rgb(0x14, 0x16, 0x19);
const MUTED: Color32 = Color32::from_rgb(0x8a, 0x92, 0x9c);
const RED: Color32 = Color32::from_rgb(0xef, 0x44, 0x44);
pub const BAR: f32 = 32.0; // pixels of the bar above the video, in place of the system's frame

/// The dark style of every widget: the prompt box, the panels.
pub fn style(ctx: &egui::Context) {
    ctx.set_theme(egui::Theme::Dark);
    ctx.all_styles_mut(|s| {
        let mut v = egui::Visuals::dark();
        v.panel_fill = GLASS;
        v.window_fill = GLASS;
        v.extreme_bg_color = Color32::from_rgb(0x10, 0x15, 0x1b);
        v.selection.bg_fill = ORANGE.gamma_multiply(0.45);
        v.selection.stroke = Stroke::new(1.0, ORANGE);
        v.text_cursor.stroke = Stroke::new(2.0, ORANGE);
        v.widgets.inactive.bg_stroke = Stroke::new(1.0, Color32::from_rgb(0x22, 0x2a, 0x33));
        v.widgets.hovered.bg_stroke = Stroke::new(1.0, BLUE.gamma_multiply(0.7));
        v.widgets.active.bg_stroke = Stroke::new(1.0, BLUE);
        s.visuals = v;
    });
}

/// Signed distance to a hexagon with flat sides left and right and inner radius r (Inigo Quilez): negative inside.
fn hexagon(p: [f32; 2], r: f32) -> f32 {
    let (kx, ky, kz) = (-0.866_025_4, 0.5, 0.577_350_3);
    let (mut x, mut y) = (p[1].abs(), p[0].abs()); // turned a quarter: points up and down
    let d = 2.0 * (kx * x + ky * y).min(0.0);
    x -= d * kx;
    y -= d * ky;
    let (cx, cy) = (x - x.clamp(-kz * r, kz * r), y - r);
    (cx * cx + cy * cy).sqrt() * cy.signum()
}

/// Signed distance to the play triangle, pointing right: negative inside.
fn play(p: [f32; 2]) -> f32 {
    let v: [[f32; 2]; 3] = [[-6.5, -9.5], [-6.5, 9.5], [10.5, 0.0]];
    (0..3).map(|i| {
        let (a, b) = (v[i], v[(i + 1) % 3]);
        let (ex, ey) = (b[0] - a[0], b[1] - a[1]);
        let len = (ex * ex + ey * ey).sqrt();
        // the triangle runs clockwise on screen (y down): inside is to the right of each edge
        (ex * (p[1] - a[1]) - ey * (p[0] - a[0])) / len
    }).fold(f32::MIN, f32::max)
}

/// The window's icon, 64×64: an orange hexagon with a play sign in it, glowing on dark glass. One colour, so that it
/// reads at the size of a taskbar.
pub fn icon() -> egui::IconData {
    const N: u32 = 64;
    let mut rgba = vec![0u8; (N * N * 4) as usize];
    let c = N as f32 / 2.0;
    for y in 0..N {
        for x in 0..N {
            let p = [x as f32 + 0.5 - c, y as f32 + 0.5 - c];
            // a rounded square of glass, the hexagon's rim, its glow, and the play sign, each antialiased over a pixel
            let q = [(p[0].abs() - 18.0).max(0.0), (p[1].abs() - 18.0).max(0.0)];
            let glass = (0.5 - ((q[0] * q[0] + q[1] * q[1]).sqrt() - 12.0)).clamp(0.0, 1.0);
            let h = hexagon(p, 21.0);
            let rim = (0.5 - (h.abs() - 2.0)).clamp(0.0, 1.0);
            let glow = 0.45 * (-h.abs() / 3.5).exp();
            let sign = (0.5 - play(p)).clamp(0.0, 1.0);
            let light = rim.max(sign).max(glow);
            let mix = |a: u8, b: u8| (a as f32 * (1.0 - light) + b as f32 * light) as u8;
            let i = ((y * N + x) * 4) as usize;
            rgba[i..i + 4].copy_from_slice(&[mix(GLASS.r(), ORANGE.r()), mix(GLASS.g(), ORANGE.g()), mix(GLASS.b(), ORANGE.b()),
                                             (glass * 255.0) as u8]);
        }
    }
    egui::IconData { rgba, width: N, height: N }
}

/// The corners of a hexagon of outer radius r round c, points up and down: the mark in the bar.
fn hexagon_points(c: Pos2, r: f32) -> Vec<Pos2> {
    (0..6).map(|k| { let a = std::f32::consts::FRAC_PI_3 * k as f32 + std::f32::consts::FRAC_PI_6;
                     Pos2::new(c.x + r * a.cos(), c.y + r * a.sin()) }).collect()
}

/// The bar above the video, in place of the system's frame: the mark, the name, what Live is doing, and the window's
/// own buttons. It moves the window when dragged and maximizes it on a double click.
pub fn bar(ui: &mut egui::Ui, status: &str) {
    let rect = ui.max_rect();
    let ctx = ui.ctx().clone();
    let drag = ui.interact(rect, ui.scope_id().with("drag"), egui::Sense::click_and_drag());
    let maximized = ctx.input(|i| i.viewport().maximized.unwrap_or(false));
    if drag.drag_started() { ctx.send_viewport_cmd(egui::ViewportCommand::StartDrag); }
    if drag.double_clicked() { ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(!maximized)); }
    let p = ui.painter();
    p.rect_filled(rect, 0.0, GLASS);
    p.line_segment([rect.left_bottom(), rect.right_bottom()], Stroke::new(1.0, Color32::from_rgb(0x1c, 0x23, 0x2b)));
    p.line_segment([rect.left_bottom(), rect.left_bottom() + Vec2::new(150.0, 0.0)], Stroke::new(2.0, ORANGE));
    let mid = rect.center().y;
    let mark = Pos2::new(rect.left() + 18.0, mid);
    p.add(egui::Shape::closed_line(hexagon_points(mark, 9.0), Stroke::new(1.6, ORANGE)));
    p.add(egui::Shape::convex_polygon(vec![mark + Vec2::new(-2.5, -4.0), mark + Vec2::new(4.5, 0.0), mark + Vec2::new(-2.5, 4.0)],
                                      ORANGE, Stroke::NONE));
    p.text(Pos2::new(rect.left() + 36.0, mid), Align2::LEFT_CENTER, "WAN · LIVE", FontId::monospace(12.0), SIGN);
    // the buttons, right to left: close, maximize, minimize
    let w = 44.0;
    let button = |k: f32, id: &str| {
        let r = Rect::from_min_max(Pos2::new(rect.right() - w * (k + 1.0), rect.top()), Pos2::new(rect.right() - w * k, rect.bottom()));
        (r, ui.interact(r, ui.scope_id().with(id), egui::Sense::click()))
    };
    let (close, close_r) = button(0.0, "close");
    let (max, max_r) = button(1.0, "maximize");
    let (min, min_r) = button(2.0, "minimize");
    let glyph = Stroke::new(1.3, SIGN);
    if close_r.hovered() { p.rect_filled(close, 0.0, RED.gamma_multiply(0.85)); }
    let c = close.center();
    p.line_segment([c + Vec2::new(-5.0, -5.0), c + Vec2::new(5.0, 5.0)], glyph);
    p.line_segment([c + Vec2::new(-5.0, 5.0), c + Vec2::new(5.0, -5.0)], glyph);
    if max_r.hovered() { p.rect_filled(max, 0.0, Color32::from_white_alpha(18)); }
    let square = Rect::from_center_size(max.center(), Vec2::splat(10.0));
    p.rect_stroke(square, 1.0, glyph, StrokeKind::Inside);
    if maximized { p.line_segment([square.left_top() + Vec2::new(3.0, -3.0), square.right_top() + Vec2::new(3.0, -3.0)], glyph); }
    if min_r.hovered() { p.rect_filled(min, 0.0, Color32::from_white_alpha(18)); }
    p.line_segment([min.center() + Vec2::new(-5.0, 0.0), min.center() + Vec2::new(5.0, 0.0)], glyph);
    p.text(Pos2::new(min.left() - 12.0, mid), Align2::RIGHT_CENTER, status, FontId::monospace(11.0), MUTED);
    if close_r.clicked() { ctx.send_viewport_cmd(egui::ViewportCommand::Close); }
    if max_r.clicked() { ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(!maximized)); }
    if min_r.clicked() { ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(true)); }
}

/// Without the system's frame the window is resized here: near an edge the pointer shows the resize cursor, and a
/// press there hands the resize to the system.
pub fn edges(ctx: &egui::Context) {
    const GRIP: f32 = 6.0;
    let r = ctx.viewport_rect();
    let Some(at) = ctx.input(|i| i.pointer.hover_pos()) else { return };
    if ctx.input(|i| i.viewport().maximized.unwrap_or(false)) { return; }
    let (w, e, n, s) = (at.x - r.left() < GRIP, r.right() - at.x < GRIP, at.y - r.top() < GRIP, r.bottom() - at.y < GRIP);
    use egui::{CursorIcon as C, ResizeDirection as D};
    let (direction, cursor) = match (w, e, n, s) {
        (true, _, true, _) => (D::NorthWest, C::ResizeNorthWest),
        (_, true, true, _) => (D::NorthEast, C::ResizeNorthEast),
        (true, _, _, true) => (D::SouthWest, C::ResizeSouthWest),
        (_, true, _, true) => (D::SouthEast, C::ResizeSouthEast),
        (true, ..) => (D::West, C::ResizeWest),
        (_, true, ..) => (D::East, C::ResizeEast),
        (_, _, true, _) => (D::North, C::ResizeNorth),
        (_, _, _, true) => (D::South, C::ResizeSouth),
        _ => return,
    };
    ctx.set_cursor_icon(cursor);
    if ctx.input(|i| i.pointer.primary_pressed()) { ctx.send_viewport_cmd(egui::ViewportCommand::BeginResize(direction)); }
}

/// One point running along the attractor, and the trail it leaves.
struct Particle {
    p: [f32; 3],
    trail: VecDeque<[f32; 3]>,
}

const PARTICLES: usize = 28;
const TRAIL: usize = 90;
const H: f32 = 0.005; // the step of the integration

/// The Lorenz system, σ = 10, ρ = 28, β = 8/3.
fn lorenz([x, y, z]: [f32; 3]) -> [f32; 3] {
    [10.0 * (y - x), x * (28.0 - z) - y, x * y - 8.0 / 3.0 * z]
}

fn rk4(p: [f32; 3], h: f32) -> [f32; 3] {
    let add = |a: [f32; 3], b: [f32; 3], k: f32| [a[0] + k * b[0], a[1] + k * b[1], a[2] + k * b[2]];
    let k1 = lorenz(p);
    let k2 = lorenz(add(p, k1, h / 2.0));
    let k3 = lorenz(add(p, k2, h / 2.0));
    let k4 = lorenz(add(p, k3, h));
    [0, 1, 2].map(|i| p[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
}

/// A star of the loading screen's sky: where it stands (in widths and heights of the window from its centre, Gaussian,
/// so that they crowd towards the middle like a galaxy), how big, how fast it twinkles.
struct Star {
    at: [f32; 2],
    size: f32,
    phase: f32,
    speed: f32,
}

const STARS: usize = 420;

pub struct Loading {
    particles: Vec<Particle>,
    stars: Vec<Star>,
    time: f32,
    angle: f32,
}

impl Default for Loading {
    fn default() -> Self {
        // nearby starts that the chaos pulls apart; warmed up so that they are already on the attractor
        let particles = (0..PARTICLES).map(|i| {
            let mut p = [1.0 + i as f32 * 0.02, 1.0, 1.0 + i as f32 * 0.01];
            for _ in 0..600 + i * 37 { p = rk4(p, H); }
            Particle { p, trail: VecDeque::with_capacity(TRAIL) }
        }).collect();
        let mut rng = rand::rng();
        let stars = (0..STARS).map(|_| {
            let (x, y): (f32, f32) = (rng.sample(StandardNormal), rng.sample(StandardNormal));
            Star { at: [0.5 + 0.22 * x, 0.5 + 0.22 * y], size: 0.4 + 1.6 * rng.random::<f32>().powi(3),
                   phase: std::f32::consts::TAU * rng.random::<f32>(), speed: 0.6 + 2.4 * rng.random::<f32>() }
        }).collect();
        Self { particles, stars, time: 0.0, angle: 0.0 }
    }
}

impl Loading {
    /// One frame of the loading screen over `rect`; dt: seconds since the frame before.
    pub fn paint(&mut self, ui: &egui::Ui, rect: Rect, progress: &Progress, error: Option<&str>, dt: f32) {
        if rect.width() < 64.0 || rect.height() < 64.0 { return; }
        let fraction = progress.fraction();
        self.time += dt;
        self.background(ui, rect, fraction);
        self.attractor(ui, rect, dt);
        sign(ui, rect, progress, fraction, error);
    }

    /// Dark glass and a sky of white stars that twinkle; the sky is brightest at the start and calms as it goes on,
    /// like noise clearing from a latent.
    fn background(&mut self, ui: &egui::Ui, rect: Rect, fraction: f32) {
        let p = ui.painter();
        // a vertical gradient from deep to glass
        let mut mesh = egui::Mesh::default();
        mesh.colored_vertex(rect.left_top(), DEEP);
        mesh.colored_vertex(rect.right_top(), DEEP);
        mesh.colored_vertex(rect.right_bottom(), GLASS);
        mesh.colored_vertex(rect.left_bottom(), GLASS);
        mesh.add_triangle(0, 1, 2);
        mesh.add_triangle(0, 2, 3);
        p.add(egui::Shape::mesh(mesh));
        let strength = 0.35 + 0.65 * (1.0 - fraction);
        for star in &self.stars {
            let (x, y) = (star.at[0], star.at[1]);
            if !(0.0..1.0).contains(&x) || !(0.0..1.0).contains(&y) { continue; }
            let twinkle = 0.5 + 0.5 * (self.time * star.speed + star.phase).sin();
            let a = (twinkle * twinkle * strength).clamp(0.0, 1.0);
            let at = Pos2::new(rect.left() + x * rect.width(), rect.top() + y * rect.height());
            p.circle_filled(at, star.size, Color32::from_white_alpha((255.0 * a) as u8));
            if star.size > 1.4 { p.circle_filled(at, star.size * 3.0, Color32::from_white_alpha((30.0 * a) as u8)); }
        }
    }

    fn attractor(&mut self, ui: &egui::Ui, rect: Rect, dt: f32) {
        let steps = ((dt * 120.0).round() as usize).clamp(1, 8);
        for q in self.particles.iter_mut() {
            for _ in 0..steps {
                q.p = rk4(q.p, H);
                if q.trail.len() == TRAIL { q.trail.pop_front(); }
                q.trail.push_back(q.p);
            }
        }
        self.angle += 0.18 * dt;
        // the right of the window, beside the sign; the whole window when it is narrow
        let area = if rect.width() > 700.0 { Rect::from_min_max(Pos2::new(rect.left() + 330.0, rect.top()), rect.max) } else { rect };
        let scale = (area.height() * 0.8 / 50.0).min(area.width() * 0.8 / 50.0);
        let (sin, cos) = self.angle.sin_cos();
        let centre = area.center();
        let project = |[x, y, z]: [f32; 3]| Pos2::new(centre.x + (x * cos - y * sin) * scale, centre.y + (25.0 - z) * scale);
        let p = ui.painter();
        for (i, q) in self.particles.iter().enumerate() {
            let colour = if i % 2 == 0 { BLUE } else { ORANGE };
            let n = q.trail.len();
            let mut last = None;
            for (k, &point) in q.trail.iter().enumerate() {
                let at = project(point);
                if let Some(from) = last {
                    let a = k as f32 / n as f32;
                    p.line_segment([from, at], Stroke::new(0.6 + 1.4 * a, colour.gamma_multiply(a * a)));
                }
                last = Some(at);
            }
            if let Some(head) = last {
                p.circle_filled(head, 2.2, colour);
                p.circle_filled(head, 5.0, colour.gamma_multiply(0.18));
            }
        }
    }
}

/// The sign of the steps, white with black type, as the signs at the door of a test chamber.
fn sign(ui: &egui::Ui, rect: Rect, progress: &Progress, fraction: f32, error: Option<&str>) {
    let p = ui.painter();
    let width = 280.0_f32.min(rect.width() - 32.0);
    let card = Rect::from_min_size(rect.left_top() + Vec2::new(24.0, 24.0), Vec2::new(width, 0.0));
    let x = card.left() + 18.0;
    let mut y = card.top() + 14.0;
    let mut shapes = Vec::new(); // laid out first, the card drawn under them once its height is known
    let mut text = |pos: Pos2, align: Align2, s: &str, font: FontId, colour: Color32| {
        let galley = ui.painter().layout_no_wrap(s.to_owned(), font, colour);
        let at = align.anchor_size(pos, galley.size()).min;
        shapes.push(egui::Shape::galley(at, galley, colour));
    };
    let current = progress.step.index();
    text(Pos2::new(x, y), Align2::LEFT_TOP, &format!("{:02}", current + 1), FontId::monospace(56.0), INK);
    text(Pos2::new(x + 72.0, y + 34.0), Align2::LEFT_TOP, &format!("/{:02}", STEPS.len()), FontId::monospace(18.0), MUTED);
    y += 70.0;
    let rule = y;
    y += 10.0;
    let title = if error.is_some() { "ERROR".to_owned() } else { progress.step.name().to_uppercase() };
    text(Pos2::new(x, y), Align2::LEFT_TOP, &title, FontId::proportional(20.0), if error.is_some() { RED } else { INK });
    y += 34.0;
    let blink = (ui.input(|i| i.time) * 3.0).sin() as f32 * 0.35 + 0.65;
    let mut boxes = Vec::new();
    for (i, step) in STEPS.iter().enumerate() {
        let seconds = if i < current { progress.took.get(i).map(|s| format!("{s:.1} s")) }
                      else if i == current { Some(format!("{:.0} s", progress.since.elapsed().as_secs_f32())) }
                      else { None };
        let colour = if i <= current { INK } else { MUTED };
        boxes.push((Rect::from_min_size(Pos2::new(x, y + 3.0), Vec2::splat(11.0)), i.cmp(&current)));
        text(Pos2::new(x + 20.0, y), Align2::LEFT_TOP, step.name(), FontId::proportional(14.0), colour);
        if let Some(s) = seconds {
            text(Pos2::new(card.left() + width - 18.0, y), Align2::RIGHT_TOP, &s, FontId::monospace(12.0), colour);
        }
        y += 22.0;
    }
    y += 8.0;
    let track = Rect::from_min_size(Pos2::new(x, y), Vec2::new(width - 36.0, 6.0));
    y += 16.0;
    let detail = error.unwrap_or(&progress.detail);
    let chars = ((width - 36.0) / 6.6) as usize;
    let mut lines = 0;
    for line in wrap(detail, chars).into_iter().take(if error.is_some() { 8 } else { 2 }) {
        text(Pos2::new(x, y), Align2::LEFT_TOP, &line, FontId::monospace(11.0), if error.is_some() { RED } else { MUTED });
        y += 15.0;
        lines += 1;
    }
    if lines == 0 { y += 15.0; }
    let card = Rect::from_min_max(card.min, Pos2::new(card.max.x, y + 10.0));
    p.rect_filled(card.translate(Vec2::new(0.0, 4.0)), 6.0, Color32::from_black_alpha(90));
    p.rect_filled(card, 6.0, SIGN);
    p.rect_filled(Rect::from_min_size(card.left_top(), Vec2::new(5.0, card.height())), 0.0, ORANGE);
    p.line_segment([Pos2::new(x, rule), Pos2::new(card.right() - 18.0, rule)], Stroke::new(2.0, INK));
    p.extend(shapes);
    for (b, order) in boxes {
        match order {
            std::cmp::Ordering::Less => { p.rect_filled(b, 1.0, INK); }
            std::cmp::Ordering::Equal => { p.rect_filled(b, 1.0, ORANGE.gamma_multiply(blink)); }
            std::cmp::Ordering::Greater => { p.rect_stroke(b, 1.0, Stroke::new(1.0, MUTED), StrokeKind::Inside); }
        }
    }
    p.rect_filled(track, 3.0, Color32::from_rgb(0xc9, 0xcd, 0xd2));
    p.rect_filled(Rect::from_min_size(track.min, Vec2::new(track.width() * fraction.clamp(0.0, 1.0), track.height())), 3.0,
                  if error.is_some() { RED } else { ORANGE });
}

/// Lines of at most n characters (at least one), broken at spaces where it can.
fn wrap(text: &str, n: usize) -> Vec<String> {
    let n = n.max(1); // the first frame may lay out in a window of no size yet
    let mut lines = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        let mut word: String = word.into();
        while word.chars().count() > n {
            if !line.is_empty() { lines.push(std::mem::take(&mut line)); }
            let head: String = word.chars().take(n).collect();
            word = word.chars().skip(n).collect();
            lines.push(head);
        }
        if !line.is_empty() && line.chars().count() + 1 + word.chars().count() > n {
            lines.push(std::mem::take(&mut line));
        }
        if !line.is_empty() { line.push(' '); }
        line.push_str(&word);
    }
    if !line.is_empty() { lines.push(line); }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// Metric: the whole of a loading frame's arithmetic (every particle's steps for a 20 Hz frame, eight at most, and the
    /// stars' twinkle) costs under a millisecond of the CPU: the screen is free beside the work behind it.
    #[test]
    fn a_loading_frame_costs_under_a_millisecond_of_arithmetic() {
        let mut loading = Loading::default();
        let t = Instant::now();
        let frames = 200;
        for f in 0..frames {
            for q in loading.particles.iter_mut() {
                for _ in 0..8 { q.p = rk4(q.p, H); }
            }
            let time = f as f32 * 0.016;
            let lit: f32 = loading.stars.iter().map(|s| (time * s.speed + s.phase).sin()).sum();
            std::hint::black_box(lit);
        }
        let per_frame = t.elapsed() / frames;
        println!("one loading frame, the worst case: {per_frame:?}");
        assert!(per_frame < Duration::from_millis(1), "{per_frame:?}");
    }

    #[test]
    fn the_stars_crowd_towards_the_middle() {
        let loading = Loading::default();
        let near = loading.stars.iter().filter(|s| (s.at[0] - 0.5).abs() < 0.22 && (s.at[1] - 0.5).abs() < 0.22).count();
        assert!(near as f32 > 0.4 * STARS as f32, "{near} of {STARS}"); // 68% x 68% = 47% for a Gaussian
        assert!(loading.stars.iter().all(|s| s.size > 0.0 && s.speed > 0.0));
    }

    #[test]
    fn the_attractor_stays_bounded() {
        let mut p = [1.0, 1.0, 1.0];
        for _ in 0..20_000 {
            p = rk4(p, H);
            assert!(p[0].abs() < 30.0 && p[1].abs() < 40.0 && (0.0..60.0).contains(&p[2]), "{p:?}");
        }
    }

    #[test]
    fn wrap_breaks_at_spaces_and_cuts_long_words() {
        assert_eq!(wrap("compiling packages (one compiler worker)", 20), vec!["compiling packages", "(one compiler", "worker)"]);
        assert_eq!(wrap("abcdefghij", 4), vec!["abcd", "efgh", "ij"]);
        assert!(wrap("", 10).is_empty());
        assert_eq!(wrap("ab", 0), vec!["a", "b"]); // a window of no width hung the first frame here
    }

    #[test]
    fn the_icon_is_an_orange_hexagon_with_a_play_sign_on_glass() {
        let icon = icon();
        let at = |x: u32, y: u32| { let i = ((y * 64 + x) * 4) as usize; [icon.rgba[i], icon.rgba[i + 1], icon.rgba[i + 2], icon.rgba[i + 3]] };
        assert_eq!(at(33, 32), [ORANGE.r(), ORANGE.g(), ORANGE.b(), 255]); // the play sign
        assert_eq!(at(31, 32 - 24), [ORANGE.r(), ORANGE.g(), ORANGE.b(), 255]); // the rim, at the top corner
        let between = at(20, 32); // inside the hexagon, left of the play sign: dark glass
        assert!(between[0] < 120 && between[3] == 255, "{between:?}");
        assert_eq!(at(0, 0)[3], 0); // outside the rounded square
    }

    #[test]
    fn the_shapes_have_their_inside_negative() {
        assert!(hexagon([0.0, 0.0], 10.0) < 0.0);
        assert!((hexagon([10.0, 0.0], 10.0)).abs() < 1e-4); // a flat side, left and right
        assert!(hexagon([0.0, 11.0], 10.0) < 0.0); // a corner, up: the outer radius is 2r/√3
        assert!(hexagon([0.0, 12.0], 10.0) > 0.0);
        assert!(play([0.0, 0.0]) < 0.0);
        assert!(play([-8.0, 0.0]) > 0.0);
        assert!(play([11.0, 0.0]) > 0.0);
    }
}
