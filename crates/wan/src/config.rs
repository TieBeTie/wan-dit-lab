//! The approved [export, generate] configuration. Parsed once, before preparation or GPU work.
//! Offloading owns one calibration policy; the runtime never reads JSON inside the chunk loop.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Exactly two preparation/execution components, in the order of the JSON array.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub export: Export,
    pub generate: Generate,
}

#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
#[serde(tag = "component", rename_all = "snake_case")]
enum Component { Export(Export), Generate(Generate) }

#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Export {
    pub compiler: String,
    pub rebuild: String,
    pub output: std::path::PathBuf,
    pub model: Model,
    pub video: VideoSpec,
    pub sampling: Sampling,
    pub context: Context,
    pub rope: Rope,
    pub kernels: Kernels,
}

#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Model {
    pub architecture: std::path::PathBuf,
    pub checkpoint: std::path::PathBuf,
    pub block_weights: std::path::PathBuf,
    pub text_encoder: TextEncoder,
    pub decoder: std::path::PathBuf,
}
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TextEncoder {
    pub checkpoint: std::path::PathBuf,
    pub block_weights: std::path::PathBuf,
    pub tokenizer: std::path::PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub precision: Option<String>,
}
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct VideoSpec { pub size: String, pub fps: f64 }
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Sampling { pub algorithm: String, pub steps: usize, pub timestep_shift: f64 }
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Context { pub chunk_frames: usize, pub window_frames: usize, pub sink_frames: usize }
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Rope {
    pub interpolate: bool, pub native_size: String, pub scene_offset_frames: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stretch: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub by_parts: Option<RopeParts>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ntk_aware: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_aware: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attn_logit_scale: Option<f64>,
    /// M of height and width given directly (the scale of the slowest pair of each side), in place of the stretch
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub m: Option<RopeM>,
    /// NTK only, mu in [0, 1]: pair j of n is scaled by M^(mu + (1 - mu) j / (n - 1)), NTK at 0, PI at 1
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ntk_pi_mix: Option<f64>,
}
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RopeParts { pub alpha: f64, pub beta: f64 }
#[derive(Deserialize, Serialize, Debug, Clone, Copy, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RopeM { pub height: f64, pub width: f64 }
/// A point of generate.rope_sweep: M of height and width and, when given, the NTK to PI mix of that point
#[derive(Deserialize, Serialize, Debug, Clone, Copy, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SweepPoint {
    pub height: f64, pub width: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ntk_pi_mix: Option<f64>,
}

#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Generate {
    pub packages_from: String,
    pub mode: Mode,
    #[serde(default)]
    pub debug: DebugSelection,
    #[serde(default)]
    pub drain: Drain,
    pub prompts: PromptSource,
    pub seed: u64,
    /// where a chunk's keys and values come from: a pass at t = 0 over the clean chunk (as trained) or the sampler's
    /// last pass (one pass a chunk fewer); video.rs chunk_passes
    pub cache_write: CacheWrite,
    /// how the descent starts and walks: the noise of every chunk and the sampler's timestep schedule
    pub sampling: GenerateSampling,
    pub offloading: Offloading,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<Output>,
    /// the knobs of the descent in resolution (descent.rs), applied by the host without a new export
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub descent: Option<Descent>,
    /// offline: the chains again for every M of height and width, in one process (the packages loaded once): the
    /// RoPE table of export.rope with this m (rope.rs), the noise of the seed anew, into <run>/<point tag>/
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_sweep: Option<Vec<SweepPoint>>,
    /// debug kv_write only: timesteps in (0, 1000) of probe passes over the clean chunk noised to t, whose keys and
    /// values are compared with the pass at t = 0 (kvwrite.rs)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_probe: Option<Vec<f64>>,
}

#[derive(Deserialize, Serialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CacheWrite {
    /// a pass at t = 0 over the clean chunk with the whole window, as trained
    CleanPass,
    /// the same pass attending to the sink and the chunk only (the frame mask of kernels.sparse)
    CleanPassSink,
    /// the sampler's last pass, no third pass
    LastStep,
}

#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Descent {
    /// lambda: a number, a number per block, or per block a number or a list per head (descent::lambda_table)
    pub attn_logit_scale: serde_json::Value,
    /// a run folder (descent.json, latent.f32, <chain>.mp4) to compare with, or null
    pub reference: Option<std::path::PathBuf>,
}

/// generate.sampling: the noise every chunk starts from and the schedule the sampler walks, each with all its keys.
/// (They were generate.descent's noise and timestep_shift until 2026-11-25.)
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GenerateSampling {
    pub noise: Noise,
    pub schedule: Schedule,
}

/// The sampler's timestep schedule: null keeps the packages' sampler, a number recomputes UniPC with that shift
/// (descent::sampler), refused unless the packages carry UniPC.
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Schedule {
    pub timestep_shift: Option<f64>,
}

/// The noise of a chunk. "fresh": drawn at the size of the run. "native_resized": the native size's noise of the same
/// seed resized by area with unit variance (descent::resize_noise; correlated, the model answers with colour
/// blotches). "native_spectral": the native noise's lowest frequencies, white again (descent::spectral_noise). The
/// native modes name the native size, "HxW" in pixels.
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum Noise {
    Fresh,
    NativeResized { native: String },
    NativeSpectral { native: String },
}

impl Noise {
    /// The native latent's height and width (a sixteenth of the size in pixels) and whether the noise is the spectral
    /// one, or None for the fresh noise; an error for a size that is no HxW of multiples of 16.
    pub fn native_grid(&self) -> Result<Option<(usize, usize, bool)>, String> {
        let (native, spectral) = match self {
            Self::Fresh => return Ok(None),
            Self::NativeResized { native } => (native, false),
            Self::NativeSpectral { native } => (native, true),
        };
        let bad = || format!("generate.sampling.noise.native must be HxW in pixels, multiples of 16, not {native}");
        let (h, w) = native.split_once('x').ok_or_else(bad)?;
        match (h.parse::<usize>(), w.parse::<usize>()) {
            (Ok(h), Ok(w)) if h > 0 && w > 0 && h % 16 == 0 && w % 16 == 0 => Ok(Some((h / 16, w / 16, spectral))),
            _ => Err(bad()),
        }
    }
}

#[derive(Deserialize, Serialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DebugTool { Fps, Memory, Transfers, Prompts,
    /// the metrics of the descent in resolution (descent.rs): heavy, so only when named
    Descent,
    /// the last step's keys and values against the pass at t = 0 (kvwrite.rs): copies of every cache, only when named
    KvWrite }

#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
#[serde(untagged)]
pub enum DebugSelection { Switch(bool), Tool(DebugTool), Tools(Vec<DebugTool>) }
impl Default for DebugSelection { fn default() -> Self { Self::Switch(false) } }
impl DebugSelection {
    pub fn includes(&self, tool: DebugTool) -> bool {
        match self { Self::Switch(all) => *all && !matches!(tool, DebugTool::Descent | DebugTool::KvWrite), Self::Tool(selected) => *selected == tool,
                     Self::Tools(selected) => selected.contains(&tool) }
    }
    pub fn enabled(&self) -> bool { !matches!(self, Self::Switch(false)) }
}

#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Drain { pub every: String }
impl Default for Drain { fn default() -> Self { Self { every: "5s".into() } } }
impl Drain {
    pub fn interval(&self) -> Result<Option<std::time::Duration>, String> {
        if self.every == "end" { return Ok(None); }
        let seconds = self.every.strip_suffix('s').and_then(|s| s.parse::<f64>().ok())
            .ok_or("drain.every must be a positive duration such as 5s, or end")?;
        if seconds < 1e-9 { return Err("drain.every must be at least one nanosecond".into()); }
        let duration = std::time::Duration::try_from_secs_f64(seconds)
            .map_err(|_| "drain.every is outside the supported duration range")?;
        if duration.is_zero() { return Err("drain.every must be at least one nanosecond".into()); }
        Ok(Some(duration))
    }
}
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
#[serde(tag = "source", rename_all = "snake_case", deny_unknown_fields)]
pub enum PromptSource {
    Interactive { #[serde(default, skip_serializing_if = "Option::is_none")] initial: Option<String> },
    Script { file: std::path::PathBuf, #[serde(default)] chains: Vec<String> },
}
impl PromptSource {
    pub fn initial(&self) -> Option<&str> {
        match self { Self::Interactive { initial } => initial.as_deref(), _ => None }
    }
    pub fn interactive(&self) -> bool { matches!(self, Self::Interactive { .. }) }
}
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Output {
    pub directory: std::path::PathBuf,
    pub codec: String,
    pub crf: u8,
    pub pixel_format: String,
}

/// Implementations baked into the AOT packages, not switches the host can change at run time.
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Kernels {
    pub attention: String,
    pub cross_attention: Vec<String>,
    pub linear: String,
    pub kv_cache: String,
    pub kv_store: String,
    /// the RoPE arithmetic; with fuse.rope_into_sage the joint kernel reproduces it (fused_float32: its own float32)
    pub rope: String,
    pub rope_dtype: String,
    pub fuse: Fuse,
    /// how the W4A4 low-rank branch adds the partial products of its column blocks: "atomic" (Nunchaku's float atomics,
    /// the last bits change from run to run) or "ordered" (each block stores its own, added in a fixed order: the same
    /// bits on every run; live::w4a4_*_ordered, ops/w4a4_ops.cpp)
    pub lora_sum: String,
    /// Block-sparse self-attention of the joint Sage/RoPE kernel; absent or disabled, the packages hold none of it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sparse: Option<Sparse>,
}

/// Each kernel fusion a switch of its own. rope_into_sage is baked into the block packages (RMSNorm, RoPE and the INT8
/// quantization of Q and K inside Sage's preparation); qkv (q, k, v as one W4A4 layer) and gelu_mlp (Nunchaku's fused
/// feed forward) into the weight pack, which export checks against them.
#[derive(Deserialize, Serialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Fuse {
    pub rope_into_sage: bool,
    pub qkv: bool,
    pub gelu_mlp: bool,
}

#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Sparse {
    pub enabled: bool,
    /// the weakest key blocks are skipped while their estimated shares of the attention add up to less: 0 skips none
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skip_mass: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frames: Option<SparseFrames>,
}

/// The static frame mask: the chunk always; the sink (and a shot's pinned chunk) with `sink`; of the ring, the
/// `recent_frames` frames before the chunk.
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SparseFrames {
    pub sink: bool,
    pub recent_frames: usize,
}

impl Kernels {
    /// The block sparsity the packages implement, if any: (skip mass, frames).
    pub fn block_sparsity(&self) -> Option<(f64, &SparseFrames)> {
        match &self.sparse {
            Some(Sparse { enabled: true, skip_mass: Some(mass), frames: Some(frames) }) => Some((*mass, frames)),
            _ => None,
        }
    }
}

impl Kernels {
    pub fn validate(&self) -> Result<(), String> {
        if !matches!(self.linear.as_str(), "nunchaku_w4a4" | "torch_bf16" | "torch_fp8") {
            return Err(format!("kernels.linear: {:?} is unsupported", self.linear));
        }
        for (field, got, supported) in [
            ("rope_dtype", self.rope_dtype.as_str(), "float32"),
        ] {
            if got != supported {
                return Err(format!("kernels.{field}: {got:?} is unsupported; export implements {supported:?}"));
            }
        }
        let dense = self.attention == "torch_sdpa";
        let storage = (self.kv_cache.as_str(), self.kv_store.as_str());
        if !matches!(storage, ("bfloat16", "torch") | ("fp8_e4m3fn", "triton"))
            || (!dense && storage != ("fp8_e4m3fn", "triton")) {
            return Err("kernels.kv_cache/kv_store: choose BF16/torch or FP8/triton; Sage requires FP8/triton".into());
        }
        if dense && self.cross_attention == ["sage_sm89"] {
            return Err("kernels.cross_attention: torch_sdpa takes SDPA backends for both attention paths".into());
        }
        if !matches!(self.attention.as_str(), "sage_sm89" | "torch_sdpa") {
            return Err(format!("kernels.attention: {:?} is unsupported", self.attention));
        }
        // the fused kernel must add the error of the separate kernel it replaces: LongLive's roundings, or its own
        // float32 by name (no separate kernel computes that one)
        match (self.fuse.rope_into_sage, self.rope.as_str()) {
            (false, "torch_inductor" | "triton" | "longlive_triton") => {},
            (false, "fused_float32") => return Err("kernels.rope: fused_float32 exists only with kernels.fuse.rope_into_sage".into()),
            (false, _) => return Err("kernels.rope: choose torch_inductor, triton or longlive_triton".into()),
            (true, _) if self.attention != "sage_sm89" => return Err("kernels.fuse.rope_into_sage needs attention sage_sm89".into()),
            (true, "longlive_triton" | "fused_float32") => {},
            (true, other) => return Err(format!("kernels.fuse.rope_into_sage rounds as longlive_triton or fused_float32 only, not {other:?}")),
        }
        if self.linear != "nunchaku_w4a4" && (self.fuse.qkv || self.fuse.gelu_mlp) {
            return Err("kernels.fuse.qkv/gelu_mlp: only W4A4 weight packs fuse layers".into());
        }
        match (self.lora_sum.as_str(), self.linear.as_str()) {
            ("atomic", _) | ("ordered", "nunchaku_w4a4") => {}
            ("ordered", _) => return Err("kernels.lora_sum: ordered exists only for the W4A4 linear (nunchaku_w4a4)".into()),
            (other, _) => return Err(format!("kernels.lora_sum: {other:?}; choose atomic or ordered")),
        }
        if self.sparse.is_some() && !self.fuse.rope_into_sage {
            return Err("kernels.sparse needs kernels.fuse.rope_into_sage".into());
        }
        if let Some((mass, _)) = self.block_sparsity() {
            if !(mass.is_finite() && (0.0..1.0).contains(&mass)) {
                return Err("kernels.sparse.skip_mass must be in [0, 1)".into());
            }
        }
        if self.cross_attention != ["sage_sm89"] && (self.cross_attention.is_empty() || self.cross_attention.iter().enumerate().any(|(i, b)|
            !matches!(b.as_str(), "cudnn" | "efficient" | "math") || self.cross_attention[..i].contains(b))) {
            return Err("kernels.cross_attention must be [sage_sm89] or list cudnn/efficient once each in priority order".into());
        }
        Ok(())
    }
}

#[derive(Deserialize, Serialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    #[default]
    Live,
    Offline,
}

/// The resources of the model that can be offloaded.
#[derive(Deserialize, Serialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum Resource {
    /// the immutable byte rows of the 30 blocks, dense or quantized
    BlockWeights,
    /// the keys and values of every block's attention over the frames already made
    KvCache,
}

impl Resource {
    pub const ALL: [Resource; 2] = [Resource::BlockWeights, Resource::KvCache];

    /// Capacity decides admission; mutable KV has a temporary disk backing.
    pub fn tiers(self) -> &'static [Tier] {
        &[Tier::Disk, Tier::Ram, Tier::Vram]
    }
}

/// Where a resource is kept, from the least GPU and RAM taken to the most.
#[derive(Deserialize, Serialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    Disk,
    Ram,
    Vram,
    Auto,
}

#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct Offloading {
    /// off: every resource on the GPU for good
    pub enabled: bool,
    /// on: "auto" resources are placed by the calibration (Video::calibrate); off: "auto" is "ram"
    pub calibration: CalibrationPolicy,
    pub resources: BTreeMap<Resource, Tier>,
    /// a copy must take no more than its share of compute divided by this
    pub copy_spare: f32,
    /// frames per second the prefetch calibration may give up for fewer slots (RAM and VRAM buffers of the weights)
    pub prefetch_fps_tolerance: f64,
    /// GiB of the GPU the calibration leaves to other programs when it keeps rows there
    pub vram_margin_gib: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram_margin_fraction: Option<f64>,
    pub ram_margin_fraction: f64,
    /// GiB T5's activations take on the GPU while a prompt is encoded, beside its two row slots
    pub t5_activations_gib: f64,
    pub t5: T5Weights,
    /// how the calibration places "auto" resources (last, so that the default leaves the key of calibration.json as
    /// it was): greedy_ram, as little GPU as there can be at the same speed, or greedy_vram, the GPU filled first
    #[serde(skip_serializing_if = "Policy::is_greedy_ram")]
    pub policy: Policy,
}

/// The rule of the calibration for "auto" resources.
#[derive(Deserialize, Serialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum Policy {
    /// as little GPU memory as there can be without losing speed: the cache and the rows go to RAM (or the disk)
    /// whenever their copies hide behind the blocks' compute
    #[default]
    GreedyRam,
    /// the GPU filled first, up to all of it less the margin: the cache, then rows of weights for good, the rest
    /// streamed from pinned RAM, the disk last (greedy_vram)
    GreedyVram,
}
impl Policy {
    pub fn is_greedy_ram(&self) -> bool { *self == Policy::GreedyRam }
}

/// Where greedy_vram puts the cache and how many rows of weights it keeps on the GPU, and where the others stream from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Greedy { pub kv: Tier, pub resident: usize, pub streamed_from: Tier }

/// greedy_vram: `room` bytes of the GPU (free less the margin and the workspace) take the KV cache (`kv_bytes`) first,
/// then rows of `row_bytes` for good, leaving `slots` rows for the streamed ones if any are streamed; the streamed
/// rows (and the cache, if it did not fit) go to pinned RAM if `ram` bytes hold them, else the rows to the disk.
pub fn greedy_vram(n: usize, room: usize, kv_bytes: usize, row_bytes: usize, slots: usize, ram: usize) -> Greedy {
    let (kv, left) = if kv_bytes <= room { (Tier::Vram, room - kv_bytes) } else { (Tier::Ram, room) };
    if left >= n * row_bytes { return Greedy { kv, resident: n, streamed_from: Tier::Vram }; }
    let resident = (left.saturating_sub(slots * row_bytes) / row_bytes).min(n);
    let ram_need = (n - resident) * row_bytes + if kv == Tier::Ram { kv_bytes } else { 0 };
    Greedy { kv, resident, streamed_from: if ram_need <= ram { Tier::Ram } else { Tier::Disk } }
}

/// What the key of calibration.json hashes: the offloading with calibration if_needed; the default policy is left
/// out, so that a configuration without it keeps the key it had before the policy existed.
pub fn placement_text(o: &Offloading) -> String {
    let mut c = o.clone();
    c.calibration = CalibrationPolicy::IfNeeded;
    format!("{c:?}").replace(", policy: GreedyRam", "")
}

/// T5's blocks (5.6 GB, their reads most of a prompt's time) go through the same streamed rows as the DiT's, from the
/// disk. The first rows that fit in `resident_gib` of the GPU stay there between prompts (the GPU has room, RAM has
/// not), each one a read fewer when a prompt comes; the others are streamed, their slots given back between prompts.
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct T5Weights {
    pub resident_gib: f64,
    /// passes per second of T5's block chain the prefetch calibration may give up for fewer slots
    pub prefetch_tolerance: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resident: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub activations: Option<String>,
}

impl Default for T5Weights {
    fn default() -> T5Weights {
        T5Weights { resident_gib: 1.0, prefetch_tolerance: 0.0, resident: None, activations: None }
    }
}

impl T5Weights {
    /// The rows of `row_bytes` that stay on the GPU, of `blocks`.
    pub fn resident(&self, row_bytes: usize, blocks: usize) -> usize {
        ((self.resident_gib * (1u64 << 30) as f64) as usize / row_bytes).min(blocks)
    }
}

#[derive(Deserialize, Serialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationPolicy { #[default] IfNeeded, Always, Disabled }

impl Default for Offloading {
    fn default() -> Offloading {
        Offloading {
            enabled: true,
            calibration: CalibrationPolicy::IfNeeded,
            resources: Resource::ALL.iter().map(|&r| (r, Tier::Auto)).collect(),
            copy_spare: 1.1,
            prefetch_fps_tolerance: 0.5,
            vram_margin_gib: 1.5,
            vram_margin_fraction: None,
            ram_margin_fraction: 0.0,
            t5_activations_gib: 0.33,
            t5: T5Weights::default(),
            policy: Policy::GreedyRam,
        }
    }
}

impl Offloading {
    pub fn activation_bytes(&self, measured: usize) -> Result<usize, String> {
        if self.t5.activations.as_deref() == Some("auto") {
            if measured == 0 { return Err("auto T5 requires a measured export; rebuild packages".into()); }
            Ok(measured)
        } else { Ok((self.t5_activations_gib * (1u64 << 30) as f64) as usize) }
    }
    pub fn margin_bytes(&self, total: usize) -> usize {
        self.vram_margin_fraction.map_or((self.vram_margin_gib * (1u64 << 30) as f64) as usize,
                                        |f| (f * total as f64) as usize)
    }
    pub fn memory_reserve(&self, total: usize, measured: usize, row: usize) -> Result<usize, String> {
        Ok(self.margin_bytes(total) + self.activation_bytes(measured)? + 2 * row)
    }

    /// The tier `resource` is configured to (a resource the file leaves out is "auto").
    pub fn tier(&self, resource: Resource) -> Tier {
        if !self.enabled {
            return Tier::Vram;
        }
        self.resources.get(&resource).copied().unwrap_or(Tier::Auto)
    }

    /// Whether the calibration places `resource`.
    pub fn calibrates(&self, resource: Resource) -> bool {
        self.calibration != CalibrationPolicy::Disabled && self.tier(resource) == Tier::Auto
    }

    /// The tier `resource` starts on: an "auto" resource starts on the lowest tier it can be on, and the calibration
    /// brings it up as far as its copies need (without calibration "auto" is "ram").
    pub fn initial(&self, resource: Resource) -> Tier {
        match self.tier(resource) {
            // greedy_vram: the cache wants the GPU first (it falls back to RAM at load if it does not fit)
            Tier::Auto if resource == Resource::KvCache && self.policy == Policy::GreedyVram => Tier::Vram,
            Tier::Auto if resource == Resource::KvCache => Tier::Ram,
            Tier::Auto if self.calibration != CalibrationPolicy::Disabled => resource.tiers()[0],
            Tier::Auto => Tier::Ram,
            tier => tier,
        }
    }

    /// Checks that every tier is one the resource can be on.
    pub fn validate(&self) -> Result<(), String> {
        if !self.copy_spare.is_finite() || self.copy_spare < 1.0
            || !self.vram_margin_gib.is_finite() || self.vram_margin_gib < 0.0
            || !self.t5_activations_gib.is_finite() || self.t5_activations_gib < 0.0
            || !self.t5.resident_gib.is_finite() || self.t5.resident_gib < 0.0
            || !self.prefetch_fps_tolerance.is_finite() || self.prefetch_fps_tolerance < 0.0
            || !self.t5.prefetch_tolerance.is_finite() || self.t5.prefetch_tolerance < 0.0 {
            return Err("invalid offloading budgets".into());
        }
        if !self.ram_margin_fraction.is_finite() || !(0.0..1.0).contains(&self.ram_margin_fraction)
            || self.vram_margin_fraction.is_some_and(|f| !f.is_finite() || !(0.0..1.0).contains(&f))
            || self.t5.resident.as_deref().is_some_and(|s| s != "auto")
            || self.t5.activations.as_deref().is_some_and(|s| s != "auto") {
            return Err("invalid automatic memory policy or VRAM fraction".into());
        }
        for (&resource, &tier) in &self.resources {
            if tier != Tier::Auto && !resource.tiers().contains(&tier) {
                return Err(format!("{resource:?} cannot be on the tier {tier:?}"));
            }
        }
        Ok(())
    }
}

/// The calibration's choice from the times of one block: its own compute and, for each resource, the copy it needs
/// over PCIe and, for the weights, the read of a row from the disk (all in ms). A block's budget is block_ms / spare;
/// the KV cache comes first (the largest resource): RAM if its copy fits the budget. The weights get what is left:
/// the disk if a row's read and its copy both fit, else RAM for as many rows as fit spread evenly (a row then has
/// n / k blocks' time), the rest on the GPU. Returns (KV cache tier, weights tier, weight rows streamed).
pub struct Times {
    pub block_ms: f32,
    pub row_copy_ms: f32,
    pub row_read_ms: f32,
    pub kv_copy_ms: f32,
}

pub fn choose(n: usize, t: &Times, spare: f32, kv_auto: bool, weights_auto: bool) -> (Tier, Tier, usize) {
    let budget = t.block_ms / spare;
    let kv = if !kv_auto { Tier::Auto } else if t.kv_copy_ms <= budget { Tier::Ram } else { Tier::Vram };
    let left = budget - if kv == Tier::Ram { t.kv_copy_ms } else { 0.0 };
    if !weights_auto {
        return (kv, Tier::Auto, n);
    }
    if t.row_read_ms <= left && t.row_copy_ms <= left {
        return (kv, Tier::Disk, n);
    }
    let rows = if left > 0.0 { ((n as f32 * left / t.row_copy_ms).floor() as usize).min(n) } else { 0 };
    (kv, if rows > 0 { Tier::Ram } else { Tier::Vram }, rows)
}

/// The KV cache's tier at load. A configured tier is kept as it is; "auto" never puts the cache on the disk (it is
/// written every chunk, and a run must not wear the SSD): the wanted tier if it fits, else the other memory that
/// fits, else an error with the reason.
pub fn kv_tier(configured: Tier, wanted: Tier, ram_fits: bool, vram_fits: bool) -> Result<Tier, String> {
    if configured != Tier::Auto {
        return Ok(configured);
    }
    match (wanted, ram_fits, vram_fits) {
        (Tier::Vram, _, true) => Ok(Tier::Vram),
        (_, true, _) => Ok(Tier::Ram),
        (_, false, true) => Ok(Tier::Vram),
        _ => Err("the KV cache fits neither RAM above the reserve nor the GPU; \"auto\" does not put it on the disk                   (free memory, or set offloading.resources.kv_cache to \"disk\" explicitly)".into()),
    }
}

/// `required` includes KV and the weight buffers that must coexist with it. `available` is already net of
/// the system reserve and pending T5/pixel buffers; do not subtract existing allocations again.
pub fn kv_tier_with_ram_budget(configured: Tier, wanted: Tier, available: usize, required: usize,
                              vram_fits: bool) -> Result<Tier, String> {
    kv_tier(configured, wanted, required <= available, vram_fits)
}

/// Where the calibration put every resource, kept in <packages>/calibration.json for the next run of the same
/// configuration and packages (`key`), so that it makes every resource on its tier at once.
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
pub struct Placement {
    pub key: String,
    pub weights: Tier,
    /// weight rows streamed (the others on the GPU), when the weights are in RAM
    pub rows_streamed: usize,
    pub kv: Tier,
    #[serde(default = "default_prefetch_slots")]
    pub weight_slots: [usize; 2],
}

fn default_prefetch_slots() -> [usize; 2] { [2, 2] }

impl Placement {
    const FILE: &'static str = "calibration.json";

    /// The key of a configuration of the offloading over packages whose manifest is `manifest` (mvp1.json's text).
    pub fn key(offloading: &Offloading, manifest: &str) -> String {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        "native-prefetch-v2".hash(&mut h);
        placement_text(offloading).hash(&mut h);
        manifest.hash(&mut h);
        format!("{:016x}", h.finish())
    }

    /// The placement kept in `directory` for `key`, if there is one.
    pub fn read(directory: &Path, key: &str) -> Option<Placement> {
        let text = std::fs::read_to_string(directory.join(Self::FILE)).ok()?;
        serde_json::from_str::<Placement>(&text).ok().filter(|p| p.key == key
            && p.weight_slots.iter().all(|&n| n > 0 && n <= 30))
    }

    pub fn write(&self, directory: &Path) -> std::io::Result<()> {
        std::fs::write(directory.join(Self::FILE), serde_json::to_string_pretty(self).map_err(std::io::Error::other)?)
    }
}

impl Config {
    pub fn read(path: &Path) -> std::io::Result<Config> {
        let text = std::fs::read_to_string(path)?;
        Self::parse(&text).map_err(|e| std::io::Error::other(format!("{}: {e}", path.display())))
    }
    pub fn parse(text: &str) -> serde_json::Result<Config> {
        // the strict contract first, on the JSON itself: the cheapest check, and the only one of keys and branches
        crate::contract::check(&serde_json::from_str(text)?).map_err(<serde_json::Error as serde::de::Error>::custom)?;
        let stages: [Component; 2] = serde_json::from_str(text)?;
        let config = match stages {
            [Component::Export(export), Component::Generate(generate)] => Config { export, generate },
            _ => return Err(<serde_json::Error as serde::de::Error>::custom("expected [export, generate]")),
        };
        config.validate().map_err(<serde_json::Error as serde::de::Error>::custom)?;
        Ok(config)
    }
    pub fn value(&self) -> serde_json::Value {
        let mut value = serde_json::to_value([Component::Export(self.export.clone()), Component::Generate(self.generate.clone())]).unwrap();
        crate::contract::normalize(&mut value);
        value
    }
    pub fn validate(&self) -> Result<(), String> {
        let e = &self.export;
        if !matches!(e.model.text_encoder.precision.as_deref(), None | Some("int8" | "bfloat16")) {
            return Err("text_encoder.precision must be int8 or bfloat16".into());
        }
        e.kernels.validate()?;
        self.generate.offloading.validate()?;
        self.generate.drain.interval()?;
        if let DebugSelection::Tools(tools) = &self.generate.debug {
            if tools.is_empty() || tools.iter().enumerate().any(|(i, t)| tools[..i].contains(t)) {
                return Err("debug must select nonempty, unique tools".into());
            }
        }
        let size = |s: &str| -> Option<(usize, usize)> {
            let (h,w) = s.split_once('x')?;
            Some((h.parse().ok()?, w.parse().ok()?))
        };
        let (h,w) = size(&e.video.size).ok_or("video.size must be HxW")?;
        if h == 0 || w == 0 || h % 32 != 0 || w % 32 != 0 || e.video.fps != 24.0 {
            return Err("LongLive2 needs positive multiples of 32 and 24 fps".into());
        }
        if e.compiler != "torch_aot_inductor" || !["if_missing_or_changed", "always", "never"].contains(&e.rebuild.as_str()) {
            return Err("unsupported compiler or rebuild policy".into());
        }
        if e.output.as_os_str().is_empty() || self.generate.packages_from != "export" {
            return Err("generate must consume export.output".into());
        }
        if e.sampling.algorithm != "unipc" || e.sampling.steps != 2 || !(e.sampling.timestep_shift.is_finite() && e.sampling.timestep_shift > 0.0 && e.sampling.timestep_shift <= 20.0) {
            return Err("the current distilled checkpoint supports unipc, 2 steps, and a timestep shift in (0, 20]".into());
        }
        // the stretch is bounded by the factor it gives a side of the token grid, (native / now)^stretch in [1/4, 4]
        let factor_ok = |stretch: f64| -> bool {
            let native = size(&e.rope.native_size).unwrap_or((h, w));
            [(native.0, h), (native.1, w)].iter().all(|&(n, now)| {
                let f = (n as f64 / now as f64).powf(stretch);
                f.is_finite() && (0.25..=4.0).contains(&f)
            })
        };
        if e.rope.stretch.is_some_and(|v| !v.is_finite() || !factor_ok(v))
            || e.rope.by_parts.as_ref().is_some_and(|p| !p.alpha.is_finite() || !p.beta.is_finite() || p.alpha <= 0.0 || p.alpha >= p.beta)
            || ([e.rope.ntk_aware == Some(true), e.rope.by_parts.is_some(), e.rope.frequency_aware == Some(true)].into_iter().filter(|v| *v).count() > 1)
            || (!e.rope.interpolate && (e.rope.stretch.is_some() || e.rope.by_parts.is_some() || e.rope.ntk_aware.is_some() || e.rope.frequency_aware.is_some() || e.rope.attn_logit_scale.is_some())) || e.rope.attn_logit_scale.is_some_and(|v| !v.is_finite() || v <= 0.0 || v > 4.0) {
            return Err("invalid RoPE profile: enable interpolation, choose one profile and valid finite parameters".into());
        }
        if let Some(m) = e.rope.m {
            if !e.rope.interpolate || e.rope.stretch.is_some() || ![m.height, m.width].iter().all(|v| v.is_finite() && (0.25..=4.0).contains(v)) {
                return Err("rope.m: M of height and width in [1/4, 4], with interpolation on and stretch null".into());
            }
        }
        if self.generate.cache_write == CacheWrite::CleanPassSink && e.kernels.block_sparsity().is_none() {
            return Err("generate.cache_write clean_pass_sink: the frame mask of block-sparse packages, enable kernels.sparse".into());
        }
        if let Some(probe) = &self.generate.kv_probe {
            if probe.is_empty() || !probe.iter().all(|t| t.is_finite() && *t > 0.0 && *t < 1000.0)
                || !self.generate.debug.includes(DebugTool::KvWrite) {
                return Err("generate.kv_probe: timesteps in (0, 1000), with debug kv_write".into());
            }
        }
        if let Some(points) = &self.generate.rope_sweep {
            if points.is_empty() || !e.rope.interpolate || self.generate.mode != Mode::Offline
                || !points.iter().all(|m| [m.height, m.width].iter().all(|v| v.is_finite() && (0.25..=4.0).contains(v)))
                || points.iter().any(|p| p.ntk_pi_mix.is_some_and(|v| !(0.0..=1.0).contains(&v) || e.rope.ntk_aware != Some(true))) {
                return Err("generate.rope_sweep: offline, interpolation on, M of height and width in [1/4, 4], a mix in [0, 1] with NTK".into());
            }
        }
        if e.rope.ntk_pi_mix.is_some_and(|v| !(0.0..=1.0).contains(&v) || e.rope.ntk_aware != Some(true) || !e.rope.interpolate) {
            return Err("rope.ntk_pi_mix: in [0, 1], of the NTK profile (ntk_aware true)".into());
        }
        // LongLive 2's chunk and sink; its KV window as trained (32) or shorter by whole chunks down to the sink alone
        if e.context.chunk_frames != 8 {
            return Err("export.context.chunk_frames: LongLive 2 runs chunks of 8 frames".into());
        }
        if e.context.sink_frames != 8 {
            return Err("export.context.sink_frames: LongLive 2 keeps a sink of 8 frames".into());
        }
        if ![32, 24, 16].contains(&e.context.window_frames) {
            return Err(format!("export.context.window_frames: {} is untested; choose 32 (as trained), 24 or 16", e.context.window_frames));
        }
        if e.rope.native_size != "704x1280" {
            return Err("export.rope.native_size: LongLive 2 is trained at 704x1280".into());
        }
        let ring_frames = e.context.window_frames - e.context.chunk_frames - e.context.sink_frames;
        if e.kernels.block_sparsity().is_some_and(|(_, f)| f.recent_frames > ring_frames) {
            return Err(format!("kernels.sparse.frames.recent_frames: the ring holds {ring_frames} frames"));
        }
        match &self.generate.prompts {
            PromptSource::Interactive { initial: Some(p) } if p.trim().is_empty() => return Err("initial prompt must be nonempty or absent".into()),
            PromptSource::Script { file, chains } if file.as_os_str().is_empty() || chains.iter().any(|c| c.is_empty()) => return Err("invalid prompt script".into()),
            _ => {}
        }
        if self.generate.sampling.schedule.timestep_shift.is_some_and(|v| !(v.is_finite() && v > 0.0 && v <= 20.0)) {
            return Err("generate.sampling.schedule.timestep_shift must be in (0, 20] or null".into());
        }
        self.generate.sampling.noise.native_grid()?;
        if let Some(d) = &self.generate.descent {
            // the sizes are checked against the model at load (lambda_table), the numbers here
            crate::descent::check_scale(&d.attn_logit_scale)?;
        }
        if self.generate.mode == Mode::Offline && self.generate.output.is_none() {
            return Err("offline output settings are required".into());
        }
        if let Some(o) = &self.generate.output {
            if o.directory.as_os_str().is_empty() || o.codec != "libx264" || o.pixel_format != "yuv420p" || o.crf > 51 {
                return Err("unsupported output encoding".into());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn point_a_uses_fp8_storage_with_math_attention() {
        let c = Config::parse(include_str!("../../../configs/experiments/director-original/point-a-fp8-kv.proposed.json")).unwrap();
        assert_eq!(c.export.kernels.kv_cache, "fp8_e4m3fn");
        assert_eq!(c.export.kernels.attention, "torch_sdpa");
    }
    #[test]
    fn fused_sage_rope_takes_longlive_rounding_only() {
        let mut value = Config::parse(include_str!("../../../configs/experiments/director-original/point-b2r-w4a4-sagerope-704x1280.json")).unwrap().value();
        value[0]["kernels"]["rope"] = serde_json::json!("longlive_triton");
        assert!(Config::parse(&value.to_string()).is_ok());
        value[0]["kernels"]["rope"] = serde_json::json!("triton");
        assert!(Config::parse(&value.to_string()).is_err());
    }
    /// The KV window of LongLive 2 (export.context.window_frames): 32 as trained, or 24 and 16 (the ring of 16, 8 or 0
    /// frames between the sink and the chunk); the export sizes the cache and the block steps from it. Any other window,
    /// chunk or sink is refused by its key.
    #[test]
    fn the_kv_window_is_32_24_or_16_frames() {
        let base = repository().value();
        for window in [32, 24, 16] {
            let mut value = base.clone();
            value[0]["context"]["window_frames"] = serde_json::json!(window);
            assert_eq!(Config::parse(&value.to_string()).unwrap().export.context.window_frames, window);
        }
        for window in [8, 20, 40] {
            let mut value = base.clone();
            value[0]["context"]["window_frames"] = serde_json::json!(window);
            let refused = Config::parse(&value.to_string()).expect_err("an untested window must be refused").to_string();
            assert!(refused.contains("export.context.window_frames"), "{refused}");
        }
        for (key, bad) in [("chunk_frames", 4), ("sink_frames", 4)] {
            let mut value = base.clone();
            value[0]["context"][key] = serde_json::json!(bad);
            assert!(Config::parse(&value.to_string()).expect_err("refused").to_string().contains(&format!("export.context.{key}")));
        }
    }
    #[test]
    fn prefetch_tolerances_come_from_the_config_and_are_part_of_the_placement_key() {
        let mut value = repository().value();
        assert_eq!(Config::parse(&value.to_string()).unwrap().generate.offloading.prefetch_fps_tolerance, 0.5);
        assert_eq!(Config::parse(&value.to_string()).unwrap().generate.offloading.t5.prefetch_tolerance, 0.0);
        value[1]["offloading"]["prefetch_fps_tolerance"] = serde_json::json!(0.25);
        value[1]["offloading"]["t5"]["prefetch_tolerance"] = serde_json::json!(0.1);
        let c = Config::parse(&value.to_string()).unwrap();
        assert_eq!(c.generate.offloading.prefetch_fps_tolerance, 0.25);
        assert_eq!(c.generate.offloading.t5.prefetch_tolerance, 0.1);
        let base = Offloading::default();
        let mut other = base.clone();
        other.prefetch_fps_tolerance = 0.25;
        assert_ne!(Placement::key(&base, "m"), Placement::key(&other, "m"));
        for bad in [-0.1, f64::NAN] {
            let mut o = Offloading::default();
            o.prefetch_fps_tolerance = bad;
            assert!(o.validate().is_err());
            let mut o = Offloading::default();
            o.t5.prefetch_tolerance = bad;
            assert!(o.validate().is_err());
        }
    }
    #[test]
    fn automatic_kv_never_goes_to_the_disk() {
        assert_eq!(kv_tier(Tier::Auto, Tier::Ram, true, false), Ok(Tier::Ram));
        assert_eq!(kv_tier(Tier::Auto, Tier::Ram, false, true), Ok(Tier::Vram));
        assert_eq!(kv_tier(Tier::Auto, Tier::Vram, true, true), Ok(Tier::Vram));
        assert_eq!(kv_tier(Tier::Auto, Tier::Vram, true, false), Ok(Tier::Ram));
        // a calibration.json of an older run that chose the disk
        assert_eq!(kv_tier(Tier::Auto, Tier::Disk, true, false), Ok(Tier::Ram));
        assert!(kv_tier(Tier::Auto, Tier::Ram, false, false).unwrap_err().contains("disk"));
        assert_eq!(kv_tier(Tier::Disk, Tier::Disk, false, false), Ok(Tier::Disk));
    }
    #[test]
    fn kv_admission_includes_weight_staging_before_choosing_ram() {
        // KV alone fits in 5 units, but KV + mandatory weight staging requires 6.
        assert_eq!(kv_tier_with_ram_budget(Tier::Auto, Tier::Ram, 5, 6, true), Ok(Tier::Vram));
        assert!(kv_tier_with_ram_budget(Tier::Auto, Tier::Ram, 5, 6, false).is_err());
        assert_eq!(kv_tier_with_ram_budget(Tier::Auto, Tier::Ram, 6, 6, true), Ok(Tier::Ram));
    }
    #[test]
    fn point_a_prime_takes_memory_efficient_sdpa_and_rejects_sage_cross() {
        let c = Config::parse(include_str!("../../../configs/experiments/director-original/point-a-prime.json")).unwrap();
        assert_eq!(c.export.kernels.attention, "torch_sdpa");
        assert_eq!(c.export.kernels.cross_attention, ["efficient"]);
        let mut value = c.value();
        value[0]["kernels"]["cross_attention"] = serde_json::json!(["sage_sm89"]);
        assert!(Config::parse(&value.to_string()).is_err());
    }
    #[test]
    fn fp8_recipe_resolves_measured_t5_memory_and_fractional_margin() {
        let c = Config::parse(include_str!("../../../configs/experiments/director-original/s2-fp8-native.proposed.json")).unwrap();
        let o = &c.generate.offloading;
        assert_eq!(o.memory_reserve(8 << 30, 400 << 20, 200 << 20).unwrap(),
                   (1.5 * (1u64 << 30) as f64) as usize + (400 << 20) + (400 << 20));
        assert!(o.memory_reserve(8 << 30, 0, 200 << 20).is_err());
    }
    #[test]
    fn dense_linear_uses_the_same_native_configuration() {
        let mut value = repository().value();
        value[0]["kernels"]["linear"] = serde_json::json!("torch_bf16");
        assert!(Config::parse(&value.to_string()).unwrap_err().to_string().contains("only W4A4 weight packs fuse layers"));
        value[0]["kernels"]["fuse"] = serde_json::json!({"rope_into_sage": true, "qkv": false, "gelu_mlp": false});
        assert_eq!(Config::parse(&value.to_string()).unwrap().export.kernels.linear, "torch_bf16");
    }
    #[test]
    fn dense_attention_rejects_quantized_cache_combinations() {
        let mut value = repository().value();
        value[0]["kernels"]["attention"] = serde_json::json!("torch_sdpa");
        value[0]["kernels"]["rope"] = serde_json::json!("torch_inductor");
        value[0]["kernels"]["fuse"]["rope_into_sage"] = serde_json::json!(false);
        value[0]["kernels"]["kv_cache"] = serde_json::json!("bfloat16");
        value[0]["kernels"]["kv_store"] = serde_json::json!("torch");
        value[0]["kernels"]["cross_attention"] = serde_json::json!(["math"]);
        assert!(Config::parse(&value.to_string()).is_ok());
        value[0]["kernels"]["kv_cache"] = serde_json::json!("fp8_e4m3fn");
        assert!(Config::parse(&value.to_string()).is_err());
    }
    #[test]
    fn dense_text_precision_is_explicit_and_typos_are_rejected() {
        let mut value = repository().value();
        value[0]["model"]["text_encoder"]["precision"] = serde_json::json!("bfloat16");
        assert!(Config::parse(&value.to_string()).is_ok());
        value[0]["model"]["text_encoder"]["precision"] = serde_json::json!("bf116");
        assert!(Config::parse(&value.to_string()).is_err());
    }
    use super::*;
    fn repository() -> Config { Config::parse(include_str!("../../../configs/mvp1.json")).unwrap() }
    #[test]
    fn debug_selects_only_known_tools_and_drain_accepts_intervals_or_end() {
        use serde_json::json;
        for debug in [json!(false), json!(true), json!("fps"), json!(["fps"]), json!("memory"), json!("transfers"), json!("prompts"), json!(["fps", "memory", "transfers", "prompts"])] {
            for every in ["5s", "10s", "0.5s", "end"] {
                let mut value = repository().value();
                value[1]["debug"] = debug.clone();
                if debug == json!(false) {
                    // no diagnostics, no drain: the contract refuses a drain beside them
                    value[1].as_object_mut().unwrap().remove("drain");
                    assert!(Config::parse(&value.to_string()).is_ok());
                    value[1]["drain"] = json!({"every": every});
                    assert!(Config::parse(&value.to_string()).is_err());
                    continue;
                }
                value[1]["drain"] = json!({"every": every});
                let parsed = Config::parse(&value.to_string()).expect("valid debug/drain config refused");
                assert_eq!(parsed.value()[1]["drain"], value[1]["drain"]);
            }
        }
        let mut value = repository().value();
        value[1].as_object_mut().unwrap().remove("debug");
        assert!(Config::parse(&value.to_string()).is_err(), "debug must be written out");
        value[1]["drain"] = json!({"every": "5s"});
        for debug in [json!("unknown"), json!(["fps", "unknown"]), json!([]), json!(["fps", "fps"]), json!(1)] {
            value[1]["debug"] = debug;
            assert!(Config::parse(&value.to_string()).is_err());
        }
        value[1]["debug"] = json!("fps");
        for every in ["never", "0s", "-5s", "NaNs", "infs", "1e999s", "", "5", "0.00000000001s", "0.0000000009s", "1e20s"] {
            value[1]["drain"] = json!({"every": every});
            assert!(Config::parse(&value.to_string()).is_err(), "{every}");
        }
        value[1]["drain"] = json!({"every": "5s", "typo": true});
        assert!(Config::parse(&value.to_string()).is_err());
    }
    #[test]
    fn the_approved_component_array_is_a_valid_configuration() {
        assert!(Config::parse(include_str!("../../../configs/pipeline.proposed.json")).is_ok());
        assert_eq!(Config::parse(&repository().value().to_string()).unwrap(), repository());
    }
    #[test]
    fn kernels_and_every_one_of_their_fields_are_required() {
        let value = repository().value();
        for key in value[0]["kernels"].as_object().unwrap().keys() {
            let mut missing = value.clone();
            missing[0]["kernels"].as_object_mut().unwrap().remove(key);
            assert!(Config::parse(&missing.to_string()).is_err(), "{key}");
        }
        assert!(Config::parse("{}").is_err());
        assert!(Config::read(Path::new("Z:/nowhere/config.json")).is_err());
    }
    #[test]
    fn triton_and_inductor_rope_are_explicit_export_choices() {
        let mut value = repository().value();
        value[0]["kernels"]["fuse"]["rope_into_sage"] = serde_json::json!(false);
        for backend in ["torch_inductor", "triton", "longlive_triton"] {
            value[0]["kernels"]["rope"] = serde_json::json!(backend);
            let config = Config::parse(&value.to_string()).expect("supported RoPE refused");
            assert_eq!(config.export.kernels.rope, backend);
        }
    }
    #[test]
    fn each_fusion_is_its_own_switch_and_fused_rope_rounds_as_a_separate_one() {
        use serde_json::json;
        let parse = |v: &serde_json::Value| Config::parse(&v.to_string()).map_err(|e| e.to_string());
        let mut value = repository().value();
        for (fused, rope) in [(true, "longlive_triton"), (true, "fused_float32"), (false, "longlive_triton")] {
            value[0]["kernels"]["fuse"]["rope_into_sage"] = json!(fused);
            value[0]["kernels"]["rope"] = json!(rope);
            let k = parse(&value).expect("supported fusion refused").export.kernels;
            assert_eq!((k.fuse.rope_into_sage, k.rope.as_str()), (fused, rope));
        }
        value[0]["kernels"]["fuse"]["rope_into_sage"] = json!(true);
        for rope in ["triton", "torch_inductor"] {
            value[0]["kernels"]["rope"] = json!(rope);
            assert!(parse(&value).unwrap_err().contains("rounds as longlive_triton or fused_float32 only"), "{rope}");
        }
        value[0]["kernels"]["rope"] = json!("fused_float32");
        value[0]["kernels"]["fuse"]["rope_into_sage"] = json!(false);
        assert!(parse(&value).unwrap_err().contains("fused_float32 exists only with kernels.fuse.rope_into_sage"));
        value[0]["kernels"]["rope"] = json!("longlive_triton");
        value[0]["kernels"]["attention"] = json!("sage_rope_sm89");
        assert!(parse(&value).is_err(), "the old joint name is a switch now");
        value[0]["kernels"]["attention"] = json!("torch_sdpa");
        value[0]["kernels"]["kv_cache"] = json!("bfloat16");
        value[0]["kernels"]["kv_store"] = json!("torch");
        value[0]["kernels"]["cross_attention"] = json!(["efficient"]);
        value[0]["kernels"]["fuse"]["rope_into_sage"] = json!(true);
        assert!(parse(&value).unwrap_err().contains("rope_into_sage needs attention sage_sm89"));
        for (qkv, gelu) in [(true, false), (false, true), (false, false)] {
            let mut layers = repository().value();
            layers[0]["kernels"]["fuse"]["qkv"] = json!(qkv);
            layers[0]["kernels"]["fuse"]["gelu_mlp"] = json!(gelu);
            let k = parse(&layers).expect("layer fusions are independent switches").export.kernels;
            assert_eq!((k.fuse.qkv, k.fuse.gelu_mlp), (qkv, gelu));
        }
    }
    #[test]
    fn sage_cross_is_an_explicit_complete_backend() {
        let mut value = repository().value();
        value[0]["kernels"]["cross_attention"] = serde_json::json!(["sage_sm89"]);
        assert!(Config::parse(&value.to_string()).is_ok());
        value[0]["kernels"]["cross_attention"] = serde_json::json!(["sage_sm89", "cudnn"]);
        assert!(Config::parse(&value.to_string()).is_err());
    }
    #[test]
    fn unknown_kernels_bad_component_order_and_legacy_flags_are_refused() {
        let value = repository().value();
        for key in value[0]["kernels"].as_object().unwrap().keys() {
            let mut unknown = value.clone();
            unknown[0]["kernels"][key] = if key == "cross_attention" { serde_json::json!(["unknown"]) } else { serde_json::json!("unknown") };
            assert!(Config::parse(&unknown.to_string()).is_err(), "{key}");
        }
        for cross in [serde_json::json!([]), serde_json::json!(["cudnn", "cudnn"])] {
            let mut bad = value.clone(); bad[0]["kernels"]["cross_attention"] = cross;
            assert!(Config::parse(&bad.to_string()).is_err());
        }
        let mut bad = value.clone(); bad.as_array_mut().unwrap().reverse();
        assert!(Config::parse(&bad.to_string()).is_err());
        let mut bad = value.clone(); bad[1]["recalibrate"] = serde_json::json!(true);
        assert!(Config::parse(&bad.to_string()).is_err());
        let mut bad = value; bad[0]["kernels"]["atention"] = serde_json::json!("sage_sm89");
        assert!(Config::parse(&bad.to_string()).is_err());
    }
    #[test]
    fn every_resource_is_explicit_and_its_calibration_is_on() {
        let c = repository().generate.offloading;
        assert!(c.enabled && c.calibration == CalibrationPolicy::IfNeeded);
        for r in Resource::ALL { assert!(c.resources.contains_key(&r)); assert!(c.calibrates(r)); }
    }
    #[test]
    fn calibration_policies_preserve_their_placement_and_tier_semantics() {
        let mut o = Offloading::default();
        assert_eq!(o.initial(Resource::BlockWeights), Tier::Disk);
        assert_eq!(o.initial(Resource::KvCache), Tier::Ram);
        let key = Placement::key(&o,"packages");
        o.calibration = CalibrationPolicy::Always;
        assert_eq!(Placement::key(&o,"packages"), key);
        assert!(o.calibrates(Resource::BlockWeights));
        o.calibration = CalibrationPolicy::Disabled;
        assert_eq!(o.initial(Resource::BlockWeights), Tier::Ram);
        assert!(!o.calibrates(Resource::KvCache));
        o.enabled = false;
        for r in Resource::ALL { assert_eq!(o.initial(r),Tier::Vram); }
    }
    #[test]
    fn ram_reserve_validation() {
        let mut o = Offloading::default();
        for fraction in [-0.1, 1.0, f64::NAN, f64::INFINITY] {
            o.ram_margin_fraction = fraction; assert!(o.validate().is_err());
        }
        o.ram_margin_fraction = 0.125; assert!(o.validate().is_ok());
    }
    #[test]
    fn configured_tiers_are_kept_and_mutable_resources_can_go_to_disk() {
        let mut o = Offloading::default();
        o.resources.insert(Resource::BlockWeights,Tier::Disk);
        o.resources.insert(Resource::KvCache,Tier::Vram);
        assert_eq!(o.initial(Resource::BlockWeights),Tier::Disk);
        assert_eq!(o.initial(Resource::KvCache),Tier::Vram);
        assert!(!o.calibrates(Resource::KvCache));
        o.resources.insert(Resource::KvCache,Tier::Disk);
        assert!(o.validate().is_ok());
        for r in Resource::ALL { assert!(r.tiers().contains(&Tier::Disk)); }
    }
    #[test]
    fn live_and_offline_use_the_same_export_and_memory_recipe() {
        // offline runs follow the latest version
        let live=Config::parse(include_str!("../../../configs/mvp3.json")).unwrap();
        let offline=Config::parse(include_str!("../../../configs/offline.json")).unwrap();
        assert_eq!(live.export,offline.export);
        assert_eq!(live.generate.offloading,offline.generate.offloading);
        assert_eq!(offline.generate.mode,Mode::Offline);
        assert!(matches!(offline.generate.prompts,PromptSource::Script {..}));
    }
    #[test]
    fn prompt_sources_are_independent_of_output_and_have_no_hidden_initial_prompt() {
        let mut value=repository().value();
        value[1]["prompts"] = serde_json::json!({"source":"interactive","initial":null});
        assert_eq!(Config::parse(&value.to_string()).unwrap().generate.prompts.initial(),None);
        value[1]["prompts"] = serde_json::json!({"source":"interactive"});
        assert!(Config::parse(&value.to_string()).is_err(), "initial must be written out, null or a prompt");
        value[1]["prompts"] = serde_json::json!({"source":"script","file":"a.json","chains":[]});
        assert!(Config::parse(&value.to_string()).is_ok());
        value[1]["prompts"] = serde_json::json!({"source":"script","file":"a.json"});
        assert!(Config::parse(&value.to_string()).is_err(), "chains must be written out");
        value[1]["prompts"] = serde_json::json!({"source":"script","file":"a.json","chains":[]});
        value[1]["prompts"]["initial"] = serde_json::json!("bad");
        assert!(Config::parse(&value.to_string()).is_err());
    }
    #[test]
    fn t5_keeps_the_rows_its_budget_holds() {
        let t5 = repository().generate.offloading.t5;
        assert_eq!(t5.resident(235_000_000,24),4);
        assert_eq!(T5Weights { resident_gib: 0.0, ..Default::default() }.resident(235_000_000,24),0);
        assert_eq!(T5Weights { resident_gib: 100.0, ..Default::default() }.resident(235_000_000,24),24);
    }

    /// The calibration's result is kept beside the packages, so that the next run makes every resource on its tier at
    /// once and nothing moves; it is used only for the same configuration and packages.
    #[test]
    fn a_placement_is_used_again_for_the_same_configuration_only() {
        let dir = std::env::temp_dir().join(format!("live-placement-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = Placement { key: "k1".into(), weights: Tier::Disk, rows_streamed: 30, kv: Tier::Ram, weight_slots: [3, 4] };
        assert_eq!(Placement::read(&dir, "k1"), None);
        p.write(&dir).unwrap();
        assert_eq!(Placement::read(&dir, "k1"), Some(p.clone()));
        assert_eq!(Placement::read(&dir, "k2"), None);
        let invalid = Placement { weight_slots: [0, 4], ..p.clone() };
        invalid.write(&dir).unwrap();
        assert_eq!(Placement::read(&dir, "k1"), None);
        std::fs::remove_dir_all(&dir).unwrap();
        // the key changes with the configuration of the offloading and with the packages
        let o = Offloading::default();
        let mut other = o.clone();
        other.copy_spare = 1.2;
        assert_ne!(Placement::key(&o, "manifest a"), Placement::key(&other, "manifest a"));
        assert_ne!(Placement::key(&o, "manifest a"), Placement::key(&o, "manifest b"));
        assert_eq!(Placement::key(&o, "manifest a"), Placement::key(&o.clone(), "manifest a"));
    }

    /// As little memory as there can be at the most speed.
    #[test]
    fn the_lowest_tier_whose_copies_hide() {
        // 704x1280: a block 57.5 ms, a row's copy 7.8 ms and its read from the disk about 26 ms, the cache 15 ms
        let t = Times { block_ms: 57.5, row_copy_ms: 7.8, row_read_ms: 26.0, kv_copy_ms: 15.0 };
        assert_eq!(choose(30, &t, 1.1, true, true), (Tier::Ram, Tier::Disk, 30));
        // 480x864: a block 18.8 ms; the disk read does not fit beside the cache's copy, the rows go to RAM
        let t = Times { block_ms: 18.8, row_copy_ms: 7.1, row_read_ms: 26.0, kv_copy_ms: 6.3 };
        assert_eq!(choose(30, &t, 1.1, true, true), (Tier::Ram, Tier::Ram, 30));
    }

    #[test]
    fn what_cannot_hide_stays_on_the_gpu() {
        // the cache alone takes longer than a block: it stays, the rows get the whole budget
        let t = Times { block_ms: 10.0, row_copy_ms: 30.0, row_read_ms: 100.0, kv_copy_ms: 12.0 };
        assert_eq!(choose(30, &t, 1.1, true, true), (Tier::Vram, Tier::Ram, 9));
        // the cache hides, the rows get the time it leaves: 30 * (9.09 - 5) / 30
        let t = Times { kv_copy_ms: 5.0, ..t };
        assert_eq!(choose(30, &t, 1.1, true, true), (Tier::Ram, Tier::Ram, 4));
        // a resource with a configured tier is not placed
        assert_eq!(choose(30, &t, 1.1, false, true).0, Tier::Auto);
        assert_eq!(choose(30, &t, 1.1, true, false).1, Tier::Auto);
    }
}
    /// greedy_vram: the cache on the GPU first, then rows of weights for good, the rest streamed from pinned RAM, the
    /// disk last; the GPU is filled up to `room` bytes (free less the margin and the workspace)
    #[test]
    fn greedy_vram_fills_the_gpu_with_the_cache_then_rows_then_streams_from_ram_then_disk() {
        let (n, row, kv, slots) = (30, 100, 1000, 2);
        // everything fits: cache and all rows on the GPU, nothing streamed, no slots needed
        assert_eq!(greedy_vram(n, 1000 + 30 * 100, kv, row, slots, 10_000),
                   Greedy { kv: Tier::Vram, resident: 30, streamed_from: Tier::Vram });
        // the cache and 4 rows besides the 2 streaming slots: 26 rows streamed from RAM
        assert_eq!(greedy_vram(n, 1000 + 6 * 100, kv, row, slots, 10_000),
                   Greedy { kv: Tier::Vram, resident: 4, streamed_from: Tier::Ram });
        // RAM too small for the streamed rows: from the disk
        assert_eq!(greedy_vram(n, 1000 + 6 * 100, kv, row, slots, 2_000),
                   Greedy { kv: Tier::Vram, resident: 4, streamed_from: Tier::Disk });
        // the cache does not fit: it goes to RAM, the GPU takes rows only, RAM must hold the cache and the rows
        assert_eq!(greedy_vram(n, 700, kv, row, slots, 10_000),
                   Greedy { kv: Tier::Ram, resident: 5, streamed_from: Tier::Ram });
        assert_eq!(greedy_vram(n, 700, kv, row, slots, 3_000),
                   Greedy { kv: Tier::Ram, resident: 5, streamed_from: Tier::Disk });
        // no room even for the slots: nothing resident
        assert_eq!(greedy_vram(n, 1000 + 100, kv, row, slots, 10_000).resident, 0);
    }

    /// greedy_vram decides the cache at load, before the weights' buffers take the GPU: it starts there if it fits
    #[test]
    fn greedy_vram_starts_the_cache_on_the_gpu_and_greedy_ram_in_ram() {
        let mut o = Offloading::default();
        assert_eq!(o.initial(Resource::KvCache), Tier::Ram);
        o.policy = Policy::GreedyVram;
        assert_eq!(o.initial(Resource::KvCache), Tier::Vram);
        o.resources.insert(Resource::KvCache, Tier::Ram);
        assert_eq!(o.initial(Resource::KvCache), Tier::Ram, "an explicit tier wins over the policy");
    }

    /// the default policy leaves the key of calibration.json as it was before the policy existed
    #[test]
    fn the_default_policy_keeps_the_calibration_key_and_greedy_vram_changes_it() {
        let mut o = Offloading::default();
        assert_eq!(o.policy, Policy::GreedyRam);
        assert!(!placement_text(&o).contains("policy"), "{}", placement_text(&o));
        assert_eq!(placement_text(&o), format!("{o:?}").replace(", policy: GreedyRam", ""));
        let before = Placement::key(&o, "m");
        o.policy = Policy::GreedyVram;
        assert!(placement_text(&o).contains("GreedyVram"));
        assert_ne!(Placement::key(&o, "m"), before);
    }

    /// the stretch is bounded by the factor it gives a side, (trained / now)^stretch in [1/4, 4], not by itself: near the
    /// native size a side's ratio is close to 1 and a strong factor needs a large stretch (672: M 1.45 is stretch 8)
    #[test]
    fn the_rope_stretch_is_bounded_by_the_factor_it_gives() {
        let at = |size: &str, stretch: f64| {
            let mut v: serde_json::Value = serde_json::from_str(include_str!("../../../configs/mvp1.json")).unwrap();
            v[0]["video"]["size"] = serde_json::json!(size);
            v[0]["rope"] = serde_json::json!({"interpolate": true, "native_size": "704x1280", "scene_offset_frames": 8,
                                              "stretch": stretch, "by_parts": null, "ntk_aware": true});
            Config::parse(&v.to_string()).map_err(|e| e.to_string()).and_then(|c| c.validate())
        };
        assert!(at("672x1216", 8.0).is_ok(), "M 1.45 at 672");
        assert!(at("512x928", 1.0).is_ok());
        assert!(at("512x928", 5.0).is_err(), "a factor of 4.9");
        assert!(at("512x928", -5.0).is_err(), "a factor of 1/4.9");
        assert!(at("672x1216", 1e300).is_err(), "an infinite factor");
    }

    /// rope.m gives M per side in place of the stretch (each in [1/4, 4]), rope.ntk_pi_mix in [0, 1] is of NTK only
    #[test]
    fn rope_m_per_side_and_the_ntk_pi_mix_are_checked() {
        let at = |rope: serde_json::Value| {
            let mut v: serde_json::Value = serde_json::from_str(include_str!("../../../configs/mvp1.json")).unwrap();
            v[0]["video"]["size"] = serde_json::json!("608x1120");
            let mut base = serde_json::json!({"interpolate": true, "native_size": "704x1280", "scene_offset_frames": 8,
                                              "stretch": null, "by_parts": null, "ntk_aware": true});
            for (k, value) in rope.as_object().unwrap() { base[k] = value.clone(); }
            v[0]["rope"] = base;
            Config::parse(&v.to_string()).map_err(|e| e.to_string()).and_then(|c| c.validate())
        };
        let m = |h: f64, w: f64| serde_json::json!({"height": h, "width": w});
        assert!(at(serde_json::json!({"m": m(1.2, 1.4)})).is_ok());
        assert!(at(serde_json::json!({"m": m(1.2, 1.2), "ntk_pi_mix": 0.5})).is_ok());
        assert!(at(serde_json::json!({"m": m(1.2, 5.0)})).is_err(), "width M 5");
        assert!(at(serde_json::json!({"m": m(0.2, 1.0)})).is_err(), "height M 0.2");
        assert!(at(serde_json::json!({"m": m(1.2, 1.2), "stretch": 1.0})).is_err(), "m and a stretch");
        assert!(at(serde_json::json!({"m": m(1.2, 1.2), "ntk_pi_mix": 1.5})).is_err(), "mix 1.5");
        assert!(at(serde_json::json!({"m": m(1.2, 1.2), "ntk_aware": false, "ntk_pi_mix": 0.5})).is_err(), "mix without NTK");
        assert!(at(serde_json::json!({"stretch": 1.0})).is_ok(), "the stretch alone as before");
    }

    /// the descent's metrics are heavy (a probe and a copy per block): only when named, never by debug: true
    #[test]
    fn descent_metrics_are_selected_only_by_name() {
        assert!(!DebugSelection::Switch(true).includes(DebugTool::Descent));
        assert!(DebugSelection::Tools(vec![DebugTool::Fps, DebugTool::Descent]).includes(DebugTool::Descent));
        let mut value: serde_json::Value = serde_json::from_str(include_str!("../../../configs/mvp1.json")).unwrap();
        value[1]["debug"] = serde_json::json!(["fps", "descent"]);
        value[1]["drain"] = serde_json::json!({"every": "end"});
        let parsed = Config::parse(&value.to_string()).unwrap();
        assert!(parsed.generate.debug.includes(DebugTool::Descent));
    }

    /// the comparison of the last step's keys and values with the pass at t = 0 (kvwrite.rs) copies a sample of every
    /// block's cache twice a chunk: only when named
    #[test]
    fn the_kv_write_comparison_is_selected_only_by_name() {
        assert!(!DebugSelection::Switch(true).includes(DebugTool::KvWrite));
        let mut value: serde_json::Value = serde_json::from_str(include_str!("../../../configs/mvp1.json")).unwrap();
        value[1]["debug"] = serde_json::json!(["fps", "kv_write"]);
        value[1]["drain"] = serde_json::json!({"every": "end"});
        assert!(Config::parse(&value.to_string()).unwrap().generate.debug.includes(DebugTool::KvWrite));
    }

    #[test]
    fn debug_true_selects_all_metrics_but_fps_stays_lightweight() {
        for tool in [DebugTool::Fps, DebugTool::Memory, DebugTool::Transfers, DebugTool::Prompts] {
            assert!(DebugSelection::Switch(true).includes(tool));
            assert!(!DebugSelection::Switch(false).includes(tool));
            assert_eq!(DebugSelection::Tool(DebugTool::Fps).includes(tool), tool == DebugTool::Fps);
        }
    }
