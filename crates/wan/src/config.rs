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
pub struct Rope { pub interpolate: bool, pub native_size: String, pub scene_offset_frames: usize }

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
    pub offloading: Offloading,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<Output>,
}

#[derive(Deserialize, Serialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DebugTool { Fps }

#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
#[serde(untagged)]
pub enum DebugSelection { Switch(bool), Tool(DebugTool), Tools(Vec<DebugTool>) }
impl Default for DebugSelection { fn default() -> Self { Self::Switch(false) } }
impl DebugSelection {
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
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Kernels {
    pub attention: String,
    pub cross_attention: Vec<String>,
    pub linear: String,
    pub kv_cache: String,
    pub kv_store: String,
    #[serde(default, deserialize_with = "explicit_rope", skip_serializing_if = "Option::is_none")]
    pub rope: Option<String>,
    pub rope_dtype: String,
}

fn explicit_rope<'de, D: serde::Deserializer<'de>>(input: D) -> Result<Option<String>, D::Error> {
    String::deserialize(input).map(Some)
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
        match self.attention.as_str() {
            "sage_sm89" | "torch_sdpa" if matches!(self.rope.as_deref(), Some("torch_inductor" | "triton" | "longlive_triton")) => {},
            "sage_rope_sm89" if self.rope.is_none() => {},
            "sage_rope_sm89" => return Err("kernels.rope must be absent: sage_rope_sm89 owns RoPE".into()),
            "sage_sm89" | "torch_sdpa" => return Err("kernels.rope: choose torch_inductor, triton or longlive_triton".into()),
            _ => return Err(format!("kernels.attention: {:?} is unsupported", self.attention)),
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
    /// GiB of the GPU the calibration leaves to other programs when it keeps rows there
    pub vram_margin_gib: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram_margin_fraction: Option<f64>,
    pub ram_margin_fraction: f64,
    /// GiB T5's activations take on the GPU while a prompt is encoded, beside its two row slots
    pub t5_activations_gib: f64,
    pub t5: T5Weights,
}

/// T5's blocks (5.6 GB, their reads most of a prompt's time) go through the same streamed rows as the DiT's, from the
/// disk. The first rows that fit in `resident_gib` of the GPU stay there between prompts (the GPU has room, RAM has
/// not), each one a read fewer when a prompt comes; the others are streamed, their slots given back between prompts.
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct T5Weights {
    pub resident_gib: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resident: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub activations: Option<String>,
}

impl Default for T5Weights {
    fn default() -> T5Weights {
        T5Weights { resident_gib: 1.0, resident: None, activations: None }
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
            vram_margin_gib: 1.5,
            vram_margin_fraction: None,
            ram_margin_fraction: 0.0,
            t5_activations_gib: 0.33,
            t5: T5Weights::default(),
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
            || !self.t5.resident_gib.is_finite() || self.t5.resident_gib < 0.0 {
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

/// Where the calibration put every resource, kept in <packages>/calibration.json for the next run of the same
/// configuration and packages (`key`), so that it makes every resource on its tier at once.
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
pub struct Placement {
    pub key: String,
    pub weights: Tier,
    /// weight rows streamed (the others on the GPU), when the weights are in RAM
    pub rows_streamed: usize,
    pub kv: Tier,
}

impl Placement {
    const FILE: &'static str = "calibration.json";

    /// The key of a configuration of the offloading over packages whose manifest is `manifest` (live.json's text).
    pub fn key(offloading: &Offloading, manifest: &str) -> String {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        let mut placement_config = offloading.clone();
        placement_config.calibration = CalibrationPolicy::IfNeeded;
        format!("{placement_config:?}").hash(&mut h);
        manifest.hash(&mut h);
        format!("{:016x}", h.finish())
    }

    /// The placement kept in `directory` for `key`, if there is one.
    pub fn read(directory: &Path, key: &str) -> Option<Placement> {
        let text = std::fs::read_to_string(directory.join(Self::FILE)).ok()?;
        serde_json::from_str::<Placement>(&text).ok().filter(|p| p.key == key)
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
        let stages: [Component; 2] = serde_json::from_str(text)?;
        let config = match stages {
            [Component::Export(export), Component::Generate(generate)] => Config { export, generate },
            _ => return Err(<serde_json::Error as serde::de::Error>::custom("expected [export, generate]")),
        };
        config.validate().map_err(<serde_json::Error as serde::de::Error>::custom)?;
        Ok(config)
    }
    pub fn value(&self) -> serde_json::Value {
        serde_json::to_value([Component::Export(self.export.clone()), Component::Generate(self.generate.clone())]).unwrap()
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
        if e.sampling.algorithm != "unipc" || e.sampling.steps != 2 || e.sampling.timestep_shift != 5.0 {
            return Err("the current distilled checkpoint supports unipc, 2 steps, shift 5".into());
        }
        if e.context.chunk_frames != 8 || e.context.window_frames != 32 || e.context.sink_frames != 8 || e.rope.native_size != "704x1280" {
            return Err("unsupported LongLive2 context or native RoPE size".into());
        }
        match &self.generate.prompts {
            PromptSource::Interactive { initial: Some(p) } if p.trim().is_empty() => return Err("initial prompt must be nonempty or absent".into()),
            PromptSource::Script { file, chains } if file.as_os_str().is_empty() || chains.iter().any(|c| c.is_empty()) => return Err("invalid prompt script".into()),
            _ => {}
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
        assert_eq!(Config::parse(&value.to_string()).unwrap().export.kernels.linear, "torch_bf16");
    }
    #[test]
    fn dense_attention_rejects_quantized_cache_combinations() {
        let mut value = repository().value();
        value[0]["kernels"]["attention"] = serde_json::json!("torch_sdpa");
        value[0]["kernels"]["rope"] = serde_json::json!("torch_inductor");
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
    fn repository() -> Config { Config::parse(include_str!("../../../configs/live.json")).unwrap() }
    #[test]
    fn debug_selects_only_known_tools_and_drain_accepts_intervals_or_end() {
        use serde_json::json;
        for debug in [json!(false), json!(true), json!("fps"), json!(["fps"])] {
            for every in ["5s", "10s", "0.5s", "end"] {
                let mut value = repository().value();
                value[1]["debug"] = debug.clone();
                value[1]["drain"] = json!({"every": every});
                let parsed = Config::parse(&value.to_string()).expect("valid debug/drain config refused");
                assert_eq!(parsed.value()[1]["drain"], value[1]["drain"]);
            }
        }
        let mut value = repository().value();
        value[1].as_object_mut().unwrap().remove("debug");
        value[1].as_object_mut().unwrap().remove("drain");
        assert!(Config::parse(&value.to_string()).is_ok());
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
        value[0]["kernels"]["attention"] = serde_json::json!("sage_sm89");
        for backend in ["torch_inductor", "triton"] {
            value[0]["kernels"]["rope"] = serde_json::json!(backend);
            let config = Config::parse(&value.to_string()).expect("supported RoPE refused");
            assert_eq!(config.export.kernels.rope.as_deref(), Some(backend));
        }
    }
    #[test]
    fn joint_sage_rope_replaces_the_independent_rope_choice() {
        let mut value = repository().value();
        value[0]["kernels"]["attention"] = serde_json::json!("sage_rope_sm89");
        value[0]["kernels"].as_object_mut().unwrap().remove("rope");
        let config = Config::parse(&value.to_string()).expect("joint backend refused");
        assert!(config.export.kernels.rope.is_none());
        assert!(config.value()[0]["kernels"].get("rope").is_none());
        value[0]["kernels"]["rope"] = serde_json::json!("triton");
        assert!(Config::parse(&value.to_string()).is_err());
        value[0]["kernels"]["rope"] = serde_json::Value::Null;
        assert!(Config::parse(&value.to_string()).is_err());
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
        let live=repository();
        let offline=Config::parse(include_str!("../../../configs/offline.json")).unwrap();
        assert_eq!(live.export,offline.export);
        assert_eq!(live.generate.offloading,offline.generate.offloading);
        assert_eq!(offline.generate.mode,Mode::Offline);
        assert!(matches!(offline.generate.prompts,PromptSource::Script {..}));
    }
    #[test]
    fn prompt_sources_are_independent_of_output_and_have_no_hidden_initial_prompt() {
        let mut value=repository().value();
        value[1]["prompts"] = serde_json::json!({"source":"interactive"});
        assert_eq!(Config::parse(&value.to_string()).unwrap().generate.prompts.initial(),None);
        value[1]["prompts"] = serde_json::json!({"source":"script","file":"a.json"});
        assert!(Config::parse(&value.to_string()).is_ok());
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
        let p = Placement { key: "k1".into(), weights: Tier::Disk, rows_streamed: 30, kv: Tier::Ram };
        assert_eq!(Placement::read(&dir, "k1"), None);
        p.write(&dir).unwrap();
        assert_eq!(Placement::read(&dir, "k1"), Some(p));
        assert_eq!(Placement::read(&dir, "k2"), None);
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
