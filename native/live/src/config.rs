//! configs/live.json: what a run of Live is, beside the packages it runs; flags of the command line override it.
//!
//!     {"size": "480x864", "seed": 0,
//!      "offloading": {"enabled": true, "calibrate": true,
//!                     "resources": {"block_weights": "auto", "kv_cache": "auto"},
//!                     "copy_spare": 1.1, "vram_margin_gib": 1.5, "t5_activations_gib": 0.33}}
//!
//! The rule: as little memory as there can be at the most speed. Every resource of the model has a tier: "vram" (on
//! the GPU for good), "ram" (pinned RAM, the part a block needs copied to the GPU ahead of use), "disk" (read from its
//! file without the page cache into a pinned buffer, then copied) or "auto": the calibration measures a pass and puts
//! the resource on the lowest tier whose copies hide behind compute. T5's weights are always on the disk tier (they are
//! read from their file only while a prompt is encoded).

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Deserialize, Debug, Clone, PartialEq)]
#[serde(default)]
pub struct Config {
    /// the packages of artifacts/live/<size>
    pub size: String,
    pub prompt: String,
    pub seed: u64,
    pub offloading: Offloading,
}

/// The resources of the model that can be offloaded.
#[derive(Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum Resource {
    /// the W4A4 rows of the 30 blocks
    BlockWeights,
    /// the FP8 keys and values of every block's attention over the frames already made
    KvCache,
}

impl Resource {
    pub const ALL: [Resource; 2] = [Resource::BlockWeights, Resource::KvCache];

    /// The tiers a resource can be on: the KV cache is written every chunk, the disk would be too slow for it.
    pub fn tiers(self) -> &'static [Tier] {
        match self {
            Resource::BlockWeights => &[Tier::Disk, Tier::Ram, Tier::Vram],
            Resource::KvCache => &[Tier::Ram, Tier::Vram],
        }
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

#[derive(Deserialize, Debug, Clone, PartialEq)]
#[serde(default)]
pub struct Offloading {
    /// off: every resource on the GPU for good
    pub enabled: bool,
    /// on: "auto" resources are placed by the calibration (Video::calibrate); off: "auto" is "ram"
    pub calibrate: bool,
    pub resources: BTreeMap<Resource, Tier>,
    /// a copy must take no more than its share of compute divided by this
    pub copy_spare: f32,
    /// GiB of the GPU the calibration leaves to other programs when it keeps rows there
    pub vram_margin_gib: f64,
    /// GiB T5's activations take on the GPU while a prompt is encoded, beside its two row slots
    pub t5_activations_gib: f64,
}

impl Default for Config {
    fn default() -> Config {
        Config {
            size: "480x864".into(),
            prompt: "A lone oak tree on a grassy hill sways in the wind".into(),
            seed: 0,
            offloading: Offloading::default(),
        }
    }
}

impl Default for Offloading {
    fn default() -> Offloading {
        Offloading {
            enabled: true,
            calibrate: true,
            resources: Resource::ALL.iter().map(|&r| (r, Tier::Auto)).collect(),
            copy_spare: 1.1,
            vram_margin_gib: 1.5,
            t5_activations_gib: 0.33,
        }
    }
}

impl Offloading {
    /// The tier `resource` is configured to (a resource the file leaves out is "auto").
    pub fn tier(&self, resource: Resource) -> Tier {
        if !self.enabled {
            return Tier::Vram;
        }
        self.resources.get(&resource).copied().unwrap_or(Tier::Auto)
    }

    /// Whether the calibration places `resource`.
    pub fn calibrates(&self, resource: Resource) -> bool {
        self.calibrate && self.tier(resource) == Tier::Auto
    }

    /// The tier `resource` starts on: an "auto" resource starts on the lowest tier it can be on, and the calibration
    /// brings it up as far as its copies need (without calibration "auto" is "ram").
    pub fn initial(&self, resource: Resource) -> Tier {
        match self.tier(resource) {
            Tier::Auto if self.calibrate => resource.tiers()[0],
            Tier::Auto => Tier::Ram,
            tier => tier,
        }
    }

    /// Checks that every tier is one the resource can be on.
    pub fn validate(&self) -> Result<(), String> {
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
        format!("{offloading:?}").hash(&mut h);
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
    /// The configuration in `path`, the defaults for what it leaves out or if there is no file.
    pub fn read(path: &Path) -> std::io::Result<Config> {
        let config = match std::fs::read_to_string(path) {
            Ok(text) => Config::parse(&text).map_err(|e| std::io::Error::other(format!("{}: {e}", path.display())))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Config::default(),
            Err(e) => return Err(e),
        };
        config.offloading.validate().map_err(|e| std::io::Error::other(format!("{}: {e}", path.display())))?;
        Ok(config)
    }

    pub fn parse(text: &str) -> serde_json::Result<Config> {
        serde_json::from_str(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repository() -> Config {
        Config::read(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/live.json")).unwrap()
    }

    #[test]
    fn every_resource_of_the_model_is_in_the_repository_config() {
        // a resource added to the model and forgotten in configs/live.json fails here, rather than staying on the GPU
        let c = repository();
        for r in Resource::ALL {
            assert!(c.offloading.resources.contains_key(&r), "{r:?} is not in configs/live.json");
        }
    }

    #[test]
    fn offloading_and_its_calibration_are_on_by_default_for_every_resource() {
        let c = repository();
        assert!(c.offloading.enabled && c.offloading.calibrate);
        for r in Resource::ALL {
            assert_eq!(c.offloading.tier(r), Tier::Auto, "{r:?}");
            assert!(c.offloading.calibrates(r));
        }
        assert_eq!(Config::parse("{}").unwrap(), Config::default());
    }

    #[test]
    fn nothing_is_put_on_the_gpu_silently() {
        // an "auto" resource starts offloaded, and so does one the file leaves out
        let c = Config::parse(r#"{"offloading": {"resources": {"kv_cache": "ram"}}}"#).unwrap();
        for r in Resource::ALL {
            assert_ne!(c.offloading.initial(r), Tier::Vram, "{r:?}");
        }
    }

    #[test]
    fn an_auto_resource_starts_on_its_lowest_tier() {
        // the calibration measures from there and brings it up only as far as its copies need; pinned RAM is
        // limited (on Windows about half of the RAM), so the weights start on the disk and leave it to the cache
        let c = Config::default();
        assert_eq!(c.offloading.initial(Resource::BlockWeights), Tier::Disk);
        assert_eq!(c.offloading.initial(Resource::KvCache), Tier::Ram);
        let c = Config::parse(r#"{"offloading": {"calibrate": false}}"#).unwrap();
        assert_eq!(c.offloading.initial(Resource::BlockWeights), Tier::Ram);
    }

    #[test]
    fn offloading_off_keeps_every_resource_on_the_gpu() {
        let c = Config::parse(r#"{"offloading": {"enabled": false}}"#).unwrap();
        for r in Resource::ALL {
            assert_eq!(c.offloading.initial(r), Tier::Vram);
        }
    }

    #[test]
    fn a_configured_tier_is_kept() {
        let c = Config::parse(r#"{"offloading": {"resources": {"block_weights": "disk", "kv_cache": "vram"}}}"#).unwrap();
        assert_eq!(c.offloading.initial(Resource::BlockWeights), Tier::Disk);
        assert_eq!(c.offloading.initial(Resource::KvCache), Tier::Vram);
        assert!(!c.offloading.calibrates(Resource::KvCache));
    }

    #[test]
    fn a_tier_a_resource_cannot_be_on_is_refused() {
        let c = Config::parse(r#"{"offloading": {"resources": {"kv_cache": "disk"}}}"#).unwrap();
        assert!(c.offloading.validate().is_err());
    }

    #[test]
    fn a_missing_file_is_the_defaults() {
        assert_eq!(Config::read(Path::new("Z:/nowhere/live.json")).unwrap(), Config::default());
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
