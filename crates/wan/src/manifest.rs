//! live.json, written by scripts/conversion/export_live.py beside the packages it exports.

use std::path::{Path, PathBuf};

use serde::Deserialize;
use crate::config::Kernels;

#[derive(Deserialize, Debug, Clone)]
pub struct Manifest {
    pub kernels: Kernels,
    pub height: usize,
    pub width: usize,
    pub fps: f64,
    pub chunk: usize,
    /// [channels, chunk, latent height, latent width]
    pub latent: [usize; 4],
    pub tokens_per_frame: usize,
    pub chunk_tokens: usize,
    pub max_tokens: usize,
    pub sink_tokens: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub text_len: usize,
    pub text_dim: usize,
    pub blocks: usize,
    pub block_steps: usize,
    pub positions: usize,
    pub time_pairs: usize,
    pub shot_rope_offset: usize,
    pub scene_prefix: String,
    pub pack: PathBuf,
    pub row_bytes: usize,
    pub stream_blocks: usize,
    pub timesteps: Vec<f32>,
    /// x_{k+1} = x_k - sampler[k] flow_k
    pub sampler: Vec<f32>,
    pub decoder_memory: Vec<Vec<i64>>,
    pub decoder_trim: usize,
    pub t5: Option<T5>,
    #[serde(default)]
    pub generation_workspace_bytes: usize,
}

#[derive(Deserialize, Debug, Clone)]
pub struct T5 {
    pub tokenizer: PathBuf,
    pub rows: PathBuf,
    pub row_bytes: usize,
    pub blocks: usize,
    /// the token embedding table [vocabulary, text_dim] bfloat16, read in place from the checkpoint
    pub embedding: PathBuf,
    pub embedding_offset: u64,
    pub vocabulary: usize,
    #[serde(default)]
    pub activation_bytes: usize,
}

impl Manifest {
    pub fn read(directory: &Path) -> std::io::Result<Manifest> {
        if directory.join(".exporting").try_exists()? {
            return Err(std::io::Error::other("unfinished package export: complete a full export before starting"));
        }
        let text = std::fs::read_to_string(directory.join("live.json"))?;
        let m: Manifest = serde_json::from_str(&text).map_err(std::io::Error::other)?;
        m.kernels.validate().map_err(std::io::Error::other)?;
        // sizes that follow from one another: a manifest written wrong fails here, not as a CUDA error later
        let consistent = m.chunk > 0 && m.chunk_tokens > 0 && m.tokens_per_frame > 0
            && m.blocks > 0 && m.heads > 0 && m.head_dim > 0 && m.row_bytes > 0
            && m.fps.is_finite() && m.fps > 0.0 && m.latent.iter().all(|&n| n > 0)
            && m.chunk.checked_mul(m.tokens_per_frame) == Some(m.chunk_tokens)
            && m.latent[1] == m.chunk
            && (m.latent[2] / 2).checked_mul(m.latent[3] / 2) == Some(m.tokens_per_frame)
            && m.max_tokens % m.chunk_tokens == 0
            && (m.max_tokens / m.chunk_tokens).checked_add(1) == Some(m.block_steps)
            && !m.timesteps.is_empty() && m.timesteps.len() == m.sampler.len();
        if !consistent {
            return Err(std::io::Error::other(format!("{}: sizes that do not agree", directory.join("live.json").display())));
        }
        Ok(m)
    }

    /// Check the package contract before LibTorch, operator DLLs or the GPU are touched.
    pub fn read_for(directory: &Path, requested: &Kernels) -> std::io::Result<Manifest> {
        let m = Self::read(directory).map_err(|e| std::io::Error::other(format!(
            "{}: {e}; export packages with export_live.py --config <run config>", directory.join("live.json").display())))?;
        if &m.kernels != requested {
            return Err(std::io::Error::other(format!(
                "{}: kernels do not match the run config: requested {}, packages {}; export packages with the same --config",
                directory.join("live.json").display(), serde_json::to_string(requested)?, serde_json::to_string(&m.kernels)?)));
        }
        Ok(m)
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::config::Config;

    pub fn fixture() -> serde_json::Value {
        let kernels = serde_json::from_str::<serde_json::Value>(include_str!("../../../configs/live.json")).unwrap()[0]["kernels"].clone();
        serde_json::json!({
            "kernels": kernels, "height": 32, "width": 32, "fps": 24.0, "chunk": 8,
            "latent": [48, 8, 2, 2], "tokens_per_frame": 1, "chunk_tokens": 8,
            "max_tokens": 24, "sink_tokens": 8, "heads": 24, "head_dim": 128,
            "text_len": 512, "text_dim": 4096, "blocks": 30, "block_steps": 4,
            "positions": 65536, "time_pairs": 22, "shot_rope_offset": 8,
            "scene_prefix": "The scene transitions. ", "pack": "weights.w4a4",
            "row_bytes": 128, "stream_blocks": 24, "timesteps": [999.0, 833.0],
            "sampler": [0.1, 0.9], "decoder_memory": [], "decoder_trim": 3
        })
    }

    #[test]
    fn a_manifest_without_kernel_provenance_is_refused() {
        let mut value = fixture();
        value.as_object_mut().unwrap().remove("kernels");
        assert!(serde_json::from_value::<Manifest>(value).is_err());
    }

    #[test]
    fn zero_token_dimensions_are_reported_before_gpu_initialization() {
        let dir = std::env::temp_dir().join(format!("wan-zero-manifest-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut value = fixture();
        value["chunk_tokens"] = serde_json::json!(0);
        value["chunk"] = serde_json::json!(0);
        value["latent"][1] = serde_json::json!(0);
        std::fs::write(dir.join("live.json"), value.to_string()).unwrap();
        let result = std::panic::catch_unwind(|| Manifest::read(&dir));
        assert!(result.is_ok(), "malformed metadata panicked instead of returning an error");
        assert!(result.unwrap().is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn an_interrupted_export_cannot_use_the_previous_manifest() {
        let dir = std::env::temp_dir().join(format!("wan-kernel-interrupted-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("live.json"), fixture().to_string()).unwrap();
        std::fs::write(dir.join(".exporting"), "{}").unwrap();
        let error = Manifest::read(&dir).unwrap_err().to_string();
        assert!(error.contains("export"), "{error}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn only_packages_with_the_requested_kernels_can_be_loaded() {
        let config = Config::read(&crate::root().join("configs/live.json")).unwrap();
        let dir = std::env::temp_dir().join(format!("wan-kernel-manifest-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut value = fixture();
        std::fs::write(dir.join("live.json"), value.to_string()).unwrap();
        assert!(Manifest::read_for(&dir, &config.export.kernels).is_ok());
        value["kernels"]["cross_attention"] = serde_json::json!(["efficient"]);
        std::fs::write(dir.join("live.json"), value.to_string()).unwrap();
        let error = Manifest::read_for(&dir, &config.export.kernels).unwrap_err().to_string();
        assert!(error.contains("cross_attention") && error.contains("export"), "{error}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
