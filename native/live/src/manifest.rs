//! live.json, written by scripts/conversion/export_live.py beside the packages it exports.

use std::path::{Path, PathBuf};

use serde::Deserialize;

#[derive(Deserialize, Debug, Clone)]
pub struct Manifest {
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
}

impl Manifest {
    pub fn read(directory: &Path) -> std::io::Result<Manifest> {
        let text = std::fs::read_to_string(directory.join("live.json"))?;
        let m: Manifest = serde_json::from_str(&text).map_err(std::io::Error::other)?;
        // sizes that follow from one another: a manifest written wrong fails here, not as a CUDA error later
        let consistent = m.chunk_tokens == m.chunk * m.tokens_per_frame
            && m.latent[1] == m.chunk
            && m.tokens_per_frame == (m.latent[2] / 2) * (m.latent[3] / 2)
            && m.max_tokens % m.chunk_tokens == 0
            && m.block_steps == m.max_tokens / m.chunk_tokens + 1
            && m.timesteps.len() == m.sampler.len();
        if !consistent {
            return Err(std::io::Error::other(format!("{}: sizes that do not agree", directory.join("live.json").display())));
        }
        Ok(m)
    }
}
