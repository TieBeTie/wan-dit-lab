//! umT5-XXL of a prompt, as FastTextEncoder (models/text_encoder.py) computes it: the tokens of the prompt, their rows
//! of the token embedding (read in place from Wan's checkpoint), the 24 INT8 blocks from FastTextEncoder's file read
//! row by row without the page cache (FILE_FLAG_NO_BUFFERING, RowFile) into two pinned buffers and two GPU slots,
//! then the last norm, over the prompt's tokens only (their number is a dynamic size of the packages). The output is
//! [1, text_len, 4096] float32 on the GPU, zeros past the prompt's tokens. The
//! slots and buffers exist only while a prompt is encoded; the embeddings of the last prompts are kept.

use std::collections::VecDeque;
use std::fs::File;
use std::path::Path;

use aoti::{DType, Output, Runtime, Tensor};
use tokenizers::Tokenizer;

use crate::config::T5Weights;
use crate::rows::Rows;
use crate::gpu::{DeviceBuffer, Stream};
use crate::manifest::T5 as Spec;
use crate::packages::{self, Loaded};
use crate::video::Result;

const KEEP: usize = 8; // embeddings of the last prompts kept (FastTextEncoder keeps 16)

pub struct T5 {
    spec: Spec,
    text_len: usize,
    dim: usize,
    tokenizer: Tokenizer,
    embedding: memmap2::Mmap,
    block: Loaded,
    last: Loaded,
    kept: VecDeque<(String, Vec<f32>)>,
    /// its blocks' rows: the first on the GPU, the others streamed from the disk, parked between prompts
    rows: Rows,
}

/// The prompt as HuggingfaceTokenizer(clean="whitespace") cleans it: whitespace collapsed, ends stripped.
pub fn clean(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

impl T5 {
    pub fn load(spec: &Spec, text_len: usize, dim: usize, runtime: &Runtime, directory: &Path, stream: &Stream,
                weights: &T5Weights) -> Result<T5> {
        let resident = weights.resident(spec.row_bytes, spec.blocks);
        let streamed: Vec<usize> = (resident..spec.blocks).collect();
        let mut rows = Rows::load(&spec.rows, spec.blocks, spec.row_bytes, &streamed, true, stream)?;
        rows.park(stream)?;
        let tokenizer = Tokenizer::from_file(&spec.tokenizer).map_err(|e| format!("{}: {e}", spec.tokenizer.display()))?;
        let file = File::open(&spec.embedding)?;
        let embedding = unsafe { memmap2::Mmap::map(&file)? };
        assert!(embedding.len() as u64 >= spec.embedding_offset + (spec.vocabulary * dim * 2) as u64);
        Ok(T5 {
            spec: spec.clone(),
            text_len,
            dim,
            tokenizer,
            embedding,
            block: packages::load(runtime, directory, "t5_block", stream)?,
            last: packages::load(runtime, directory, "t5_final", stream)?,
            kept: VecDeque::new(),
            rows,
        })
    }

    /// The ids of the prompt's tokens (with the end token), at most text_len - 1 (the packages take no more).
    /// debug-tools: the waits for its rows' copies from the disk that the GPU has passed since the last call.
    #[cfg(feature = "debug-tools")]
    pub fn copy_waits(&self) -> Result<crate::rows::CopyWaits> {
        Ok(self.rows.passed_copy_waits()?)
    }

    pub fn tokens(&self, prompt: &str) -> Result<Vec<u32>> {
        let encoding = self.tokenizer.encode(clean(prompt), true).map_err(|e| e.to_string())?;
        let mut ids = encoding.get_ids().to_vec();
        ids.truncate(self.text_len - 1);
        Ok(ids)
    }

    /// The embeddings of `prompt` [1, text_len, dim] float32, into `out` on the GPU.
    pub fn encode(&mut self, prompt: &str, out: &DeviceBuffer, stream: &Stream) -> Result<()> {
        let floats = self.text_len * self.dim;
        assert_eq!(out.bytes, floats * 4);
        if let Some(i) = self.kept.iter().position(|(p, _)| p == prompt) {
            let (p, values) = self.kept.remove(i).unwrap();
            stream.copy_to_device(out.ptr, bytes(&values))?;
            stream.synchronize()?;
            self.kept.push_back((p, values));
            return Ok(());
        }
        let ids = self.tokens(prompt)?;
        let (n, dim) = (ids.len(), self.dim);
        // the rows of the embedding, bfloat16
        let table = &self.embedding[self.spec.embedding_offset as usize..];
        let mut x = vec![0u8; n * dim * 2];
        for (i, &id) in ids.iter().enumerate() {
            let row = id as usize * dim * 2;
            x[i * dim * 2..(i + 1) * dim * 2].copy_from_slice(&table[row..row + dim * 2]);
        }
        let mask = vec![1i64; n];
        let x_device = DeviceBuffer::new(x.len())?;
        stream.copy_to_device(x_device.ptr, &x)?;
        let mask_device = DeviceBuffer::new(n * 8)?;
        stream.copy_to_device(mask_device.ptr, bytes(&mask))?;
        let mask_tensor = mask_device.tensor(0, DType::I64, &[1, n as i64]);
        let mut hidden = x_device.tensor(0, DType::BF16, &[1, n as i64, dim as i64]);

        // the blocks through the rows, as the DiT's: each streamed one copied while the block before it computes
        self.rows.unpark()?;
        let mut kept_output: Option<Output> = None;
        for b in 0..self.spec.blocks {
            let row = self.rows.row(b, stream)?;
            let next = self.block.run(&[row, hidden, mask_tensor], stream.raw())?.remove(0);
            self.rows.release(b, stream)?;
            hidden = *next;
            kept_output = Some(next);
        }
        let embeddings = self.last.run(&[hidden, mask_tensor], stream.raw())?.remove(0);
        unsafe { cudarc::driver::result::memset_d8_async(out.ptr, 0, floats * 4, stream.0)? };
        stream.copy_on_device(out.ptr, embeddings.data as u64, n * dim * 4)?;
        let mut values = vec![0f32; floats];
        stream.copy_to_host(bytes_mut(&mut values), out.ptr)?;
        stream.synchronize()?;
        drop(kept_output);
        self.rows.park(stream)?;
        if self.kept.len() >= KEEP {
            self.kept.pop_front();
        }
        self.kept.push_back((prompt.to_owned(), values));
        Ok(())
    }

    pub fn shape(&self) -> [i64; 3] {
        [1, self.text_len as i64, self.dim as i64]
    }

    pub fn tensor(&self, out: &DeviceBuffer) -> Tensor {
        out.tensor(0, DType::F32, &self.shape())
    }
}

fn bytes<T>(values: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values)) }
}

fn bytes_mut<T>(values: &mut [T]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, std::mem::size_of_val(values)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T5 through the streamed rows gives the embeddings of FastTextEncoder as before (2-3%, the same prompts as the
    /// reference of live check), twice the same, and between prompts only the rows of its budget stay on the GPU.
    /// Needs the GPU and artifacts/live/480x864: cargo test -- --ignored
    #[test]
    #[ignore]
    fn t5_through_the_rows_gives_the_embeddings_of_fast_text_encoder() {
        static GPU: std::sync::OnceLock<crate::gpu::Gpu> = std::sync::OnceLock::new();
        let operators: Vec<std::path::PathBuf> = ["live_w4a4", "live_sage"].iter()
            .map(|n| crate::root().join("native").join("ops").join("build").join(format!("{n}.dll"))).collect();
        let runtime = aoti::Runtime::load(&aoti::Runtime::default_torch_lib(), &operators).unwrap();
        GPU.get_or_init(|| crate::gpu::Gpu::new().unwrap()).bind().unwrap();
        let directory = crate::root().join("artifacts").join("live").join("480x864");
        let m = crate::manifest::Manifest::read(&directory).unwrap();
        let stream = Stream::new().unwrap();
        let free = || cudarc::driver::result::mem_get_info().unwrap().0;
        let before = free();
        let mut t5 = T5::load(m.t5.as_ref().unwrap(), m.text_len, m.text_dim, &runtime, &directory, &stream,
                              &T5Weights::default()).unwrap();
        let taken = before.saturating_sub(free());
        let row = m.t5.as_ref().unwrap().row_bytes;
        println!("T5 between prompts on the GPU: {:.0} MB (a row is {:.0} MB)", taken as f64 / 1e6, row as f64 / 1e6);
        let budget = T5Weights::default();
        let resident = budget.resident(row, m.t5.as_ref().unwrap().blocks);
        assert!(resident > 0);
        assert!(taken >= resident * row && taken < resident * row + row,
                "{taken} bytes of T5 on the GPU between prompts, {resident} rows of {row} in its budget");
        let file = std::fs::File::open(directory.join("reference.safetensors")).unwrap();
        let map = unsafe { memmap2::Mmap::map(&file).unwrap() };
        let r = safetensors::SafeTensors::deserialize(&map).unwrap();
        let (_, meta) = safetensors::SafeTensors::read_metadata(&map).unwrap();
        let prompts: std::collections::HashMap<String, String> =
            serde_json::from_str(&meta.metadata().as_ref().unwrap()["prompts"]).unwrap();
        let out = DeviceBuffer::new(m.text_len * m.text_dim * 4).unwrap();
        for name in ["A", "C"] {
            let mut got = Vec::new();
            for _ in 0..2 {
                t5.kept.clear(); // computed each time, not taken from the kept embeddings
                t5.encode(&prompts[name], &out, &stream).unwrap();
                let mut host = vec![0u8; out.bytes];
                stream.copy_to_host(&mut host, out.ptr).unwrap();
                stream.synchronize().unwrap();
                got.push(host);
            }
            assert!(got[0] == got[1], "{name}: the same prompt twice, other embeddings");
            let floats = |b: &[u8]| b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<_>>();
            let (a, b) = (floats(&got[0]), floats(r.tensor(&format!("embedding_{name}")).unwrap().data()));
            let d: f64 = a.iter().zip(&b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
            let n: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
            println!("{name}: relative L2 to FastTextEncoder {:.2e}", (d / n).sqrt());
            assert!((d / n).sqrt() < 0.05, "{name}");
        }
        assert!(free() + 2 * row >= before - taken, "T5's slots are given back between prompts");
    }

    #[test]
    fn whitespace_as_the_reference_cleans_it() {
        assert_eq!(clean("  a fox\n\t walks  "), "a fox walks");
    }

    /// The ids of HuggingfaceTokenizer(clean="whitespace") of Wan's reference for the same prompts.
    #[test]
    fn tokens_as_the_reference_gives_them() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../checkpoints/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(path).unwrap();
        let ids = |p: &str| tokenizer.encode(clean(p), true).unwrap().get_ids().to_vec();
        assert_eq!(ids("A lone oak tree on a grassy hill sways in the wind"),
                   [320, 273, 34074, 316, 1263, 37703, 369, 289, 70279, 313, 71169, 382, 50386, 301, 312, 15854, 1]);
        assert_eq!(ids("The scene transitions. Snow starts to fall on the oak tree"),
                   [517, 26866, 44295, 281, 274, 27726, 62241, 315, 5923, 369, 312, 316, 1263, 37703, 1]);
        assert_eq!(ids("  Лиса  в снегу, ночь\n"), [122680, 351, 327, 74766, 407, 275, 81334, 1]);
    }
}
