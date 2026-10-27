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

use crate::gpu::{DeviceBuffer, Event, PinnedBuffer, Stream};
use crate::manifest::T5 as Spec;
use crate::packages::{self, Loaded};
use crate::video::Result;

const SECTOR: usize = 4096;
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
}

/// The prompt as HuggingfaceTokenizer(clean="whitespace") cleans it: whitespace collapsed, ends stripped.
pub fn clean(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(windows)]
fn open_unbuffered(path: &Path) -> std::io::Result<File> {
    use std::os::windows::fs::OpenOptionsExt;
    const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
    const FILE_FLAG_SEQUENTIAL_SCAN: u32 = 0x0800_0000;
    std::fs::OpenOptions::new().read(true).custom_flags(FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN).open(path)
}

#[cfg(not(windows))]
fn open_unbuffered(path: &Path) -> std::io::Result<File> {
    File::open(path)
}

/// Reads row `index` of `row_bytes` of `file` into `buffer` (sector-aligned reads of whole sectors around it);
/// returns where the row starts in the buffer.
fn read_row(file: &File, index: usize, row_bytes: usize, buffer: &mut [u8]) -> std::io::Result<usize> {
    #[cfg(windows)]
    use std::os::windows::fs::FileExt;
    let start = index * row_bytes;
    let low = start / SECTOR * SECTOR;
    let size = (start + row_bytes - low).div_ceil(SECTOR) * SECTOR;
    let mut done = 0;
    while done < size {
        #[cfg(windows)]
        let got = file.seek_read(&mut buffer[done..size], (low + done) as u64)?;
        #[cfg(not(windows))]
        let got = std::os::unix::fs::FileExt::read_at(file, &mut buffer[done..size], (low + done) as u64)?;
        if got == 0 {
            break; // the end of the file: its last sector was not whole
        }
        done += got;
    }
    Ok(start - low)
}

impl T5 {
    pub fn load(spec: &Spec, text_len: usize, dim: usize, runtime: &Runtime, directory: &Path, stream: &Stream)
        -> Result<T5> {
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
        })
    }

    /// The ids of the prompt's tokens (with the end token), at most text_len - 1 (the packages take no more).
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

        let row_bytes = self.spec.row_bytes;
        let span = (row_bytes + 2 * SECTOR).div_ceil(SECTOR) * SECTOR;
        let file = open_unbuffered(&self.spec.rows)?;
        let copy = Stream::new()?;
        let mut staging = [PinnedBuffer::new(span)?, PinnedBuffer::new(span)?];
        let slots = [DeviceBuffer::new(row_bytes)?, DeviceBuffer::new(row_bytes)?];
        let copied = [Event::new()?, Event::new()?];
        let freed = [Event::new()?, Event::new()?];
        let uploaded = [Event::new()?, Event::new()?];
        let mut kept_output: Option<Output> = None;
        for b in 0..self.spec.blocks {
            let s = b % 2;
            if b >= 2 {
                uploaded[s].synchronize()?; // the staging buffer's last row is on the GPU
            }
            let at = read_row(&file, b, row_bytes, staging[s].as_mut_slice())?;
            copy.wait(&freed[s])?;
            copy.copy_to_device(slots[s].ptr, &staging[s].as_slice()[at..at + row_bytes])?;
            uploaded[s].record(&copy)?;
            copied[s].record(&copy)?;
            stream.wait(&copied[s])?;
            let row = slots[s].tensor(0, DType::U8, &[row_bytes as i64]);
            let next = self.block.run(&[row, hidden, mask_tensor], stream.raw())?.remove(0);
            freed[s].record(stream)?;
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
    use super::clean;

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
