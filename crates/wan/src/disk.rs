//! Rows of a file read without the page cache (FILE_FLAG_NO_BUFFERING on Windows, RowFile of offload.py): the file
//! takes no memory beyond the buffer it is read into, and sequential reads of whole rows keep the NVMe busy. T5's
//! blocks and the weights on the disk tier are read so.

use std::fs::File;
use std::path::Path;

pub const SECTOR: usize = 4096;

/// Bytes a buffer needs to read a row of `row_bytes` at any offset: the sectors around it.
pub fn span(row_bytes: usize) -> usize {
    (row_bytes + 2 * SECTOR).div_ceil(SECTOR) * SECTOR
}

#[cfg(windows)]
pub fn open_unbuffered(path: &Path) -> std::io::Result<File> {
    use std::os::windows::fs::OpenOptionsExt;
    const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
    const FILE_FLAG_SEQUENTIAL_SCAN: u32 = 0x0800_0000;
    std::fs::OpenOptions::new().read(true).custom_flags(FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN).open(path)
}

#[cfg(not(windows))]
pub fn open_unbuffered(path: &Path) -> std::io::Result<File> {
    File::open(path)
}

/// Reads row `index` of `row_bytes` of `file` into `buffer` (sector-aligned reads of whole sectors around it);
/// returns where the row starts in the buffer.
pub fn read_row(file: &File, index: usize, row_bytes: usize, buffer: &mut [u8]) -> std::io::Result<usize> {
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

