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
        done += got;
        if got == 0 || got % SECTOR != 0 {
            break; // the end of the file: its last sector was not whole, and a read after it would be unaligned
        }
    }
    if done < start - low + row_bytes {
        return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof,
            format!("weight row {index}: needed {} bytes from offset {low}, read {done}", start - low + row_bytes)));
    }
    Ok(start - low)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncated_row_is_an_error_instead_of_stale_buffer_contents() {
        let path = std::env::temp_dir().join(format!("wan-disk-truncated-{}.bin", std::process::id()));
        std::fs::write(&path, vec![7u8; 6000]).unwrap();
        let mut storage = vec![0xAA; span(5000) + SECTOR];
        let skip = storage.as_ptr().align_offset(SECTOR);
        let result = read_row(&open_unbuffered(&path).unwrap(), 1, 5000, &mut storage[skip..]);
        std::fs::remove_file(path).unwrap();
        assert_eq!(result.unwrap_err().kind(), std::io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn the_last_row_reads_when_the_file_ends_inside_a_sector() {
        // 30 rows of the FP8 pack end 4912071360 bytes in, not on a sector: an unbuffered read must not go on at
        // an unaligned offset after the short read of the file's last sector
        let row_bytes = 5000;
        let data: Vec<u8> = (0..3 * row_bytes).map(|i| (i % 251) as u8).collect();
        let path = std::env::temp_dir().join(format!("wan-disk-test-{}.bin", std::process::id()));
        std::fs::write(&path, &data).unwrap();
        let mut storage = vec![0u8; span(row_bytes) + SECTOR];
        let skip = storage.as_ptr().align_offset(SECTOR);
        let buffer = &mut storage[skip..skip + span(row_bytes)];
        let read = read_row(&open_unbuffered(&path).unwrap(), 2, row_bytes, buffer);
        std::fs::remove_file(&path).unwrap();
        let at = read.unwrap();
        assert_eq!(&buffer[at..at + row_bytes], &data[2 * row_bytes..]);
    }
}
