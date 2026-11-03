//! Temporary mutable KV backing, separate from output artifacts.
use std::fs::{File, OpenOptions};
use std::io::{Read, Write, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
static SERIAL: AtomicU64 = AtomicU64::new(0);
pub struct Store {
    file: Option<File>,
    path: PathBuf,
    len: usize,
    pub written: u64,
    pub read_bytes: u64,
}
impl Store {
    pub fn new(len: usize) -> std::io::Result<Self> {
        let id = SERIAL.fetch_add(1, Ordering::Relaxed);
        let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let path = std::env::temp_dir().join(format!("wan-kv-{}-{stamp}-{id}.bin", std::process::id()));
        let mut options = OpenOptions::new();
        options.read(true).write(true).create_new(true);
        #[cfg(windows)] {
            use std::os::windows::fs::OpenOptionsExt;
            // Delete on close also cleans up if the process is terminated.
            options.custom_flags(0x0400_0000);
        }
        let file = options.open(&path)?;
        let store = Self { file: Some(file), path, len, written: 0, read_bytes: 0 };
        store.file.as_ref().unwrap().set_len(len as u64)?;
        Ok(store)
    }
    fn range(&self, at: usize, len: usize) -> std::io::Result<()> {
        if at.checked_add(len).is_some_and(|end| end <= self.len) { Ok(()) }
        else { Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "KV disk range exceeds backing size")) }
    }
    pub fn read(&mut self, at: usize, bytes: &mut [u8]) -> std::io::Result<()> {
        self.range(at, bytes.len())?;
        let file = self.file.as_mut().unwrap();
        file.seek(SeekFrom::Start(at as u64))?;
        file.read_exact(bytes)?;
        self.read_bytes += bytes.len() as u64;
        Ok(())
    }
    pub fn write(&mut self, at: usize, bytes: &[u8]) -> std::io::Result<()> {
        self.range(at, bytes.len())?;
        let file = self.file.as_mut().unwrap();
        file.seek(SeekFrom::Start(at as u64))?;
        file.write_all(bytes)?;
        self.written += bytes.len() as u64;
        Ok(())
    }
}
impl Drop for Store {
    fn drop(&mut self) { drop(self.file.take()); let _ = std::fs::remove_file(&self.path); }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dirty_ranges_preserve_other_blocks_and_file_is_removed() {
        let mut store = Store::new(64).unwrap();
        let path = store.path.clone();
        store.write(8, &[1, 2, 3]).unwrap();
        store.write(32, &[9, 8]).unwrap();
        let mut bytes = [0; 40];
        store.read(0, &mut bytes).unwrap();
        assert_eq!(&bytes[8..11], &[1, 2, 3]);
        assert_eq!(&bytes[32..34], &[9, 8]);
        assert_eq!(bytes[11], 0);
        assert!(store.write(63, &[1, 2]).is_err());
        assert_eq!(store.written, 5);
        drop(store);
        assert!(!path.exists());
    }
}
