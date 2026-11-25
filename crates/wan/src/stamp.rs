//! The warm start. After a preparation succeeds, the host stamps the export recipe with the size and modification
//! time of every input and package file (the list is written by the preparation itself, scripts/conversion/
//! prepare_live.py) and the entry names of their folders. A start that finds the same recipe and the same stamps
//! has nothing to prepare and runs no Python at all: it costs a few thousand `stat` calls, not tens of gigabytes
//! of hashing.
use std::path::Path;
use std::time::UNIX_EPOCH;

use serde_json::{json, Value};

use crate::video::Result;

fn modified(path: &Path) -> Option<(u64, u128)> {
    let meta = std::fs::metadata(path).ok()?;
    let nanos = meta.modified().ok()?.duration_since(UNIX_EPOCH).ok()?.as_nanos();
    Some((meta.len(), nanos))
}

fn names(folder: &Path) -> Option<Vec<String>> {
    let mut names: Vec<String> = std::fs::read_dir(folder).ok()?
        .filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()).collect();
    names.sort();
    Some(names)
}

fn strings(value: &Value, key: &str) -> Option<Vec<String>> {
    value.get(key)?.as_array()?.iter().map(|v| v.as_str().map(str::to_owned)).collect()
}

/// Records the state the preparation left: `listing` is the file the preparation wrote.
pub fn seal(stamp: &Path, listing: &Path, export: &Value) -> Result<()> {
    let listing: Value = serde_json::from_slice(&std::fs::read(listing)?)?;
    let files = strings(&listing, "files").ok_or("the listing has no files")?;
    let folders = strings(&listing, "folders").ok_or("the listing has no folders")?;
    let files: Vec<Value> = files.iter().filter_map(|f| modified(Path::new(f)).map(|(len, ns)| json!([f, len, ns.to_string()]))).collect();
    let folders: Vec<Value> = folders.iter().filter_map(|f| names(Path::new(f)).map(|n| json!([f, n]))).collect();
    let temporary = stamp.with_extension(format!("json.{}", std::process::id()));
    std::fs::write(&temporary, serde_json::to_vec(&json!({"export": export, "files": files, "folders": folders}))?)?;
    std::fs::rename(&temporary, stamp)?;
    Ok(())
}

/// True when nothing the stamp saw has changed. Any doubt (no stamp, a broken stamp, a missing file) is false.
pub fn holds(stamp: &Path, export: &Value) -> bool {
    let Some(value) = std::fs::read(stamp).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()) else { return false };
    if value.get("export") != Some(export) { return false; }
    let (Some(files), Some(folders)) = (value.get("files").and_then(Value::as_array), value.get("folders").and_then(Value::as_array)) else { return false };
    if files.is_empty() { return false; }
    files.iter().all(|f| {
        let Some([path, len, ns]) = f.as_array().map(Vec::as_slice) else { return false };
        let (Some(path), Some(len), Some(ns)) = (path.as_str(), len.as_u64(), ns.as_str()) else { return false };
        modified(Path::new(path)).is_some_and(|(l, n)| l == len && n.to_string() == ns)
    }) && folders.iter().all(|f| {
        let Some([path, listed]) = f.as_array().map(Vec::as_slice) else { return false };
        let (Some(path), Some(listed)) = (path.as_str(), strings(&json!({"n": listed}), "n")) else { return false };
        names(Path::new(path)).is_some_and(|n| n == listed)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Scene { dir: std::path::PathBuf }
    impl Scene {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!("wan-stamp-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(dir.join("src")).unwrap();
            std::fs::create_dir_all(dir.join("state")).unwrap();
            std::fs::write(dir.join("src/a.py"), "a").unwrap();
            std::fs::write(dir.join("weights.bin"), "0123456789").unwrap();
            let files = [dir.join("src/a.py"), dir.join("weights.bin")];
            let folders = [dir.join("src")];
            std::fs::write(dir.join("state/inputs.json"), json!({"files": files, "folders": folders}).to_string()).unwrap();
            Scene { dir }
        }
        fn seal(&self, export: &Value) { seal(&self.dir.join("state/stamp.json"), &self.dir.join("state/inputs.json"), export).unwrap(); }
        fn holds(&self, export: &Value) -> bool { holds(&self.dir.join("state/stamp.json"), export) }
    }
    impl Drop for Scene { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.dir); } }

    #[test]
    fn nothing_changed_means_nothing_to_prepare() {
        let scene = Scene::new("same");
        let export = json!({"rebuild": "auto"});
        assert!(!scene.holds(&export), "no stamp yet");
        scene.seal(&export);
        assert!(scene.holds(&export));
    }

    #[test]
    fn a_changed_recipe_file_or_folder_asks_for_the_preparation() {
        let scene = Scene::new("changed");
        let export = json!({"steps": 4});
        scene.seal(&export);
        assert!(!scene.holds(&json!({"steps": 5})), "recipe");
        std::fs::write(scene.dir.join("weights.bin"), "01234567890").unwrap();
        assert!(!scene.holds(&export), "size");
        scene.seal(&export);
        assert!(scene.holds(&export));
        std::fs::write(scene.dir.join("src/b.py"), "b").unwrap();
        assert!(!scene.holds(&export), "a new file in a watched folder");
        std::fs::remove_file(scene.dir.join("src/b.py")).unwrap();
        std::fs::remove_file(scene.dir.join("src/a.py")).unwrap();
        assert!(!scene.holds(&export), "a missing file");
    }

    #[test]
    fn a_broken_stamp_is_no_stamp() {
        let scene = Scene::new("broken");
        std::fs::write(scene.dir.join("state/stamp.json"), "{").unwrap();
        assert!(!scene.holds(&json!({})));
    }

    #[test]
    fn the_check_of_many_files_costs_milliseconds() {
        let scene = Scene::new("many");
        let many = scene.dir.join("many");
        std::fs::create_dir_all(&many).unwrap();
        let files: Vec<_> = (0..3000).map(|i| { let p = many.join(format!("{i}.bin")); std::fs::write(&p, "x").unwrap(); p }).collect();
        std::fs::write(scene.dir.join("state/inputs.json"), json!({"files": files, "folders": [many]}).to_string()).unwrap();
        let export = json!({});
        scene.seal(&export);
        let start = std::time::Instant::now();
        assert!(scene.holds(&export));
        let took = start.elapsed();
        assert!(took < std::time::Duration::from_millis(250), "3000 files took {took:?}");
    }
}
