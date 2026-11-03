//! The cold preparation boundary. The child process exits before the native runtime is loaded.
use crate::config::Config;
use crate::video::Result;

pub fn prepare(config: &Config) -> Result<()> {
    // Capacity is checked against live placement at allocation time, after export.
    let path = std::env::temp_dir().join(format!("wan-pipeline-{}.json", std::process::id()));
    std::fs::write(&path, serde_json::to_vec_pretty(&config.value())?)?;
    let result = std::process::Command::new(crate::root().join(".venv/Scripts/python.exe"))
        .arg(crate::root().join("scripts/conversion/prepare_live.py"))
        .arg("--config").arg(&path).current_dir(crate::root())
        .env("PYTHONUTF8", "1").status();
    let _ = std::fs::remove_file(&path);
    if !result?.success() { return Err("package preparation failed; generation was not started".into()); }
    Ok(())
}
