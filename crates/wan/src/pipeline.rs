//! The cold preparation boundary. The child process exits before the native runtime is loaded.
use std::io::BufRead;

use crate::config::Config;
use crate::video::Result;

/// Exports and checks the packages in a child (scripts/conversion/prepare_live.py). Its output goes on to this
/// process's output line by line, and each line is the loading screen's detail of the step.
pub fn prepare(config: &Config) -> Result<()> {
    let export = config.value()[0].clone();
    let stamp = crate::root().join("artifacts/prepare_stamp.json");
    let listing = crate::root().join("artifacts/prepare_inputs.json");
    if crate::stamp::holds(&stamp, &export) {
        crate::stage::detail("packages match the config; nothing to prepare");
        println!("packages match the config; nothing to prepare");
        return Ok(());
    }
    let _ = std::fs::remove_file(&listing);
    // Capacity is checked against live placement at allocation time, after export.
    let path = std::env::temp_dir().join(format!("wan-pipeline-{}.json", std::process::id()));
    std::fs::write(&path, serde_json::to_vec_pretty(&config.value())?)?;
    let child = std::process::Command::new(crate::root().join(".venv/Scripts/python.exe"))
        .arg(crate::root().join("scripts/conversion/prepare_live.py"))
        .arg("--config").arg(&path).current_dir(crate::root())
        .env("PYTHONUTF8", "1").stdout(std::process::Stdio::piped()).spawn();
    let result = child.and_then(|mut child| {
        if let Some(out) = child.stdout.take() {
            let mut out = std::io::BufReader::new(out);
            let mut line = Vec::new();
            while out.read_until(b'\n', &mut line)? > 0 {
                let text = String::from_utf8_lossy(&line);
                let text = text.trim_end();
                println!("{text}");
                crate::stage::detail(text);
                line.clear();
            }
        }
        child.wait()
    });
    let _ = std::fs::remove_file(&path);
    if !result?.success() { return Err("package preparation failed; generation was not started".into()); }
    // A listing that failed to be written only costs the next start its warm path.
    if let Err(error) = crate::stamp::seal(&stamp, &listing, &export) { println!("no warm stamp: {error}"); }
    Ok(())
}
