//! `wan migrate [--check]`: brings every pipeline configuration under configs/ to the current contract, writing out
//! what each file already meant, so no run changes behaviour.
//!
//! Kernel fusions became switches of their own (`export.kernels.fuse`, 2026-11-06): the old attention name
//! `sage_rope_sm89` is `sage_sm89` with `fuse.rope_into_sage`, whose float32 rounding (no `rope` named) is
//! `rope: "fused_float32"`; `fuse.qkv` and `fuse.gelu_mlp` are what the weight pack holds (the layout in its
//! `<pack>.json`; dense and FP8 packs fuse no layers). The text is edited in place, so the
//! files keep their order and spacing, and the edit is checked against the same change made on the parsed JSON.
//! `export.kernels.lora_sum` (2026-11-11): every recipe names the sum it ran, "atomic" (upgrade_lora_sum).
//! `generate.cache_write` (2026-11-12): every recipe names how it writes the cache, "clean_pass" (upgrade_cache_write).
//! Idempotent: a migrated file comes out unchanged.

use serde_json::{json, Value};
use std::path::Path;

/// The fusions (qkv, gelu_mlp) a weight pack holds, read from the layout in its `<pack>.json` as its slot installs
/// them (w4a4.fusions): a layer self_attn.qkv; the GELU MLP where an entry is marked gelu_mlp or self_attn.qkv is.
pub fn pack_fusions(root: &Path, pack: &str) -> Result<(bool, bool), String> {
    let path = root.join(format!("{pack}.json"));
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let meta: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let layout = meta["layout"].as_array().ok_or_else(|| format!("{}: no layout", path.display()))?;
    let qkv = layout.iter().any(|e| e["name"].as_str().is_some_and(|n| n.starts_with("self_attn.qkv.")));
    let gelu_mlp = qkv || layout.iter().any(|e| e["gelu_mlp"] == Value::Bool(true));
    Ok((qkv, gelu_mlp))
}

/// The migrated text of one file, or None when it is no pipeline configuration or already migrated.
pub fn upgrade(text: &str, fusions: impl Fn(&str) -> Result<(bool, bool), String>) -> Result<Option<String>, String> {
    let Ok(value) = serde_json::from_str::<Value>(text) else { return Ok(None) };
    let pipeline = value.as_array().is_some_and(|s| s.len() == 2 && s[0]["component"] == "export");
    if !pipeline || value[0]["kernels"].get("fuse").is_some() { return Ok(None); }
    let kernels = &value[0]["kernels"];
    let joint = kernels["attention"] == "sage_rope_sm89";
    let (qkv, gelu_mlp) = if kernels["linear"] == "nunchaku_w4a4" {
        fusions(value[0]["model"]["block_weights"].as_str().ok_or("export.model.block_weights is not a path")?)?
    } else { (false, false) };
    let fuse = json!({"rope_into_sage": joint, "qkv": qkv, "gelu_mlp": gelu_mlp});
    let mut expected = value.clone();
    let k = expected[0]["kernels"].as_object_mut().ok_or("export.kernels is not an object")?;
    if joint {
        k.insert("attention".into(), json!("sage_sm89"));
        k.entry("rope").or_insert(json!("fused_float32"));
    }
    k.insert("fuse".into(), fuse);

    // the same change on the text: after the attention line of the kernels block, its own indentation
    let newline = if text.contains("\r\n") { "\r\n" } else { "\n" };
    let block = text.find("\"kernels\"").ok_or("no kernels block in the text")?;
    let at = block + text[block..].find("\"attention\"").ok_or("no attention line in the kernels block")?;
    let start = text[..at].rfind('\n').map_or(0, |i| i + 1);
    let end = at + text[at..].find(newline).ok_or("the attention line does not end")?;
    let indent = &text[start..at];
    let line = &text[start..end];
    let line = if joint { line.replace("\"sage_rope_sm89\"", "\"sage_sm89\"") } else { line.to_string() };
    let (attention, more) = match line.strip_suffix(',') { Some(body) => (body.to_string(), true), None => (line, false) };
    let mut items = Vec::new();
    if joint && kernels.get("rope").is_none() { items.push(format!("{indent}\"rope\": \"fused_float32\"")); }
    let flags = [("rope_into_sage", joint), ("qkv", qkv), ("gelu_mlp", gelu_mlp)]
        .map(|(name, on)| format!("{indent}  \"{name}\": {on}")).join(&format!(",{newline}"));
    items.push(format!("{indent}\"fuse\": {{{newline}{flags}{newline}{indent}}}"));
    let inserted = format!("{attention},{newline}{}{}", items.join(&format!(",{newline}")), if more { "," } else { "" });
    let edited = format!("{}{inserted}{}", &text[..start], &text[end..]);
    let parsed: Value = serde_json::from_str(&edited).map_err(|e| format!("the edited text is no JSON: {e}"))?;
    if parsed != expected { return Err("the text edit differs from the JSON change".into()); }
    Ok(Some(edited))
}

/// `export.kernels.lora_sum` (2026-11-11) became a key of every recipe: what each one ran until then, "atomic"
/// (Nunchaku's own sum of the low-rank branch), written after its `linear` line. None when the file is no pipeline
/// configuration, has no fusions yet (upgrade comes first) or names the sum already.
pub fn upgrade_lora_sum(text: &str) -> Result<Option<String>, String> {
    let Ok(value) = serde_json::from_str::<Value>(text) else { return Ok(None) };
    let pipeline = value.as_array().is_some_and(|s| s.len() == 2 && s[0]["component"] == "export");
    let kernels = &value[0]["kernels"];
    if !pipeline || kernels.get("fuse").is_none() || kernels.get("lora_sum").is_some() { return Ok(None); }
    let mut expected = value.clone();
    expected[0]["kernels"]["lora_sum"] = json!("atomic");

    let newline = if text.contains("\r\n") { "\r\n" } else { "\n" };
    let block = text.find("\"kernels\"").ok_or("no kernels block in the text")?;
    let at = block + text[block..].find("\"linear\"").ok_or("no linear line in the kernels block")?;
    let start = text[..at].rfind('\n').map_or(0, |i| i + 1);
    let end = at + text[at..].find(newline).ok_or("the linear line does not end")?;
    let indent = &text[start..at];
    let line = &text[start..end];
    let inserted = match line.strip_suffix(',') {
        Some(_) => format!("{line}{newline}{indent}\"lora_sum\": \"atomic\","),
        None => format!("{line},{newline}{indent}\"lora_sum\": \"atomic\""),
    };
    let edited = format!("{}{inserted}{}", &text[..start], &text[end..]);
    let parsed: Value = serde_json::from_str(&edited).map_err(|e| format!("the edited text is no JSON: {e}"))?;
    if parsed != expected { return Err("the text edit differs from the JSON change".into()); }
    Ok(Some(edited))
}

/// `generate.cache_write` (2026-11-12) became a key of every recipe: what each one ran until then, "clean_pass" (the
/// pass at t = 0 that writes the chunk), written after its generate `seed` line. None when the file is no pipeline
/// configuration or names it already.
pub fn upgrade_cache_write(text: &str) -> Result<Option<String>, String> {
    let Ok(value) = serde_json::from_str::<Value>(text) else { return Ok(None) };
    let pipeline = value.as_array().is_some_and(|s| s.len() == 2 && s[1]["component"] == "generate");
    if !pipeline || value[1].get("cache_write").is_some() { return Ok(None); }
    let mut expected = value.clone();
    expected[1]["cache_write"] = json!("clean_pass");
    let newline = if text.contains("\r\n") { "\r\n" } else { "\n" };
    let block = text.find("\"generate\"").ok_or("no generate block in the text")?;
    let at = block + text[block..].find("\"seed\"").ok_or("no seed line in the generate block")?;
    let start = text[..at].rfind('\n').map_or(0, |i| i + 1);
    let end = at + text[at..].find(newline).ok_or("the seed line does not end")?;
    let indent = &text[start..at];
    let line = &text[start..end];
    let inserted = match line.strip_suffix(',') {
        Some(_) => format!("{line}{newline}{indent}\"cache_write\": \"clean_pass\","),
        None => format!("{line},{newline}{indent}\"cache_write\": \"clean_pass\""),
    };
    let edited = format!("{}{inserted}{}", &text[..start], &text[end..]);
    let parsed: Value = serde_json::from_str(&edited).map_err(|e| format!("the edited text is no JSON: {e}"))?;
    if parsed != expected { return Err("the text edit differs from the JSON change".into()); }
    Ok(Some(edited))
}

/// `generate.sampling` (2026-11-25) became a key of every recipe and took over the noise and the timestep shift of
/// `generate.descent`: what each one ran until then, the descent's noise (with the native size of `export.rope`, the
/// one it was always resized from) and shift, or the fresh noise and the packages' schedule where there was no
/// descent. The two keys leave the descent, which keeps `attn_logit_scale` and `reference`. The node is written after
/// the generate `seed` line. None when the file is no pipeline configuration or has the node already.
pub fn upgrade_sampling(text: &str) -> Result<Option<String>, String> {
    let Ok(value) = serde_json::from_str::<Value>(text) else { return Ok(None) };
    let pipeline = value.as_array().is_some_and(|s| s.len() == 2 && s[1]["component"] == "generate");
    if !pipeline || value[1].get("sampling").is_some() { return Ok(None); }
    let descent = value[1].get("descent");
    let mode = descent.and_then(|d| d.get("noise")).and_then(Value::as_str).unwrap_or("fresh");
    let shift = descent.and_then(|d| d.get("timestep_shift")).cloned().unwrap_or(Value::Null);
    let noise = if mode == "fresh" { json!({"mode": "fresh"}) } else {
        let native = value[0]["rope"]["native_size"].as_str().ok_or("export.rope.native_size is not a size")?;
        json!({"mode": mode, "native": native})
    };
    let mut expected = value.clone();
    if let Some(d) = expected[1].get_mut("descent").and_then(Value::as_object_mut) { d.remove("noise"); d.remove("timestep_shift"); }
    expected[1]["sampling"] = json!({"noise": noise, "schedule": {"timestep_shift": shift}});

    let newline = if text.contains("\r\n") { "\r\n" } else { "\n" };
    let block = text.find("\"generate\"").ok_or("no generate block in the text")?;
    // the two keys leave the descent block: their lines go, and the comma of what is last in the block with them
    let mut edited = text.to_string();
    if descent.is_some() {
        let key = block + text[block..].find("\"descent\":").ok_or("no descent block in the text")?;
        let open = key + text[key..].find('{').ok_or("the descent block does not open")?;
        let mut depth = 0;
        let mut close = None;
        for (i, c) in text[open..].char_indices() {
            match c { '{' => depth += 1, '}' => { depth -= 1; if depth == 0 { close = Some(open + i); break; } } _ => {} }
        }
        let close = close.ok_or("the descent block does not close")?;
        let start = text[..open].rfind('\n').map_or(0, |i| i + 1);
        let end = close + text[close..].find(newline).map_or(text.len() - close, |i| i);
        let mut lines: Vec<&str> = text[start..end].split(newline).collect();
        // the first line holds the opening brace, the last the closing one
        let last = lines.len() - 1;
        let mut kept: Vec<String> = Vec::new();
        for (i, line) in lines.drain(..).enumerate() {
            let trimmed = line.trim_start();
            let moved = i > 0 && i < last && (trimmed.starts_with("\"noise\":") || trimmed.starts_with("\"timestep_shift\":"));
            if !moved { kept.push(line.to_string()); }
        }
        let closing = kept.len() - 1;
        if let Some(line) = kept[closing - 1].trim_end().strip_suffix(',').map(str::to_string) { kept[closing - 1] = line; }
        edited = format!("{}{}{}", &text[..start], kept.join(newline), &text[end..]);
    }
    let block = edited.find("\"generate\"").ok_or("no generate block in the text")?;
    let at = block + edited[block..].find("\"seed\"").ok_or("no seed line in the generate block")?;
    let start = edited[..at].rfind('\n').map_or(0, |i| i + 1);
    let end = at + edited[at..].find(newline).ok_or("the seed line does not end")?;
    let indent = edited[start..at].to_string();
    let line = &edited[start..end];
    let mut node = vec![format!("\"sampling\": {{"), "  \"noise\": {".to_string(), format!("    \"mode\": \"{mode}\"")];
    if mode != "fresh" {
        node[2].push(',');
        node.push(format!("    \"native\": {}", noise["native"]));
    }
    node.extend(["  },".to_string(), "  \"schedule\": {".to_string(), format!("    \"timestep_shift\": {shift}"), "  }".to_string(), "}".to_string()]);
    let node = node.iter().map(|l| format!("{indent}{l}")).collect::<Vec<_>>().join(newline);
    let inserted = match line.strip_suffix(',') {
        Some(_) => format!("{line}{newline}{node},"),
        None => format!("{line},{newline}{node}"),
    };
    let edited = format!("{}{inserted}{}", &edited[..start], &edited[end..]);
    let parsed: Value = serde_json::from_str(&edited).map_err(|e| format!("the edited text is no JSON: {e}"))?;
    if parsed != expected { return Err("the text edit differs from the JSON change".into()); }
    Ok(Some(edited))
}

pub fn main(args: &[String]) -> i32 {
    let check = args.iter().any(|a| a == "--check");
    if let Some(bad) = args.iter().find(|a| *a != "--check") {
        eprintln!("wan migrate [--check]: unknown argument {bad}");
        return 2;
    }
    let root = crate::root();
    let mut files = Vec::new();
    let mut stack = vec![root.join("configs")];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() { stack.push(path) } else if path.extension().is_some_and(|e| e == "json") { files.push(path) }
        }
    }
    files.sort();
    let (mut changed, mut failed) = (0, 0);
    for path in files {
        let Ok(text) = std::fs::read_to_string(&path) else { continue };
        let steps = upgrade(&text, |pack| pack_fusions(&root, pack)).and_then(|fused| {
            let current = fused.as_deref().unwrap_or(&text);
            let summed = upgrade_lora_sum(current)?.or(fused);
            let current = summed.as_deref().unwrap_or(&text);
            let cached = upgrade_cache_write(current)?.or(summed);
            let current = cached.as_deref().unwrap_or(&text);
            Ok(upgrade_sampling(current)?.or(cached))
        });
        match steps {
            Ok(None) => {}
            Ok(Some(new)) => {
                changed += 1;
                println!("{} {}", if check { "would change" } else { "migrated" }, path.display());
                if !check {
                    if let Err(e) = std::fs::write(&path, new) { eprintln!("{}: {e}", path.display()); failed += 1; }
                }
            }
            Err(e) => { eprintln!("{}: {e}", path.display()); failed += 1; }
        }
    }
    println!("{changed} files{}, {failed} failed", if check { " would change" } else { " migrated" });
    if failed > 0 { 1 } else { 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    const JOINT: &str = "[\n  {\n    \"component\": \"export\",\n    \"model\": {\n      \"block_weights\": \"artifacts/p.w4a4\"\n    },\n    \"kernels\": {\n      \"attention\": \"sage_rope_sm89\",\n      \"linear\": \"nunchaku_w4a4\",\n      \"rope_dtype\": \"float32\"\n    }\n  },\n  {\n    \"component\": \"generate\"\n  }\n]\n";

    fn v2(_: &str) -> Result<(bool, bool), String> { Ok((true, true)) }

    #[test]
    fn the_old_joint_name_becomes_a_switch_and_keeps_its_float32_rounding() {
        let new = upgrade(JOINT, v2).unwrap().unwrap();
        let value: Value = serde_json::from_str(&new).unwrap();
        let k = &value[0]["kernels"];
        assert_eq!(k["attention"], "sage_sm89");
        assert_eq!(k["rope"], "fused_float32");
        assert_eq!(k["fuse"], json!({"rope_into_sage": true, "qkv": true, "gelu_mlp": true}));
        assert!(new.contains("      \"attention\": \"sage_sm89\",\n      \"rope\": \"fused_float32\",\n      \"fuse\": {\n        \"rope_into_sage\": true,\n        \"qkv\": true,\n        \"gelu_mlp\": true\n      },\n      \"linear\""), "{new}");
        assert_eq!(upgrade(&new, v2).unwrap(), None, "a migrated file comes out unchanged");
    }

    #[test]
    fn a_named_rounding_stays_and_separate_paths_get_no_fused_rope() {
        let named = JOINT.replace("\"rope_dtype\"", "\"rope\": \"longlive_triton\",\n      \"rope_dtype\"");
        let k = serde_json::from_str::<Value>(&upgrade(&named, v2).unwrap().unwrap()).unwrap()[0]["kernels"].clone();
        assert_eq!((k["rope"].as_str(), k["fuse"]["rope_into_sage"].as_bool()), (Some("longlive_triton"), Some(true)));
        let separate = named.replace("sage_rope_sm89", "sage_sm89");
        let k = serde_json::from_str::<Value>(&upgrade(&separate, v2).unwrap().unwrap()).unwrap()[0]["kernels"].clone();
        assert_eq!((k["attention"].as_str(), k["fuse"]["rope_into_sage"].as_bool()), (Some("sage_sm89"), Some(false)));
    }

    #[test]
    fn layer_fusions_come_from_the_pack_and_dense_packs_fuse_nothing() {
        let v1 = |_: &str| Ok((false, false));
        let k = serde_json::from_str::<Value>(&upgrade(JOINT, v1).unwrap().unwrap()).unwrap()[0]["kernels"].clone();
        assert_eq!(k["fuse"], json!({"rope_into_sage": true, "qkv": false, "gelu_mlp": false}));
        let fp8 = JOINT.replace("sage_rope_sm89", "torch_sdpa").replace("nunchaku_w4a4", "torch_fp8");
        let k = serde_json::from_str::<Value>(&upgrade(&fp8, |_| Err("read".into())).unwrap().unwrap()).unwrap()[0]["kernels"].clone();
        assert_eq!(k["fuse"], json!({"rope_into_sage": false, "qkv": false, "gelu_mlp": false}));
        assert!(upgrade(JOINT, |_| Err("no pack metadata".into())).is_err());
    }

    #[test]
    fn a_pack_fuses_what_its_slot_installs() {
        let dir = std::env::temp_dir().join(format!("wan-migrate-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for (name, layout, want) in [
            ("v2", json!([{"name": "self_attn.qkv.qweight"}, {"name": "ffn.2.qweight"}]), (true, true)),
            ("sqkv", json!([{"name": "self_attn.q.qweight"}, {"name": "ffn.2.qweight", "gelu_mlp": true}]), (false, true)),
            ("v1", json!([{"name": "self_attn.q.qweight"}, {"name": "ffn.2.qweight"}]), (false, false)),
        ] {
            std::fs::write(dir.join(format!("{name}.w4a4.json")), json!({"layout": layout}).to_string()).unwrap();
            assert_eq!(pack_fusions(&dir, &format!("{name}.w4a4")), Ok(want), "{name}");
        }
        assert!(pack_fusions(&dir, "missing.w4a4").is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn every_recipe_gets_the_atomic_low_rank_sum_it_always_had() {
        let fused = upgrade(JOINT, v2).unwrap().unwrap();
        for text in [fused.clone(), fused.replace('\n', "\r\n")] {
            let new = upgrade_lora_sum(&text).unwrap().unwrap();
            let value: Value = serde_json::from_str(&new).unwrap();
            assert_eq!(value[0]["kernels"]["lora_sum"], "atomic");
            let mut expected: Value = serde_json::from_str(&text).unwrap();
            expected[0]["kernels"]["lora_sum"] = json!("atomic");
            assert_eq!(value, expected, "nothing else changes");
            assert!(new.contains("\"linear\": \"nunchaku_w4a4\",") && new.contains("\"lora_sum\": \"atomic\""), "{new}");
            assert_eq!(upgrade_lora_sum(&new).unwrap(), None, "a migrated file comes out unchanged");
        }
        assert_eq!(upgrade_lora_sum(JOINT).unwrap(), None, "the fusions come first");
        assert_eq!(upgrade_lora_sum("{\"prompts\": []}").unwrap(), None);
    }

    #[test]
    fn every_recipe_writes_its_cache_with_the_clean_pass_it_always_had() {
        let text = "[\n  {\n    \"component\": \"export\"\n  },\n  {\n    \"component\": \"generate\",\n    \"seed\": 0,\n    \"mode\": \"offline\"\n  }\n]\n";
        for t in [text.to_string(), text.replace('\n', "\r\n"), text.replace(",\n    \"mode\": \"offline\"", "")] {
            let new = upgrade_cache_write(&t).unwrap().unwrap();
            let mut expected: Value = serde_json::from_str(&t).unwrap();
            expected[1]["cache_write"] = json!("clean_pass");
            assert_eq!(serde_json::from_str::<Value>(&new).unwrap(), expected, "{new}");
            assert_eq!(upgrade_cache_write(&new).unwrap(), None);
        }
        assert_eq!(upgrade_cache_write("{\"prompts\": []}").unwrap(), None);
    }

    const SAMPLED: &str = "[\n  {\n    \"component\": \"export\",\n    \"rope\": {\n      \"native_size\": \"704x1280\"\n    }\n  },\n  {\n    \"component\": \"generate\",\n    \"seed\": 0,\n    \"descent\": {\n      \"attn_logit_scale\": 1.0,\n      \"timestep_shift\": 3.63,\n      \"reference\": null,\n      \"noise\": \"native_spectral\"\n    }\n  }\n]\n";

    /// the value the text must parse to: sampling written, the noise and the shift gone from the descent
    fn sampled(text: &str, noise: Value, shift: Value) -> Value {
        let mut expected: Value = serde_json::from_str(text).unwrap();
        if let Some(d) = expected[1].get_mut("descent").and_then(Value::as_object_mut) { d.remove("noise"); d.remove("timestep_shift"); }
        expected[1]["sampling"] = json!({"noise": noise, "schedule": {"timestep_shift": shift}});
        expected
    }

    #[test]
    fn the_descents_noise_and_shift_move_into_the_sampling_node() {
        let native = json!({"mode": "native_spectral", "native": "704x1280"});
        for text in [SAMPLED.to_string(), SAMPLED.replace('\n', "\r\n")] {
            let new = upgrade_sampling(&text).unwrap().unwrap();
            assert_eq!(serde_json::from_str::<Value>(&new).unwrap(), sampled(&text, native.clone(), json!(3.63)), "{new}");
            assert!(!new.contains("\"noise\": \"native_spectral\""), "the descent keeps no noise: {new}");
            assert_eq!(upgrade_sampling(&new).unwrap(), None, "a migrated file comes out unchanged");
        }
        let new = upgrade_sampling(SAMPLED).unwrap().unwrap();
        let block = "    \"seed\": 0,\n    \"sampling\": {\n      \"noise\": {\n        \"mode\": \"native_spectral\",\n        \"native\": \"704x1280\"\n      },\n      \"schedule\": {\n        \"timestep_shift\": 3.63\n      }\n    },\n    \"descent\": {";
        assert!(new.contains(block), "{new}");
        let crlf = upgrade_sampling(&SAMPLED.replace('\n', "\r\n")).unwrap().unwrap();
        assert!(!crlf.replace("\r\n", "").contains('\n'), "mixed line endings");
    }

    #[test]
    fn whatever_key_order_the_descent_had_its_commas_stay_valid() {
        let noise = "\"noise\": \"native_resized\"";
        let orders: [[&str; 4]; 4] = [
            [noise, "\"attn_logit_scale\": 1.0", "\"timestep_shift\": null", "\"reference\": null"],
            ["\"attn_logit_scale\": 1.0", noise, "\"reference\": null", "\"timestep_shift\": null"],
            ["\"attn_logit_scale\": 1.0", "\"reference\": null", "\"timestep_shift\": null", noise],
            ["\"timestep_shift\": null", noise, "\"attn_logit_scale\": 1.0", "\"reference\": null"],
        ];
        for order in orders {
            let body = order.iter().map(|l| format!("      {l}")).collect::<Vec<_>>().join(",\n");
            let text = SAMPLED.replace("      \"attn_logit_scale\": 1.0,\n      \"timestep_shift\": 3.63,\n      \"reference\": null,\n      \"noise\": \"native_spectral\"", &body);
            let new = upgrade_sampling(&text).unwrap().unwrap_or_else(|| panic!("{order:?}"));
            let want = sampled(&text, json!({"mode": "native_resized", "native": "704x1280"}), json!(null));
            assert_eq!(serde_json::from_str::<Value>(&new).unwrap(), want, "{order:?}\n{new}");
        }
    }

    #[test]
    fn a_recipe_without_a_descent_gets_the_fresh_noise_and_the_packages_schedule() {
        let text = "[\n  {\n    \"component\": \"export\"\n  },\n  {\n    \"component\": \"generate\",\n    \"seed\": 0,\n    \"mode\": \"offline\"\n  }\n]\n";
        for t in [text.to_string(), text.replace(",\n    \"mode\": \"offline\"", "")] {
            let new = upgrade_sampling(&t).unwrap().unwrap();
            assert_eq!(serde_json::from_str::<Value>(&new).unwrap(), sampled(&t, json!({"mode": "fresh"}), json!(null)), "{new}");
        }
        // a descent that already named the fresh noise and no shift
        let fresh = SAMPLED.replace("\"native_spectral\"", "\"fresh\"").replace("3.63", "null");
        let new = upgrade_sampling(&fresh).unwrap().unwrap();
        assert_eq!(serde_json::from_str::<Value>(&new).unwrap(), sampled(&fresh, json!({"mode": "fresh"}), json!(null)), "{new}");
        assert_eq!(upgrade_sampling("{\"prompts\": []}").unwrap(), None);
        assert_eq!(upgrade_sampling("not json").unwrap(), None);
    }

    #[test]
    fn crlf_files_and_other_json_are_handled() {
        let crlf = JOINT.replace('\n', "\r\n");
        let new = upgrade(&crlf, v2).unwrap().unwrap();
        assert!(!new.replace("\r\n", "").contains('\n'), "mixed line endings");
        assert_eq!(upgrade("{\"prompts\": []}", v2).unwrap(), None);
        assert_eq!(upgrade("not json", v2).unwrap(), None);
    }
}
