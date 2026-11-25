//! The strict contract of a configuration file, checked on the JSON itself before anything is deserialized.
//!
//! Every object lists all of its keys: nothing is filled in silently. A branch (the rope's interpolation, the
//! offloading, the diagnostics, the offline output) needs its keys only while its switch is on; while the switch is
//! off its keys must be absent, so that a file never carries a value that does nothing.

use serde_json::{Map, Value};

type Check = Result<(), String>;

fn object<'a>(value: &'a Value, path: &str) -> Result<&'a Map<String, Value>, String> {
    value.as_object().ok_or_else(|| format!("{path} must be an object"))
}

/// The keys of `path`: `always` must be there, `branch` only while `on`, and no other key may be.
fn keys(value: &Value, path: &str, always: &[&str], switch: &str, on: bool, branch: &[&str]) -> Check {
    let map = object(value, path)?;
    for key in always {
        if !map.contains_key(*key) { return Err(format!("{path}.{key} is missing")); }
    }
    for key in branch {
        match (on, map.contains_key(*key)) {
            (true, false) => return Err(format!("{path}.{key} is missing while {switch} is on")),
            (false, true) => return Err(format!("{path}.{key} is set while {switch} is off: remove it")),
            _ => {}
        }
    }
    for key in map.keys() {
        if !always.contains(&key.as_str()) && !branch.contains(&key.as_str()) {
            return Err(format!("{path}.{key} is not a key of this object"));
        }
    }
    Ok(())
}

fn flat(value: &Value, path: &str, always: &[&str]) -> Check { keys(value, path, always, "", false, &[]) }

/// Exactly one of two alternative keys of `map` (a size in GiB or a fraction, say).
fn one_of(map: &Map<String, Value>, path: &str, a: &str, b: &str) -> Check {
    match (map.contains_key(a), map.contains_key(b)) {
        (true, false) | (false, true) => Ok(()),
        _ => Err(format!("{path} needs exactly one of {a} and {b}")),
    }
}

pub fn check(config: &Value) -> Check {
    let stages = config.as_array().filter(|s| s.len() == 2).ok_or("expected [export, generate]")?;
    export(&stages[0])?;
    generate(&stages[1])
}

fn export(e: &Value) -> Check {
    flat(e, "export", &["component", "compiler", "rebuild", "output", "model", "video", "sampling", "context", "rope", "kernels"])?;
    flat(&e["model"], "export.model", &["architecture", "checkpoint", "block_weights", "text_encoder", "decoder"])?;
    flat(&e["model"]["text_encoder"], "export.model.text_encoder", &["checkpoint", "block_weights", "tokenizer", "precision"])?;
    flat(&e["video"], "export.video", &["size", "fps"])?;
    flat(&e["sampling"], "export.sampling", &["algorithm", "steps", "timestep_shift"])?;
    flat(&e["context"], "export.context", &["chunk_frames", "window_frames", "sink_frames"])?;
    let rope = &e["rope"];
    let on = object(rope, "export.rope")?.get("interpolate").and_then(Value::as_bool)
        .ok_or("export.rope.interpolate is missing or not a boolean")?;
    // A new explicit profile selector is optional for compatibility with existing NTK/PI recipes.
    // It is never inserted into old recipes, and is forbidden when interpolation is off.
    let mut rope_keys = vec!["interpolate", "native_size", "scene_offset_frames"];
    if rope.get("frequency_aware").is_some() {
        if !on { return Err("export.rope.frequency_aware is set while export.rope.interpolate is off: remove it".into()); }
        rope_keys.push("frequency_aware");
    }
    // Attention logit multiplier carried by the RoPE table: optional, never inserted into old recipes.
    if rope.get("attn_logit_scale").is_some() {
        if !on { return Err("export.rope.attn_logit_scale is set while export.rope.interpolate is off: remove it".into()); }
        rope_keys.push("attn_logit_scale");
    }
    // M per side and the NTK to PI mix (rope.rs): optional, never inserted into old recipes
    for key in ["m", "ntk_pi_mix"] {
        if rope.get(key).is_some() {
            if !on { return Err(format!("export.rope.{key} is set while export.rope.interpolate is off: remove it")); }
            rope_keys.push(key);
        }
    }
    if let Some(m) = rope.get("m") { flat(m, "export.rope.m", &["height", "width"])?; }
    keys(rope, "export.rope", &rope_keys, "export.rope.interpolate", on,
         &["stretch", "by_parts", "ntk_aware"])?;
    if on {
        if let Some(parts) = rope.get("by_parts").filter(|p| !p.is_null()) {
            flat(parts, "export.rope.by_parts", &["alpha", "beta"])?;
        }
    }
    kernels(&e["kernels"])
}

/// Every fusion is a switch of its own in `fuse`: RoPE into Sage (the block packages), q, k, v as one layer and the
/// GELU MLP (the weight pack). `rope` always names the RoPE arithmetic, which the fused kernel reproduces.
fn kernels(k: &Value) -> Check {
    let map = object(k, "export.kernels")?;
    let always = ["attention", "cross_attention", "linear", "kv_cache", "kv_store", "rope", "rope_dtype", "fuse", "lora_sum"];
    for key in always {
        if !map.contains_key(key) { return Err(format!("export.kernels.{key} is missing")); }
    }
    if let Some(key) = map.keys().find(|key| key.as_str() != "sparse" && !always.contains(&key.as_str())) {
        return Err(format!("export.kernels.{key} is not a key of this object"));
    }
    let fusions = ["rope_into_sage", "qkv", "gelu_mlp"];
    flat(&k["fuse"], "export.kernels.fuse", &fusions)?;
    if let Some(flag) = fusions.iter().find(|f| !k["fuse"][**f].is_boolean()) {
        return Err(format!("export.kernels.fuse.{flag} must be true or false"));
    }
    let joint = k["fuse"]["rope_into_sage"] == Value::Bool(true);
    // Block sparsity: optional, never inserted into old recipes; its kernels exist in the joint path only
    match map.get("sparse") {
        None => Ok(()),
        Some(_) if !joint => Err("export.kernels.sparse needs export.kernels.fuse.rope_into_sage".into()),
        Some(sparse) => {
            let on = object(sparse, "export.kernels.sparse")?.get("enabled").and_then(Value::as_bool)
                .ok_or("export.kernels.sparse.enabled is missing or not a boolean")?;
            keys(sparse, "export.kernels.sparse", &["enabled"], "export.kernels.sparse.enabled", on, &["skip_mass", "frames"])?;
            if on { flat(&sparse["frames"], "export.kernels.sparse.frames", &["sink", "recent_frames"])?; }
            Ok(())
        }
    }
}

fn generate(g: &Value) -> Check {
    let map = object(g, "generate")?;
    let debug_on = !matches!(map.get("debug"), Some(Value::Bool(false)));
    let offline = map.get("mode").and_then(Value::as_str) == Some("offline");
    let always = ["component", "packages_from", "mode", "debug", "prompts", "seed", "cache_write", "sampling", "offloading"];
    // keys of two switches: the output is needed offline only, the drain while diagnostics are on
    for key in always {
        if !map.contains_key(key) { return Err(format!("generate.{key} is missing")); }
    }
    for (key, on, switch) in [("drain", debug_on, "generate.debug"), ("output", offline, "generate.mode offline")] {
        match (on, map.contains_key(key)) {
            (true, false) => return Err(format!("generate.{key} is missing while {switch} is on")),
            (false, true) => return Err(format!("generate.{key} is set while {switch} is off: remove it")),
            _ => {}
        }
    }
    for key in map.keys() {
        if !always.contains(&key.as_str()) && !["drain", "output", "descent", "rope_sweep", "kv_probe"].contains(&key.as_str()) {
            return Err(format!("generate.{key} is not a key of this object"));
        }
    }
    if debug_on { flat(&g["drain"], "generate.drain", &["every"])?; }
    if offline { flat(&g["output"], "generate.output", &["directory", "codec", "crf", "pixel_format"])?; }
    sampling(&g["sampling"])?;
    // the search in resolution: optional, never inserted into old configs, complete when present; the noise and the
    // schedule it once held moved to generate.sampling
    if let Some(d) = map.get("descent") {
        for (key, to) in [("noise", "generate.sampling.noise"), ("timestep_shift", "generate.sampling.schedule")] {
            if object(d, "generate.descent")?.contains_key(key) {
                return Err(format!("generate.descent.{key} moved to {to}: run wan migrate"));
            }
        }
        flat(d, "generate.descent", &["attn_logit_scale", "reference"])?;
    }
    if let Some(points) = map.get("rope_sweep") {
        for point in points.as_array().ok_or("generate.rope_sweep must be a list")? {
            let mix = point.get("ntk_pi_mix").is_some();
            keys(point, "generate.rope_sweep[]", &["height", "width"], "a mix of the point", mix, &if mix { vec!["ntk_pi_mix"] } else { vec![] })?;
        }
    }
    prompts(&g["prompts"])?;
    offloading(&g["offloading"])
}

/// The noise and the schedule: a fresh noise has no other key, the two native ones name their native size.
fn sampling(s: &Value) -> Check {
    flat(s, "generate.sampling", &["noise", "schedule"])?;
    let mode = object(&s["noise"], "generate.sampling.noise")?.get("mode").and_then(Value::as_str)
        .ok_or("generate.sampling.noise.mode is missing or not a string")?;
    if !["fresh", "native_resized", "native_spectral"].contains(&mode) {
        return Err(format!("generate.sampling.noise.mode must be fresh, native_resized or native_spectral, not {mode}"));
    }
    keys(&s["noise"], "generate.sampling.noise", &["mode"], "generate.sampling.noise.mode native_resized or native_spectral",
         mode != "fresh", &["native"])?;
    flat(&s["schedule"], "generate.sampling.schedule", &["timestep_shift"])
}

fn prompts(p: &Value) -> Check {
    match p.get("source").and_then(Value::as_str) {
        Some("interactive") => flat(p, "generate.prompts", &["source", "initial"]),
        Some("script") => flat(p, "generate.prompts", &["source", "file", "chains"]),
        _ => Err("generate.prompts.source must be interactive or script".into()),
    }
}

fn offloading(o: &Value) -> Check {
    let map = object(o, "generate.offloading")?;
    let on = map.get("enabled").and_then(Value::as_bool).ok_or("generate.offloading.enabled is missing or not a boolean")?;
    // the two sizes of a margin and of T5's activations each come as one of two alternatives
    let mut required: Vec<&str> = vec!["calibration", "resources", "copy_spare", "prefetch_fps_tolerance", "ram_margin_fraction", "t5"];
    let mut optional: Vec<&str> = vec![];
    if on {
        one_of(map, "generate.offloading", "vram_margin_gib", "vram_margin_fraction")?;
        optional.extend(["vram_margin_gib", "vram_margin_fraction", "t5_activations_gib", "policy"]);
    } else {
        required.clear();
    }
    for key in &required {
        if !map.contains_key(*key) { return Err(format!("generate.offloading.{key} is missing while generate.offloading.enabled is on")); }
    }
    for key in map.keys() {
        let allowed = key == "enabled" || required.contains(&key.as_str()) || optional.contains(&key.as_str());
        if !allowed {
            return Err(if on { format!("generate.offloading.{key} is not a key of this object") }
                       else { format!("generate.offloading.{key} is set while generate.offloading.enabled is off: remove it") });
        }
    }
    if !on { return Ok(()); }
    flat(&o["resources"], "generate.offloading.resources", &["block_weights", "kv_cache"])?;
    let t5 = object(&o["t5"], "generate.offloading.t5")?;
    one_of(t5, "generate.offloading.t5", "resident", "resident_gib")?;
    if !t5.contains_key("prefetch_tolerance") { return Err("generate.offloading.t5.prefetch_tolerance is missing".into()); }
    if !t5.contains_key("activations") && !map.contains_key("t5_activations_gib") {
        return Err("generate.offloading needs t5.activations or t5_activations_gib".into());
    }
    for key in t5.keys() {
        if !["resident", "resident_gib", "activations", "prefetch_tolerance"].contains(&key.as_str()) {
            return Err(format!("generate.offloading.t5.{key} is not a key of this object"));
        }
    }
    Ok(())
}

/// A configuration read into the structs and written out again carries their defaults: a T5 size beside "auto", a
/// drain beside diagnostics that are off. This writes it in the form of the contract, so that it can be parsed again
/// (the export gets this form). It drops only what the code ignores anyway.
pub fn normalize(config: &mut Value) {
    let Some(stages) = config.as_array_mut().filter(|s| s.len() == 2) else { return };
    let rope = &mut stages[0]["rope"];
    if rope["interpolate"].as_bool() == Some(true) {
        let map = rope.as_object_mut().unwrap();
        let stretch = if map.contains_key("m") { Value::Null } else { Value::from(1.0) };
        map.entry("stretch").or_insert(stretch);
        map.entry("by_parts").or_insert(Value::Null);
        map.entry("ntk_aware").or_insert(Value::Bool(false));
    } else if let Some(map) = rope.as_object_mut() {
        for key in ["stretch", "by_parts", "ntk_aware", "frequency_aware", "attn_logit_scale", "m", "ntk_pi_mix"] { map.remove(key); }
    }
    let generate = stages[1].as_object_mut().unwrap();
    if generate.get("debug") == Some(&Value::Bool(false)) { generate.remove("drain"); }
    if generate.get("mode").and_then(Value::as_str) != Some("offline") { generate.remove("output"); }
    if let Some(prompts) = generate.get_mut("prompts").and_then(Value::as_object_mut) {
        if prompts.get("source").and_then(Value::as_str) == Some("interactive") { prompts.entry("initial").or_insert(Value::Null); }
    }
    let Some(off) = generate.get_mut("offloading").and_then(Value::as_object_mut) else { return };
    if off.get("enabled") == Some(&Value::Bool(false)) {
        off.retain(|key, _| key == "enabled");
        return;
    }
    if off.contains_key("vram_margin_fraction") { off.remove("vram_margin_gib"); }
    let t5 = off.get_mut("t5").and_then(Value::as_object_mut);
    if let Some(t5) = t5 {
        if t5.contains_key("resident") { t5.remove("resident_gib"); }
        if t5.contains_key("activations") { off.remove("t5_activations_gib"); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn shipped(name: &str) -> Value {
        serde_json::from_str(&std::fs::read_to_string(crate::root().join("configs").join(name)).unwrap()).unwrap()
    }
    fn remove(config: &mut Value, pointer: &str) {
        let (parent, key) = pointer.rsplit_once('/').unwrap();
        config.pointer_mut(parent).unwrap().as_object_mut().unwrap().remove(key).unwrap_or_else(|| panic!("{pointer}"));
    }
    fn message(config: &Value) -> String { check(config).expect_err("the contract must refuse this") }

    #[test]
    fn every_required_key_is_refused_when_missing_and_named_by_its_path() {
        for (file, pointer, path) in [
            ("mvp1.json", "/0/compiler", "export.compiler"),
            ("mvp1.json", "/0/model/text_encoder/precision", "export.model.text_encoder.precision"),
            ("mvp1.json", "/0/rope/native_size", "export.rope.native_size"),
            ("mvp1.json", "/0/kernels/rope_dtype", "export.kernels.rope_dtype"),
            ("mvp1.json", "/1/mode", "generate.mode"),
            ("mvp1.json", "/1/debug", "generate.debug"),
            ("mvp1.json", "/1/seed", "generate.seed"),
            ("mvp1.json", "/1/sampling", "generate.sampling"),
            ("mvp1.json", "/1/offloading/enabled", "generate.offloading.enabled"),
            ("mvp1.json", "/1/offloading/copy_spare", "generate.offloading.copy_spare"),
            ("mvp1.json", "/1/offloading/prefetch_fps_tolerance", "generate.offloading.prefetch_fps_tolerance"),
            ("mvp1.json", "/1/offloading/ram_margin_fraction", "generate.offloading.ram_margin_fraction"),
            ("mvp1.json", "/1/offloading/t5/prefetch_tolerance", "prefetch_tolerance"),
            ("mvp1.json", "/1/offloading/resources/kv_cache", "generate.offloading.resources.kv_cache"),
            ("offline.json", "/1/output/crf", "generate.output.crf"),
        ] {
            let mut config = shipped(file);
            remove(&mut config, pointer);
            assert!(message(&config).contains(path), "{pointer}: {}", message(&config));
        }
    }

    /// export.kernels.lora_sum: how the W4A4 low-rank branch adds its column blocks, named in every recipe (atomic,
    /// Nunchaku's own, or ordered, the same bits on every run); ordered exists only for the W4A4 linear
    #[test]
    fn the_low_rank_sum_is_named_and_ordered_only_for_w4a4() {
        let mut config = shipped("mvp1.json");
        remove(&mut config, "/0/kernels/lora_sum");
        assert!(message(&config).contains("export.kernels.lora_sum"), "{}", message(&config));
        for sum in ["atomic", "ordered"] {
            config[0]["kernels"]["lora_sum"] = json!(sum);
            assert_eq!(crate::config::Config::parse(&config.to_string()).unwrap().value()[0]["kernels"]["lora_sum"], json!(sum));
        }
        config[0]["kernels"]["lora_sum"] = json!("fast");
        assert!(crate::config::Config::parse(&config.to_string()).is_err());
        let mut dense = shipped("mvp1.json");
        dense[0]["kernels"]["linear"] = json!("torch_bf16");
        dense[0]["kernels"]["fuse"] = json!({"rope_into_sage": false, "qkv": false, "gelu_mlp": false});
        dense[0]["kernels"]["lora_sum"] = json!("ordered");
        let refused = crate::config::Config::parse(&dense.to_string()).expect_err("ordered without W4A4 must be refused");
        assert!(refused.to_string().contains("kernels.lora_sum"), "{refused}");
        dense[0]["kernels"]["lora_sum"] = json!("atomic");
        crate::config::Config::parse(&dense.to_string()).unwrap();
    }

    /// generate.cache_write: where a chunk's keys and values come from, named in every recipe: "clean_pass" (a pass
    /// at t = 0 over the clean chunk, as trained) or "last_step" (the sampler's last pass, no third pass)
    /// generate.kv_probe (debug kv_write only): timesteps in (0, 1000) whose keys and values are compared with the
    /// pass at t = 0; optional, never inserted
    #[test]
    fn kv_probe_timesteps_come_with_the_kv_write_comparison_only() {
        let mut config = shipped("mvp1.json");
        config[1]["debug"] = json!(["kv_write"]);
        config[1]["drain"] = json!({"every": "end"});
        config[1]["kv_probe"] = json!([600, 400, 250, 100]);
        crate::config::Config::parse(&config.to_string()).unwrap();
        for bad in [json!([0]), json!([1000]), json!("600"), json!([])] {
            config[1]["kv_probe"] = bad;
            assert!(crate::config::Config::parse(&config.to_string()).is_err());
        }
        config[1]["kv_probe"] = json!([600]);
        config[1]["debug"] = json!(["fps"]);
        assert!(crate::config::Config::parse(&config.to_string()).is_err(), "probes without debug kv_write");
    }

    #[test]
    fn the_cache_write_is_named_and_one_of_two() {
        let mut config = shipped("mvp1.json");
        remove(&mut config, "/1/cache_write");
        assert!(message(&config).contains("generate.cache_write"), "{}", message(&config));
        for mode in ["clean_pass", "last_step"] {
            config[1]["cache_write"] = json!(mode);
            assert_eq!(crate::config::Config::parse(&config.to_string()).unwrap().value()[1]["cache_write"], json!(mode));
        }
        config[1]["cache_write"] = json!("both");
        assert!(crate::config::Config::parse(&config.to_string()).is_err());
        // clean_pass_sink: the pass at t = 0 attends to the sink and the chunk only, through the frame mask of the
        // block-sparse packages, so only with kernels.sparse on
        config[1]["cache_write"] = json!("clean_pass_sink");
        let refused = crate::config::Config::parse(&config.to_string()).expect_err("no mask input without sparsity");
        assert!(refused.to_string().contains("clean_pass_sink"), "{refused}");
        config[0]["kernels"]["attention"] = json!("sage_sm89");
        config[0]["kernels"]["rope"] = json!("longlive_triton");
        config[0]["kernels"]["fuse"]["rope_into_sage"] = json!(true);
        config[0]["kernels"]["sparse"] = json!({"enabled": true, "skip_mass": 0.0, "frames": {"sink": true, "recent_frames": 8}});
        config[0]["context"]["window_frames"] = json!(24);
        crate::config::Config::parse(&config.to_string()).unwrap();
    }

    #[test]
    fn a_branch_is_required_while_its_switch_is_on() {
        let mut config = shipped("mvp1.json");
        config[0]["rope"]["interpolate"] = json!(true);
        for key in ["stretch", "by_parts", "ntk_aware"] {
            let mut missing = config.clone();
            if missing[0]["rope"].get(key).is_some() { remove(&mut missing, &format!("/0/rope/{key}")); }
            else { missing[0]["rope"]["stretch"] = json!(1.0); missing[0]["rope"]["by_parts"] = json!(null); missing[0]["rope"]["ntk_aware"] = json!(false); remove(&mut missing, &format!("/0/rope/{key}")); }
            assert!(message(&missing).contains(&format!("export.rope.{key} is missing")), "{key}");
        }
        let mut offline = shipped("offline.json");
        remove(&mut offline, "/1/output");
        assert!(message(&offline).contains("generate.output is missing"));
        let mut diagnostics = shipped("mvp1.json");
        diagnostics[1]["debug"] = json!("fps");
        if diagnostics[1].get("drain").is_some() { remove(&mut diagnostics, "/1/drain"); }
        assert!(message(&diagnostics).contains("generate.drain is missing"));
    }

    #[test]
    fn a_branch_key_is_an_error_while_its_switch_is_off() {
        let mut rope = shipped("mvp1.json");
        rope[0]["rope"]["interpolate"] = json!(false);
        rope[0]["rope"]["stretch"] = json!(1.1);
        assert!(message(&rope).contains("export.rope.stretch is set while export.rope.interpolate is off"));
        let mut offloading = shipped("mvp1.json");
        offloading[1]["offloading"]["enabled"] = json!(false);
        assert!(message(&offloading).contains("is set while generate.offloading.enabled is off"));
        let mut live = shipped("mvp1.json");
        live[1]["output"] = json!({"directory": "x", "codec": "libx264", "crf": 16, "pixel_format": "yuv420p"});
        assert!(message(&live).contains("generate.output is set while generate.mode offline is off"));
        let mut quiet = shipped("mvp1.json");
        quiet[1]["debug"] = json!(false);
        quiet[1]["drain"] = json!({"every": "5s"});
        assert!(message(&quiet).contains("generate.drain is set while generate.debug is off"));
    }

    #[test]
    fn an_off_switch_needs_none_of_its_branch() {
        let mut config = shipped("mvp1.json");
        config[0]["rope"]["interpolate"] = json!(false);
        for key in ["stretch", "by_parts", "ntk_aware"] {
            if config[0]["rope"].get(key).is_some() { remove(&mut config, &format!("/0/rope/{key}")); }
        }
        config[1]["debug"] = json!(false);
        if config[1].get("drain").is_some() { remove(&mut config, "/1/drain"); }
        config[1]["offloading"] = json!({"enabled": false});
        check(&config).unwrap();
    }

    #[test]
    fn an_unknown_key_is_refused() {
        let mut config = shipped("mvp1.json");
        config[1]["offloading"]["copy_sparse"] = json!(1.1);
        assert!(message(&config).contains("copy_sparse is not a key of this object"));
    }

    #[test]
    fn every_shipped_configuration_satisfies_the_contract() {
        fn walk(dir: &std::path::Path, found: &mut Vec<std::path::PathBuf>) {
            for entry in std::fs::read_dir(dir).unwrap().flatten() {
                let path = entry.path();
                if path.is_dir() { walk(&path, found) } else if path.extension().is_some_and(|e| e == "json") { found.push(path) }
            }
        }
        let mut files = vec![];
        walk(&crate::root().join("configs"), &mut files);
        let mut checked = 0;
        for path in files {
            let value: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
            // prompt files are objects; a pipeline configuration is an array of two components
            if value.as_array().is_some_and(|a| a.iter().any(|c| c.get("component").is_some())) {
                if let Err(error) = check(&value) { panic!("{}: {error}", path.display()); }
                checked += 1;
            }
        }
        assert!(checked >= 2, "only {checked} configurations found");
    }

    #[test]
    fn a_parsed_configuration_written_out_again_still_satisfies_the_contract() {
        for file in ["mvp1.json", "offline.json"] {
            let text = std::fs::read_to_string(crate::root().join("configs").join(file)).unwrap();
            let again = crate::config::Config::parse(&text).unwrap().value();
            check(&again).unwrap_or_else(|error| panic!("{file}: {error}"));
            assert_eq!(crate::config::Config::parse(&again.to_string()).unwrap().value(), again);
        }
    }

    #[test]
    fn invalid_pipelines_are_refused_before_anything_starts() {
        // what the Python loader used to refuse: it no longer checks, the host does, once, first
        let refuse: Vec<(&str, Box<dyn Fn(&mut Value)>)> = vec![
            ("reversed stages", Box::new(|c| c.as_array_mut().unwrap().reverse())),
            ("a third stage", Box::new(|c| { let first = c[0].clone(); c.as_array_mut().unwrap().push(first) })),
            ("packages_from absent", Box::new(|c| c[1]["packages_from"] = json!("absent"))),
            ("an unknown offloading key", Box::new(|c| c[1]["offloading"]["recalibrate"] = json!(true))),
            ("an unknown calibration policy", Box::new(|c| c[1]["offloading"]["calibration"] = json!("unknown"))),
            ("an initial prompt in a script", Box::new(|c| c[1]["prompts"] = json!({"source": "script", "file": "a.json", "chains": [], "initial": "wrong"}))),
            ("three steps", Box::new(|c| c[0]["sampling"]["steps"] = json!(3))),
            ("a chunk of no frames", Box::new(|c| c[0]["context"]["chunk_frames"] = json!(0))),
            ("an unknown text precision", Box::new(|c| c[0]["model"]["text_encoder"]["precision"] = json!("bf16"))),
            ("a negative RAM margin", Box::new(|c| c[1]["offloading"]["ram_margin_fraction"] = json!(-0.1))),
            ("a whole RAM margin", Box::new(|c| c[1]["offloading"]["ram_margin_fraction"] = json!(1))),
            ("a RAM margin that is a string", Box::new(|c| c[1]["offloading"]["ram_margin_fraction"] = json!("x"))),
            ("a negative prefetch tolerance", Box::new(|c| c[1]["offloading"]["prefetch_fps_tolerance"] = json!(-0.1))),
            ("a stretch that is a boolean", Box::new(|c| { c[0]["rope"]["interpolate"] = json!(true); c[0]["rope"]["stretch"] = json!(true) })),
            ("an ntk_aware that is a string", Box::new(|c| { c[0]["rope"]["interpolate"] = json!(true); c[0]["rope"]["ntk_aware"] = json!("true") })),
            ("by_parts with alpha above beta", Box::new(|c| { c[0]["rope"]["interpolate"] = json!(true); c[0]["rope"]["by_parts"] = json!({"alpha": 2, "beta": 1}) })),
            ("by_parts and ntk_aware together", Box::new(|c| { c[0]["rope"]["interpolate"] = json!(true); c[0]["rope"]["by_parts"] = json!({"alpha": 0.25, "beta": 2}); c[0]["rope"]["ntk_aware"] = json!(true) })),
        ];
        for (what, change) in refuse {
            let mut config = shipped("mvp1.json");
            // the rope branch of the cases above needs its three keys, whatever the file had
            config[0]["rope"]["interpolate"] = json!(true);
            for (key, value) in [("stretch", json!(1.0)), ("by_parts", json!(null)), ("ntk_aware", json!(false))] {
                config[0]["rope"][key] = value;
            }
            check(&config).unwrap_or_else(|error| panic!("the base of {what}: {error}"));
            change(&mut config);
            assert!(crate::config::Config::parse(&config.to_string()).is_err(), "{what} was accepted");
        }
    }

    #[test]
    fn frequency_aware_profile_roundtrips_without_changing_legacy_recipes() {
        let mut config = shipped("mvp1.json");
        config[0]["rope"]["interpolate"] = json!(true);
        config[0]["rope"]["stretch"] = json!(0.3);
        config[0]["rope"]["by_parts"] = Value::Null;
        config[0]["rope"]["ntk_aware"] = json!(false);
        config[0]["rope"].as_object_mut().unwrap().remove("frequency_aware");
        let legacy = crate::config::Config::parse(&config.to_string()).unwrap().value();
        assert!(legacy[0]["rope"].get("frequency_aware").is_none());
        config[0]["rope"]["frequency_aware"] = json!(true);
        let parsed = crate::config::Config::parse(&config.to_string()).unwrap().value();
        assert_eq!(parsed[0]["rope"]["frequency_aware"], json!(true));
        crate::config::Config::parse(&parsed.to_string()).unwrap();
        for other in ["ntk_aware", "by_parts"] {
            let mut invalid = config.clone();
            invalid[0]["rope"][other] = if other == "by_parts" { json!({"alpha":0.25,"beta":2}) } else { json!(true) };
            assert!(crate::config::Config::parse(&invalid.to_string()).is_err());
        }
        config[0]["rope"]["frequency_aware"] = json!("true");
        assert!(crate::config::Config::parse(&config.to_string()).is_err());
        config[0]["rope"]["frequency_aware"] = json!(true);
        config[0]["rope"]["interpolate"] = json!(false);
        assert!(message(&config).contains("frequency_aware"));
    }

    /// offloading.policy: optional (old configs keep the greedy_ram rule), greedy_ram or greedy_vram, never inserted
    #[test]
    fn the_placement_policy_is_optional_and_named() {
        let mut config = shipped("mvp1.json");
        assert!(crate::config::Config::parse(&config.to_string()).unwrap().value()[1]["offloading"].get("policy").is_none());
        // the default is written out of the normalized form, as if absent
        for (policy, normalized) in [("greedy_ram", Value::Null), ("greedy_vram", json!("greedy_vram"))] {
            config[1]["offloading"]["policy"] = json!(policy);
            let parsed = crate::config::Config::parse(&config.to_string()).unwrap().value();
            assert_eq!(parsed[1]["offloading"]["policy"], normalized);
            crate::config::Config::parse(&parsed.to_string()).unwrap();
        }
        config[1]["offloading"]["policy"] = json!("greedy_disk");
        assert!(crate::config::Config::parse(&config.to_string()).is_err());
    }

    /// generate.descent (the knobs of the search in resolution): optional, never inserted, both keys when present;
    /// the noise and the schedule are generate.sampling's now
    #[test]
    fn the_descent_section_keeps_only_the_search_keys() {
        let mut config = shipped("mvp1.json");
        assert!(crate::config::Config::parse(&config.to_string()).unwrap().value()[1].get("descent").is_none());
        config[1]["descent"] = json!({"attn_logit_scale": [[1.0, 0.9], 0.8], "reference": null});
        let parsed = crate::config::Config::parse(&config.to_string()).unwrap().value();
        assert_eq!(parsed[1]["descent"]["attn_logit_scale"], json!([[1.0, 0.9], 0.8]));
        crate::config::Config::parse(&parsed.to_string()).unwrap();
        for key in ["attn_logit_scale", "reference"] {
            let mut missing = config.clone();
            missing[1]["descent"].as_object_mut().unwrap().remove(key);
            assert!(message(&missing).contains(&format!("generate.descent.{key}")), "{key}");
        }
        config[1]["descent"]["extra"] = json!(1);
        assert!(message(&config).contains("generate.descent.extra"));
        config[1]["descent"].as_object_mut().unwrap().remove("extra");
        // the two keys that moved are refused with the place they moved to
        for (key, value, to) in [("noise", json!("fresh"), "generate.sampling.noise"), ("timestep_shift", json!(null), "generate.sampling.schedule")] {
            let mut old = config.clone();
            old[1]["descent"][key] = value;
            let text = message(&old);
            assert!(text.contains(&format!("generate.descent.{key}")) && text.contains(to), "{key}: {text}");
        }
    }

    /// generate.sampling: how the descent starts and walks, named in every recipe: the noise and the schedule, each
    /// with all its keys
    #[test]
    fn the_sampling_section_is_required_and_complete() {
        let mut config = shipped("mvp1.json");
        for (pointer, path) in [("/1/sampling", "generate.sampling"), ("/1/sampling/noise", "generate.sampling.noise"),
                                ("/1/sampling/noise/mode", "generate.sampling.noise.mode"), ("/1/sampling/schedule", "generate.sampling.schedule"),
                                ("/1/sampling/schedule/timestep_shift", "generate.sampling.schedule.timestep_shift")] {
            let mut missing = config.clone();
            remove(&mut missing, pointer);
            assert!(message(&missing).contains(path), "{pointer}: {}", message(&missing));
        }
        config[1]["sampling"]["extra"] = json!(1);
        assert!(message(&config).contains("generate.sampling.extra"));
        config[1]["sampling"].as_object_mut().unwrap().remove("extra");
        config[1]["sampling"]["schedule"]["extra"] = json!(1);
        assert!(message(&config).contains("generate.sampling.schedule.extra"));
    }

    /// the noise modes: fresh needs no other key, the two native ones need the native size and no mode may carry
    /// the key of another
    #[test]
    fn a_native_noise_names_its_native_size_and_a_fresh_one_has_none() {
        let mut config = shipped("mvp1.json");
        config[1]["sampling"]["noise"] = json!({"mode": "fresh"});
        crate::config::Config::parse(&config.to_string()).unwrap();
        config[1]["sampling"]["noise"] = json!({"mode": "fresh", "native": "704x1280"});
        let text = message(&config);
        assert!(text.contains("generate.sampling.noise.native") && text.contains("remove it"), "{text}");
        for mode in ["native_resized", "native_spectral"] {
            config[1]["sampling"]["noise"] = json!({"mode": mode});
            let text = message(&config);
            assert!(text.contains("generate.sampling.noise.native") && text.contains("missing"), "{mode}: {text}");
            config[1]["sampling"]["noise"] = json!({"mode": mode, "native": "704x1280"});
            let parsed = crate::config::Config::parse(&config.to_string()).unwrap().value();
            assert_eq!(parsed[1]["sampling"]["noise"], json!({"mode": mode, "native": "704x1280"}));
            crate::config::Config::parse(&parsed.to_string()).unwrap();
            for bad in [json!("704"), json!("0x1280"), json!("700x1280"), json!(704), json!("704x1280x1")] {
                config[1]["sampling"]["noise"] = json!({"mode": mode, "native": bad});
                assert!(crate::config::Config::parse(&config.to_string()).is_err(), "{mode} {bad}");
            }
        }
        config[1]["sampling"]["noise"] = json!({"mode": "other"});
        assert!(crate::config::Config::parse(&config.to_string()).is_err());
    }

    /// the schedule's timestep shift: null keeps the packages' sampler, a number in (0, 20] recomputes it
    #[test]
    fn the_schedule_shift_is_null_or_a_number_in_range() {
        let mut config = shipped("mvp1.json");
        for good in [json!(null), json!(3.63), json!(20.0)] {
            config[1]["sampling"]["schedule"]["timestep_shift"] = good.clone();
            let parsed = crate::config::Config::parse(&config.to_string()).unwrap().value();
            assert_eq!(parsed[1]["sampling"]["schedule"]["timestep_shift"], good);
        }
        for bad in [json!(0.0), json!(-1.0), json!(25.0), json!("3.6")] {
            config[1]["sampling"]["schedule"]["timestep_shift"] = bad.clone();
            assert!(crate::config::Config::parse(&config.to_string()).is_err(), "{bad}");
        }
    }

    /// the parsed sampling is what the file says: the grid of the native latent (a sixteenth of the size) for the
    /// native noises, none for the fresh one
    #[test]
    fn the_parsed_noise_gives_the_native_latent_grid() {
        use crate::config::{Config, Noise};
        let mut config = shipped("mvp1.json");
        for (noise, want) in [(json!({"mode": "fresh"}), None),
                              (json!({"mode": "native_resized", "native": "704x1280"}), Some((44, 80, false))),
                              (json!({"mode": "native_spectral", "native": "704x1280"}), Some((44, 80, true)))] {
            config[1]["sampling"]["noise"] = noise;
            let parsed = Config::parse(&config.to_string()).unwrap();
            let noise: &Noise = &parsed.generate.sampling.noise;
            assert_eq!(noise.native_grid().unwrap(), want);
        }
    }

    #[test]
    fn attention_logit_scale_is_optional_positive_and_needs_interpolation() {
        let mut config = shipped("mvp1.json");
        config[0]["rope"]["interpolate"] = json!(true);
        config[0]["rope"]["stretch"] = json!(0.3);
        config[0]["rope"]["by_parts"] = Value::Null;
        config[0]["rope"]["ntk_aware"] = json!(true);
        config[0]["rope"].as_object_mut().unwrap().remove("attn_logit_scale");
        let legacy = crate::config::Config::parse(&config.to_string()).unwrap().value();
        assert!(legacy[0]["rope"].get("attn_logit_scale").is_none());
        config[0]["rope"]["attn_logit_scale"] = json!(0.93);
        let parsed = crate::config::Config::parse(&config.to_string()).unwrap().value();
        assert_eq!(parsed[0]["rope"]["attn_logit_scale"], json!(0.93));
        crate::config::Config::parse(&parsed.to_string()).unwrap();
        for bad in [json!(0.0), json!(-1.0), json!("0.93")] {
            let mut invalid = config.clone();
            invalid[0]["rope"]["attn_logit_scale"] = bad;
            assert!(crate::config::Config::parse(&invalid.to_string()).is_err());
        }
        config[0]["rope"]["interpolate"] = json!(false);
        assert!(message(&config).contains("attn_logit_scale"));
    }

    #[test]
    fn block_sparsity_is_optional_joint_only_and_needs_its_keys_while_enabled() {
        let parse = |c: &Value| crate::config::Config::parse(&c.to_string());
        let mut config = shipped("mvp1.json");
        config[0]["kernels"]["fuse"]["rope_into_sage"] = json!(true);
        config[0]["kernels"].as_object_mut().unwrap().remove("sparse");
        assert!(parse(&config).unwrap().value()[0]["kernels"].get("sparse").is_none());
        config[0]["kernels"]["sparse"] = json!({"enabled": false});
        assert_eq!(parse(&config).unwrap().value()[0]["kernels"]["sparse"], json!({"enabled": false}));
        config[0]["kernels"]["sparse"] = json!({"enabled": false, "skip_mass": 0.1});
        assert!(message(&config).contains("export.kernels.sparse.skip_mass is set while export.kernels.sparse.enabled is off"));
        let on = json!({"enabled": true, "skip_mass": 0.1, "frames": {"sink": true, "recent_frames": 8}});
        config[0]["kernels"]["sparse"] = on.clone();
        let parsed = parse(&config).unwrap();
        assert_eq!(parsed.value()[0]["kernels"]["sparse"], on);
        assert_eq!(parse(&parsed.value()).unwrap(), parsed);
        config[0]["kernels"]["sparse"] = json!({"enabled": true, "skip_mass": 0.1});
        assert!(message(&config).contains("export.kernels.sparse.frames is missing while export.kernels.sparse.enabled is on"));
        for (what, bad) in [
            ("a negative skip mass", json!({"enabled": true, "skip_mass": -0.1, "frames": {"sink": true, "recent_frames": 8}})),
            ("a whole skip mass", json!({"enabled": true, "skip_mass": 1.0, "frames": {"sink": true, "recent_frames": 8}})),
            ("a skip mass that is a string", json!({"enabled": true, "skip_mass": "0.1", "frames": {"sink": true, "recent_frames": 8}})),
            ("more recent frames than the ring holds", json!({"enabled": true, "skip_mass": 0.0, "frames": {"sink": true, "recent_frames": 17}})),
            ("negative recent frames", json!({"enabled": true, "skip_mass": 0.0, "frames": {"sink": true, "recent_frames": -1}})),
            ("an unknown frames key", json!({"enabled": true, "skip_mass": 0.0, "frames": {"sink": true, "recent_frames": 8, "local": 2}})),
        ] {
            config[0]["kernels"]["sparse"] = bad;
            assert!(parse(&config).is_err(), "{what} was accepted");
        }
        config[0]["kernels"]["sparse"] = on;
        config[0]["kernels"]["fuse"]["rope_into_sage"] = json!(false);
        assert!(message(&config).contains("export.kernels.sparse needs export.kernels.fuse.rope_into_sage"));
    }

    #[test]
    fn every_fusion_is_an_explicit_switch_of_its_own() {
        let config = shipped("mvp1.json");
        let mut missing = config.clone();
        missing[0]["kernels"].as_object_mut().unwrap().remove("fuse");
        assert!(message(&missing).contains("export.kernels.fuse is missing"));
        for flag in ["rope_into_sage", "qkv", "gelu_mlp"] {
            let mut bad = config.clone();
            bad[0]["kernels"]["fuse"].as_object_mut().unwrap().remove(flag);
            assert!(message(&bad).contains(&format!("export.kernels.fuse.{flag} is missing")), "{flag}");
            bad[0]["kernels"]["fuse"][flag] = json!("yes");
            assert!(message(&bad).contains(&format!("export.kernels.fuse.{flag} must be true or false")), "{flag}");
        }
        let mut extra = config.clone();
        extra[0]["kernels"]["fuse"]["kv_store"] = json!(true);
        assert!(message(&extra).contains("export.kernels.fuse.kv_store is not a key of this object"));
        let mut rope = config;
        rope[0]["kernels"].as_object_mut().unwrap().remove("rope");
        assert!(message(&rope).contains("export.kernels.rope is missing"), "rope is named even when fused");
    }

    #[test]
    fn timestep_shift_is_a_positive_finite_number() {
        let mut config = shipped("mvp1.json");
        for ok in [json!(5.0), json!(3.63)] {
            config[0]["sampling"]["timestep_shift"] = ok;
            crate::config::Config::parse(&config.to_string()).unwrap();
        }
        for bad in [json!(0.0), json!(-5.0), json!("5"), json!(100.0)] {
            config[0]["sampling"]["timestep_shift"] = bad;
            assert!(crate::config::Config::parse(&config.to_string()).is_err());
        }
    }

    #[test]
    fn the_configuration_parser_runs_the_contract_first() {
        let mut config = shipped("mvp1.json");
        remove(&mut config, "/1/offloading/copy_spare");
        let error = crate::config::Config::parse(&config.to_string()).unwrap_err().to_string();
        assert!(error.contains("generate.offloading.copy_spare is missing"), "{error}");
    }
}
