"""One-time migration of every pipeline configuration to the strict contract (crates/wan/src/contract.rs).

Every key the old formats filled in silently is written out with the value it had, so no behaviour changes; the keys
of a switched-off branch are removed. Idempotent: a migrated file comes out unchanged.

    .venv\\Scripts\\python.exe scripts\\conversion\\migrate_configs.py [--check]
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def migrate(config):
    export, generate = config
    export["model"]["text_encoder"].setdefault("precision", "int8")
    rope = export["rope"]
    branch = ("stretch", "by_parts", "ntk_aware")
    if rope["interpolate"]:
        rope.setdefault("stretch", 1.0)
        rope.setdefault("by_parts", None)
        rope.setdefault("ntk_aware", False)
    else:
        for key in branch:
            rope.pop(key, None)
    generate.setdefault("debug", False)
    if generate["debug"] is False:
        generate.pop("drain", None)
    else:
        generate.setdefault("drain", {"every": "5s"})
    if generate["mode"] != "offline":
        generate.pop("output", None)
    prompts = generate["prompts"]
    if prompts["source"] == "interactive":
        prompts.setdefault("initial", None)
    else:
        prompts.setdefault("chains", [])
    off = generate["offloading"]
    if not off["enabled"]:
        for key in [k for k in off if k != "enabled"]:
            del off[key]
        return config
    off.setdefault("prefetch_fps_tolerance", 0.5)
    off.setdefault("ram_margin_fraction", 0.0)
    if "vram_margin_gib" not in off and "vram_margin_fraction" not in off:
        off["vram_margin_gib"] = 1.5
    t5 = off["t5"]
    t5.setdefault("prefetch_tolerance", 0.0)
    if "resident" not in t5 and "resident_gib" not in t5:
        t5["resident_gib"] = 1.0
    if "activations" not in t5 and "t5_activations_gib" not in off:
        off["t5_activations_gib"] = 0.33
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="only list the files that would change")
    args = parser.parse_args()
    changed = 0
    for path in sorted((ROOT / "configs").rglob("*.json")):
        text = path.read_text(encoding="utf-8")
        value = json.loads(text)
        if not (isinstance(value, list) and len(value) == 2 and all(isinstance(c, dict) and "component" in c for c in value)):
            continue
        migrated = migrate(json.loads(text))
        if migrated == value:
            continue
        changed += 1
        print(("would change " if args.check else "migrated ") + str(path.relative_to(ROOT)))
        if not args.check:
            path.write_text(json.dumps(migrated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{changed} files")


if __name__ == "__main__":
    main()
