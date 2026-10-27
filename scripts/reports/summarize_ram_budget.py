"""Report completed RAM-budget runs without inferring results from incomplete runs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import argparse
import csv
import json
import statistics


def report(root):
    rows = []
    expected = None
    for name in ("calibration", "median_1g", "guarded_1g", "median_05g", "median_025g"):
        folder = root / name
        if not (folder / "complete.json").exists():
            continue
        record = json.loads((folder / "record.json").read_text())
        hashes = [(r["latent_hash"], r["decoded_hash"]) for r in record["chunks"]]
        expected = hashes if expected is None else expected
        if hashes != expected:
            raise RuntimeError(f"Numerical mismatch: {name}")
        events = record.get("tensor_resource_events", [])
        deliveries = [r["delivery_s"] for r in record["chunks"] if r["delivery_s"] is not None]
        rows.append(dict(method=name, chunks=len(hashes), bitwise=True,
                         full_run_s=record["full_run_s"],
                         chunk_median_s=statistics.median(deliveries) if deliveries else None,
                         ram_rss_peak_gib=max(r["rss"] for r in record["memory_samples"]) / 2**30,
                         vram_tensor_peak_gib=max(r["peak_allocated"] for r in record["memory_samples"]) / 2**30,
                         ram_cache_peak_gib=record["backing_peak_bytes"]["ram"] / 2**30,
                         disk_read_logical_gib=sum(r["bytes"] for r in events if r["event"] == "read") / 2**30,
                         disk_write_logical_gib=sum(r["bytes"] for r in events if r["event"] == "write") / 2**30,
                         read_wall_s=sum(r["duration_s"] for r in events if r["event"] == "read"),
                         write_wall_s=sum(r["duration_s"] for r in events if r["event"] == "write")))
    (root / "comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        with (root / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(report(args.directory), indent=2))
