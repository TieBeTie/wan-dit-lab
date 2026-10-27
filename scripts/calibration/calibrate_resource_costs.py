"""Merge real resource traces into one frozen median/guarded calibration input."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import argparse
import json

from wan_dit_lab.offloading.resource_cost import calibration_from_events


def merge(records):
    samples = {}
    for record in records:
        current = calibration_from_events(record["tensor_resource_events"])
        for row in record.get("dit_trace", []):
            current.setdefault(f"dit/block/{row['block']}", {}).setdefault(
                "compute", []).append(row["compute_s"])
        for name, operations in current.items():
            for operation, values in operations.items():
                samples.setdefault(name, {}).setdefault(operation, []).extend(values)
    return dict(format="resource_cost_v1", samples=samples,
                timing="end-to-end observed costs; filesystem cache state unknown")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("records", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = merge([json.loads(Path(path).read_text(encoding="utf-8")) for path in args.records])
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
