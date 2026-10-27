
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import argparse
import json
import time

from wan_dit_lab.offloading.resource_budget_model import predict_budgets

parser = argparse.ArgumentParser()
parser.add_argument("reference", type=Path)
parser.add_argument("calibration", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
started = time.perf_counter()
record = json.loads(args.reference.read_text(encoding="utf-8"))
calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
result = predict_budgets(record, calibration, [m * 2**20 for m in range(64, 2049, 64)])
result["elapsed_s"] = time.perf_counter() - started
args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({key: result[key] for key in ("selected_cache_bytes", "elapsed_s", "intervals")}))
for row in result["rows"]:
    print(round(row["cache_bytes"] / 2**20), round(row["predicted_mean_chunk_s"], 3),
          round(row["predicted_slowdown"] * 100, 2), row["eligible"])
