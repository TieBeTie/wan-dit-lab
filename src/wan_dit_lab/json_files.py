"""JSON files of configs and results, one way for every script."""

import json
import os
import time
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    """Whole or not at all: written beside and then replaced. Windows readers and indexers can briefly hold the
    destination open, so the replacement is retried instead of losing the run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for attempt in range(30):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(min(0.05 * 2**attempt, 0.5))
