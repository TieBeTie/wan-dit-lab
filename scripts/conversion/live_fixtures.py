"""Fixtures of the Rust host's tests (native/live/tests/fixtures), made by the Python code it replaces.

    python scripts/conversion/live_fixtures.py

ring.json: the bookkeeping of causal.KVCache over a script of chunks, shots and new videos, at Live's proportions
(chunk 8 frames, window 32, sink 8) with 3 tokens a frame: after each commit where the chunk was written (at), the
tokens saved, and the ring indices its tokens went to (the chunk's token i to sink + slots[i] when the ring is full).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch

from wan_dit_lab.generation.causal import KVCache

OUT = ROOT / "native" / "live" / "tests" / "fixtures"


def ring():
    per, chunk, window, sink = 3, 8, 32, 8
    n = chunk * per
    script = ["chunk"] * 6 + ["shot"] + ["chunk"] * 4 + ["shot"] + ["chunk"] * 3 + ["new"] + ["chunk"] * 5 + \
        ["shot", "chunk", "shot"] + ["chunk"] * 4
    cache = KVCache((window - chunk) * per, sink * per, chunk=chunk)
    steps, pin_next = [], False
    for op in script:
        if op == "shot":
            pin_next = True
        elif op == "new":
            cache.reset()
            cache.last_slots = None
            pin_next = False
        else:
            K = torch.randn(1, n, 1, 4)
            cache.attend(K, K, K)
            at = cache.saved
            cache.commit()
            if pin_next:
                cache.pin_last()
            pin_next = False
            slots = None if cache.last_slots is None else cache.last_slots.tolist()
            steps.append(dict(at=at, saved=cache.saved, slots=slots))
    return dict(tokens_per_frame=per, chunk=chunk, window=window, sink=sink, script=script, commits=steps)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ring.json").write_text(json.dumps(ring()), encoding="utf-8")
    print(OUT / "ring.json")
