"""The height and width RoPE tables wan.exe computes (crates/wan/src/rope.rs) are checked there against these files:
the table of live_presets.rope_table for every profile, float32 [2, rows, 42] little-endian, rows = the longer side in
tokens. Regenerate with WAN_REGENERATE_ROPE_FIXTURE=1 after a change of rope_table."""
import json
import math
import os
from pathlib import Path

import numpy as np

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "rope_hw"
S = math.log(1.2) / math.log(22 / 19)
CASES = {
    "native": ("608x1120", {"interpolate": False}),
    "pi": ("608x1120", {"interpolate": True, "stretch": S, "ntk_aware": False, "by_parts": None}),
    "ntk": ("608x1120", {"interpolate": True, "stretch": S, "ntk_aware": True, "by_parts": None}),
    "by_parts": ("608x1120", {"interpolate": True, "stretch": S, "by_parts": {"alpha": .25, "beta": 2}}),
    "frequency_aware": ("608x1120", {"interpolate": True, "stretch": .3, "frequency_aware": True, "by_parts": None}),
    "m_axes": ("608x1120", {"interpolate": True, "stretch": None, "m": {"height": 1.2, "width": 1.4}, "ntk_aware": True}),
    "mix": ("608x1120", {"interpolate": True, "stretch": None, "m": {"height": 1.2, "width": 1.2}, "ntk_aware": True,
                         "ntk_pi_mix": .5}),
    "logit_scale": ("608x1120", {"interpolate": True, "stretch": S, "ntk_aware": True, "attn_logit_scale": .9}),
    "ntk_576": ("576x1056", {"interpolate": True, "stretch": .7, "ntk_aware": True, "by_parts": None}),
}


def table(size, rope):
    from wan_dit_lab.runtime.live_presets import rope_table
    height, width = map(int, size.split("x"))
    spec = dict(height=height, width=width, native=(704, 1280), vae_stride=16, rope_dtype="float32",
                rope_interpolate=rope["interpolate"], rope_stretch=rope.get("stretch") or 1.0,
                rope_by_parts=rope.get("by_parts"), rope_ntk_aware=bool(rope.get("ntk_aware")),
                rope_frequency_aware=bool(rope.get("frequency_aware")), rope_m=rope.get("m"),
                rope_ntk_pi_mix=rope.get("ntk_pi_mix"), attn_logit_scale=rope.get("attn_logit_scale", 1.0))
    rows = max(height, width) // 32
    return rope_table(spec, 128, "cpu")[:, :rows, 22:].contiguous().numpy()


def test_the_rust_fixture_is_the_table_of_rope_table():
    if os.environ.get("WAN_REGENERATE_ROPE_FIXTURE") == "1":
        FIXTURE.mkdir(parents=True, exist_ok=True)
        for name, (size, rope) in CASES.items():
            table(size, rope).astype("<f4").tofile(FIXTURE / f"{name}.bin")
        (FIXTURE / "cases.json").write_text(json.dumps({n: {"size": s, "rope": r} for n, (s, r) in CASES.items()},
                                                       indent=1), encoding="utf-8")
    cases = json.loads((FIXTURE / "cases.json").read_text(encoding="utf-8"))
    assert set(cases) == set(CASES)
    for name, (size, rope) in CASES.items():
        stored = np.fromfile(FIXTURE / f"{name}.bin", dtype="<f4")
        assert np.array_equal(stored, table(size, rope).reshape(-1)), name
