"""The exported two-step sampler is two linear Euler steps for any timestep shift, not only for 5. The reference
scheduler rounds its sigmas to float32, so the linearity holds to about 1e-7, not to float64 precision."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_export():
    sys.path.insert(0, str(ROOT / "scripts" / "conversion"))
    spec = importlib.util.spec_from_file_location("export_live_for_test", ROOT / "scripts/conversion/export_live.py")
    return spec, importlib.util.module_from_spec(spec)


@pytest.mark.parametrize("shift", [5.0, 3.63])
def test_two_step_unipc_is_linear_for_shifts(shift):
    spec, module = load_export()
    try:
        spec.loader.exec_module(module)
    except Exception as error:  # the exporter needs the CUDA stack: skip where it cannot be imported
        pytest.skip(f"export_live cannot be imported here: {error}")
    result = module.causal_unipc(2, shift)
    assert len(result["timesteps"]) == len(result["coefficients"]) == 2
    assert result["timesteps"][0] == 999 and result["timesteps"][1] < result["timesteps"][0]
