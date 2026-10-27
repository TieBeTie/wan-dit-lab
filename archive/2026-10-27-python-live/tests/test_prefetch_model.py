import importlib.util
from pathlib import Path


def model():
    path = Path(__file__).resolve().parents[1] / "benchmarks/bench_prefetch.py"
    spec = importlib.util.spec_from_file_location("bench_prefetch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.simulate


def test_overlap_and_slot_reuse():
    simulate = model()
    # One slot: next copy must wait for the preceding compute to release it.
    assert simulate([2] * 6, [3] * 6, [10] * 6, 1) == 80
    # Two slots hide the next preparation and copy behind compute.
    assert simulate([2] * 6, [3] * 6, [10] * 6, 2) == 65
    assert simulate([2] * 6, [3] * 6, [10] * 6, 4) == 65


def test_more_slots_do_not_remove_copy_bandwidth_limit():
    simulate = model()
    assert simulate([1] * 6, [20] * 6, [5] * 6, 2) == 126
    assert simulate([1] * 6, [20] * 6, [5] * 6, 4) == 126
