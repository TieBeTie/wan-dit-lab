"""The original baseline must not silently become our optimized pipeline."""
from scripts.checks.check_original_ready import blockers

def test_missing_official_checkpoint_blocks_run_even_when_gpu_is_large():
    issues = blockers(checkpoint_exists=False, gpu_bytes=80 << 30, generator_bytes=10_000_000_000)
    assert any("checkpoint" in s for s in issues)

def test_dense_weights_that_do_not_fit_cannot_be_marked_ready():
    issues = blockers(checkpoint_exists=True, gpu_bytes=8 << 30, generator_bytes=9_999_575_424)
    assert any("VRAM" in s for s in issues)

def test_weight_fit_is_only_a_prerequisite_not_a_full_memory_estimate():
    assert blockers(checkpoint_exists=True, gpu_bytes=80 << 30, generator_bytes=10_000_000_000) == []

def test_cpu_offloading_does_not_require_all_weights_to_fit_vram():
    assert blockers(checkpoint_exists=True, gpu_bytes=8 << 30,
                    generator_bytes=10_000_000_000, offloaded=True) == []
