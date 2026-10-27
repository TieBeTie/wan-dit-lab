import pytest
import torch

from wan_dit_lab.harness.memory import inventory
from wan_dit_lab.offloading.offload import BlockOffloader
from wan_dit_lab.offloading.offload_trace import OffloadTrace


def test_inventory_deduplicates_parameter_views_and_categories():
    data = torch.zeros(20)
    records = inventory([("weights", "a", data[:10]), ("data", "b", data[10:])])
    assert len(records) == 1 and records[0]["bytes"] == 80


def test_inventory_includes_callable_encoder_and_decoder_owners():
    class Owner:
        def __init__(self):
            self.layer = torch.nn.Linear(2, 2)

        def __call__(self, x):
            return self.layer(x)

    rows = inventory([("weights", "encoder", Owner())])
    assert len(rows) == 2
    assert sum(row["bytes"] for row in rows) == 24


@pytest.mark.gpu
@pytest.mark.parametrize("ram,gpu", [(1, 1), (1, 3), (2, 2), (3, 1), (4, 4)])
def test_async_slots_and_trace_preserve_outputs(ram, gpu):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(2)
    blocks = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(4)])
    expected = torch.ones(1, 8)
    with torch.no_grad():
        for _ in range(3):
            for block in blocks:
                expected = block(expected)
        trace = OffloadTrace("test")
        offloader = BlockOffloader(blocks, gpu, "cuda", ram_window=ram, trace=trace)
        try:
            offloader.start()
            actual = torch.ones(1, 8, device="cuda")
            for _ in range(12):
                block = offloader.get_next()
                actual = block(actual)
                offloader.release(block)
            torch.cuda.synchronize()
            assert torch.allclose(actual.cpu(), expected, atol=1e-6)
            rows = trace.resolve()
            assert len(rows) == 12
            assert all(row["compute_s"] >= 0 and row["copy_s"] >= 0 for row in rows)
        finally:
            offloader.close()


@pytest.mark.gpu
def test_pack_worker_error_reaches_caller(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    offloader = BlockOffloader([torch.nn.Linear(2, 2)], 1, "cuda")

    def fail(*a, **k):
        raise ValueError("injected pack error")

    monkeypatch.setattr(torch, "cat", fail)
    try:
        with pytest.raises(RuntimeError, match="preparation failed"):
            offloader.start()
    finally:
        offloader.close()
