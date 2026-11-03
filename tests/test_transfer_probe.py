import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA transfer measurements')
def test_probe_counts_real_transfers_without_changing_values():
    from scripts.experiments.transfer_probe import TransferProbe
    source = torch.arange(16, dtype=torch.float32)
    with TransferProbe() as probe:
        gpu = source.to('cuda')
        gpu = gpu.to(dtype=torch.float32)
        result = torch.empty_like(source)
        result.copy_(gpu)
    report = probe.report()
    assert torch.equal(result, source)
    assert report['copies'] == 2
    assert report['bytes'] == source.numel() * source.element_size() * 2
    assert report['host_blocked_seconds'] >= 0
    assert report['copy_stream_seconds'] >= 0
