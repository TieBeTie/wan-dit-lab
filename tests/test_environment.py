import pytest
import torch

from wan_dit_lab.cli import environment


def test_environment_reports_torch():
    assert environment()["packages"]["torch"] == torch.__version__


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_matmul_against_cpu():
    # Exactly representable integer inputs separate setup failures from tolerances.
    x = torch.arange(32, dtype=torch.float32).reshape(4, 8) - 16
    w = torch.arange(24, dtype=torch.float32).reshape(8, 3) - 12
    actual = (x.cuda().half() @ w.cuda().half()).float().cpu()
    torch.testing.assert_close(actual, x @ w, rtol=0, atol=0)
