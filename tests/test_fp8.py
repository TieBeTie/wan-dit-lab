"""fp8.DynamicFp8Linear: W8A8 per tensor and per row against bfloat16, and a weight changed in place."""

import pytest
import torch

from wan_dit_lab.quantization.fp8 import DynamicFp8Linear


@pytest.mark.gpu
@pytest.mark.parametrize("rows", [False, True])
def test_close_to_bfloat16_and_follows_the_weight(rows):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0)
    linear = torch.nn.Linear(256, 512, device="cuda", dtype=torch.bfloat16)
    fp8 = DynamicFp8Linear(linear, rows)
    x = torch.randn(4, 96, 256, device="cuda", dtype=torch.bfloat16)
    for _ in range(2):  # a second block copied into the same weight, as BlockOffloader does
        ref = linear(x).float()
        err = (fp8(x).float() - ref).norm() / ref.norm()
        assert err < 0.06
        with torch.no_grad():
            linear.weight.copy_(torch.randn_like(linear.weight) * 0.05)
