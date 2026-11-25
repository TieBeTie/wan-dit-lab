"""One call of a block package for Nsight Compute: block_<step> on the inputs of block_profile (full window for step 3),
burned in first, then exactly one call between cudaProfilerStart and cudaProfilerStop. Measurement only.

    ncu --profile-from-start off ... .venv\\Scripts\\python.exe -m wan_dit_lab.harness.ncu_block <package> [step] [calls]
"""

import sys
from pathlib import Path

from wan_dit_lab.harness import block_profile as bp
from wan_dit_lab.harness import bottleneck_bench as bb


def main():
    import torch
    from wan_dit_lab.kernels import operators
    operators.load()
    package = Path(sys.argv[1])
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    calls = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    m, inputs = bp.block_inputs(package, step)
    blk = bp.load(package, f"block_{step}")
    call = lambda: blk(*inputs)
    bb.burn(call, 3.0)
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(calls):
        call()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print("profiled", calls, "call(s) of block", step, flush=True)


if __name__ == "__main__":
    main()
