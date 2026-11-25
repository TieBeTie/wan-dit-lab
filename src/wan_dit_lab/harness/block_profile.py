"""Where the time of a block goes: a compiled block package (artifacts/live/<size>/block_<k>.pt2) run on the inputs its
export builds (text, cross, pre packages, a row of the weight pack, a random FP8 cache), its kernels timed with
torch.profiler and grouped into the parts of a block.

    .venv\\Scripts\\python.exe -m wan_dit_lab.harness.block_profile <package folder> [--step K] [--runs N]
"""

import argparse
import json
import time
from pathlib import Path

# (part, substrings of kernel names), first match wins; the names as torch.profiler reports them
PARTS = [
    ("self-attention: Sage kernel", ("qk_int_sv_f8_attn",)),
    ("self-attention: Sage preparation", ("_sum_max", "_quant_k", "_quant_v", "sage_sm89", "abs_amax")),
    ("self-attention: RoPE", ("_rope_apply",)),
    ("KV store", ("_store_kernel", "_store_rows")),
    ("linear W4A4: GEMM", ("gemm_w4a4_kernel",)),
    ("linear W4A4: activation quantization", ("quantize_w4a4",)),
    ("linear W4A4: ordered low-rank sum", ("sum_functor",)),
    ("cross-attention", ("fmha_cutlass", "efficient_attention")),
    ("norms, modulation, residuals", ("triton_",)),
]
RUNTIME = ("cuda", "cuLaunch", "Activity Buffer")


def part_of(kernel):
    """The part of a block a kernel belongs to, "other" when none matches, None for a runtime call (no kernel)."""
    if kernel.startswith(RUNTIME):
        return None
    for part, keys in PARTS:
        if any(k in kernel for k in keys):
            return part
    return "other"


def parts(kernel_ms):
    """[(part, ms, share)] from [(kernel, ms)], largest first; runtime calls left out."""
    totals = {}
    for kernel, ms in kernel_ms:
        part = part_of(kernel)
        if part is not None:
            totals[part] = totals.get(part, 0.0) + ms
    whole = sum(totals.values())
    return [(p, ms, ms / whole) for p, ms in sorted(totals.items(), key=lambda kv: -kv[1])]


def block_inputs(package, step):
    """The inputs of block_<step> as the export builds them: the outputs of its text, cross and pre packages, row 0 of
    the weight pack, the time and height-width RoPE rows, a random FP8 cache of the package's size."""
    import numpy as np
    import torch
    m = json.loads((package / "mvp1.json").read_text(encoding="utf-8"))
    dev, g = "cuda", torch.Generator("cuda").manual_seed(0)
    rows = np.memmap(m["pack"], dtype=np.uint8, mode="r").reshape(m["blocks"], m["row_bytes"])
    row = torch.from_numpy(np.array(rows[0])).to(dev)
    text = load(package, "text")(torch.randn(1, m["text_len"], m["text_dim"], device=dev, generator=g) * 0.1)
    text = text[0] if isinstance(text, (list, tuple)) else text
    ck, cv = load(package, "cross")(row, text)
    lat = m["latent"]
    X, e0, _ = load(package, "pre")(torch.randn(*lat, device=dev, generator=g), torch.tensor([999.0], device=dev))
    tp = m["time_pairs"]
    time_rope = torch.from_numpy(np.fromfile(package / "time_rope.bin", dtype=np.float32)).to(dev).view(2, -1, tp)
    time_rope = time_rope[:, 3 * m["chunk"]:4 * m["chunk"]].contiguous()
    hw = torch.randn(2, m["hw_rows"], m["hw_pairs"], device=dev, generator=g)
    n, mt, H, D = m["chunk_tokens"], m["max_tokens"], m["heads"], m["head_dim"]
    k = torch.randn(1, mt + n, H, D, device=dev, generator=g).to(torch.float8_e4m3fn)
    ks = torch.rand(1, mt + n, H, 1, device=dev, generator=g) * 0.05
    return m, (row, X, e0, ck, cv, time_rope, hw, k, ks, k.clone(), ks.clone())


def load(package, name):
    import torch
    from safetensors.torch import load_file
    runner = torch._inductor.aoti_load_package(str(package / f"{name}.pt2"))
    constants = {k: v.cuda() for k, v in load_file(str(package / f"{name}.safetensors")).items()}
    runner.load_constants(constants, check_full_update=True, user_managed=True)
    return runner


def profile(package, step=None, runs=5):
    """{block_ms, kernel_ms, parts} of block_<step> (the last, full window, by default)."""
    import torch
    from torch.profiler import ProfilerActivity, profile as torch_profile
    from wan_dit_lab.kernels import operators
    operators.load()
    package = Path(package)
    m, inputs = block_inputs(package, step)
    step = m["block_steps"] - 1 if step is None else step
    block = load(package, f"block_{step}")
    for _ in range(3):
        block(*inputs)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        block(*inputs)
    torch.cuda.synchronize()
    block_ms = (time.perf_counter() - start) / runs * 1000
    with torch_profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(runs):
            block(*inputs)
        torch.cuda.synchronize()
    kernels = [(e.key, e.device_time_total / runs / 1000) for e in p.key_averages()]
    return {"package": str(package), "step": step, "block_ms": block_ms,
            "kernels": sorted(((k, ms) for k, ms in kernels if part_of(k)), key=lambda r: -r[1]),
            "parts": parts(kernels)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("package")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--out", default=None, help="write the result as JSON here")
    args = parser.parse_args()
    result = profile(args.package, args.step, args.runs)
    print(f"block_{result['step']}: {result['block_ms']:.2f} ms")
    for part, ms, share in result["parts"]:
        print(f"  {ms:6.2f} ms  {share:5.1%}  {part}")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
