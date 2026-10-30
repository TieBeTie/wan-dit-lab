"""Measure achievable FP16 matmul TFLOPS and device copy bandwidth, print the ridge point."""
import torch


def timed(fn, iters):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000 / iters


n = 4096
a = torch.randn(n, n, device="cuda", dtype=torch.float16)
b = torch.randn(n, n, device="cuda", dtype=torch.float16)
t = timed(lambda: a @ b, 50)
tflops = 2 * n**3 / t / 1e12

x = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.uint8)
y = torch.empty_like(x)
t = timed(lambda: y.copy_(x), 50)
gbs = 2 * x.numel() / t / 1e9

print(torch.cuda.get_device_name())
print(f"FP16 matmul: {tflops:.1f} TFLOPS")
print(f"copy bandwidth: {gbs:.1f} GB/s")
print(f"ridge point: {tflops * 1e12 / (gbs * 1e9):.1f} FLOP/byte")
