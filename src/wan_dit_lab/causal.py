"""Self-Forcing: Wan 1.3B distilled to four steps without guidance, generating a video chunk by chunk.

Causal Forcing, the same model distilled otherwise, was tried and dropped on 2026-10-16: at 480x832 its video was
saturated and far from the look of Wan, Self-Forcing's close to it, at the same speed.

A chunk (3 latent frames) is denoised in four steps while it attends to itself and to the keys and values of the
frames before it, kept in a cache. Its own keys and values are noisy and change every step, so the cache is only read
then. Once the chunk is clean, one more pass with t = 0 writes its keys and values: from then on they never change.

The sampler is the one of Self-Forcing (the homework notebook): predict x0 from the flow, then noise x0 again to the
next level with fresh noise.
"""

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

SHIFT = 5.0  # timestep_shift of the distilled model
STEPS = (1000, 750, 500, 250)  # its four timesteps before the shift
CHUNK = 3  # latent frames generated together
CONTEXT = 21  # latent frames the model was trained to look back at


def sigmas(shift=SHIFT, steps=STEPS):
    """Noise levels of the four steps: the shifted flow-matching schedule of FlowMatchScheduler at these timesteps,
    sigma = shift u / (1 + (shift - 1) u) with u = t / 1000. For (1000, 750, 500, 250): 1, 0.9375, 0.8333, 0.625."""
    u = torch.tensor(steps, dtype=torch.float64) / 1000
    return (shift * u / (1 + (shift - 1) * u)).tolist()


class KVCache:
    """Keys and values of the finished frames for one block, in one buffer allocated once:
    [B, max_tokens + chunk tokens, heads, head_dim], keys with RoPE.

    Attention does not depend on the order of the keys (their positions are in them, by RoPE), so the buffer is a
    ring. While it fills, the chunk is written right after the saved frames, and saving it is moving a counter. Once
    full, the chunk is written into the last part, and saving it copies it over the oldest chunk, past the first
    sink_tokens which never leave (sink_size of causal_model.py). Attention takes a view of the buffer, no copy;
    once the buffer is full its shape and address are the same on every pass, and a CUDA graph can replay it.

    attend writes the chunk and attends; commit, after the pass with t = 0, keeps it."""

    def __init__(self, max_tokens, sink_tokens=0, sage=False):
        self.sage = sage  # SageAttention (Triton): Q and K in INT8, 1.4-1.5 times faster here than cuDNN
        self.k = self.v = None
        self.max_tokens = max_tokens
        self.sink_tokens = sink_tokens
        self.saved = 0  # tokens of finished frames in the buffer
        self.oldest = 0  # the ring slot to overwrite next, in chunks past the sink

    def full(self):
        return self.k is not None and self.saved == self.max_tokens

    # Outside a compiled graph: the state of the cache changes every chunk, and inside a graph each change is a
    # recompile; and there SDPA ignores the kernel chosen by sdpa_kernel in generate and takes the slower one.
    @torch.compiler.disable
    def attend(self, Q, K, V):
        """softmax(Q K^T / sqrt(d)) V over the saved frames and the chunk; all [B, tokens, heads, head_dim]."""
        self.n = n = K.shape[1]
        if self.k is None:
            self.k = V.new_empty(V.shape[0], self.max_tokens + n, *V.shape[2:])
            self.v = torch.empty_like(self.k)
        at = self.saved  # where the chunk goes: right after the saved frames, the last part once they fill the ring
        self.k[:, at:at + n] = K  # RMSNorm returns float32 keys: stored in the dtype of the values
        self.v[:, at:at + n] = V
        q, k, v = Q.permute(0, 2, 1, 3), self.k[:, :at + n].permute(0, 2, 1, 3), self.v[:, :at + n].permute(0, 2, 1, 3)
        if self.sage:
            from sageattention import sageattn  # only for this option: a package outside the requirements
            A = sageattn(q.to(v.dtype), k, v, tensor_layout="HND", is_causal=False)  # autocast does not cast for it
        else:
            A = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return A.permute(0, 2, 1, 3)

    def commit(self):
        """Keeps the chunk last written by attend."""
        n, at = self.n, self.saved
        if self.saved + n <= self.max_tokens:
            self.saved += n
        elif self.max_tokens > self.sink_tokens:  # else the window is the chunk alone: nothing to keep
            ring = self.sink_tokens + self.oldest * n
            self.k[:, ring:ring + n] = self.k[:, at:at + n]
            self.v[:, ring:ring + n] = self.v[:, at:at + n]
            self.oldest = (self.oldest + 1) % ((self.max_tokens - self.sink_tokens) // n)


class Pass:
    """One pass of the model over a chunk, replayed from a CUDA graph once the caches are full.

    Until then the model runs as it is. On the first pass with full caches the whole pass, every block with its
    attention, is recorded into a graph over fixed input tensors; from then on a pass is copying x and t into
    them and one replay, with no launch of a kernel from Python."""

    def __init__(self, model, context, grid, freqs, caches, dtype, cuda_graph):
        self.model, self.context, self.grid, self.freqs, self.caches = model, context, grid, freqs, caches
        self.dtype, self.cuda_graph = dtype, cuda_graph
        self.graph = None

    def run(self, x, t):
        with torch.autocast(x.device.type, dtype=self.dtype):
            return self.model(x[None], t, self.context, self.grid, self.freqs, caches=self.caches)[0]

    def __call__(self, x, t):
        if not (self.cuda_graph and self.caches[0].full()):
            return self.run(x, t)
        if self.graph is None:
            self.x, self.t = x.clone(), t.clone()
            side = torch.cuda.Stream()  # warm-up off the default stream, as CUDA graph capture requires
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self.run(self.x, self.t)
            torch.cuda.current_stream().wait_stream(side)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out = self.run(self.x, self.t)
        self.x.copy_(x)
        self.t.copy_(t)
        self.graph.replay()
        return self.out


ATTENTION = {"cudnn": [SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION],
             "efficient": [SDPBackend.EFFICIENT_ATTENTION],
             "sage": [SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]}  # SDPA only for cross-attention


@torch.no_grad()
def generate(model, noise, context, freqs, generator=None, dtype=torch.bfloat16, max_frames=CONTEXT, on_chunk=None,
             attention="cudnn", sink_frames=0, cuda_graph=False):
    """Latents [16, F, H, W] from noise of the same shape, F a multiple of CHUNK, for one prompt (context [1, L, 4096]).

    generator: the torch.Generator of the fresh noise between steps. max_frames: how many latent frames attention
    sees, the chunk included; the first sink_frames of them stay when the cache rolls. on_chunk(index, latents of
    the chunk) is called after every chunk. cuda_graph: replay the passes from a CUDA graph once the caches are
    full."""
    _, frames, height, width = noise.shape
    assert frames % CHUNK == 0, frames
    tokens = (height // 2) * (width // 2)  # of one latent frame, patch (1, 2, 2)
    # attention sees at most max_frames with the chunk itself, as in training (causal_model.py, max_attention_size);
    # a short video needs no buffer longer than itself
    window = min(max_frames, frames)
    caches = [KVCache((window - CHUNK) * tokens, sink_frames * tokens, attention == "sage") for _ in model.blocks]
    grid = ((CHUNK, height // 2, width // 2),)  # numbers, not a tensor: .tolist() in rope_apply breaks a compiled graph
    # The time position of the chunk is data, not a number: before each chunk the rows of its frames are copied into
    # the first rows of the time part of this table, and the model runs with start_frame 0. A number would be fixed
    # in a recorded graph (and recompiled for in a compiled one).
    chunk_freqs = freqs.clone()
    time = freqs.shape[-1] - 2 * (freqs.shape[-1] // 3)  # pairs of the time axis, as rope_apply splits them
    run = Pass(model, context, grid, chunk_freqs, caches, dtype, cuda_graph)
    levels = sigmas()
    out = torch.empty_like(noise)
    # cuDNN's fused attention, a flash-style kernel: 1.6-1.9 times faster here than the memory-efficient one that
    # SDPA picks by default; the Windows build of PyTorch has no flash kernel of its own
    with sdpa_kernel(ATTENTION[attention], set_priority=True):
        for start in range(0, frames, CHUNK):
            chunk_freqs[:, :CHUNK, :time] = freqs[:, start:start + CHUNK, :time]
            x = noise[:, start:start + CHUNK]
            for i, sigma in enumerate(levels):
                t = torch.full((1,), 1000 * sigma, device=noise.device)
                with torch.profiler.record_function(f"step {i + 1}"):
                    flow = run(x, t).float()
                x0 = x - sigma * flow  # x_t = (1 - sigma) x0 + sigma noise, flow = noise - x0
                if i + 1 < len(levels):
                    fresh = torch.randn(x0.shape, generator=generator, device=noise.device)
                    x = (1 - levels[i + 1]) * x0 + levels[i + 1] * fresh
            with torch.profiler.record_function("write cache"):
                run(x0, torch.zeros(1, device=noise.device))
            for cache in caches:
                cache.commit()
            out[:, start:start + CHUNK] = x0
            if on_chunk is not None:
                on_chunk(start // CHUNK, x0)
    return out
