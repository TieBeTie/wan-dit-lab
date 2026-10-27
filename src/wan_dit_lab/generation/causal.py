"""Self-Forcing: Wan 1.3B distilled to four steps without guidance, generating a video chunk by chunk.

Causal Forcing, the same model distilled otherwise, was tried and dropped on 2026-10-16: at 480x832 its video was
saturated and far from the look of Wan, Self-Forcing's close to it, at the same speed.

A chunk (3 latent frames) is denoised in four steps while it attends to itself and to the keys and values of the
frames before it, kept in a cache. Its own keys and values are noisy and change every step, so the cache is only read
then. Once the chunk is clean, one more pass with t = 0 writes its keys and values: from then on they never change.

The sampler is the one of Self-Forcing (the homework notebook): predict x0 from the flow, then noise x0 again to the
next level with fresh noise.

Beyond the 21 latent frames of training (5 s) the video drifts. What is here against it, all without training: a sink
of the first frames that never leaves the cache, with time positions counted in the attention window (Deep Forcing,
arXiv 2512.05081); Participative Compression, which evicts the tokens the new chunk looks at least instead of the
oldest frames (the same paper); and for a prompt that changes on the way, the frames in the cache encoded again under
the new prompt (KV recache of LongLive, arXiv 2509.22622), the old ones forgotten softly (difference-aware memory
decay of Echo-Forcing, arXiv 2605.16003), or the cross-attention keys and values of the two prompts blended.
"""

import math
from collections import deque
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from wan_dit_lab.offloading.resource_store import (
    StoredSequence,
    bind_tensor_attributes,
    resident_method,
)
from wan_dit_lab.quantization.fp8 import FP8, quantize_rows

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

    The first sink_tokens are the sink: the first frames of the video, never evicted (sink_size of causal_model.py,
    Deep Sink of Deep Forcing). The rest up to max_tokens is the ring. Attention does not depend on the order of the
    keys (their positions are in them, by RoPE), so a token can sit anywhere in the ring; for each one the cache keeps
    the frame it belongs to, its place in that frame and the time position its key was turned to when written. While
    the ring fills, the chunk is written right after the saved frames, and saving it is moving a counter. Once full,
    the chunk is written into the last part, and saving it copies it over the tokens it evicts: those of the 3 oldest
    frames, or with Participative Compression (pc_recent) the ones the chunk's queries scored lowest, never from the
    pc_recent newest frames.

    attend writes the chunk and attends; commit, after the pass with t = 0, keeps it.

    time_freqs (cos and sin of the time pairs, [2, positions, pairs]) turns on positions in the window: a frame's time
    position is its place among the frames attention sees, not in the video, so the sink stays at 0, the ring follows
    in the order of the video and the chunk is last, at most CONTEXT - 1 from the sink, as in training. Ring tokens
    are counted off in that order, one time position per frame's worth of tokens: after compression the survivors of
    several frames share one, as in Deep Forcing. A key keeps the time position it was written at; before every
    attention the time pairs of the ring are turned once, from the stored keys into a copy, by the difference to their
    place now. The stored keys are never turned again: rounding does not add up over the chunks a frame stays.

    fp8: keys and values stored in FP8 (E4M3) with a scale per token and head, half the memory of bfloat16; they are
    read back into a bfloat16 copy for attention, and the chunk itself attends to its own keys unquantized. With sage
    (and no turn, forgetting or Participative Compression) SageAttention's own quantization reads them, no copy.

    forget (Echo-Forcing): after a prompt switch the keys and values already in the cache are scaled down by
    exp(-r mu), r the chunks since the first chunk of the new scene, mu larger for the tokens that differ more from
    that chunk at the same place."""

    def __init__(self, max_tokens, sink_tokens=0, sage=False, time_freqs=None, fp8=False, pc_recent=None, chunk=CHUNK,
                 entropy_ratio=None):
        self.chunk = chunk  # latent frames of a chunk
        # tokens of a frame at the trained resolution over those now (Jin et al., NeurIPS 2023, Eq. 7): with N keys,
        # T = N * entropy_ratio of the same attention in training, Q K^T is scaled by sqrt(log_T N / d) instead of
        # 1 / sqrt(d), so that the entropy of attention stays that of training; None: 1 / sqrt(d), as trained
        self.entropy_ratio = entropy_ratio
        self.sage = sage  # SageAttention (Triton): Q and K in INT8, 1.4-1.5 times faster here than cuDNN
        self.fp8 = fp8
        self.max_tokens = max_tokens
        self.sink_tokens = sink_tokens
        self.time_freqs = time_freqs
        self.pc_recent = pc_recent  # None: evict the oldest frames; else the newest frames Participative Compression keeps
        # Forcing-KV (Ji et al. 2026): the static heads, those that look almost only at the chunk and the frame just
        # before it, attend to the sink, that frame and the chunk; the other heads to the whole window (set_heads)
        self.static_heads = None
        self.k = self.v = None
        self.reset()

    def set_heads(self, static, heads):
        """static: the head numbers (of `heads`) that attend to the sink, the newest finished frame and the chunk
        only, once the cache is full; None or empty: all heads see the whole window."""
        self.static_heads = sorted(static) if static else None
        self.dynamic_heads = [h for h in range(heads) if h not in (static or ())]

    @resident_method
    def reset(self):
        """Empties the cache; the buffers stay where they are."""
        self.saved = 0  # tokens of finished frames in the buffer
        self.frames = 0  # latent frames committed so far: the number of the next chunk's first frame
        self.scores = None  # Participative Compression: score of each ring token by the chunk's queries
        self.old = self.mu = None  # forget: tokens from before the prompt switch, their forgetting strengths
        self.r = 0  # forget: chunks since the first chunk of the new scene
        self.commit_after_attention = self.committed_in_attention = False
        if getattr(self, "pinned", None) is not None:
            self.pinned.zero_()

    @resident_method
    def keep_sink(self, frames):
        """Empties the ring and keeps the sink; the next commit fills the ring from latent frame `frames` on."""
        self.saved = min(self.saved, self.sink_tokens)
        self.frames = frames
        self.scores = None
        self.old = self.mu = None
        self.r = 0
        self.commit_after_attention = self.committed_in_attention = False

    @resident_method
    def pin_last(self):
        """LongLive 2.0's multi-shot sink (_pin_current_chunk): the chunk just committed, the first of a new shot,
        stays in the ring until the next shot; the shot pinned before is released to be evicted as usual."""
        self.pinned.zero_()
        if self.last_slots is not None:
            assert int(self.last_slots.numel()) < len(self.pinned), "the ring must hold more than the pinned chunk"
            self.pinned[self.last_slots] = True

    def full(self):
        return hasattr(self, "n") and self.saved == self.max_tokens

    def scale(self, keys, head_dim):
        """Of Q K^T over `keys` keys: None (1 / sqrt(d)) or sqrt(log_T N / d), T = keys * entropy_ratio."""
        if self.entropy_ratio is None:
            return None
        return math.sqrt(math.log(keys) / math.log(keys * self.entropy_ratio) / head_dim)

    def configure_offloading(self, store, name, device, tier):
        if getattr(self, "_residency", None) is not None:
            raise RuntimeError("Cache offloading is already configured")
        self._residency = bind_tensor_attributes(self, store, name, device, tier)

    def allocate(self, V):
        B, n, heads, d = V.shape
        binding = getattr(self, "_residency", None)
        if binding is not None and binding.tier == "auto":
            # Conservative persistent KV, scales, indices, rotation and score buffers.
            element = 1 if self.fp8 else V.element_size()
            size = 2 * B * (self.max_tokens + n) * heads * (d * element + (4 if self.fp8 else 0))
            size += (self.max_tokens - self.sink_tokens) * (24 + 4 * d) + self.max_tokens * 16
            binding.store.reserve_gpu(binding, size)
        self.n, self.per_frame = n, n // self.chunk
        self.k = torch.empty(B, self.max_tokens + n, heads, d, dtype=FP8 if self.fp8 else V.dtype, device=V.device)
        self.v = torch.empty_like(self.k)
        if self.fp8:
            self.ks = torch.empty(B, self.max_tokens + n, heads, 1, device=V.device)
            self.vs = torch.empty_like(self.ks)
        ring = self.max_tokens - self.sink_tokens
        self.frame_of = torch.zeros(ring, dtype=torch.long, device=V.device)  # frame number of each ring token
        self.place_in_frame = torch.zeros_like(self.frame_of)  # its index among the tokens of its frame
        self.written = torch.zeros_like(self.frame_of)  # time position its key was turned to when written
        c = d // 2
        self.pairs = c - 2 * (c // 3)  # pairs of the time axis, as rope_apply splits a head
        # fixed tensors, so that a recorded CUDA graph reads the new turns
        self.cos = torch.ones(ring, self.pairs, device=V.device)
        self.sin = torch.zeros_like(self.cos)
        # LongLive 2.0's multi-shot sink: the ring tokens of the first chunk of the current shot, never evicted
        self.pinned = torch.zeros(ring, dtype=torch.bool, device=V.device)
        self.last_slots = None  # ring indices the last committed chunk went to

    def store(self, at, K, V):
        n = K.shape[1]
        if self.fp8:
            self.k[:, at:at + n], self.ks[:, at:at + n] = quantize_rows(K)
            self.v[:, at:at + n], self.vs[:, at:at + n] = quantize_rows(V)
        else:
            self.k[:, at:at + n] = K  # RMSNorm returns float32 keys: stored in the dtype of the values
            self.v[:, at:at + n] = V

    def keys(self, start, end, dtype):
        """Stored keys [start:end] in `dtype` (read back from FP8)."""
        k = self.k[:, start:end]
        return k.to(dtype) * self.ks[:, start:end].to(dtype) if self.fp8 else k.to(dtype)

    # Outside a compiled graph: the state of the cache changes every chunk, and inside a graph each change is a
    # recompile; and there SDPA ignores the kernel chosen by sdpa_kernel in generate and takes the slower one.
    @torch.compiler.disable
    @resident_method
    def attend(self, Q, K, V):
        """softmax(Q K^T / sqrt(d)) V over the saved frames and the chunk; all [B, tokens, heads, head_dim]."""
        if self.k is None:
            self.allocate(V)
        n, at = K.shape[1], self.saved  # where the chunk goes: after the saved frames, the last part once they fill
        self.store(at, K, V)
        turn = self.time_freqs is not None and self.full() and self.max_tokens > self.sink_tokens  # else no ring
        decay = self.mu is not None and self.r > 0
        if self.sage and self.fp8 and not (turn or decay) and self.pc_recent is None:
            # Sage's quantization read straight from the FP8 cache, no bfloat16 copy of it (sage_kv): 32 ms instead
            # of 43 for the attention of a block at 704x1280; the chunk attends to its own keys unquantized
            from wan_dit_lab.kernels.sage_kv import attend
            if self.static_heads is not None and self.full():
                A = self.attend_split(Q, K, V, at, n)
            else:
                A = attend(Q, self.k, self.ks, self.v, self.vs, at, K, V, sm_scale=self.scale(at + n, Q.shape[-1]))
        else:
            A = self.attend_unpacked(Q, K, V, at, n, turn, decay)
        if self.commit_after_attention:
            # Final write only. Attention has already consumed these K/V on the
            # same CUDA stream. Update them before the existing lease evicts
            # them, without extending residency across the block's FFN.
            self.commit_after_attention = False
            self.commit()
            self.committed_in_attention = True
        return A

    def attend_split(self, Q, K, V, at, n):
        """Forcing-KV over the FP8 cache: the dynamic heads over the window, the static ones over the sink and the
        newest finished frame (anchor_tokens, set by commit); both read the one cache, nothing copied but Q, K, V."""
        from wan_dit_lab.kernels.sage_kv import attend
        if not hasattr(self, "anchor_tokens"):
            self.static_idx = torch.tensor(self.static_heads, device=Q.device, dtype=torch.int32)
            self.dynamic_idx = torch.tensor(self.dynamic_heads, device=Q.device, dtype=torch.int32)
            self.anchor_tokens = torch.zeros(self.sink_tokens + self.per_frame, device=Q.device, dtype=torch.int32)
            self.anchor_tokens[:self.sink_tokens] = torch.arange(self.sink_tokens, device=Q.device, dtype=torch.int32)
            self.place_anchor()
        A = torch.empty(*Q.shape[:2], Q.shape[2], Q.shape[3], device=Q.device, dtype=torch.bfloat16)
        for idx, tokens, length in ((self.dynamic_idx, None, at), (self.static_idx, self.anchor_tokens,
                                                                  self.anchor_tokens.numel())):
            if idx.numel() == 0:
                continue
            i = idx.long()
            A[:, :, i] = attend(Q[:, :, i].contiguous(), self.k, self.ks, self.v, self.vs, length, K[:, :, i].contiguous(),
                                V[:, :, i].contiguous(), sm_scale=self.scale(length + n, Q.shape[-1]), heads=idx,
                                tokens=tokens)
        return A

    def place_anchor(self):
        """The buffer tokens of the newest finished frame (the largest frame and place of the ring) into
        anchor_tokens after the sink, in place, so that a recorded CUDA graph reads them."""
        order = self.frame_of * self.per_frame + self.place_in_frame
        newest = torch.topk(order, self.per_frame).indices
        self.anchor_tokens[self.sink_tokens:] = (self.sink_tokens + newest).to(torch.int32)

    def attend_unpacked(self, Q, K, V, at, n, turn, decay):
        """attend over keys and values in 16 bits: the buffer itself, or the copy of finished (FP8, turn, decay)."""
        if not (self.fp8 or turn or decay):
            k, v = self.k[:, :at + n], self.v[:, :at + n]  # views of the buffer, no copy
        else:
            k, v = self.finished(at, V.dtype, turn, decay, extra=n)
            k[:, at:] = K
            if v.data_ptr() != self.v.data_ptr():  # a copy: the chunk goes in at its end (else it is already there)
                v[:, at:] = V
        if self.pc_recent is not None and self.full() and self.max_tokens > self.sink_tokens and self.scores is None:
            # Participative Compression scores, once per chunk at its first step (Deep Forcing, Section 4.3): the
            # sum of the chunk's queries against every ring key, over all heads, as their fusion "sum"
            ring = k[:, self.sink_tokens:self.max_tokens]
            self.scores = torch.einsum("bhd,brhd->r", Q.sum(dim=1).to(ring.dtype), ring).float()
        scale = self.scale(k.shape[1], Q.shape[-1])
        q, k, v = Q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
        if self.sage:
            # Only for this option: a package outside the requirements.
            from sageattention import sageattn
            A = sageattn(q.to(v.dtype), k, v, tensor_layout="HND", is_causal=False, sm_scale=scale)  # autocast does
            # not cast for it
        else:
            A = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)
        return A.permute(0, 2, 1, 3)

    def finished(self, at, dtype, turn, decay, extra=0):
        """Keys and values of the finished frames, [:at] of tensors with `extra` tokens more left for the chunk: the
        time pairs of the ring turned to the places of their frames now (turn), scaled by the forgetting weights
        (decay). One copy each, filled in place; without FP8 and decay the values are the buffer itself."""
        k = torch.empty(self.k.shape[0], at + extra, *self.k.shape[2:], dtype=dtype, device=self.k.device)
        k[:, :at] = self.keys(0, at, dtype)
        if self.fp8 or decay:
            v = torch.empty_like(k)
            v[:, :at] = self.v[:, :at].to(dtype) * self.vs[:, :at].to(dtype) if self.fp8 else self.v[:, :at]
        else:
            v = self.v[:, :at + extra]
        if turn:  # full: at is max_tokens
            sink, pairs = self.sink_tokens, 2 * self.pairs
            x1, x2 = k[:, sink:at, :, :pairs:2].float(), k[:, sink:at, :, 1:pairs:2].float()
            c, s = self.cos[None, :, None], self.sin[None, :, None]  # [1, ring token, 1, pair]
            k[:, sink:at, :, :pairs] = torch.stack((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1).flatten(-2)
        if decay:
            w = torch.exp(-self.r * self.mu[:at]).to(dtype)[None, :, None, None]
            k[:, :at] *= w
            v[:, :at] *= w
        return k, v

    @resident_method
    def commit(self):
        """Keeps the chunk last written by attend."""
        n, at, per, sink = self.n, self.saved, self.per_frame, self.sink_tokens
        i = torch.arange(n, device=self.k.device)  # token of the chunk
        if self.old is not None and self.mu is None:
            self.reference(at)
        slots = None  # ring indices the chunk's tokens went to
        if at + n <= self.max_tokens:  # filling: the chunk stays where it was written
            self.saved += n
            o = torch.arange(max(at, sink), max(at + n, sink), device=self.k.device)  # its tokens past the sink
            slots, i = o - sink, o - at
        elif self.max_tokens > sink:  # else the window is the chunk alone: nothing to keep
            slots = self.evicted(n)
            self.copy(sink + slots, at, n)
            if self.mu is not None:
                self.mu[sink + slots] = 0
                self.old[sink + slots] = False
        self.last_slots = slots
        if slots is not None and len(slots):
            self.frame_of[slots] = self.frames + i // per
            self.place_in_frame[slots] = i % per
            self.written[slots] = at // per + i // per
        self.frames += self.chunk
        self.scores = None
        if self.mu is not None:
            self.r += 1
        if self.time_freqs is not None and self.full() and self.max_tokens > sink:
            self.place()
        if self.static_heads is not None and hasattr(self, "anchor_tokens") and self.max_tokens > sink:
            self.place_anchor()

    def evicted(self, n):
        """Ring indices of the n tokens the chunk replaces."""
        order = self.frame_of * self.per_frame + self.place_in_frame  # order of the ring tokens in the video
        order = order.masked_fill(self.pinned, torch.iinfo(order.dtype).max)  # the pinned shot sink stays
        if self.pc_recent is not None and self.scores is not None:
            candidate = (self.frame_of < self.frames - self.pc_recent) & ~self.pinned  # not the newest, not pinned
            if int(candidate.sum()) >= n:
                return torch.topk(self.scores.masked_fill(~candidate, float("inf")), n, largest=False).indices
        return torch.topk(order, n, largest=False).indices  # the oldest frames

    def copy(self, to, at, n):
        """Tokens [at:at + n] of the buffers to the indices `to`; FP8 is copied as bytes (index_put has no FP8)."""
        for buffer in (self.k, self.v, self.ks, self.vs) if self.fp8 else (self.k, self.v):
            raw = buffer.view(torch.uint8) if buffer.dtype == FP8 else buffer
            raw[:, to] = raw[:, at:at + n].clone()  # the same buffer: index_put will not read what it writes

    def place(self):
        """The turns of the ring tokens to their places now: counted off after the sink in the order of the video,
        one time position per frame's worth of tokens."""
        order = self.frame_of * self.per_frame + self.place_in_frame
        rank = torch.empty_like(order)
        rank[order.argsort()] = torch.arange(len(order), device=order.device)
        turn = self.sink_tokens // self.per_frame + rank // self.per_frame - self.written
        self.cos.copy_(self.time_freqs[0, turn.abs()])
        self.sin.copy_(self.time_freqs[1, turn.abs()] * turn.sign()[:, None])

    @resident_method
    def forget(self, mu_min, mu_max):
        """Echo-Forcing's difference-aware memory decay from the next chunk on, which is the reference block of the
        new scene: every token now in the cache becomes old memory."""
        if self.k is None:  # nothing generated yet: nothing to forget
            return
        self.old = torch.zeros(self.max_tokens, dtype=torch.bool, device=self.k.device)
        self.old[:self.saved] = True
        self.mu, self.r, self.mu_range = None, 0, (mu_min, mu_max)

    def reference(self, at):
        """Forgetting strengths from the chunk just written, the first of the new scene (Echo-Forcing, Eq. 19-21):
        d = 1 - cos between an old key and the key at the same place in the chunk's last frame, normalized over the
        old tokens, mapped linearly onto [mu_min, mu_max]. Only the height and width pairs are compared: they are
        turned alike for the same place, so the cosine sees the content and not the time between the two."""
        per, pairs = self.per_frame, 2 * self.pairs
        reference = self.keys(at + self.n - per, at + self.n, torch.float32)[..., pairs:]  # [B, per, heads, ...]
        old = self.keys(0, self.max_tokens, torch.float32)[..., pairs:]
        place = torch.cat([torch.arange(self.sink_tokens, device=old.device) % per, self.place_in_frame])
        d = 1 - torch.nn.functional.cosine_similarity(old.flatten(2), reference[:, place].flatten(2), dim=-1).mean(0)
        lo, hi = d[self.old].min(), d[self.old].max()
        mu_min, mu_max = self.mu_range
        self.mu = torch.where(self.old, mu_min + (mu_max - mu_min) * (d - lo) / (hi - lo + 1e-6), torch.zeros_like(d))
        self.r = 0


class Pass:
    """One pass of the model over a chunk, replayed from a CUDA graph once the caches are full.

    Until then the model runs as it is. On the first pass with full caches the whole pass, every block with its
    attention, is recorded into a graph over fixed input tensors; from then on a pass is copying x and t into
    them and one replay, with no launch of a kernel from Python."""

    def __init__(self, model, context, grid, freqs, caches, dtype, cuda_graph, context_kvs=None):
        self.model, self.context, self.grid, self.freqs, self.caches = model, context, grid, freqs, caches
        self.dtype, self.cuda_graph, self.context_kvs = dtype, cuda_graph, context_kvs
        self.graph = None

    def run(self, x, t, commit_next=False):
        with torch.autocast(x.device.type, dtype=self.dtype):
            return self.model(x[None], t, self.context, self.grid, self.freqs, caches=self.caches,
                              context_kvs=self.context_kvs, commit_next=commit_next)[0]

    def __call__(self, x, t, commit_next=False):
        if not (self.cuda_graph and self.caches[0].full()):
            return self.run(x, t, commit_next=commit_next)
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
SWITCHES = ("swap", "recache", "blend", "decay", "reanchor", "recache_ring", "shot", "new")  # ways to change the prompt, joined by "+"


class Stream:
    """A video from the model, chunk by chunk, as long as it is asked for; the prompt can change between chunks.

    Holds the caches of every block and the position of the next chunk. next_chunk denoises it in four steps,
    writes its keys and values and returns its latents. set_prompt replaces the prompt in place, so that a recorded
    CUDA graph, which reads the same tensor, sees the new one, in one of the ways of SWITCHES:
    swap: the prompt alone; the frames already in the cache stay, and the video goes on from them towards it.
    recache: the frames in the cache are encoded again under the new prompt (KV recache of LongLive).
    blend: the cross-attention keys and values go from the old prompt to the new one linearly over blend_chunks
        chunks (needs cross_cache).
    decay: the old keys and values fade out, the faster the more they differ from the new scene (Echo-Forcing).
    reanchor: once the new scene has as many frames as the sink, they become the sink, and the cache is encoded
        again (needs positions "window").
    recache_ring: LongLive's KV recache with its global sink (interactive_causal_inference.py of LongLive1.0): the
        sink keeps the old prompt, the frames of the ring are encoded again under the new one at their places in
        the video (needs positions "video", as LongLive is trained).
    shot: LongLive 2.0's scene cut (pipeline/causal_diffusion_inference.py: multi-shot RoPE and sink): the frames
        of the new shot are placed shot_rope_offset frames further in time than the video so far, and its first
        chunk stays in the cache, pinned, until the next shot (needs positions "video"). The prompt of a shot
        begins with "The scene transitions. ", as LongLive 2.0 detects a cut.
    new: a new video under the new prompt: the caches emptied, the frames counted from 0 again; nothing of the old
        scene is seen any more.

    cross_cache: the keys and values of cross-attention over the prompt are computed once per prompt, not on every
    pass of every block."""

    def __init__(self, model, context, freqs, height, width, generator=None, dtype=torch.bfloat16,  # height, width: pixels
                 max_frames=CONTEXT, sink_frames=0, attention="cudnn", cuda_graph=False, frames=None, positions="video",
                 context_noise=0, kv_fp8=False, pc_recent=None, cross_cache=False, write_pass=True, shift=SHIFT,
                 resource_store=None, state_tier="vram", cross_tier="vram", history_tier="vram",
                 commit_in_attention=False, chunk=CHUNK, vae_stride=8, sampler="dmd", shot_rope_offset=8,
                 steps=None, attention_entropy_ratio=None, static_heads=None):
        # chunk: latent frames generated together (3 for Wan2.1 Self-Forcing and LongLive-1.3B, 8 for LongLive 2.0);
        # vae_stride: pixels per latent cell (8 for the Wan2.1 VAE, 16 for Wan2.2)
        self.chunk = chunk
        if commit_in_attention and (state_tier != "auto" or not write_pass):
            raise ValueError("Commit in attention requires managed KV and a final write pass")
        self.commit_in_attention = commit_in_attention
        if any(tier not in ("vram", "ram", "disk", "auto")
               for tier in (state_tier, cross_tier, history_tier)):
            raise ValueError("Unknown resource placement")
        if resource_store is None and (state_tier, cross_tier, history_tier) != ("vram",) * 3:
            raise ValueError("Resource offloading requires a TensorStore")
        if cuda_graph and (state_tier, cross_tier, history_tier) != ("vram",) * 3:
            raise ValueError("Resource offloading changes addresses; CUDA graph capture is unsupported")
        self.resource_store = resource_store
        self.state_tier, self.cross_tier, self.history_tier = state_tier, cross_tier, history_tier
        h, w = height // vae_stride, width // vae_stride  # the latent grid
        self.tokens = (h // 2) * (w // 2)  # of one latent frame, patch (1, 2, 2)
        # attention sees at most max_frames with the chunk itself, as in training (causal_model.py,
        # max_attention_size); a video of known, shorter length needs no buffer longer than itself
        window = max_frames if frames is None else min(max_frames, frames)
        self.time = freqs.shape[-1] - 2 * (freqs.shape[-1] // 3)  # pairs of the time axis, as rope_apply splits them
        # positions "video": a frame's time position is its frame in the video, as in causal_model.py; "window": its
        # place among the frames attention sees (KVCache), so that no distance exceeds those of training
        assert positions in ("video", "window"), positions
        # the ring takes a whole chunk at a time: none at all (sink + chunk fill the window), or at least a chunk's worth
        assert sink_frames == window - self.chunk or 0 <= sink_frames <= window - 2 * self.chunk, (sink_frames, window)
        assert pc_recent is None or 0 <= pc_recent <= window - 2 * self.chunk - sink_frames, \
            "Participative Compression needs a chunk's worth of ring frames beyond the newest pc_recent"
        assert not (cuda_graph and pc_recent is not None), "the tokens it keeps change from chunk to chunk"
        self.positions, self.window, self.sink_frames = positions, window, sink_frames
        time_freqs = freqs[:, :window + 1, :self.time] if positions == "window" else None
        self.caches = [KVCache((window - self.chunk) * self.tokens, sink_frames * self.tokens, attention == "sage",
                               time_freqs, kv_fp8, pc_recent, self.chunk, attention_entropy_ratio)
                       for _ in model.blocks]
        if static_heads:  # Forcing-KV: {block: [static heads]}, from forcing_kv_heads.py
            assert attention == "sage" and kv_fp8 and positions == "video", "Forcing-KV is on Sage over the FP8 cache"
            heads = model.blocks[0].self_attn.num_heads
            for index, cache in enumerate(self.caches):
                cache.set_heads(static_heads.get(index, static_heads.get(str(index), [])), heads)
        if state_tier != "vram":
            for index, cache in enumerate(self.caches):
                cache.configure_offloading(resource_store, f"self_kv/{index}", context.device, state_tier)
            if state_tier == "auto":
                resource_store.register_cycle([cache._residency.name for cache in self.caches])
        grid = ((self.chunk, h // 2, w // 2),)  # numbers, not a tensor: .tolist() in rope_apply breaks a compiled graph
        # The time position of the chunk is data, not a number: before each chunk the rows of its frames are copied
        # into the first rows of the time part of this table, and the model runs with start_frame 0. A number would
        # be fixed in a recorded graph (and recompiled for in a compiled one).
        self.freqs, self.chunk_freqs = freqs, freqs.clone()
        self.model, self.dtype = model, dtype
        self.context = context.clone()
        if state_tier == "auto":
            resource_store.reserve_device(
                "stream/inputs", self.context.numel() * self.context.element_size()
                + self.chunk_freqs.numel() * self.chunk_freqs.element_size()
                + self.freqs.numel() * self.freqs.element_size(),
            )
        self.context_kvs = self.cross(self.context) if cross_cache else None
        if self.context_kvs is not None and cross_tier != "vram":
            self.context_kvs = StoredSequence(resource_store, "cross_kv", cross_tier, self.context_kvs)
            if cross_tier == "auto":
                resource_store.register_cycle(self.context_kvs.names)
                if self.context_kvs.names:
                    resource_store.prefetch_device(self.context_kvs.names[0])
        self.run = Pass(model, self.context, grid, self.chunk_freqs, self.caches, dtype, cuda_graph, self.context_kvs)
        self.shape = (model.out_dim, self.chunk, h, w)
        self.generator, self.attention, self.levels = generator, attention, sigmas(shift=shift)
        # sampler "dmd": Self-Forcing's, x0 then fresh noise at the next level (also LongLive-1.3B); "unipc":
        # LongLive 2.0's deterministic multistep solver over the same number of steps
        assert sampler in ("dmd", "unipc"), sampler
        # steps: of the UniPC sampler (2 for the 2-step LongLive 2.0); Self-Forcing's sampler keeps its STEPS
        assert steps is None or sampler == "unipc", "the dmd sampler has the fixed timesteps of STEPS"
        self.sampler, self.shift, self.steps = sampler, shift, steps or len(STEPS)
        self.shot_rope_offset, self.rope_offset, self.pin_next = shot_rope_offset, 0, False  # shots (set_prompt)
        self.start = 0  # latent frame of the next chunk
        self.context_noise = context_noise  # timestep the chunk is written to the cache at, sigma = context_noise / 1000
        # write_pass False: no pass with t = 0, the cache keeps the keys and values of the last step (of x at the
        # last noise level, not of the clean chunk): a fifth less work per chunk, outside what the model was trained on
        self.write_pass = write_pass
        # latents of the frames in the caches, to encode them again under a new prompt
        self.sink_latents, self.recent_latents = [], deque(maxlen=window - self.chunk - sink_frames)
        if history_tier != "vram":
            self.sink_latents = StoredSequence(resource_store, "history/sink", history_tier)
            self.recent_latents = StoredSequence(
                resource_store, "history/recent", history_tier,
                maxlen=window - self.chunk - sink_frames,
            )
        self.blend = None  # (old keys and values, new ones, chunks done, chunks in all)
        self.reanchor = None  # frames of the new scene so far

    def close(self):
        for cache in self.caches:
            if getattr(cache, "_residency", None) is not None:
                cache._residency.close()
        if isinstance(self.context_kvs, StoredSequence):
            self.context_kvs.close()
        self.sink_latents.clear()
        self.recent_latents.clear()
        if self.state_tier == "auto":
            self.resource_store.release_device("stream/inputs")

    @torch.no_grad()
    def cross(self, context):
        """Keys and values of cross-attention over the prompt, one pair per block: they depend on the prompt alone."""
        with torch.autocast(context.device.type, dtype=self.dtype):
            text = self.model.text_embedding(context)
            kvs = []
            for _ in self.model.blocks:  # one round of the blocks, as a pass takes them
                block = self.model.block_offloader.get_next()
                kv = block.cross_attn.keys_values(text)
                kvs.append(kv if self.cross_tier == "vram" else tuple(t.cpu() for t in kv))
                self.model.block_offloader.release(block)
        return kvs

    def set_context(self, context):
        self.set_prompt(context, "swap")

    def set_prompt(self, context, switch="swap", blend_chunks=4, mu=(0.1, 0.7)):
        """Goes on under the prompt `context` [1, L, 4096], changed in one of the ways of SWITCHES, joined by "+"."""
        ways = set(switch.split("+"))
        assert ways <= set(SWITCHES), switch
        if self.run.cuda_graph and ways - {"swap", "blend"} and self.run.graph is not None:
            # a recorded graph has the old cache state: recorded again once the cache is full. Its output lives in
            # the graph's own memory pool (0.4 GB at 512x928), freed only with the output: without it the passes
            # until the cache fills again run out of memory
            self.run.graph = self.run.out = None
            torch.cuda.empty_cache()
        if self.context_kvs is not None:
            new = self.cross(context)
            if "blend" in ways:
                self.blend = ([(k.clone(), v.clone()) for k, v in self.context_kvs], new, 0, blend_chunks)
            else:
                for index, ((k, v), (k_new, v_new)) in enumerate(zip(self.context_kvs, new)):
                    if isinstance(self.context_kvs, StoredSequence):
                        self.context_kvs[index] = (k_new, v_new)
                    else:
                        k.copy_(k_new)
                        v.copy_(v_new)
        else:
            assert "blend" not in ways, "blending the prompts needs cross_cache"
        self.context.copy_(context)
        if "recache" in ways:
            self.recache()
        if "decay" in ways:
            for cache in self.caches:
                cache.forget(*mu)
        if "reanchor" in ways:
            assert self.positions == "window" and self.sink_frames > 0, "reanchor moves the sink"
            self.reanchor = []
        if "recache_ring" in ways:
            self.recache_ring()
        if "new" in ways:
            for cache in self.caches:
                cache.reset()
                cache.last_slots = None
            self.start, self.rope_offset, self.pin_next = 0, 0, False
            self.sink_latents.clear()
            self.recent_latents.clear()
            self.blend = self.reanchor = None
        if "shot" in ways:
            assert self.positions == "video", "a shot moves the positions in the video"
            self.rope_offset += self.shot_rope_offset
            self.pin_next = True

    def unipc(self, x):
        """The clean chunk from noise x by LongLive 2.0's sampler (pipeline/causal_diffusion_inference.py, Step
        3.1): a fresh FlowUniPCMultistepScheduler per chunk, its steps over the shifted timesteps, deterministic."""
        global FlowUniPCMultistepScheduler
        if "FlowUniPCMultistepScheduler" not in globals():  # loaded by path: wan_5b/__init__ pulls the whole model
            import importlib.util
            path = (Path(__file__).resolve().parents[3] / "reference" / "LongLive" / "LongLive2.0" / "wan_5b" / "utils"
                    / "fm_solvers_unipc.py")
            spec = importlib.util.spec_from_file_location("fm_solvers_unipc", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            FlowUniPCMultistepScheduler = module.FlowUniPCMultistepScheduler
        scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
        scheduler.set_timesteps(self.steps, device=x.device, shift=self.shift)
        for i, t in enumerate(scheduler.timesteps):
            with torch.profiler.record_function(f"step {i + 1}"):
                flow = self.run(x, t.reshape(1).float()).float()
            x = scheduler.step(flow, t, x, return_dict=False)[0]
        return x

    @torch.no_grad()
    def recache_ring(self):
        """LongLive's recache with a global sink: the sink stays as it was written, under the old prompt; the frames
        of the ring are encoded again under the current prompt, a chunk's worth at a time with t = 0, at their frames
        in the video. LongLive encodes them in one pass; here chunk by chunk, each seeing the sink and those before."""
        assert self.positions == "video", "the frames are put at their places in the video, as LongLive is trained"
        frames = list(self.recent_latents)
        if not frames:
            return
        x = torch.stack(frames, dim=1).to(self.context.device)
        assert x.shape[1] % self.chunk == 0, x.shape
        first = self.start - x.shape[1]  # the frame in the video of the oldest ring frame
        for cache in self.caches:
            cache.keep_sink(first)
        with sdpa_kernel(ATTENTION[self.attention], set_priority=True):
            for g in range(0, x.shape[1], self.chunk):
                self.chunk_freqs[:, :self.chunk, :self.time] = self.freqs[:, first + g:first + g + self.chunk, :self.time]
                self.run(x[:, g:g + self.chunk], torch.zeros(1, device=x.device))
                for cache in self.caches:
                    cache.commit()

    @torch.no_grad()
    def recache(self):
        """LongLive's KV recache: the frames in the cache (sink, then the newest) are encoded again under the current
        prompt, a chunk's worth at a time with t = 0, at their places in the window, as they were written."""
        assert self.positions == "window", "re-encoded frames are put at their places in the window"
        frames = list(self.sink_latents) + list(self.recent_latents)
        if not frames:
            return
        x = torch.stack(frames, dim=1).to(self.context.device)
        assert x.shape[1] % self.chunk == 0, x.shape
        for cache in self.caches:
            cache.reset()
        with sdpa_kernel(ATTENTION[self.attention], set_priority=True):
            for g in range(0, x.shape[1], self.chunk):
                self.chunk_freqs[:, :self.chunk, :self.time] = self.freqs[:, g:g + self.chunk, :self.time]
                self.run(x[:, g:g + self.chunk], torch.zeros(1, device=x.device))
                for cache in self.caches:
                    cache.commit()

    @torch.no_grad()
    def next_chunk(self, noise=None):
        """Latents [C, chunk, H, W] of the next chunk, from `noise` of that shape or fresh noise."""
        device = self.context.device
        x = torch.randn(self.shape, generator=self.generator, device=device) if noise is None else noise
        if self.blend is not None:
            old, new, done, chunks = self.blend
            alpha = (done + 1) / chunks
            for index, ((k, v), (k_old, v_old), (k_new, v_new)) in enumerate(
                zip(self.context_kvs, old, new)
            ):
                if isinstance(self.context_kvs, StoredSequence):
                    # Placement must not move interpolation arithmetic from CUDA to CPU.
                    k_old, v_old, k_new, v_new = (
                        tensor.to(device) for tensor in (k_old, v_old, k_new, v_new)
                    )
                blended = (torch.lerp(k_old, k_new, alpha), torch.lerp(v_old, v_new, alpha))
                if isinstance(self.context_kvs, StoredSequence):
                    self.context_kvs[index] = blended
                else:
                    k.copy_(blended[0])
                    v.copy_(blended[1])
            self.blend = None if done + 1 >= chunks else (old, new, done + 1, chunks)
        # in the window the chunk is last: right after the frames the cache holds (all 21 - 3 once they fill it)
        s = self.start + self.rope_offset if self.positions == "video" else self.caches[0].saved // self.tokens
        self.chunk_freqs[:, :self.chunk, :self.time] = self.freqs[:, s:s + self.chunk, :self.time]
        # cuDNN's fused attention, a flash-style kernel: 1.6-1.9 times faster here than the memory-efficient one
        # that SDPA picks by default; the Windows build of PyTorch has no flash kernel of its own
        with sdpa_kernel(ATTENTION[self.attention], set_priority=True):
            if self.sampler == "unipc":
                x0 = self.unipc(x)
            for i, sigma in enumerate(self.levels if self.sampler == "dmd" else ()):
                t = torch.full((1,), 1000 * sigma, device=device)
                with torch.profiler.record_function(f"step {i + 1}"):
                    flow = self.run(x, t, commit_next=not self.write_pass and i + 1 == len(self.levels)).float()
                x0 = x - sigma * flow  # x_t = (1 - sigma) x0 + sigma noise, flow = noise - x0
                if i + 1 < len(self.levels):
                    fresh = torch.randn(x0.shape, generator=self.generator, device=device)
                    x = (1 - self.levels[i + 1]) * x0 + self.levels[i + 1] * fresh
            with torch.profiler.record_function("write cache"):
                # context noise (pipeline/self_forcing_training.py of Causal-Forcing): the chunk goes into the cache
                # noised to this timestep, so that the next chunks trust its details less; 0, as trained, by default
                sigma = self.context_noise / 1000
                for cache in self.caches:
                    cache.commit_after_attention = self.commit_in_attention
                if not self.write_pass:
                    pass  # attend of the last step has already written its keys and values where commit takes them
                elif sigma > 0:
                    fresh = torch.randn(x0.shape, generator=self.generator, device=device)
                    self.run((1 - sigma) * x0 + sigma * fresh, torch.full((1,), self.context_noise, device=device),
                             commit_next=not self.commit_in_attention)
                else:
                    self.run(x0, torch.zeros(1, device=device), commit_next=not self.commit_in_attention)
        for cache in self.caches:
            if not cache.committed_in_attention:
                cache.commit()
            cache.committed_in_attention = False
            if self.pin_next:  # the first chunk of a shot: LongLive 2.0 pins it after its write pass
                cache.pin_last()
        self.pin_next = False
        for f in range(self.chunk):
            frame = x0[:, f] if self.history_tier == "vram" else x0[:, f].cpu()
            (self.sink_latents if self.start + f < self.sink_frames else self.recent_latents).append(frame)
        self.start += self.chunk
        if self.reanchor is not None:
            self.reanchor.extend(x0[:, f] if self.history_tier == "vram" else x0[:, f].cpu()
                                 for f in range(self.chunk))
            if len(self.reanchor) >= self.sink_frames:  # a whole number of chunks: the sink, then the rest of them
                frames, self.reanchor = self.reanchor, None
                if isinstance(self.sink_latents, StoredSequence):
                    self.sink_latents.clear()
                    self.recent_latents.clear()
                    self.sink_latents.extend(frames[:self.sink_frames])
                    self.recent_latents.extend(frames[self.sink_frames:])
                else:
                    self.sink_latents = frames[:self.sink_frames]
                    self.recent_latents = deque(frames[self.sink_frames:], maxlen=self.recent_latents.maxlen)
                self.recache()
        return x0


@torch.no_grad()
def generate(model, noise, context, freqs, generator=None, dtype=torch.bfloat16, max_frames=CONTEXT, on_chunk=None,
             **options):
    """Latents [16, F, H, W] from noise of the same shape, F a multiple of CHUNK, for one prompt (context [1, L, 4096]).

    generator: the torch.Generator of the fresh noise between steps. max_frames: how many latent frames attention
    sees, the chunk included. on_chunk(index, latents of the chunk) is called after every chunk. options: those of
    Stream (attention, sink_frames, cuda_graph, positions, kv_fp8, pc_recent, cross_cache, ...)."""
    _, frames, height, width = noise.shape
    assert frames % CHUNK == 0, frames
    stream = Stream(model, context, freqs, height * 8, width * 8, generator, dtype, max_frames, frames=frames, **options)
    out = torch.empty_like(noise)
    for start in range(0, frames, CHUNK):
        out[:, start:start + CHUNK] = x0 = stream.next_chunk(noise[:, start:start + CHUNK])
        if on_chunk is not None:
            on_chunk(start // CHUNK, x0)
    return out
