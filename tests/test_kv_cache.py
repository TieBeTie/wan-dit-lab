"""causal.KVCache on random keys, on the CPU: positions in the window against positions in the video and against
attention written out directly, Participative Compression, FP8 storage, Echo-Forcing decay."""

import pytest
import torch

from wan_dit_lab.generation.causal import CHUNK, KVCache
from wan_dit_lab.models.rope import rope, rope_apply

HEADS, D, H, W = 2, 128, 2, 2
T = H * W  # tokens of one latent frame
FREQS = torch.cat([rope(1024, D - 4 * (D // 6)), rope(1024, 2 * (D // 6)), rope(1024, 2 * (D // 6))], dim=-1)
PAIRS = FREQS.shape[-1] - 2 * (FREQS.shape[-1] // 3)


def rotate(X, start, frames=CHUNK):
    return rope_apply(X, ((frames, H, W),), FREQS, start).float()


def attention(Q, K, V):
    return torch.nn.functional.scaled_dot_product_attention(
        Q.permute(0, 2, 1, 3), K.permute(0, 2, 1, 3), V.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)


def cache(window, sink, **options):
    window_freqs = FREQS[:, :window + 1, :PAIRS]
    return KVCache((window - CHUNK) * T, sink * T, time_freqs=window_freqs, **options)


def chunks(count, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return [tuple(torch.randn(1, CHUNK * T, HEADS, D, generator=generator) for _ in range(3)) for _ in range(count)]


@pytest.mark.parametrize("window", [9, 12, 21])
def test_window_without_sink_is_video(window):
    """No sink: every frame moves by the same amount, distances stay, attention is the same."""
    video = KVCache((window - CHUNK) * T, 0)
    win = cache(window, 0)
    for c, (Q, K, V) in enumerate(chunks(12)):
        s, p = c * CHUNK, min(c * CHUNK, window - CHUNK)
        a = video.attend(rotate(Q, s), rotate(K, s), V)
        b = win.attend(rotate(Q, p), rotate(K, p), V)
        assert (a - b).abs().max() < 1e-5
        video.commit()
        win.commit()


@pytest.mark.parametrize("window, sink", [(9, 3), (12, 4), (21, 3), (21, 9), (21, 10), (21, 14), (21, 18)])
def test_window_with_sink_is_direct(window, sink):
    """With a sink: the first `sink` frames, then the newest frames in order, then the chunk, at consecutive places."""
    win = cache(window, sink)
    frames = []  # (unturned key, value) of every finished latent frame
    for c, (Q, K, V) in enumerate(chunks(12)):
        seen = frames[:sink] + frames[sink:][max(0, len(frames) - sink - (window - CHUNK - sink)):]
        seen = seen if len(frames) > window - CHUNK else frames
        p = len(seen)
        b = win.attend(rotate(Q, p), rotate(K, p), V)
        keys = torch.cat([rotate(k, i, frames=1) for i, (k, _) in enumerate(seen)] + [rotate(K, p)], dim=1)
        values = torch.cat([v for _, v in seen] + [V], dim=1)
        assert (b - attention(rotate(Q, p), keys, values)).abs().max() < 1e-5
        win.commit()
        frames += [(K[:, f * T:(f + 1) * T], V[:, f * T:(f + 1) * T]) for f in range(CHUNK)]


def test_participative_compression_keeps_sink_and_newest():
    window, sink, recent = 21, 10, 1
    win = cache(window, sink, pc_recent=recent)
    first = None
    for c, (Q, K, V) in enumerate(chunks(14)):
        p = min(c * CHUNK, window - CHUNK)
        win.attend(rotate(Q, p), rotate(K, p), V)
        win.commit()
        if first is None and win.full():
            first = win.k[:, :sink * T].clone()
        if win.full():
            assert torch.equal(win.k[:, :sink * T], first)  # the sink never moves
            newest = win.frame_of >= win.frames - recent - CHUNK  # the chunk just kept and the protected frames
            assert int(newest.sum()) == (recent + CHUNK) * T
            # one time position per frame's worth of ring tokens: the chunk lands right after them
            assert win.cos.shape[0] == (window - CHUNK - sink) * T


def test_participative_compression_evicts_lowest_scores():
    window, sink = 21, 10
    win = cache(window, sink, pc_recent=1)
    for c, (Q, K, V) in enumerate(chunks(8)):
        p = min(c * CHUNK, window - CHUNK)
        win.attend(rotate(Q, p), rotate(K, p), V)
        if win.scores is not None:
            candidate = win.frame_of < win.frames - 1
            scores = win.scores.clone()
            before = win.frame_of * T + win.place_in_frame
            win.commit()
            kept = win.frame_of * T + win.place_in_frame
            gone = candidate & (before != kept)
            stayed = candidate & (before == kept)
            assert int(gone.sum()) == CHUNK * T
            assert scores[gone].max() <= scores[stayed].min()
        else:
            win.commit()


def test_fp8_close_to_bfloat16():
    window, sink = 21, 10
    exact, fp8 = cache(window, sink), cache(window, sink, fp8=True)
    worst = 0
    for c, (Q, K, V) in enumerate(chunks(10)):
        p = min(c * CHUNK, window - CHUNK)
        a = exact.attend(rotate(Q, p), rotate(K, p), V)
        b = fp8.attend(rotate(Q, p), rotate(K, p), V)
        worst = max(worst, ((a - b).norm() / a.norm()).item())
        exact.commit()
        fp8.commit()
    assert worst < 0.05, worst


def test_decay_scales_old_keys_and_values():
    window, sink = 21, 10
    win = cache(window, sink)
    data = chunks(10)
    for c, (Q, K, V) in enumerate(data[:8]):
        p = min(c * CHUNK, window - CHUNK)
        win.attend(rotate(Q, p), rotate(K, p), V)
        win.commit()
    win.forget(0.1, 0.7)
    old = win.old.clone()
    Q, K, V = data[8]
    win.attend(rotate(Q, 18), rotate(K, 18), V)  # the reference chunk: no decay yet
    win.commit()
    assert win.r == 1 and win.mu is not None
    assert float(win.mu[old & (win.mu > 0)].min()) >= 0.1 - 1e-6 and float(win.mu.max()) <= 0.7 + 1e-6
    Q, K, V = data[9]
    k, v = win.finished(win.saved, torch.float32, True, False)
    w = torch.exp(-win.mu)[None, :, None, None]
    b = win.attend(rotate(Q, 18), rotate(K, 18), V)
    expected = attention(rotate(Q, 18), torch.cat([k * w, rotate(K, 18)], dim=1), torch.cat([v * w, V], dim=1))
    assert (b - expected).abs().max() < 1e-5
