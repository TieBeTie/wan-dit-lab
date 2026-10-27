"""RoPE tables at positions stretched off the integers: a token grid smaller than the one the model is trained on, its
height and width spread over the trained positions (Position Interpolation of height and width, E27, E28)."""

import torch


def rope_at(count, dim, scale):
    """rope.rope(count, dim) at the positions i * scale (rope itself at scale 1): cos and sin, [2, count, dim / 2]."""
    positions = torch.arange(count, dtype=torch.float64)[:, None] * scale
    angles = positions * 10000 ** (-2 * torch.arange(dim // 2, dtype=torch.float64) / dim)
    return torch.stack((torch.cos(angles), torch.sin(angles)))


def stretch(trained, now, strength=1.0):
    """Scales of height and width: (trained / now) ** strength per side, sizes in tokens; 1 where they are equal."""
    return [(t / n) ** strength for t, n in zip(trained, now)]
