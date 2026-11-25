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


def spatial_rope(count, dim, scale, trained, *, by_parts=None, ntk_aware=False, frequency_aware=False, ntk_pi_mix=None):
    """Spatial frequency profiles; shrinking-grid use remains an experimental adaptation. ntk_pi_mix (NTK only), mu in
    [0, 1]: pair j of n is scaled by scale ** (mu + (1 - mu) * j / (n - 1)), NTK at 0, Position Interpolation at 1."""
    import math
    if sum((by_parts is not None, bool(ntk_aware), bool(frequency_aware))) > 1:
        raise ValueError("choose one RoPE frequency profile")
    if ntk_pi_mix is not None and (not ntk_aware or not 0 <= ntk_pi_mix <= 1):
        raise ValueError("ntk_pi_mix is of the NTK profile and in [0, 1]")
    if dim <= 2 or dim % 2 or not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid spatial RoPE dimensions or scale")
    theta = 10000 ** (-2 * torch.arange(dim // 2, dtype=torch.float64) / dim)
    if ntk_aware:
        # Equivalent to b_new = b / scale ** (dim / (dim - 2)).
        mu = ntk_pi_mix or 0.0
        factors = scale ** (mu + (1 - mu) * torch.arange(dim // 2, dtype=torch.float64) / (dim // 2 - 1))
    elif frequency_aware:
        if not math.isfinite(trained) or trained <= 0:
            raise ValueError("frequency-aware RoPE requires a positive finite trained axis length")
        if trained <= 2 * math.pi:
            return rope_at(count, dim, scale)
        # Lumina-Next's anchor is one rotation over the trained axis, lambda = trained.
        # With theta_j = b**(-2*j/dim), its index need not be an integer.
        target_pair = dim / 2 * math.log(trained / (2 * math.pi), 10000)
        blend = (torch.arange(dim // 2, dtype=torch.float64) / target_pair).clamp(max=1)
        # M<1 clips from below by PI; M>1 reverses clipping for the shrinking-grid adaptation.
        factors = scale ** blend
    elif by_parts is not None:
        alpha, beta = by_parts["alpha"], by_parts["beta"]
        if not (math.isfinite(alpha) and math.isfinite(beta) and 0 < alpha < beta):
            raise ValueError("RoPE requires 0 < alpha < beta")
        gamma = ((trained * theta / (2 * math.pi) - alpha) / (beta - alpha)).clamp(0, 1)
        factors = (1 - gamma) * scale + gamma
    else:
        return rope_at(count, dim, scale)
    angles = torch.arange(count, dtype=torch.float64)[:, None] * theta * factors
    return torch.stack((torch.cos(angles), torch.sin(angles)))
