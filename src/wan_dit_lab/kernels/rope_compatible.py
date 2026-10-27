"""Experimental batch/padding-compatible RoPE; not enabled in the model.

Uses our cos/sin frequency layout and returns the input dtype. Not a drop-in
replacement for upstream Wan's complex-frequency, FP32-output interface.
"""

import torch


def rope(pos_count, pos_dim):
    pairs_count = pos_dim // 2
    positions = torch.arange(pos_count)[:, None]
    pair_indices = torch.arange(pairs_count, dtype=torch.float64)
    frequencies = 10000 ** (-2 * pair_indices / pos_dim)
    angles = positions * frequencies
    return torch.stack((torch.cos(angles), torch.sin(angles)))


def rope_apply(X, grid_sizes, freq, start_frame=0):
    # X = (batch, token, heads, dim); start_frame: the time position of the first frame of X (a chunk of a longer video)
    # Export can specialize Python sizes; tensor values would become data-dependent symbols.
    grids = grid_sizes.tolist() if isinstance(grid_sizes, torch.Tensor) else grid_sizes
    if len(grids) != X.shape[0]:
        raise ValueError("One grid is required per batch sample")
    f, h, w = grids[0]
    if f * h * w == X.shape[1] and all(tuple(grid) == (f, h, w) for grid in grids):
        return _rope_apply_grid(X, (f, h, w), freq, start_frame)
    output = []
    for i, grid in enumerate(grids):
        length = grid[0] * grid[1] * grid[2]
        if length > X.shape[1]:
            raise ValueError("Grid exceeds the padded sequence length")
        rotated = _rope_apply_grid(X[i:i + 1, :length], grid, freq, start_frame)
        output.append(torch.cat((rotated, X[i:i + 1, length:]), dim=1))
    return torch.cat(output, dim=0)


def _rope_apply_grid(X, grid, freq, start_frame):
    f, h, w = grid
    c = X.size(3) // 2
    sizes = [c - 2 * (c // 3), c // 3, c // 3]
    # freq = (2, position, pair): cos and sin on the first axis, built by rope()
    time, height, width = freq.split(sizes, dim=-1)
    # (2, token_pos, 1, 64 angles)
    C, S = torch.cat([
        time[:, start_frame:start_frame + f].view(2, f, 1, 1, -1).expand(2, f, h, w, -1),
        height[:, :h].view(2, 1, h, 1, -1).expand(2, f, h, w, -1),
        width[:, :w].view(2, 1, 1, w, -1).expand(2, f, h, w, -1),
    ], dim=-1).reshape(2, f * h * w, 1, -1)
    # (x1 + ix2)e^(i(pt)) = (x1 + ix2)(cos(pt) + isin(pt)
    # x1' = (x1cos(pt) - x2sin(pt))
    # x2' = (x2cos(pt) + x1sin(pt))
    # Match Wan's promotion before rotation, including backward accumulation.
    values = X.to(freq.dtype)
    X1 = values[:, :, :, ::2]
    X2 = values[:, :, :, 1::2]
    return torch.stack((X1 * C - X2 * S, X2 * C + X1 * S), dim=-1).flatten(-2).type_as(X)
