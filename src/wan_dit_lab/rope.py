import torch


def rope(pos_count, pos_dim):
    pairs_count = pos_dim // 2
    positions = torch.arange(pos_count)[:, None]
    pair_indices = torch.arange(pairs_count, dtype=torch.float64)
    frequencies = 10000 ** (-2 * pair_indices / pos_dim)
    angles = positions * frequencies
    return torch.stack((torch.cos(angles), torch.sin(angles)))

def rope_apply(X, grid_sizes, freq):
    # X = (batch, token, heads, dim)
    f, h, w = grid_sizes[0].tolist()
    c = X.size(3) // 2
    sizes = [c - 2 * (c // 3), c // 3, c // 3]
    # freq = (2, position, pair): cos and sin on the first axis, built by rope()
    time, height, width = freq.split(sizes, dim=-1)
    # (2, token_pos, 1, 64 angles)
    C, S = torch.cat([
        time[:, :f].view(2, f, 1, 1, -1).expand(2, f, h, w, -1),
        height[:, :h].view(2, 1, h, 1, -1).expand(2, f, h, w, -1),
        width[:, :w].view(2, 1, 1, w, -1).expand(2, f, h, w, -1),
    ], dim=-1).reshape(2, f * h * w, 1, -1)
    # (x1 + ix2)e^(i(pt)) = (x1 + ix2)(cos(pt) + isin(pt)
    # x1' = (x1cos(pt) - x2sin(pt))
    # x2' = (x2cos(pt) + x1sin(pt))
    X1 = X[:, :, : , ::2]
    X2 = X[:, :, : , 1::2]
    return torch.stack((X1 * C - X2 * S, X2 * C + X1 * S), dim=-1).flatten(-2).type_as(X)
